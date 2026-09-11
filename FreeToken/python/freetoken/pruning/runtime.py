"""Opt-in eager collection; ordinary serving never installs an active observer."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct

import torch
import torch.nn.functional as F

from . import ADAPTER_VERSION, METRIC
from .artifacts import checkpoint_identity, read_json, validate_mask, write_new

_ACTIVE = ContextVar("expert_pruning_capture", default=None)


def active_capture():
    return _ACTIVE.get()


def apply_expert_mask(scores, mask):
    return scores if mask is None else scores.masked_fill(~mask, float("-inf"))


def record_weighted_outputs(down):
    capture = active_capture()
    if capture is not None:
        capture.record_outputs(down)


def validate_collection_config(config):
    if getattr(config, "expert_mask_path", None) and getattr(config, "expert_stats_path", None):
        raise ValueError("Collect importance from the unmasked model; mask and statistics modes are exclusive")
    if not getattr(config, "expert_stats_path", None):
        if getattr(config, "expert_stats_decode", False):
            raise ValueError("Live expert tracing requires --expert-stats")
        return
    if config.moe_strategy != "offload" or config.moe_cpu_layers:
        raise ValueError("Statistics collection requires --moe-strategy offload with no CPU expert layers")
    if config.max_running_req != 1 or getattr(config, "cache_type", None) != "naive":
        raise ValueError("Statistics collection requires --max-running-requests 1 --cache-type naive")
    if config.cuda_graph_max_bs != 0 or config.cuda_graph_bs:
        raise ValueError("Statistics collection requires --cuda-graph-max-bs 0 (eager execution)")
    if getattr(config, "expert_stats_decode", False) and os.environ.get("FREETOKEN_DISABLE_OVERLAP_SCHEDULING") != "1":
        raise ValueError("Live expert tracing requires FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1")


class BatchCapture:
    def __init__(self, geometry, tokens, trace=False):
        self.geometry = geometry
        self.tokens = tokens
        self.rows = []
        self.pending = None
        self.trace = trace
        self.trace_rows = []

    def begin_layer(self, layer, ids, weights):
        if self.pending is not None or layer != len(self.rows):
            raise RuntimeError("Expert statistics observed out-of-order layers")
        if tuple(ids.shape) != (self.tokens, self.geometry["n_activated_experts"]):
            raise RuntimeError("Unexpected routing shape during collection")
        self.pending = {"layer": layer, "ids": ids.clone(), "weights": weights.clone()}

    def record_outputs(self, down):
        p = self.pending
        if p is None or "norms" in p:
            raise RuntimeError("Expected exactly one expert-output observation per layer")
        if down.shape[:2] != p["ids"].shape:
            raise RuntimeError("Per-route outputs do not match logical expert IDs")
        # down is already gate-weighted. Multiplying by the gate again would square it.
        p["norms"] = torch.linalg.vector_norm(down.float(), dim=-1)

    def set_routed(self, routed):
        if self.pending is None:
            raise RuntimeError("No active routed layer")
        self.pending["routed"] = routed

    def routed_output(self, layer):
        if self.pending is None or self.pending["layer"] != layer:
            raise RuntimeError("Missing routed output for hyper-connection scoring")
        return self.pending["routed"]

    def finish_layer(self, layer, before, after_routed, post):
        p = self.pending
        if p is None or p["layer"] != layer or "norms" not in p:
            raise RuntimeError("Missing per-expert outputs; unsupported collection backend")
        if post.ndim != 2 or post.shape[0] != self.tokens or post.shape[1] == 0:
            raise RuntimeError("Unexpected hyper-connection post shape during collection")
        before = before.reshape(self.tokens, -1).float()
        after = after_routed.reshape(self.tokens, -1).float()
        sensitivity = (1 - F.cosine_similarity(before, after, dim=-1)).clamp(0, 2)
        # ||post outer weighted_output|| = ||post|| * ||weighted_output||.
        residual_norms = p["norms"] * torch.linalg.vector_norm(post.float(), dim=-1)[:, None]
        ids = p["ids"].reshape(-1).long()
        experts = self.geometry["n_routed_experts"]
        score = torch.zeros(experts, dtype=torch.float32, device=ids.device)
        score.scatter_add_(0, ids, (residual_norms * sensitivity[:, None]).reshape(-1))
        gate = torch.zeros_like(score)
        gate.scatter_add_(0, ids, p["weights"].float().reshape(-1))
        counts = torch.bincount(ids, minlength=experts)
        self.rows.append((score, gate, counts))
        if self.trace:
            self.trace_rows.append((p["ids"], p["weights"].float(), residual_norms, sensitivity))
        self.pending = None

    def result(self):
        if self.pending is not None or len(self.rows) != self.geometry["n_layers"]:
            raise RuntimeError("Incomplete layer coverage; discarding this batch")
        score, gate, counts = [torch.stack([row[i] for row in self.rows]).cpu() for i in range(3)]
        if not torch.isfinite(score).all() or not torch.isfinite(gate).all():
            raise RuntimeError("Non-finite expert statistics")
        return score, gate, counts

    def trace_document(self):
        if not self.trace or len(self.trace_rows) != self.geometry["n_layers"]:
            raise RuntimeError("Incomplete per-token expert trace")
        names = ("expert_ids", "routing_weights", "residual_weighted_output_norms", "sensitivity")
        return {name: torch.stack([row[i] for row in self.trace_rows]).cpu().tolist()
                for i, name in enumerate(names)}


class PruningSession:
    def __init__(self, config):
        validate_collection_config(config)
        if config.tp_info.size != 1:
            raise ValueError("DeepSeek-V4 pruning currently requires TP=1")
        self.identity = checkpoint_identity(config.model_path)
        self.mask = (validate_mask(read_json(config.expert_mask_path), self.identity)
                     if config.expert_mask_path else None)
        self.path = Path(config.expert_stats_path) if config.expert_stats_path else None
        self.live = bool(getattr(config, "expert_stats_decode", False))
        self.trace_path = self.path.with_suffix(".trace.jsonl") if self.path and self.live else None
        self.geometry = self.identity["geometry"]
        root = Path(__file__).resolve().parents[1]
        self.provenance = {"source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
                "pruning/runtime.py", "pruning/artifacts.py", "engine/engine.py",
                "models/deepseek_v4/moe.py", "models/deepseek_v4/model.py", "moe/fused_ds_fp4.py",
                "scheduler/scheduler.py",
            )}, "packages": {}}
        for name in ("torch", "triton", "transformers", "freetoken"):
            try:
                self.provenance["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                self.provenance["packages"][name] = None
        self.provenance["collection_settings"] = {
            key: getattr(config, key, None) for key in ("moe_strategy", "max_running_req", "cache_type",
                "cuda_graph_max_bs", "max_extend_tokens", "max_seq_len_override", "expert_stats_decode")
        }
        shape = (self.geometry["n_layers"], self.geometry["n_routed_experts"])
        self.scores = torch.zeros(shape, dtype=torch.float64, device="cpu")
        self.gates = torch.zeros_like(self.scores)
        self.counts = torch.zeros(shape, dtype=torch.int64, device="cpu")
        self.requests = {}
        self.hashers = {}
        self.total_tokens = 0
        self.batches = 0
        if self.path:
            write_new(self.path, self.document())
        if self.trace_path:
            self.trace_path.open("x").close()

    def attach(self, model):
        blocks = model.model.layers.op_list
        if len(blocks) != self.geometry["n_layers"]:
            raise ValueError("Pruning/model layer count mismatch")
        if self.mask is not None:
            for layer, block in enumerate(blocks):
                block.ffn.gate._expert_mask = torch.tensor(
                    self.mask[layer], dtype=torch.bool, device=block.ffn.gate.weight.device
                )

    @contextmanager
    def capture(self, batch):
        if self.path is None or (not batch.is_prefill and not self.live):
            yield
            return
        if len(batch.reqs) != 1 or batch.reqs[0].uid < 0:
            raise RuntimeError("Statistics mode accepts one real request per batch")
        uid = str(batch.reqs[0].uid)
        positions = batch.positions.reshape(-1).cpu().tolist()
        tokens = batch.input_ids.reshape(-1).cpu().tolist()
        start = self.requests.get(uid, {}).get("tokens", 0)
        if not tokens or positions != list(range(start, start + len(tokens))):
            raise RuntimeError("Missing/repeated tokens or reused prefix during statistics collection")
        if active_capture() is not None:
            raise RuntimeError("Nested statistics capture")
        capture = BatchCapture(self.geometry, len(tokens), trace=self.live)
        handle = _ACTIVE.set(capture)
        try:
            yield
            score, gate, counts = capture.result()
        finally:
            _ACTIVE.reset(handle)
        self.scores += score
        self.gates += gate
        self.counts += counts
        hasher = self.hashers.setdefault(uid, hashlib.sha256())
        hasher.update(struct.pack(f"<{len(tokens)}i", *tokens))
        request = self.requests.setdefault(uid, {})
        request.update(tokens=start + len(tokens), token_sha256=hasher.hexdigest())
        if self.live:
            phase = "prefill" if batch.is_prefill else "decode"
            request[phase + "_tokens"] = request.get(phase + "_tokens", 0) + len(tokens)
            event = {"uid": uid, "phase": phase, "start_position": start,
                     "token_ids": tokens, **capture.trace_document()}
            with self.trace_path.open("a") as handle:
                handle.write(json.dumps(event, allow_nan=False) + "\n")
        self.total_tokens += len(tokens)
        self.batches += 1
        if not self.live or batch.is_prefill or self.batches % 16 == 0:
            self.save()

    def finish_request(self, uid, token_ids, finish_reason):
        if not self.live or self.path is None:
            return
        request = self.requests[str(uid)]
        if request["tokens"] != len(token_ids) - 1:
            raise RuntimeError("Live trace must cover every forwarded token exactly once")
        digest = hashlib.sha256(struct.pack(f"<{len(token_ids)-1}i", *token_ids[:-1])).hexdigest()
        if digest != request["token_sha256"]:
            raise RuntimeError("Live trace token IDs differ from the completed request")
        request.update(finished=True, finish_reason=finish_reason,
                       generated_token_ids=token_ids[request["prefill_tokens"]:])
        self.save()

    def document(self):
        return {"adapter_version": ADAPTER_VERSION, "kind": "expert_statistics", "metric": METRIC,
                "checkpoint": self.identity,
                "collection": "live_prefill_decode" if self.live else "prefill_only_replay",
                "provenance": self.provenance,
                "scores": self.scores.tolist(), "gate_sums": self.gates.tolist(),
                "activation_counts": self.counts.tolist(), "requests": self.requests,
                "total_tokens": self.total_tokens, "batches": self.batches,
                "coverage_requires_replay_log": not self.live,
                "trace_file": self.trace_path.name if self.trace_path else None}

    def save(self):
        tmp = self.path.with_name(self.path.name + f".tmp-{os.getpid()}")
        try:
            with tmp.open("w") as handle:
                json.dump(self.document(), handle, allow_nan=False)
                handle.write("\n")
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)
