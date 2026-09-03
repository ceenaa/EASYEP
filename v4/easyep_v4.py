"""EASY-EP (arXiv:2504.06792) ported to DeepSeek-V4-Flash.

Faithful to the paper's score, which is a sum of per-token PRODUCTS
(expert_selection.py:38 in RUCAIBox/EASYEP):

    score[layer, e] += weight_t[e] * simibr_t * norm_t[e]

    weight   gating score for expert e on token t
    norm     ||post||_2 * ||unweighted expert output||_2
    simibr   max(1 - cos(Cr, Cr + post ⊗ y_routed), 0)

Three V4-specific adaptations, each forced by the architecture:

1. V4 folds the gating weight inside Expert.forward -- expert(x, w) computes
   w2(w * silu(gate)*up). Without activation quantisation this is exactly w times
   the unweighted output. FP4 can break that proportionality, so division is a
   recovered approximation that ``validate`` checks against explicit forwards.

2. V4's MoE returns routed+shared summed. simibr needs the routed part alone,
   so the routed sum is stashed on the module for Block to read.

3. Layers 0..n_hash_layers-1 route by token ID (Gate.tid2eid), not by content.
   Pruning them would disable specific vocabulary items outright, so they are
   protected and keep all experts.

Hyper-connections require scoring in the real hc_mult residual space. The primary
score compares the FP32 residual-mixing base against that base plus
``post ⊗ y_routed``, and scales each expert norm by ``||post||_2``. The old
single-vector approximation is retained only as ``score_reduced_legacy``.

Modes
-----
  profile   calibration forward passes -> per-layer/expert scores
  mask      scores -> keep/prune mask
  eval      generate answers with the mask applied at the gate (or without)
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import sys
import time
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import re

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_model
from transformers import AutoTokenizer

EPS = 1e-6
SCORE_SCHEMA_VERSION = 2
SCORE_SEMANTICS = "easyep-mhc-residual-generated-response-v2"
ACCUMULATION_MODE = "torch-deterministic-index-add-v1"
MODEL_IDENTITY_SCHEMA_VERSION = 1
MAX_ERROR_SAMPLES_PER_LAYER_PER_RANK = 10_000
MHC_SCORE_CHUNK_TOKENS = 1024
# Indexer.forward scores the whole prefill unblocked and materialises
# [1,T,n_local_heads,T/4] bf16 twice, so the transient is ~16*T^2 bytes against
# ~35 GiB of headroom on 4xH100 (weights are 43.5 GiB/rank). 16384 costs 4.9 GiB
# and 24576 costs 11.0 GiB; 65280 costs 77.4 GiB and OOMs. The launcher caps this
# too, but standalone invocations bypass the launcher, so enforce it here where
# every mode passes through.
MAX_SUPPORTED_SEQ_LEN = 24576
# Every prompt this pipeline encodes -- calibration, parity, questions, pairs --
# goes through one mode. Scattered literals drift, and a mismatch between the
# profiled and evaluated distributions is invisible in the numbers, so there is
# exactly one definition and it is recorded in the score artifact.
THINKING_MODE = "reasoning"


@contextmanager
def deterministic_score_reductions():
    """Require deterministic kernels only while mutating score accumulators.

    Enabling deterministic algorithms for the entire V4 forward can change or
    reject unrelated custom FP4/TileLang kernels.  The identified source of
    score nondeterminism is the duplicate-index CUDA ``index_add_`` reduction,
    so keep the strict setting scoped to those reductions and restore the
    caller's process-global setting exactly.
    """
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        # Yield what actually took effect rather than what was requested. The
        # pinned torch is not the torch these runs execute on, so a build that
        # ignores or downgrades the request must not leave the artifact
        # claiming a determinism guarantee that never held.
        yield (torch.are_deterministic_algorithms_enabled()
               and not torch.is_deterministic_algorithms_warn_only_enabled())
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)

# Keys legitimately absent from the checkpoint. DSparkBlock holds references to
# the trunk embedding and head (Transformer.__init__ assigns mtp[i].embed =
# self.embed), so those parameters appear twice in state_dict but are stored
# once; inference/convert.py drops the duplicates. Anything else missing means
# the checkpoint and the model disagree, which must not pass silently.
ALLOWED_MISSING = (
    r"^mtp\.\d+\.embed\.",
    r"^mtp\.\d+\.head\.",
)


def load_checkpoint_checked(model, path: str, say) -> dict:
    """load_model(strict=False) returns (missing, unexpected) and callers usually
    throw them away. Instrumentation bugs and checkpoint drift both surface here,
    and either would be indistinguishable from a pruning effect downstream."""
    missing, unexpected = load_model(model, path, strict=False)
    missing, unexpected = sorted(set(missing)), sorted(set(unexpected))
    unexplained = [k for k in missing if not any(re.match(p, k) for p in ALLOWED_MISSING)]
    say(f"checkpoint {Path(path).name}: {len(missing)} missing "
        f"({len(missing) - len(unexplained)} expected), {len(unexpected)} unexpected")
    if unexpected or unexplained:
        for k in unexpected[:20]:
            say(f"  UNEXPECTED  {k}")
        for k in unexplained[:20]:
            say(f"  MISSING     {k}")
        raise SystemExit(
            f"checkpoint/model mismatch: {len(unexpected)} unexpected key(s), "
            f"{len(unexplained)} unexplained missing key(s). Refusing to run -- "
            f"a partially loaded model would look like a pruning effect."
        )
    return {"missing": missing, "unexpected": unexpected}


# ---------------------------------------------------------------- accumulator


class Profiler:
    """Accumulates the EASY-EP score, plus diagnostics, on GPU."""

    def __init__(self, n_layers: int, n_experts: int, device):
        self.enabled = False
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.gate_topk = None
        # float64: this is a long-running sum of many small products. ``score`` is
        # the architecture-faithful score in V4's hc_mult residual space.
        self.score = torch.zeros((n_layers, n_experts), dtype=torch.float64, device=device)
        # Keep the literal h -> h+y_routed definition only as a diagnostic. That
        # transition is not a state transition made by a V4 block.
        self.score_reduced_legacy = torch.zeros_like(self.score)
        # diagnostics, so the paper's score can be compared against the
        # simpler weight*norm variant without a second calibration run
        self.score_no_simibr = torch.zeros_like(self.score)
        self.counts = torch.zeros((n_layers, n_experts), dtype=torch.int64, device=device)
        self.gate_sums = torch.zeros_like(self.score)
        self.tokens_seen = 0
        # None until the first accumulation; AND-reduced thereafter, so one
        # non-deterministic reduction anywhere is enough to withdraw the claim.
        self.deterministic_reductions = None
        # --- norm-recovery validation (only populated in validate mode) ---
        self.validate = False
        # bisect switches: each adds one observation on top of the previous
        self.do_norms = True       # per-expert unweighted-norm recovery + all_gather
        self.do_accum = True       # index_add_ into the score tensors
        self.do_mhc = True         # the mHC simibr, which touches hc_post inputs
        self.score_true = torch.zeros_like(self.score)   # from explicit unweighted forwards
        # Validation percentiles are diagnostic only. Cap every layer equally so
        # early layers/chunks cannot consume a process-wide reservoir and exclude
        # later layers from the reported error distribution.
        self._err = [[] for _ in range(n_layers)]         # samples, grouped by layer
        self._err_n_by_layer = [0] * n_layers

    def note_reduction_determinism(self, observed: bool) -> None:
        """Record whether strict deterministic kernels were genuinely active."""
        self.deterministic_reductions = (
            bool(observed) if self.deterministic_reductions is None
            else self.deterministic_reductions and bool(observed))

    def err_tensor(self):
        chunks = [chunk for layer_chunks in self._err for chunk in layer_chunks]
        return (torch.cat(chunks) if chunks else
                torch.empty(0, dtype=torch.float32, device=self.score.device))

    @property
    def score_mhc(self):
        """Compatibility alias: schema-v2's primary score is mHC-aware."""
        return self.score

    def state(self, n_hash_layers: int | None = None,
              model_identity: dict | None = None) -> dict[str, Any]:
        state = {
            "score_schema_version": SCORE_SCHEMA_VERSION,
            "score_semantics": SCORE_SEMANTICS,
            "score_mhc": self.score_mhc.cpu(),
            "score_reduced_legacy": self.score_reduced_legacy.cpu(),
            "score_true": self.score_true.cpu(),
            "norm_rel_err": self.err_tensor().cpu(),
            "score": self.score.cpu(),
            "score_no_simibr": self.score_no_simibr.cpu(),
            "counts": self.counts.cpu(),
            "gate_sums": self.gate_sums.cpu(),
            "tokens_seen": self.tokens_seen,
            "n_layers": self.n_layers,
            "n_experts": self.n_experts,
            "gate_topk": self.gate_topk,
            "accumulation_mode": ACCUMULATION_MODE,
            "deterministic_score_reductions": self.deterministic_reductions is True,
            "torch_version": torch.__version__,
            "norm_error_samples_per_layer": list(self._err_n_by_layer),
        }
        if (n_hash_layers is None) != (model_identity is None):
            raise ValueError("n_hash_layers and model_identity must be recorded together")
        if n_hash_layers is not None:
            if (isinstance(n_hash_layers, bool) or not isinstance(n_hash_layers, int)
                    or not 0 <= n_hash_layers <= self.n_layers):
                raise ValueError("n_hash_layers is outside the profiler layer range")
            digest = _validate_model_identity_record(model_identity, "current model")
            state["n_hash_layers"] = n_hash_layers
            state["model_identity"] = model_identity
            state["model_identity_sha256"] = digest
        return state


# ------------------------------------------------------------------ patching


def mhc_residual_base(residual: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """Return ``sum_i comb[..., i, j] * residual[..., i, :]`` in FP32.

    V4's ``hc_post`` casts the completed update back to the FFN dtype. Recovering
    the base by subtracting from that rounded result loses information, while the
    direct broadcast expression materialises an enormous [..., hc, hc, d]
    temporary. hc_mult is small, so accumulate one source stream at a time.
    """
    if residual.ndim < 2 or comb.ndim != residual.ndim:
        raise ValueError("mHC residual/comb tensors have incompatible ranks")
    if comb.shape[:-2] != residual.shape[:-2] or comb.shape[-2] != residual.shape[-2]:
        raise ValueError("mHC residual/comb tensors have incompatible shapes")
    shape = (*residual.shape[:-2], comb.shape[-1], residual.shape[-1])
    base = torch.zeros(shape, dtype=torch.float32, device=residual.device)
    residual_f = residual.float()
    comb_f = comb.float()
    for source in range(residual.shape[-2]):
        base.add_(comb_f[..., source, :].unsqueeze(-1)
                  * residual_f[..., source, :].unsqueeze(-2))
    return base


def mhc_score_observations(residual: torch.Tensor, comb: torch.Tensor,
                           post: torch.Tensor, routed: torch.Tensor,
                           chunk_tokens: int = MHC_SCORE_CHUNK_TOKENS
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute residual-space simibr and ||post|| without full-context FP32 copies."""
    if chunk_tokens < 1 or routed.shape != (*residual.shape[:-2], residual.shape[-1]):
        raise ValueError("mHC routed tensor shape or chunk size is invalid")
    if post.shape != (*residual.shape[:-2], comb.shape[-1]):
        raise ValueError("mHC post tensor shape is invalid")
    hc_mult, dim = residual.shape[-2], residual.shape[-1]
    n_tokens = residual.numel() // (hc_mult * dim)
    if n_tokens < 1:
        raise ValueError("mHC observation tensors contain no tokens")
    residual_flat = residual.reshape(n_tokens, hc_mult, dim)
    comb_flat = comb.reshape(n_tokens, comb.shape[-2], comb.shape[-1])
    post_flat = post.reshape(n_tokens, post.shape[-1]).float()
    routed_flat = routed.reshape(n_tokens, dim)
    similarities = []
    for start in range(0, n_tokens, chunk_tokens):
        stop = min(start + chunk_tokens, n_tokens)
        base = mhc_residual_base(residual_flat[start:stop], comb_flat[start:stop])
        after = (base + post_flat[start:stop].unsqueeze(-1)
                 * routed_flat[start:stop].float().unsqueeze(-2))
        similarities.append((1.0 - F.cosine_similarity(
            base.flatten(1), after.flatten(1), dim=-1)).clamp_min(0.0))
        del base, after
    return torch.cat(similarities), post_flat.norm(dim=-1)


def patch(official: Any, profiler: Profiler) -> None:
    """Patch Gate/MoE/Block forwards. Weights and module structure untouched."""

    Gate, MoE, Block = official.Gate, official.MoE, official.Block
    linear = official.linear
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # ---- Gate: identical to upstream, plus an optional keep-mask -----------
    def gate_forward(self, x, input_ids=None):
        scores = linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            # token-ID routing: never masked, see module docstring
            indices = self.tid2eid[input_ids]
        else:
            keep = getattr(self, "ep_keep", None)
            if keep is not None:
                scores = scores.masked_fill(~keep.view(1, -1), float("-inf"))
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices

    # ---- MoE: upstream maths, plus per-token unweighted norms --------------
    def moe_forward(self, x, input_ids):
        shape = x.size()
        x = x.view(-1, self.dim)
        n_tok = x.size(0)
        weights, indices = self.gate(x, input_ids.flatten())
        y = torch.zeros_like(x, dtype=torch.float32)

        # Must match block_forward's accumulate guard exactly: hash layers route by
        # token id, and the MTP/DSpark stages sit past the end of the score tensor.
        # Profiling them would compute norms and all-gather for a result that is
        # never consumed, and leave the stashes unreleased.
        prof = (profiler.enabled and profiler.do_norms and not self.gate.hash
                and self.layer_id < profiler.n_layers)
        if prof:
            norms_local = torch.zeros(
                (self.n_local_experts, n_tok), dtype=torch.float32, device=x.device
            )
            norms_true_local = (torch.zeros_like(norms_local)
                                if profiler.validate else None)

        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for e in range(self.experts_start_idx, self.experts_end_idx):
            if counts[e] == 0:
                continue
            rows, slots = torch.where(indices == e)
            w = weights[rows, slots, None]
            contribution = self.experts[e](x[rows], w)
            y[rows] += contribution
            if prof:
                # ||w * out|| / w == ||out|| holds in exact arithmetic because w2 is
                # linear, but the weight is applied BEFORE act_quant + fp4_gemm, so
                # quantisation can break proportionality. validate mode measures it.
                wn = contribution.float().norm(dim=-1) / w.squeeze(-1).float().clamp_min(EPS)
                norms_local[e - self.experts_start_idx, rows] = wn
                if profiler.validate:
                    true_out = self.experts[e](x[rows], None)
                    tn = true_out.float().norm(dim=-1)
                    norms_true_local[e - self.experts_start_idx, rows] = tn
                    sampled = profiler._err_n_by_layer[self.layer_id]
                    if sampled < MAX_ERROR_SAMPLES_PER_LAYER_PER_RANK:
                        rel = ((wn - tn).abs() / tn.clamp_min(EPS)).flatten()
                        remaining = MAX_ERROR_SAMPLES_PER_LAYER_PER_RANK - sampled
                        rel = rel[:remaining]
                        profiler._err[self.layer_id].append(rel.detach())
                        profiler._err_n_by_layer[self.layer_id] += rel.numel()

        if world_size > 1:
            dist.all_reduce(y)

        if prof:
            # each rank holds norms only for its own experts; gather all
            if world_size > 1:
                bufs = [torch.zeros_like(norms_local) for _ in range(world_size)]
                dist.all_gather(bufs, norms_local)
                norms_global = torch.cat(bufs, dim=0)
            else:
                norms_global = norms_local
            # [n_tok, n_routed] -> per activated slot
            self._ep_norms = norms_global.t().gather(1, indices)   # [n_tok, topk]
            if profiler.validate:
                if world_size > 1:
                    tb = [torch.zeros_like(norms_true_local) for _ in range(world_size)]
                    dist.all_gather(tb, norms_true_local)
                    nt = torch.cat(tb, dim=0)
                else:
                    nt = norms_true_local
                self._ep_norms_true = nt.t().gather(1, indices)
            self._ep_weights = weights.float()                     # [n_tok, topk]
            self._ep_indices = indices                             # [n_tok, topk]
            self._ep_y_routed = y.detach()                         # routed only

        y = y + self.shared_experts(x)
        return y.type_as(x).view(shape)

    def moe_release(self) -> None:
        """Release per-forward observations, including no-accumulation bisect runs."""
        self._ep_norms = self._ep_weights = self._ep_indices = self._ep_y_routed = None
        self._ep_norms_true = None

    def moe_accumulate(self, h_flat: torch.Tensor, simibr_mhc: torch.Tensor = None,
                       post_norm: torch.Tensor = None) -> None:
        """Form the per-token product and add it to the running score.

        h_flat is the pre-norm reduced residual entering the FFN sub-block --
        the V4 analogue of EASY-EP's x_before_moe, measured in the single-vector
        space the FFN actually sees.

        simibr_mhc is measured after hc_post has remixed the contribution into the
        hc_mult residual copies. post_norm maps the expert output norm into that
        same space; the reduced-vector definition is retained only as a diagnostic.
        """
        y_r = self._ep_y_routed
        simibr = (1.0 - F.cosine_similarity(h_flat.float(), (h_flat + y_r).float(), dim=-1)).clamp_min(0.0)

        w, idx, nrm = self._ep_weights, self._ep_indices, self._ep_norms
        lid = self.layer_id
        if simibr_mhc is None or post_norm is None:
            raise RuntimeError("mHC score accumulation requires simibr and post norm")
        if simibr_mhc.shape != post_norm.shape or simibr_mhc.numel() != idx.size(0):
            raise RuntimeError("mHC observations do not match routed token count")
        residual_nrm = nrm * post_norm[:, None]
        # Expert IDs repeat across tokens.  CUDA's default index_add_ reduction
        # may therefore use atomic additions in an unspecified order.  PyTorch
        # 2.10 provides a deterministic CUDA path when strict deterministic
        # algorithms are enabled; scope it to these mutations so custom model
        # kernels outside the profiler are unaffected.
        with deterministic_score_reductions() as deterministic_reduction:
            profiler.note_reduction_determinism(deterministic_reduction)
            for slot in range(idx.size(1)):
                e = idx[:, slot]
                profiler.score[lid].index_add_(
                    0, e, (w[:, slot] * simibr_mhc * residual_nrm[:, slot]).double())
                profiler.score_reduced_legacy[lid].index_add_(
                    0, e, (w[:, slot] * simibr * nrm[:, slot]).double())
                profiler.score_no_simibr[lid].index_add_(
                    0, e, (w[:, slot] * residual_nrm[:, slot]).double())
                if profiler.validate:
                    profiler.score_true[lid].index_add_(
                        0, e, (w[:, slot] * simibr_mhc * post_norm
                               * self._ep_norms_true[:, slot]).double())
                profiler.gate_sums[lid].index_add_(0, e, w[:, slot].double())
                profiler.counts[lid].index_add_(
                    0, e, torch.ones_like(e, dtype=torch.int64))
        profiler.tokens_seen += idx.size(0)

        # release: these hold [n_tok, *] tensors per layer and would otherwise stay
        # resident through the generation phases, eating KV-cache headroom
        self.ep_release()

    # ---- Block: keep the pre-FFN residual so simibr is computable ---------
    def block_forward(self, x, start_pos, input_ids, *attn_args):
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos, *attn_args)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        h = x                                   # <-- EASY-EP's x_before_moe
        x = self.ffn_norm(x)
        x_ffn = self.ffn(x, input_ids)
        observes_mhc = (profiler.enabled and profiler.do_norms and profiler.do_mhc
                        and not self.ffn.gate.hash
                        and self.layer_id < profiler.n_layers
                        and getattr(self.ffn, "_ep_y_routed", None) is not None)
        # The observation path must not replace or reorder official forward math.
        x = self.hc_post(x_ffn, residual, post, comb)
        if (profiler.enabled
                and not self.ffn.gate.hash
                and self.layer_id < profiler.n_layers):   # excludes the MTP/DSpark stages
            y_r = getattr(self.ffn, "_ep_y_routed", None)
            if observes_mhc:
                # Score in bounded token chunks so an 8k context does not create
                # several multi-GiB FP32 residual tensors between model layers.
                simibr_mhc, post_norm = mhc_score_observations(
                    residual, comb, post, y_r.view(h.shape))
                if profiler.do_accum:
                    self.ffn.ep_accumulate(
                        h.view(-1, h.size(-1)), simibr_mhc, post_norm)
                else:
                    self.ffn.ep_release()
            elif y_r is not None:
                self.ffn.ep_release()
        return x

    originals = {"Gate.forward": Gate.forward, "MoE.forward": MoE.forward,
                 "Block.forward": Block.forward}
    Gate.forward = gate_forward
    MoE.forward = moe_forward
    MoE.ep_accumulate = moe_accumulate
    MoE.ep_release = moe_release
    Block.forward = block_forward
    return originals


def unpatch(official: Any, originals: dict) -> None:
    official.Gate.forward = originals["Gate.forward"]
    official.MoE.forward = originals["MoE.forward"]
    official.Block.forward = originals["Block.forward"]


def _sha256_file(path: Path, block_bytes: int = 1024 * 1024) -> str:
    """Content-hash a small/medium provenance file, rejecting concurrent changes."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"required provenance file is missing: {path}")
    before = path.stat()
    identity = lambda stat: (
        stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
    )
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        if identity(os.fstat(fp.fileno())) != identity(before):
            raise RuntimeError(f"provenance file changed while opening: {path}")
        while True:
            block = fp.read(block_bytes)
            if not block:
                break
            digest.update(block)
        finished = os.fstat(fp.fileno())
    after = path.stat()
    if identity(before) != identity(finished) or identity(before) != identity(after):
        raise RuntimeError(f"provenance file changed while hashing: {path}")
    return digest.hexdigest()


def _sampled_checkpoint_identity(path: Path, sample_bytes: int = 64 * 1024) -> dict:
    """Cheap identity for a very large shard: metadata plus start/middle/end samples."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"required checkpoint shard is missing: {path}")
    before = path.stat()
    size = before.st_size
    offsets = sorted({0, max(0, size // 2 - sample_bytes // 2),
                      max(0, size - sample_bytes)})
    identity = lambda stat: (
        stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
    )
    digest = hashlib.sha256()
    digest.update(f"size:{size}\n".encode("ascii"))
    with path.open("rb") as fp:
        if identity(os.fstat(fp.fileno())) != identity(before):
            raise RuntimeError(f"checkpoint shard changed while opening: {path}")
        for offset in offsets:
            fp.seek(offset)
            block = fp.read(min(sample_bytes, size - offset))
            digest.update(f"offset:{offset}:bytes:{len(block)}\n".encode("ascii"))
            digest.update(block)
        finished = os.fstat(fp.fileno())
    after = path.stat()
    if identity(before) != identity(finished) or identity(before) != identity(after):
        raise RuntimeError(f"checkpoint shard changed while sampling: {path}")
    return {
        "name": path.name,
        "size": size,
        "sample_sha256": digest.hexdigest(),
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _checkpoint_attestation_identity(sidecar: Path, checkpoint: Path,
                                     config: Path, world_size: int) -> str:
    """Validate a checkpoint-local full-hash sidecar before trusting its ID."""
    helper_path = Path(__file__).with_name("checkpoint_provenance.py")
    spec = importlib.util.spec_from_file_location(
        "_easyep_checkpoint_provenance", helper_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load checkpoint provenance helper: {helper_path}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)

    before = sidecar.stat()
    try:
        payload = sidecar.read_bytes()
        record = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid checkpoint provenance sidecar: {exc}") from exc
    after = sidecar.stat()
    identity = lambda stat: (
        stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
    )
    if identity(before) != identity(after):
        raise RuntimeError("checkpoint provenance sidecar changed while reading")
    if not isinstance(record, dict) or record.get("schema_version") != helper.SCHEMA_VERSION:
        raise ValueError("checkpoint provenance sidecar has an unsupported schema")
    content_identity = record.get("content_identity_sha256")
    if (not isinstance(content_identity, str)
            or helper.SHA256_RE.fullmatch(content_identity) is None
            or content_identity != helper.checkpoint_content_identity(record)):
        raise ValueError("checkpoint provenance sidecar has an inconsistent content identity")
    if record.get("model") != {
            "id": helper.PINNED_MODEL_ID,
            "revision": helper.PINNED_REVISION,
            "inference_revision": helper.PINNED_REVISION,
    }:
        raise ValueError("checkpoint provenance sidecar identifies a different model")

    conversion = record.get("conversion")
    if (not isinstance(conversion, dict)
            or (conversion.get("n_experts"), conversion.get("model_parallel"),
                conversion.get("expert_dtype")) != (256, world_size, "fp4")):
        raise ValueError("checkpoint provenance sidecar has incompatible conversion settings")
    source_hashes = conversion.get("source_files_sha256")
    if (not isinstance(source_hashes, dict)
            or set(source_hashes) != set(helper.REQUIRED_SOURCE_FILES)
            or any(not isinstance(value, str)
                   or helper.SHA256_RE.fullmatch(value) is None
                   for value in source_hashes.values())):
        raise ValueError("checkpoint provenance sidecar has invalid source hashes")
    # The checked-in config is intentionally outside the downloaded source tree;
    # all other source hashes are checked by _model_identity's complete tree hash.
    if source_hashes["inference/config.json"] != _sha256_file(config):
        raise ValueError("checkpoint provenance config hash does not match this run")

    checkpoint_record = record.get("checkpoint")
    shards = checkpoint_record.get("shards") if isinstance(checkpoint_record, dict) else None
    if (not isinstance(checkpoint_record, dict)
            or checkpoint_record.get("nproc") != world_size
            or not isinstance(shards, list) or len(shards) != world_size):
        raise ValueError("checkpoint provenance sidecar has an invalid shard topology")
    by_name = {}
    for shard in shards:
        if (not isinstance(shard, dict)
                or not isinstance(shard.get("name"), str)
                or isinstance(shard.get("size"), bool)
                or not isinstance(shard.get("size"), int) or shard["size"] < 0
                or isinstance(shard.get("mtime_ns"), bool)
                or not isinstance(shard.get("mtime_ns"), int)
                or not isinstance(shard.get("sha256"), str)
                or helper.SHA256_RE.fullmatch(shard["sha256"]) is None
                or shard["name"] in by_name):
            raise ValueError("checkpoint provenance sidecar has a malformed shard record")
        by_name[shard["name"]] = shard
    for rank in range(world_size):
        name = f"model{rank}-mp{world_size}.safetensors"
        item = by_name.get(name)
        if item is None:
            raise ValueError(f"checkpoint provenance sidecar is missing {name}")
        stat = (checkpoint / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item["size"], item["mtime_ns"]):
            raise ValueError(
                f"checkpoint shard metadata no longer matches its full-hash record: {name}")
    return content_identity


def _model_identity(ckpt_path: str, config_path: str, code_dir: str,
                    world_size: int) -> dict:
    """Fingerprint every model input that can change an EASY-EP score.

    Official/tokenizer/config sources are small enough to hash completely. Converted
    FP4 shards can be hundreds of GiB, so each shard is identified by name, size,
    and SHA256 over bounded start/middle/end samples. When present, the launcher-
    verified sidecar adds a portable identity over every shard's full SHA256.
    """
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    config = Path(config_path)
    code = Path(code_dir)
    encoding = code.parent / "encoding"
    checkpoint = Path(ckpt_path)
    required_sources = (
        code / "model.py",
        code / "generate.py",
        encoding / "encoding_dsv4.py",
    )
    for path in required_sources:
        if not path.is_file():
            raise ValueError(f"required official source file is missing: {path}")

    source_suffixes = {
        ".py", ".pyi", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp",
        ".triton",
    }
    official_sources = {}
    for prefix, root in (("inference", code), ("encoding", encoding)):
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in source_suffixes:
                key = f"{prefix}/{path.relative_to(root).as_posix()}"
                official_sources[key] = _sha256_file(path)

    tokenizer_names = {
        "added_tokens.json", "chat_template.jinja", "merges.txt",
        "sentencepiece.bpe.model", "special_tokens_map.json", "vocab.json",
    }
    tokenizer_files = [
        path for path in sorted(checkpoint.iterdir())
        if path.is_file()
        and (path.name.startswith("tokenizer") or path.name in tokenizer_names)
    ] if checkpoint.is_dir() else []
    if not tokenizer_files:
        raise ValueError(f"no tokenizer provenance files found under {checkpoint}")
    tokenizer = {path.name: _sha256_file(path) for path in tokenizer_files}

    shard_paths = [
        checkpoint / f"model{rank}-mp{world_size}.safetensors"
        for rank in range(world_size)
    ]
    shards = [_sampled_checkpoint_identity(path) for path in shard_paths]
    checkpoint_identity = {"world_size": world_size, "shards": shards}
    full_hash_attestation = checkpoint / "EASYEP_CHECKPOINT_PROVENANCE.json"
    if full_hash_attestation.is_file():
        content_identity = _checkpoint_attestation_identity(
            full_hash_attestation, checkpoint, config, world_size)
        checkpoint_identity["full_hash_attestation"] = {
            "name": full_hash_attestation.name,
            "content_identity_sha256": content_identity,
        }
    return {
        "identity_schema_version": MODEL_IDENTITY_SCHEMA_VERSION,
        "instrumentation_sha256": _sha256_file(Path(__file__)),
        "config": {"name": config.name, "sha256": _sha256_file(config)},
        "official_sources": official_sources,
        "tokenizer": tokenizer,
        "checkpoint": checkpoint_identity,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "transformers": _package_version("transformers"),
            "safetensors": _package_version("safetensors"),
        },
    }


def _identity_digest(identity: dict) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_model_identity_record(identity: Any, source: str) -> str:
    sha = re.compile(r"^[0-9a-f]{64}$")
    valid_sha = lambda value: isinstance(value, str) and sha.fullmatch(value) is not None
    if not isinstance(identity, dict):
        raise SystemExit(f"{source} is missing a valid model_identity record")
    if identity.get("identity_schema_version") != MODEL_IDENTITY_SCHEMA_VERSION:
        raise SystemExit(f"{source} uses an incompatible model-identity schema")
    if not valid_sha(identity.get("instrumentation_sha256")):
        raise SystemExit(f"{source} has an invalid instrumentation identity")
    config = identity.get("config")
    if (not isinstance(config, dict)
            or not isinstance(config.get("name"), str) or not config["name"]
            or not valid_sha(config.get("sha256"))):
        raise SystemExit(f"{source} has an invalid config identity")
    for label in ("official_sources", "tokenizer"):
        files = identity.get(label)
        if (not isinstance(files, dict) or not files
                or any(not isinstance(name, str) or not name or not valid_sha(value)
                       for name, value in files.items())):
            raise SystemExit(f"{source} has an invalid {label} identity")
    required = {
        "inference/model.py", "inference/generate.py", "encoding/encoding_dsv4.py"
    }
    if not required.issubset(identity["official_sources"]):
        raise SystemExit(f"{source} is missing required official-source identities")
    checkpoint = identity.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"{source} has an invalid checkpoint identity")
    world_size, shards = checkpoint.get("world_size"), checkpoint.get("shards")
    if (isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1
            or not isinstance(shards, list) or len(shards) != world_size):
        raise SystemExit(f"{source} has an invalid checkpoint shard topology")
    for shard in shards:
        if (not isinstance(shard, dict) or not isinstance(shard.get("name"), str)
                or isinstance(shard.get("size"), bool)
                or not isinstance(shard.get("size"), int) or shard["size"] < 0
                or not valid_sha(shard.get("sample_sha256"))):
            raise SystemExit(f"{source} has an invalid checkpoint shard identity")
    expected_names = [
        f"model{rank}-mp{world_size}.safetensors" for rank in range(world_size)
    ]
    if [shard["name"] for shard in shards] != expected_names:
        raise SystemExit(f"{source} has inconsistent checkpoint shard names")
    attestation = checkpoint.get("full_hash_attestation")
    if (attestation is not None
            and (not isinstance(attestation, dict)
                 or attestation.get("name") != "EASYEP_CHECKPOINT_PROVENANCE.json"
                 or not valid_sha(attestation.get("content_identity_sha256")))):
        raise SystemExit(f"{source} has an invalid full-checkpoint attestation identity")
    runtime = identity.get("runtime")
    runtime_keys = {"python", "torch", "cuda", "transformers", "safetensors"}
    if (not isinstance(runtime, dict) or not runtime_keys.issubset(runtime)
            or any(not isinstance(runtime[key], str) or not runtime[key]
                   for key in runtime_keys)):
        raise SystemExit(f"{source} has an invalid runtime identity")
    return _identity_digest(identity)


def _identity_difference(artifact: Any, expected: Any, path: str = "model_identity") -> str:
    """Return a compact path to the first provenance mismatch."""
    if type(artifact) is not type(expected):
        return f"{path} type"
    if isinstance(expected, dict):
        if set(artifact) != set(expected):
            return f"{path} keys"
        for key in sorted(expected):
            difference = _identity_difference(artifact[key], expected[key], f"{path}.{key}")
            if difference:
                return difference
        return ""
    if isinstance(expected, list):
        if len(artifact) != len(expected):
            return f"{path} length"
        for index, value in enumerate(expected):
            difference = _identity_difference(artifact[index], value, f"{path}[{index}]")
            if difference:
                return difference
        return ""
    return "" if artifact == expected else path


def _validate_keep(keep_n: int, n_experts: int, topk: int | None = None) -> int:
    if isinstance(keep_n, bool) or not isinstance(keep_n, int):
        raise ValueError("keep must be an integer")
    if n_experts < 1 or not 1 <= keep_n <= n_experts:
        raise ValueError(f"keep must be in [1, {n_experts}], got {keep_n}")
    if topk is not None and keep_n < topk:
        raise ValueError(
            f"mask keeps {keep_n} experts but the gate activates top-{topk}; "
            "masked experts would otherwise be selected from -inf ties")
    return keep_n


def _validate_scores(scores: torch.Tensor, keep_n: int, n_hash: int) -> tuple[int, int]:
    if not isinstance(scores, torch.Tensor) or scores.ndim != 2:
        raise ValueError("expert scores must be a rank-2 tensor [layers, experts]")
    n_layers, n_experts = scores.shape
    _validate_keep(keep_n, n_experts)
    if not 0 <= n_hash <= n_layers:
        raise ValueError(f"n_hash_layers must be in [0, {n_layers}], got {n_hash}")
    if not bool(torch.isfinite(scores).all().item()):
        raise ValueError("expert scores contain NaN or infinity")
    return n_layers, n_experts


def _require_score_artifact(state: dict, source: str = "score artifact",
                            n_hash_layers: int | None = None,
                            model_identity: dict | None = None) -> dict:
    version = state.get("score_schema_version") if isinstance(state, dict) else None
    semantics = state.get("score_semantics") if isinstance(state, dict) else None
    if version != SCORE_SCHEMA_VERSION or semantics != SCORE_SEMANTICS:
        raise SystemExit(
            f"{source} uses incompatible score schema {version!r}/{semantics!r}; "
            "re-profile with this easyep_v4.py before building or evaluating masks")
    if state.get("accumulation_mode") != ACCUMULATION_MODE:
        raise SystemExit(
            f"{source} was accumulated as {state.get('accumulation_mode')!r}, not "
            f"{ACCUMULATION_MODE!r}; re-profile with this easyep_v4.py before "
            "building or evaluating masks")
    if state.get("deterministic_score_reductions") is not True:
        # Not a stale-artifact check: this fires when the run itself observed
        # that strict deterministic kernels were not actually in force, so the
        # expert ranking is not reproducible and must not be used.
        raise SystemExit(
            f"{source} reports that deterministic score reductions did not hold "
            f"(torch {state.get('torch_version', 'unknown')}); the expert ranking "
            "is not reproducible. Re-profile on a build whose index_add_ honours "
            "torch.use_deterministic_algorithms(True).")
    for key in ("score", "score_mhc", "score_reduced_legacy",
                "score_no_simibr", "counts", "gate_sums"):
        if key not in state:
            raise SystemExit(f"{source} is missing required tensor {key!r}")
    score = state["score"]
    if not isinstance(score, torch.Tensor) or score.ndim != 2:
        raise SystemExit(f"{source} has invalid score shape")
    shape = tuple(score.shape)
    for key in ("score_mhc", "score_reduced_legacy", "score_no_simibr",
                "counts", "gate_sums"):
        value = state[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise SystemExit(f"{source} tensor {key!r} does not match score shape {shape}")
    if state.get("n_layers") != shape[0] or state.get("n_experts") != shape[1]:
        raise SystemExit(f"{source} metadata does not match score shape {shape}")
    if not torch.equal(state["score_mhc"], score):
        raise SystemExit(f"{source} score_mhc is not the schema-v2 primary-score alias")
    gate_topk = state.get("gate_topk")
    if (isinstance(gate_topk, bool) or not isinstance(gate_topk, int)
            or not 1 <= gate_topk <= shape[1]):
        raise SystemExit(f"{source} has invalid gate_topk provenance")
    tokens_seen = state.get("tokens_seen")
    if (isinstance(tokens_seen, bool) or not isinstance(tokens_seen, int)
            or tokens_seen < 1):
        raise SystemExit(f"{source} has invalid token-observation provenance")
    if state["counts"].dtype not in (torch.int32, torch.int64):
        raise SystemExit(f"{source} counts must use an integer dtype")
    if bool((state["counts"] < 0).any().item()):
        raise SystemExit(f"{source} counts contain negative values")
    for key in ("score", "score_reduced_legacy", "score_no_simibr", "gate_sums"):
        if not bool(torch.isfinite(state[key]).all().item()):
            raise SystemExit(f"{source} tensor {key!r} contains NaN or infinity")
    stored_n_hash = state.get("n_hash_layers")
    if (isinstance(stored_n_hash, bool) or not isinstance(stored_n_hash, int)
            or not 0 <= stored_n_hash <= shape[0]):
        raise SystemExit(
            f"{source} is missing valid n_hash_layers provenance; re-profile it")
    if n_hash_layers is not None and stored_n_hash != n_hash_layers:
        raise SystemExit(
            f"{source} was profiled with n_hash_layers={stored_n_hash}, but the "
            f"current model/request uses {n_hash_layers}; refusing cross-config reuse")
    stored_identity = state.get("model_identity")
    stored_digest = _validate_model_identity_record(stored_identity, source)
    if state.get("model_identity_sha256") != stored_digest:
        raise SystemExit(f"{source} model_identity digest is missing or inconsistent")
    if model_identity is not None:
        expected_digest = _validate_model_identity_record(model_identity, "current model")
        if stored_digest != expected_digest or stored_identity != model_identity:
            difference = _identity_difference(stored_identity, model_identity)
            raise SystemExit(
                f"{source} does not match the current model/runtime ({difference}); "
                "re-profile instead of reusing scores")
    _validate_calibration_provenance(state.get("calibration"), source)
    return state


def _validate_profile_observations(profiler: Profiler, n_hash_layers: int,
                                   require_true: bool = False) -> None:
    if profiler.tokens_seen < 1:
        raise ValueError("calibration recorded no routed token-layer observations")
    if not 0 <= n_hash_layers < profiler.n_layers:
        raise ValueError("calibration has no prunable routed layers")
    names = ["score", "score_no_simibr", "score_reduced_legacy", "gate_sums"]
    if require_true:
        names.append("score_true")
    for name in names:
        tensor = getattr(profiler, name)
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"calibration produced non-finite {name}")
    for lid in range(n_hash_layers, profiler.n_layers):
        if int(profiler.counts[lid].sum().item()) < 1:
            raise ValueError(f"calibration recorded no routed experts for layer {lid}")
        if not bool((profiler.score[lid].abs().sum() > 0).item()):
            raise ValueError(f"calibration produced an all-zero primary score for layer {lid}")
        if require_true and not bool((profiler.score_true[lid].abs().sum() > 0).item()):
            raise ValueError(f"calibration produced an all-zero true score for layer {lid}")


def _validate_score_state_observations(state: dict, n_hash_layers: int,
                                       score_key: str = "score") -> None:
    score, counts = state[score_key], state["counts"]
    n_layers, _ = score.shape
    if int(state.get("tokens_seen", 0)) < 1:
        raise ValueError("score artifact contains no calibration observations")
    if not 0 <= n_hash_layers < n_layers:
        raise ValueError("score artifact has no prunable routed layers")
    for lid in range(n_hash_layers, n_layers):
        if int(counts[lid].sum().item()) < 1:
            raise ValueError(f"score artifact contains no observations for layer {lid}")
        if not bool((score[lid].abs().sum() > 0).item()):
            raise ValueError(
                f"score artifact contains an all-zero {score_key} for layer {lid}")


def set_mask(model, mask: dict | None, n_hash_layers: int, device) -> None:
    if not 0 <= n_hash_layers <= len(model.layers):
        raise ValueError("n_hash_layers is outside the model layer range")
    if mask is not None and not isinstance(mask, dict):
        raise ValueError("mask must be a JSON object")
    declared_keep = None if mask is None else mask.get("keep_per_layer")
    for lid, layer in enumerate(model.layers):
        if mask is None or lid < n_hash_layers:
            layer.ffn.gate.ep_keep = None
            continue
        gate = layer.ffn.gate
        n_experts, topk = int(layer.ffn.n_routed_experts), int(gate.topk)
        if declared_keep is not None:
            _validate_keep(declared_keep, n_experts, topk)
        if mask.get("n_experts", n_experts) != n_experts:
            raise ValueError(f"mask/model expert-count mismatch on layer {lid}")
        try:
            layer_mask = mask["layers"][str(lid)]
            kept = layer_mask["kept"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"mask is missing layer {lid} kept experts") from exc
        if (not isinstance(kept, list) or any(isinstance(e, bool) or not isinstance(e, int)
                                                  for e in kept)):
            raise ValueError(f"layer {lid} kept experts must be a list of integers")
        if len(set(kept)) != len(kept) or any(e < 0 or e >= n_experts for e in kept):
            raise ValueError(f"layer {lid} kept experts are duplicate or out of range")
        _validate_keep(len(kept), n_experts, topk)
        if declared_keep is not None and len(kept) != declared_keep:
            raise ValueError(
                f"layer {lid} keeps {len(kept)} experts, metadata declares {declared_keep}")
        if "pruned" in layer_mask:
            pruned = layer_mask["pruned"]
            if (not isinstance(pruned, list)
                    or any(isinstance(e, bool) or not isinstance(e, int) for e in pruned)
                    or len(set(pruned)) != len(pruned)
                    or set(kept).intersection(pruned)
                    or set(kept).union(pruned) != set(range(n_experts))):
                raise ValueError(f"layer {lid} kept/pruned lists do not partition experts")
        keep = torch.zeros(n_experts, dtype=torch.bool, device=device)
        keep[torch.tensor(kept, dtype=torch.long, device=device)] = True
        layer.ffn.gate.ep_keep = keep


# --------------------------------------------------------------------- model


def build(ckpt_path: str, config_path: str, code_dir: str, max_seq_len: int, max_bs: int,
          temperature: float | None = None):
    if (temperature is not None
            and (isinstance(temperature, bool)
                 or not isinstance(temperature, (int, float))
                 or not math.isfinite(temperature)
                 or temperature < 0)):
        raise ValueError("temperature must be finite and >= 0")
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int) or max_seq_len < 2:
        raise ValueError("max_seq_len must be an integer >= 2")
    if max_seq_len > MAX_SUPPORTED_SEQ_LEN:
        raise ValueError(
            f"max_seq_len={max_seq_len} exceeds the {MAX_SUPPORTED_SEQ_LEN} supported "
            f"on this topology: prefill memory is quadratic and unblocked (~16*T^2 "
            f"bytes), so this would need roughly {16 * max_seq_len ** 2 / 2**30:.1f} GiB "
            f"of transient against ~35 GiB of headroom. Token-exact chunking already "
            f"preserves whole sources, so a smaller window costs no coverage.")
    sys.path.insert(0, code_dir)
    import model as official  # noqa: E402

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)

    with open(config_path) as f:
        cfg = json.load(f)
    args = official.ModelArgs(**cfg)
    args.max_seq_len = max_seq_len
    args.max_batch_size = max_bs
    # config.json carries no "temperature", so ModelArgs defaults to 1.0 (stochastic).
    # Transformer.forward samples every token via the Gumbel-max trick, which consumes
    # RNG -- see the paired-seeding note in cmd_pipeline.
    if temperature is not None:
        args.temperature = temperature

    t0 = time.time()
    with torch.device("cuda"):
        model = official.Transformer(args)
    tok = AutoTokenizer.from_pretrained(ckpt_path)
    def _say(m):
        if rank == 0:
            print(f"[easyep] {m}", flush=True)
    load_checkpoint_checked(
        model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"), _say)
    torch.set_default_device("cuda")
    if rank == 0:
        print(f"[easyep] model ready in {time.time()-t0:.1f}s", flush=True)
    return official, model, tok, args, rank, world_size


def load_calibration(path: Path) -> list[str]:
    """Accepts a JSON list of strings, a JSONL with a 'text'/'prompt' field, or a dir of files."""
    if path.is_dir():
        out = []
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix in {".js", ".ts", ".jsx", ".tsx", ".txt", ".md", ".py"}:
                try:
                    out.append(p.read_text(encoding="utf-8", errors="ignore"))
                except OSError:
                    pass
        return out
    raw = path.read_text(encoding="utf-8")

    def text_of(row, index):
        """Never fall back to json.dumps(row).

        That turned "this file is not calibration text" into "calibrate on a
        JSON dump of the record", which profiles a distribution nobody chose and
        looks like a successful run. The same fallback in the evaluation path
        put reference answers into prompts; fail instead of guessing.
        """
        if isinstance(row, str):
            return row
        if isinstance(row, dict):
            for field in ("text", "prompt", "content"):
                value = row.get(field)
                if isinstance(value, str) and value:
                    return value
            raise SystemExit(
                f"{path}: calibration record {index} has no non-empty "
                f"text/prompt/content field (keys: {sorted(row)[:8]}); refusing "
                "to calibrate on a JSON dump of the record")
        raise SystemExit(
            f"{path}: calibration record {index} is {type(row).__name__}, "
            "expected a string or an object")

    if path.suffix == ".jsonl":
        rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
        return [text_of(r, i) for i, r in enumerate(rows)]
    data = json.loads(raw)
    if isinstance(data, list):
        return [text_of(d, i) for i, d in enumerate(data)]
    raise SystemExit(f"unsupported calibration file: {path}")


# --------------------------------------------------------------------- modes


def cmd_profile(a) -> None:
    official, model, tok, args, rank, world_size = build(
        a.ckpt_path, a.config, a.code_dir, a.max_seq_len, 1,
        getattr(a, "temperature", None)
    )
    model_identity = _model_identity(a.ckpt_path, a.config, a.code_dir, world_size)
    sys.path.insert(0, str(Path(a.code_dir).parent / "encoding"))
    from encoding_dsv4 import encode_messages
    sys.path.insert(0, a.code_dir)
    from generate import generate

    prof = Profiler(args.n_layers, args.n_routed_experts, torch.device("cuda"))
    prof.gate_topk = int(args.n_activated_experts)
    patch(official, prof)
    set_mask(model, None, args.n_hash_layers, torch.device("cuda"))

    samples = load_calibration(Path(a.calib))
    if a.limit:
        samples = samples[: a.limit]
    if rank == 0:
        print(f"[easyep] {len(samples)} calibration samples", flush=True)

    inputs = _calibration_chunks(
        [(f"sample-{i}", text) for i, text in enumerate(samples)],
        tok, encode_messages, a.max_seq_len, a.max_new_tokens, a.max_chunks,
        security_prompt=False, min_prompt_tokens=8)
    cal = _run_calibration(
        model, inputs, generate, tok.eos_token_id, prof, a.max_seq_len,
        a.max_new_tokens, a.seed, torch.device("cuda"),
        (lambda message: print(f"[easyep]{message}", flush=True)) if rank == 0 else None)
    _validate_profile_observations(prof, args.n_hash_layers)

    if rank == 0:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        st = prof.state(args.n_hash_layers, model_identity)
        st["samples_used"] = len(samples)
        st["generated_response_tokens"] = cal["response_tokens"]
        st["source"] = str(a.calib)
        st["calibration"] = _calibration_provenance(
            [(f"sample-{i}", text) for i, text in enumerate(samples)], inputs, cal,
            seed=a.seed, temperature=args.temperature,
            max_seq_len=a.max_seq_len, max_new_tokens=a.max_new_tokens,
            max_chunks=a.max_chunks,
            # synthetic sample-N names, and raw text rather than the
            # security-review prompt the evaluation uses
            source_kind="sample_texts", security_prompt=False)
        torch.save(st, out)
        print(f"[easyep] wrote {out}  ({len(samples)} sources / {len(inputs)} chunks)",
              flush=True)
    if world_size > 1:
        dist.destroy_process_group()


def cmd_mask(a) -> None:
    st = _require_score_artifact(
        torch.load(a.scores, map_location="cpu"), a.scores,
        n_hash_layers=a.n_hash_layers)
    key = ("gate_sums" if a.gating_score else
           "score_no_simibr" if a.no_simibr else "score")
    scores = st[key]
    n_layers, n_experts = _validate_scores(scores, a.keep, a.n_hash_layers)
    keep_n = a.keep
    _validate_keep(keep_n, n_experts, st.get("gate_topk"))
    _validate_score_state_observations(st, a.n_hash_layers, key)
    mask = build_mask_from_scores(scores, keep_n, a.n_hash_layers, score_key=key)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mask, indent=1), encoding="utf-8")

    dead = int((scores[a.n_hash_layers:] == 0).sum())
    print(f"[easyep] mask -> {a.out}")
    print(f"[easyep] keep {keep_n}/{n_experts} per layer on layers "
          f"{a.n_hash_layers}..{n_layers-1}; hash layers kept whole")
    print(f"[easyep] experts never activated during calibration: {dead}")


def cmd_eval(a) -> None:
    """Standalone single-variant generation.

    Delegates to _evaluate rather than repeating it. The previous copy built its
    prompt as ``q.get("question") or q.get("prompt") or json.dumps(q)``; the real
    question records carry neither field, so every prompt became a JSON dump of
    the whole record -- reference answer, CWE and grading rubric included. Two
    evaluation paths over the same data is what let that survive, so there is
    now only one.
    """
    official, model, tok, args, rank, world_size = build(
        a.ckpt_path, a.config, a.code_dir, a.max_seq_len, 1,
        getattr(a, "temperature", None)
    )
    sys.path.insert(0, str(Path(a.code_dir).parent / "encoding"))
    from encoding_dsv4 import encode_messages
    sys.path.insert(0, a.code_dir)
    from generate import generate

    dev = torch.device("cuda")
    prof = Profiler(args.n_layers, args.n_routed_experts, dev)
    patch(official, prof)

    mask = json.loads(Path(a.mask).read_text(encoding="utf-8")) if a.mask else None
    tag = "pruned" if mask else "full"
    # _evaluate reports the retained-expert count. Take it from the mask itself
    # so this mode needs no separate --keep that could drift out of agreement.
    a.keep = (mask.get("keep_per_layer", args.n_routed_experts) if mask
              else args.n_routed_experts)

    out = Path(a.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    def say(m):
        if rank == 0:
            print(f"[easyep] {m}", flush=True)

    # _evaluate applies the mask, seeds per question, and destroys the group.
    _evaluate(a, model, tok, args, rank, world_size, dev, out, say,
              encode_messages, generate, [(tag, mask)])


# ------------------------------------------------------------------- prompts
# Kept byte-identical to the earlier run_easy_ep.py so calibration and
# evaluation stay comparable with that session's outputs.


def security_review_text(relative_path: str, code: str) -> str:
    return f"""Perform a security review of this source file.

File: {relative_path}

Find concrete vulnerabilities. Trace attacker-controlled inputs to dangerous
operations, identify missing validation or authorization, and explain impact.
Do not assume the code is safe merely because context is incomplete.

<UNTRUSTED_CODE>
{code}
</UNTRUSTED_CODE>
"""


def question_text(q: dict) -> str:
    return f"""Review code snippet {q['id']} for security vulnerabilities.

Language: {q['language']}

<UNTRUSTED_CODE>
{q['snippet']}
</UNTRUSTED_CODE>

Answer in at most 160 words using this structure:
Verdict: VULNERABLE or SAFE
Vulnerability: precise name
CWE: CWE number if known
Reasoning: source, dangerous operation, and why it is unsafe
Impact: realistic consequence
Remediation: concrete fix
"""


def sample_calibration_files(root: Path, n: int, seed: int = 42) -> list[tuple[str, str]]:
    """Stratified sample across CWE directories, so no single class dominates."""
    import random
    if n < 1:
        raise ValueError("n_calib must be positive")
    if not root.is_dir():
        raise ValueError(f"calibration root is not a directory: {root}")
    by_cwe: dict[str, list[Path]] = {}
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix in {".js", ".ts", ".jsx", ".tsx"}:
            cwe = next((part for part in f.relative_to(root).parts if part.startswith("CWE-")), "other")
            by_cwe.setdefault(cwe, []).append(f)
    n_eligible = sum(len(paths) for paths in by_cwe.values())
    if n_eligible < n:
        raise ValueError(
            f"requested {n} calibration files but only {n_eligible} eligible files exist under {root}")
    rng = random.Random(seed)
    for v in by_cwe.values():
        rng.shuffle(v)
    picked: list[Path] = []
    cwes = sorted(by_cwe)
    # When n < number of strata, a sorted round-robin is a deterministic
    # alphabetical prefix. Shuffle the strata themselves, not only files within.
    rng.shuffle(cwes)
    i = 0
    while len(picked) < n and any(by_cwe[c] for c in cwes):
        c = cwes[i % len(cwes)]
        if by_cwe[c]:
            picked.append(by_cwe[c].pop())
        i += 1
    return [(str(f.relative_to(root)), f.read_text(encoding="utf-8", errors="ignore")) for f in picked]


def _calibration_chunks(files: list[tuple[str, str]], tok, encode_messages,
                        max_seq_len: int, max_new_tokens: int,
                        max_chunks: int, *, security_prompt: bool = True,
                        min_prompt_tokens: int = 32) -> list[dict]:
    """Create shared, token-exact calibration inputs without dropping source tails.

    Truncating to the first N tokens silently drops everything after it -- and a
    sink often sits far below its source, so the routing behaviour that matters
    is exactly what gets cut. Measured on this corpus: 17% of files exceed a
    4096-token prompt, and one 9,962-token file alone accounted for 19% of the
    calibration tokens in a 25-file sample. Splitting is exact, so concatenating
    the pieces reproduces every source character.
    """
    input_limit = max_seq_len - max_new_tokens
    if max_new_tokens < 1 or input_limit < min_prompt_tokens:
        raise ValueError(
            f"max_seq_len must leave at least {min_prompt_tokens} prompt tokens after reserving "
            "max_new_tokens for the generated response")
    if max_chunks < 0:
        raise ValueError("max_chunks must be non-negative (0 means unlimited)")
    out = []

    def encode_piece(relative_path: str, piece: str) -> list[int]:
        content = (security_review_text(relative_path, piece)
                   if security_prompt else piece)
        prompt = encode_messages([{"role": "user", "content": content}],
                                 thinking_mode=THINKING_MODE)
        return tok.encode(prompt)

    def split_to_fit(relative_path: str, piece: str) -> list[tuple[str, list[int]]]:
        ids = encode_piece(relative_path, piece)
        if len(ids) <= input_limit:
            return [(piece, ids)]
        if len(piece) < 2:
            raise ValueError(
                f"prompt scaffolding for {relative_path} alone exceeds the token budget")
        midpoint = len(piece) // 2
        before = piece.rfind("\n", 1, midpoint + 1)
        after = piece.find("\n", midpoint)
        candidates = [point + 1 for point in (before, after)
                      if point >= 0 and point + 1 < len(piece)]
        cut = min(candidates, key=lambda point: abs(point - midpoint)) if candidates else midpoint
        # Exact slicing is intentional: concatenating the pieces reproduces every
        # source character, including line endings.
        return (split_to_fit(relative_path, piece[:cut])
                + split_to_fit(relative_path, piece[cut:]))

    for file_index, (relative_path, code) in enumerate(files):
        pieces = split_to_fit(relative_path, code)
        if max_chunks and len(pieces) > max_chunks:
            raise ValueError(
                f"{relative_path} requires {len(pieces)} token-exact chunks, exceeding "
                f"explicit --max-chunks={max_chunks}; increase it or use 0 for unlimited")
        for chunk_index, (_, ids) in enumerate(pieces):
            if len(ids) < min_prompt_tokens:
                raise ValueError(
                    f"calibration chunk {chunk_index + 1} for {relative_path} has only "
                    f"{len(ids)} tokens")
            out.append({"path": relative_path, "file_index": file_index,
                        "chunk_index": chunk_index, "n_chunks": len(pieces),
                        "prompt_ids": ids})
    return out


def _completion_token_ids(generated, prompt_ids: list[int]) -> list[int]:
    """Normalise generate() output and tolerate APIs returning prompt+completion."""
    if generated is None or len(generated) != 1:
        raise RuntimeError("generation must return exactly one sequence")
    seq = generated[0]
    if isinstance(seq, torch.Tensor):
        seq = seq.detach().cpu().tolist()
    seq = [int(token) for token in seq]
    if len(seq) >= len(prompt_ids) and seq[:len(prompt_ids)] == prompt_ids:
        seq = seq[len(prompt_ids):]
    return seq


def _decode_completion(tok, generated, prompt_ids: list[int]) -> tuple[str, int]:
    """Decode only the generated continuation, never the prompt.

    Every generate() consumer must agree on the API's contract. The pinned
    generator returns prompt+completion, so decoding element 0 whole would store
    the prompt -- including the answer scaffolding -- as the model's answer.
    _completion_token_ids strips it when present and is a no-op otherwise, so
    this stays correct under either convention.
    """
    ids = _completion_token_ids(generated, prompt_ids)
    return tok.decode(ids), len(ids)


def _run_calibration(model, inputs: list[dict], generate, eos_token_id: int,
                     profiler: Profiler, max_seq_len: int, max_new_tokens: int,
                     seed: int, device, say=None) -> dict:
    """Generate with profiling off, then profile the exact prompt+response tokens.

    The original EASY-EP calibration examples are complete model trajectories,
    not prompt-only prefills. Generation remains unobserved so prompt tokens are
    counted once, in the subsequent teacher-forced full-sequence pass.
    """
    if not inputs:
        raise ValueError("calibration produced no eligible prompt chunks")
    used = response_tokens = 0
    generated_responses = []
    profiler.enabled = False
    for run_index, item in enumerate(inputs):
        prompt_ids = item["prompt_ids"]
        torch.manual_seed(seed + run_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + run_index)
        with torch.inference_mode():
            generated = generate(model, [prompt_ids], max_new_tokens, eos_token_id)
        completion_ids = _completion_token_ids(generated, prompt_ids)
        if not completion_ids:
            raise RuntimeError(
                f"calibration generation returned no response tokens for {item.get('path', run_index)}")
        # Official V4 generate() appends a synthetic EOS after its decode loop.
        # Only the budget-overflow form is outside max_new_tokens; discard that
        # sentinel while retaining every model-generated token.
        if (len(completion_ids) == max_new_tokens + 1
                and completion_ids[-1] == eos_token_id):
            completion_ids = completion_ids[:-1]
        if len(completion_ids) > max_new_tokens:
            raise RuntimeError("calibration generator exceeded max_new_tokens")
        if len(prompt_ids) + len(completion_ids) > max_seq_len:
            raise RuntimeError(
                "calibration generator exceeded the reserved response budget; "
                "refusing to truncate the prompt+response trajectory")
        full_ids = prompt_ids + completion_ids
        try:
            profiler.enabled = True
            with torch.inference_mode():
                model.forward(torch.tensor([full_ids], dtype=torch.long, device=device), 0)
        finally:
            profiler.enabled = False
        used += 1
        response_tokens += len(full_ids) - len(prompt_ids)
        generated_responses.append({
            "run_index": run_index,
            "path": str(item.get("path", "")),
            "file_index": int(item.get("file_index", run_index)),
            "chunk_index": int(item.get("chunk_index", 0)),
            "tokens": len(completion_ids),
            "token_ids_sha256": _ids_sha256(completion_ids),
        })
        if say is not None:
            chunk = (f" chunk {item['chunk_index'] + 1}/{item['n_chunks']}"
                     if item.get("n_chunks", 1) > 1 else "")
            say(f"  calib {item.get('file_index', run_index) + 1}{chunk}  "
                f"{len(prompt_ids)}+{len(full_ids) - len(prompt_ids)} tok  "
                f"{item.get('path', '')}")
    if used < 1 or response_tokens < 1:
        raise ValueError("calibration observed no generated-response tokens")
    return {"forwards": used, "response_tokens": response_tokens,
            "generated_responses": generated_responses}


def _ids_sha256(ids: list[int]) -> str:
    payload = json.dumps([int(token) for token in ids], separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _calibration_provenance(files: list[tuple[str, str]], inputs: list[dict],
                            result: dict, *, seed: int, temperature: float,
                            max_seq_len: int, max_new_tokens: int,
                            max_chunks: int, source_kind: str,
                            security_prompt: bool) -> dict:
    """Content-bind a reusable score tensor to its calibration trajectory.

    source_kind and security_prompt describe the calibration *distribution*.
    The per-chunk token hashes already bind it, but nothing downstream can
    interpret a hash: two artifacts profiled on different prompt distributions,
    or on samples that have no corpus paths at all, were indistinguishable.
    Consumers that depend on either property now check it explicitly.
    """
    if source_kind not in ("corpus_files", "sample_texts"):
        raise ValueError(f"unknown calibration source_kind: {source_kind!r}")
    record = {
        "provenance_schema_version": 1,
        "source_kind": source_kind,
        "security_prompt": bool(security_prompt),
        # The thinking mode changes what the model emits and therefore which
        # experts route, so it identifies the calibration distribution just as
        # much as the prompt does.
        "thinking_mode": THINKING_MODE,
        "source_samples": len(files),
        "chunks": len(inputs),
        "forwards": int(result["forwards"]),
        "response_tokens": int(result["response_tokens"]),
        "parameters": {
            "seed": int(seed),
            "temperature": float(temperature),
            "max_seq_len": int(max_seq_len),
            "max_new_tokens": int(max_new_tokens),
            "max_chunks": int(max_chunks),
        },
        "selected_sources": [
            {"file_index": index, "path": path, "characters": len(text),
             "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
            for index, (path, text) in enumerate(files)
        ],
        "prompt_chunks": [
            {"file_index": int(item["file_index"]),
             "chunk_index": int(item["chunk_index"]),
             "n_chunks": int(item["n_chunks"]),
             "path": str(item["path"]),
             "prompt_tokens": len(item["prompt_ids"]),
             "prompt_ids_sha256": _ids_sha256(item["prompt_ids"])}
            for item in inputs
        ],
        "generated_responses": list(result["generated_responses"]),
    }
    _validate_calibration_provenance(record)
    return record


def _require_evaluation_calibration(state: dict, source: str, *,
                                    need_corpus_paths: bool = False) -> dict:
    """Reject a schema-valid artifact that cannot support an evaluation.

    A score artifact can be perfectly well-formed and still be the wrong thing
    to evaluate against, so every evaluation consumer checks this -- not just
    the one where a specific symptom was noticed first.

    security_prompt is universal: masks built from raw-text calibration encode a
    different routing distribution than the security-review prompt these
    evaluations use. need_corpus_paths is for consumers that additionally rely
    on selected_sources naming real corpus files, which today is matched-pair
    selection excluding the profiled programs.
    """
    calibration = state.get("calibration")
    if not isinstance(calibration, dict):
        raise SystemExit(f"{source} has no calibration provenance; re-profile it")
    if calibration.get("security_prompt") is not True:
        raise SystemExit(
            f"{source} was profiled without the security-review prompt, so its "
            "routing statistics come from a different distribution than this "
            "evaluation. Re-profile with `pipeline --calib-dir`.")
    if calibration.get("thinking_mode") != THINKING_MODE:
        raise SystemExit(
            f"{source} was profiled in thinking_mode "
            f"{calibration.get('thinking_mode')!r} but this build evaluates in "
            f"{THINKING_MODE!r}. The mode changes what the model emits and which "
            "experts route, so the masks would come from a different "
            "distribution than the evaluation. Re-profile.")
    if need_corpus_paths and calibration.get("source_kind") != "corpus_files":
        raise SystemExit(
            f"{source} was profiled from {calibration.get('source_kind')!r}, whose "
            "sources have no corpus paths, so matched-pair selection could not "
            "exclude the profiled files and would report zero overlap while "
            "possibly reusing one. Profile with `pipeline --calib-dir`.")
    return calibration


def _validate_calibration_provenance(record: Any, source: str = "score artifact") -> None:
    sha = re.compile(r"^[0-9a-f]{64}$")
    if not isinstance(record, dict) or record.get("provenance_schema_version") != 1:
        raise SystemExit(f"{source} is missing calibration provenance; re-profile it")
    if record.get("source_kind") not in ("corpus_files", "sample_texts"):
        raise SystemExit(
            f"{source} does not record which calibration distribution produced it; "
            "re-profile with this easyep_v4.py")
    if not isinstance(record.get("security_prompt"), bool):
        raise SystemExit(
            f"{source} does not record whether calibration used the security-review "
            "prompt; re-profile with this easyep_v4.py")
    if not isinstance(record.get("thinking_mode"), str) or not record["thinking_mode"]:
        raise SystemExit(
            f"{source} does not record the calibration thinking mode; "
            "re-profile with this easyep_v4.py")
    integer_fields = ("source_samples", "chunks", "forwards", "response_tokens")
    if any(isinstance(record.get(key), bool) or not isinstance(record.get(key), int)
           or record[key] < 1 for key in integer_fields):
        raise SystemExit(f"{source} has invalid calibration counts")
    if record["forwards"] != record["chunks"]:
        raise SystemExit(f"{source} calibration forward/chunk counts disagree")
    parameters = record.get("parameters")
    if not isinstance(parameters, dict):
        raise SystemExit(f"{source} has invalid calibration parameters")
    for key, minimum in (("seed", 0), ("max_seq_len", 2),
                         ("max_new_tokens", 1), ("max_chunks", 0)):
        value = parameters.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise SystemExit(f"{source} has invalid calibration parameter {key}")
    temperature = parameters.get("temperature")
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature < 0):
        raise SystemExit(f"{source} has invalid calibration temperature")
    if parameters["max_new_tokens"] >= parameters["max_seq_len"]:
        raise SystemExit(f"{source} has an impossible calibration token budget")

    selected = record.get("selected_sources")
    chunks = record.get("prompt_chunks")
    responses = record.get("generated_responses")
    if (not isinstance(selected, list) or len(selected) != record["source_samples"]
            or not isinstance(chunks, list) or len(chunks) != record["chunks"]
            or not isinstance(responses, list) or len(responses) != record["forwards"]):
        raise SystemExit(f"{source} calibration source/chunk lists disagree with counts")
    source_indices = set()
    source_paths = {}
    for item in selected:
        if (not isinstance(item, dict)
                or isinstance(item.get("file_index"), bool)
                or not isinstance(item.get("file_index"), int)
                or item["file_index"] < 0
                or not isinstance(item.get("path"), str) or not item["path"]
                or isinstance(item.get("characters"), bool)
                or not isinstance(item.get("characters"), int) or item["characters"] < 0
                or not isinstance(item.get("text_sha256"), str)
                or sha.fullmatch(item["text_sha256"]) is None):
            raise SystemExit(f"{source} has malformed selected-source provenance")
        source_indices.add(item["file_index"])
        source_paths[item["file_index"]] = item["path"]
    if source_indices != set(range(record["source_samples"])):
        raise SystemExit(f"{source} calibration source indices are incomplete")
    if len(set(source_paths.values())) != len(source_paths):
        raise SystemExit(f"{source} calibration source paths are duplicated")
    seen_chunks = set()
    chunks_by_file: dict[int, set[int]] = {index: set() for index in source_indices}
    declared_chunks_by_file = {}
    for item in chunks:
        if (not isinstance(item, dict)
                or item.get("file_index") not in source_indices
                or isinstance(item.get("chunk_index"), bool)
                or not isinstance(item.get("chunk_index"), int) or item["chunk_index"] < 0
                or isinstance(item.get("n_chunks"), bool)
                or not isinstance(item.get("n_chunks"), int) or item["n_chunks"] < 1
                or not isinstance(item.get("path"), str) or not item["path"]
                or isinstance(item.get("prompt_tokens"), bool)
                or not isinstance(item.get("prompt_tokens"), int) or item["prompt_tokens"] < 1
                or not isinstance(item.get("prompt_ids_sha256"), str)
                or sha.fullmatch(item["prompt_ids_sha256"]) is None):
            raise SystemExit(f"{source} has malformed prompt-chunk provenance")
        if item["path"] != source_paths[item["file_index"]]:
            raise SystemExit(f"{source} calibration chunk/source paths disagree")
        key = (item["file_index"], item["chunk_index"])
        if key in seen_chunks or item["chunk_index"] >= item["n_chunks"]:
            raise SystemExit(f"{source} has duplicate/inconsistent calibration chunks")
        seen_chunks.add(key)
        chunks_by_file[item["file_index"]].add(item["chunk_index"])
        previous = declared_chunks_by_file.setdefault(item["file_index"], item["n_chunks"])
        if previous != item["n_chunks"]:
            raise SystemExit(f"{source} has inconsistent per-source chunk counts")
    for index in source_indices:
        declared = declared_chunks_by_file.get(index)
        if declared is None or chunks_by_file[index] != set(range(declared)):
            raise SystemExit(f"{source} calibration chunks are incomplete for source {index}")
    total_response_tokens = 0
    seen_runs = set()
    seen_response_chunks = set()
    for item in responses:
        if (not isinstance(item, dict)
                or isinstance(item.get("run_index"), bool)
                or not isinstance(item.get("run_index"), int) or item["run_index"] < 0
                or item.get("file_index") not in source_indices
                or isinstance(item.get("chunk_index"), bool)
                or not isinstance(item.get("chunk_index"), int) or item["chunk_index"] < 0
                or not isinstance(item.get("path"), str) or not item["path"]
                or isinstance(item.get("tokens"), bool)
                or not isinstance(item.get("tokens"), int) or item["tokens"] < 1
                or not isinstance(item.get("token_ids_sha256"), str)
                or sha.fullmatch(item["token_ids_sha256"]) is None):
            raise SystemExit(f"{source} has malformed generated-response provenance")
        response_key = (item["file_index"], item["chunk_index"])
        if (item["path"] != source_paths[item["file_index"]]
                or response_key not in seen_chunks):
            raise SystemExit(f"{source} response/source chunk provenance disagrees")
        if item["run_index"] in seen_runs:
            raise SystemExit(f"{source} has duplicate calibration response indices")
        if response_key in seen_response_chunks:
            raise SystemExit(f"{source} has duplicate calibration response chunks")
        seen_runs.add(item["run_index"])
        seen_response_chunks.add(response_key)
        total_response_tokens += item["tokens"]
    if seen_runs != set(range(record["forwards"])):
        raise SystemExit(f"{source} calibration response indices are incomplete")
    if seen_response_chunks != seen_chunks:
        raise SystemExit(f"{source} calibration responses do not cover every prompt chunk")
    if total_response_tokens != record["response_tokens"]:
        raise SystemExit(f"{source} calibration response-token counts disagree")



def build_mask_from_scores(scores: torch.Tensor, keep_n: int, n_hash: int,
                           score_key: str | None = None) -> dict:
    n_layers, n_experts = _validate_scores(scores, keep_n, n_hash)
    mask = {"keep_per_layer": keep_n, "n_experts": n_experts,
            "protected_hash_layers": list(range(n_hash)), "layers": {}}
    if score_key is not None:
        mask["score_key"] = score_key
    for lid in range(n_layers):
        if lid < n_hash:
            mask["layers"][str(lid)] = {"kept": list(range(n_experts)), "pruned": []}
            continue
        order = torch.argsort(scores[lid], descending=True)
        mask["layers"][str(lid)] = {"kept": sorted(order[:keep_n].tolist()),
                                    "pruned": sorted(order[keep_n:].tolist())}
    return mask


def score_cutoff_diagnostics(scores: torch.Tensor, keep_n: int, n_hash: int) -> dict:
    """Report how securely the final retained expert clears the first pruned one."""
    n_layers, n_experts = _validate_scores(scores, keep_n, n_hash)
    rows = []
    for lid in range(n_hash, n_layers):
        ordered = torch.sort(scores[lid].double(), descending=True).values
        retained = float(ordered[keep_n - 1].item())
        if keep_n == n_experts:
            pruned = margin = relative = None
        else:
            pruned = float(ordered[keep_n].item())
            margin = retained - pruned
            scale = max(abs(retained), abs(pruned), torch.finfo(torch.float64).tiny)
            relative = margin / scale
        # Integer-valued rankings (the frequency control) tie constantly, and
        # topk breaks ties by index, so a zero margin means the cut is index
        # order rather than signal. Report how many experts sit exactly on the
        # boundary value so that is visible instead of inferred.
        tied = (None if keep_n == n_experts
                else int((scores[lid] == ordered[keep_n - 1]).sum().item()))
        rows.append({
            "layer": lid,
            "rank_retained": keep_n,
            "rank_pruned": None if keep_n == n_experts else keep_n + 1,
            "retained_score": retained,
            "pruned_score": pruned,
            "absolute_margin": margin,
            "relative_margin": relative,
            "experts_tied_at_cutoff": tied,
            "cut_is_arbitrary": None if margin is None else margin == 0.0,
        })
    finite_rows = [row for row in rows if row["relative_margin"] is not None]
    minimum = (min(finite_rows, key=lambda row: row["relative_margin"])
               if finite_rows else None)
    return {
        "keep_per_layer": keep_n,
        "layers_compared": [n_hash, n_layers - 1] if rows else [],
        "minimum_relative_margin": (None if minimum is None
                                    else minimum["relative_margin"]),
        "minimum_margin_layer": None if minimum is None else minimum["layer"],
        "layers_with_arbitrary_cut": sum(1 for row in rows
                                         if row["cut_is_arbitrary"]),
        "max_experts_tied_at_cutoff": max(
            (row["experts_tied_at_cutoff"] for row in rows
             if row["experts_tied_at_cutoff"] is not None), default=None),
        "per_layer": rows,
    }


def compare_scorings(scores: torch.Tensor, scores_alt: torch.Tensor,
                     mask: dict, mask_alt: dict,
                     keep: int, n_hash: int, n_layers: int) -> dict:
    """Per-layer comparison of the two scoring rules.

    scores      paper's rule: sum_t weight * simibr * norm
    scores_alt  ablation:     sum_t weight * norm      (no token contribution)

    The headline is the top-`keep` expert overlap per layer: how often the
    token-contribution term actually changes which experts survive.
    """
    def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
        n = x.numel()
        rx = torch.argsort(torch.argsort(x)).float()
        ry = torch.argsort(torch.argsort(y)).float()
        rx = rx - rx.mean()
        ry = ry - ry.mean()
        d = (rx.norm() * ry.norm()).clamp_min(1e-12)
        return round(float((rx * ry).sum() / d), 4)

    paper_cutoffs = score_cutoff_diagnostics(scores, keep, n_hash)
    alt_cutoffs = score_cutoff_diagnostics(scores_alt, keep, n_hash)
    paper_cutoff_by_layer = {row["layer"]: row for row in paper_cutoffs["per_layer"]}
    alt_cutoff_by_layer = {row["layer"]: row for row in alt_cutoffs["per_layer"]}
    rows = []
    for lid in range(n_hash, n_layers):
        k1 = set(mask["layers"][str(lid)]["kept"])
        k2 = set(mask_alt["layers"][str(lid)]["kept"])
        inter = k1 & k2
        rows.append({
            "layer": lid,
            "overlap": len(inter),
            "overlap_frac": round(len(inter) / max(keep, 1), 4),
            "jaccard": round(len(inter) / max(len(k1 | k2), 1), 4),
            "spearman_full_ranking": spearman(scores[lid], scores_alt[lid]),
            "only_in_paper_score": sorted(k1 - k2),
            "only_in_no_simibr": sorted(k2 - k1),
            "n_never_activated": int((scores[lid] == 0).sum()),
            "paper_cutoff": paper_cutoff_by_layer[lid],
            "no_simibr_cutoff": alt_cutoff_by_layer[lid],
        })
    ov = [r["overlap"] for r in rows]
    worst = min(rows, key=lambda r: r["overlap"])
    return {
        "keep_per_layer": keep,
        "layers_compared": [n_hash, n_layers - 1],
        "scoring_a": "paper: sum_t weight * simibr * norm",
        "scoring_b": "ablation: sum_t weight * norm (no simibr)",
        "overlap_mean": round(sum(ov) / len(ov), 2),
        "overlap_mean_frac": round(sum(ov) / len(ov) / max(keep, 1), 4),
        "overlap_min": min(ov),
        "overlap_max": max(ov),
        "min_overlap_layer": worst["layer"],
        "spearman_mean": round(
            sum(r["spearman_full_ranking"] for r in rows) / len(rows), 4),
        "paper_cutoff_minimum_relative_margin":
            paper_cutoffs["minimum_relative_margin"],
        "paper_cutoff_minimum_margin_layer": paper_cutoffs["minimum_margin_layer"],
        "no_simibr_cutoff_minimum_relative_margin":
            alt_cutoffs["minimum_relative_margin"],
        "no_simibr_cutoff_minimum_margin_layer": alt_cutoffs["minimum_margin_layer"],
        "per_layer": rows,
    }


def build_frequency_mask(counts: torch.Tensor, keep_n: int, n_hash: int) -> dict:
    """Baseline: keep the most-often-selected experts. The naive heuristic that
    any sensible person tries first -- if the paper's score cannot beat this,
    the extra machinery is not earning its place."""
    return build_mask_from_scores(counts, keep_n, n_hash, score_key="counts")


def build_gating_mask(gate_sums: torch.Tensor, keep_n: int, n_hash: int) -> dict:
    """Paper baseline: keep experts with the largest total activated gate weight."""
    return build_mask_from_scores(gate_sums, keep_n, n_hash, score_key="gate_sums")


def build_random_mask(n_layers: int, n_experts: int, keep_n: int, n_hash: int, seed: int) -> dict:
    """Floor: a seeded random subset. If this matches the scored masks, the
    scoring carries no signal and the model is simply robust to expert removal."""
    _validate_keep(keep_n, n_experts)
    if not 0 <= n_hash <= n_layers:
        raise ValueError("n_hash_layers is outside the requested layer range")
    # build() sets the default device to cuda, so tensor creation must name CPU
    # explicitly or it will try to pair a CPU generator with a CUDA allocation.
    g = torch.Generator(device="cpu").manual_seed(seed)
    mask = {"keep_per_layer": keep_n, "n_experts": n_experts,
            "protected_hash_layers": list(range(n_hash)),
            "score_key": "random", "seed": seed, "layers": {}}
    for lid in range(n_layers):
        if lid < n_hash:
            mask["layers"][str(lid)] = {"kept": list(range(n_experts)), "pruned": []}
            continue
        perm = torch.randperm(n_experts, generator=g, device="cpu")
        mask["layers"][str(lid)] = {"kept": sorted(perm[:keep_n].tolist()),
                                    "pruned": sorted(perm[keep_n:].tolist())}
    return mask


def all_variants(scores, scores_alt, counts, gate_sums,
                 keep_n, n_hash, n_layers, n_experts,
                 seed: int, controls: bool, scores_mhc=None,
                 scores_reduced_legacy=None):
    """The variant set, in a fixed order so runs stay comparable.

    ``pruned_paper`` is the architecture-aware schema-v2 score. The old reduced
    h -> h+y_routed approximation is explicitly labelled as a legacy diagnostic.
    """
    expected = (n_layers, n_experts)
    for name, tensor in (("score", scores), ("score_no_simibr", scores_alt),
                         ("counts", counts), ("gate_sums", gate_sums)):
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected:
            raise ValueError(f"{name} shape must be {expected}, got {getattr(tensor, 'shape', None)}")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} contains NaN or infinity")
    _validate_scores(scores, keep_n, n_hash)
    _validate_scores(scores_alt, keep_n, n_hash)
    if scores_mhc is not None:
        if (not isinstance(scores_mhc, torch.Tensor)
                or tuple(scores_mhc.shape) != expected
                or not torch.equal(scores_mhc, scores)):
            raise ValueError(
                "score_mhc must be an exact compatibility alias of primary score in schema v2")
    if scores_reduced_legacy is not None:
        if (not isinstance(scores_reduced_legacy, torch.Tensor)
                or tuple(scores_reduced_legacy.shape) != expected):
            raise ValueError(f"score_reduced_legacy shape must be {expected}")
        if not bool(torch.isfinite(scores_reduced_legacy).all().item()):
            raise ValueError("score_reduced_legacy contains NaN or infinity")
    v = [("full", None),
         ("pruned_paper", build_mask_from_scores(
             scores, keep_n, n_hash, score_key="score"))]
    if scores_reduced_legacy is not None:
        v.append(("pruned_reduced_legacy",
                  build_mask_from_scores(
                      scores_reduced_legacy, keep_n, n_hash,
                      score_key="score_reduced_legacy")))
    v.append(("pruned_no_simibr", build_mask_from_scores(
        scores_alt, keep_n, n_hash, score_key="score_no_simibr")))
    if controls:
        v.append(("pruned_gating", build_gating_mask(gate_sums, keep_n, n_hash)))
        v.append(("pruned_frequency", build_frequency_mask(counts, keep_n, n_hash)))
        v.append(("pruned_random", build_random_mask(n_layers, n_experts, keep_n, n_hash, seed)))
    return v


VERDICT_PROMPT = """Perform a security review of this source file.

File: {path}

<UNTRUSTED_CODE>
{code}
</UNTRUSTED_CODE>

Answer in at most 60 words, exactly this structure:
Verdict: VULNERABLE or SAFE
Vulnerability: precise name, or NONE
Reasoning: one sentence
"""


def load_pairs(manifest: Path, root: Path, n: int, seed: int = 42,
               accept=None, stats: dict | None = None,
               exclude_relative_paths: set[str] | None = None) -> list[dict]:
    """Matched vulnerable/secure file pairs from the CodeQL manifest.

    The secure file is the SAME file with the flagged expression neutralised and
    re-scanned clean, so a model that simply calls everything vulnerable scores
    100% on one half and 0% on the other. Recall-only metrics cannot see that;
    this can.
    """
    import random
    if n < 1:
        raise ValueError("n_pairs must be positive")
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    selection = {"manifest_rows": len(rows), "rows_examined": 0,
                 "skipped_no_alert": 0, "skipped_missing_file": 0,
                 "skipped_unusable_content": 0, "skipped_by_acceptance": 0,
                 "skipped_calibration_overlap": 0}
    rng = random.Random(seed)
    rng.shuffle(rows)
    root = root.resolve()
    base = root.parent
    excluded = {Path(path).as_posix() for path in (exclude_relative_paths or set())}
    out = []
    for r in rows:
        selection["rows_examined"] += 1
        # Ground truth here is "CodeQL raised alerts on this file, and the
        # neutralised copy rescans clean" -- not a human exploitability judgement.
        # Require actual alerts so the vulnerable label means something.
        if int(r.get("alert_locations", 0)) < 1:
            selection["skipped_no_alert"] += 1
            continue
        try:
            vp = (base / r["original_file"]).resolve()
            sp = (base / r["secure_file"]).resolve()
            vrel = vp.relative_to(root).as_posix()
            srel = sp.relative_to(root).as_posix()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("pair manifest path escapes or does not identify the data root") from exc
        if vrel in excluded or srel in excluded:
            selection["skipped_calibration_overlap"] += 1
            continue
        if not (vp.is_file() and sp.is_file()):
            selection["skipped_missing_file"] += 1
            continue
        try:
            vb, sb = vp.read_bytes(), sp.read_bytes()
        except OSError:
            selection["skipped_missing_file"] += 1
            continue
        for label, payload, expected in (
                ("original", vb, r.get("original_sha256")),
                ("secure", sb, r.get("secure_sha256"))):
            if expected and hashlib.sha256(payload).hexdigest().lower() != str(expected).lower():
                raise ValueError(
                    f"{label} SHA-256 mismatch for manifest row {r.get('original_file')}")
        vc, sc = vb.decode("utf-8", errors="ignore"), sb.decode("utf-8", errors="ignore")
        if not vc.strip() or not sc.strip() or vc == sc:
            selection["skipped_unusable_content"] += 1
            continue
        candidate = {"pair_id": len(out), "queries": r.get("queries", []),
                     "alert_locations": int(r.get("alert_locations", 0)),
                     "vuln_path": r["original_file"], "vuln_code": vc,
                     "safe_path": r["secure_file"], "safe_code": sc,
                     "original_sha256": r.get("original_sha256"),
                     "secure_sha256": r.get("secure_sha256")}
        if accept is not None and not bool(accept(candidate)):
            selection["skipped_by_acceptance"] += 1
            continue
        out.append(candidate)
        if len(out) >= n:
            break
    selection["selected"] = len(out)
    if stats is not None:
        stats.clear()
        stats.update(selection)
    if len(out) != n:
        raise ValueError(
            f"requested {n} valid/fitting matched pairs but found only {len(out)}; "
            f"selection={selection}")
    return out


def _pair_eval_items(pairs: list[dict]) -> list[tuple[int, str, str, str]]:
    """Return label-neutral pair inputs; both members receive the same display path."""
    items = []
    for pair in pairs:
        suffix = Path(pair["vuln_path"]).suffix.lower()
        display_path = f"review_target{suffix if suffix else '.txt'}"
        items.append((pair["pair_id"], "VULNERABLE", display_path, pair["vuln_code"]))
        items.append((pair["pair_id"], "SAFE", display_path, pair["safe_code"]))
    return items


def final_message(parse_message, completion: str) -> str:
    """Return the answer without the reasoning trace.

    In reasoning mode the completion carries the chain of thought ahead of the
    answer, and deliberation legitimately contains discarded hypotheses -- a
    "Verdict: SAFE" the model then argues itself out of. parse_verdict treats
    two conflicting valid verdicts as an abstention, so scanning the trace would
    turn ordinary reasoning into systematic unparsed rows. Grade the answer, not
    the thinking; the full completion is still stored and still judged.
    """
    try:
        message = parse_message(completion, thinking_mode=THINKING_MODE)
    except Exception:
        return completion
    if isinstance(message, str):
        return message if message.strip() else completion
    if isinstance(message, dict):
        for key in ("content", "message", "text"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return completion


def parse_verdict(text: str) -> str | None:
    verdicts = set()
    for line in text.splitlines():
        clean = line.strip().replace("**", "")
        field = re.fullmatch(r"Verdict\s*:\s*(.*)", clean, flags=re.IGNORECASE)
        if field:
            value = field.group(1).strip()
            match = re.fullmatch(
                r"\(?\s*(VULNERABLE|SAFE)\s*\)?"
                r"(?:\s*\((?![^)]*\b(?:VULNERABLE|SAFE)\b)[^()]*\))?\s*[.!]?",
                value, flags=re.IGNORECASE)
            if match:
                verdicts.add(match.group(1).upper())
    # Repeated agreement is fine; contradictory valid answers are an abstention.
    return next(iter(verdicts)) if len(verdicts) == 1 else None


def discrimination_stats(rows: list[dict]) -> dict:
    """Coverage-aware discrimination statistics.

    The headline TPR/TNR penalise an unparsed verdict as an error.  Because a
    rejected SAFE item is neither an explicit false positive nor a true
    negative, report the explicit false-positive, abstention, and combined
    safe-error rates separately.  Youden's J uses the combined safe-error rate:
    ``J = TPR + TNR - 1 = TPR - safe_error_rate``.
    """
    v = [r for r in rows if r["truth"] == "VULNERABLE"]
    s = [r for r in rows if r["truth"] == "SAFE"]
    def ratio(num, den):
        return num / den if den else None

    def rounded(value):
        return round(value, 4) if value is not None else None

    # Abstention is an error in the headline metrics for either class.  Keep it
    # distinct from an explicit VULNERABLE false alarm so the output remains a
    # complete three-way accounting rather than calling a rejection a positive.
    tpr = ratio(sum(1 for r in v if r["verdict"] == "VULNERABLE"), len(v))
    tnr = ratio(sum(1 for r in s if r["verdict"] == "SAFE"), len(s))
    explicit_fpr = ratio(sum(1 for r in s if r["verdict"] == "VULNERABLE"), len(s))
    safe_abstention_rate = ratio(sum(1 for r in s if r["verdict"] is None), len(s))
    safe_error_rate = ratio(sum(1 for r in s if r["verdict"] != "SAFE"), len(s))
    j = None if tpr is None or tnr is None else tpr + tnr - 1
    balacc = None if tpr is None or tnr is None else (tpr + tnr) / 2
    pv = [r for r in v if r["verdict"] is not None]
    ps = [r for r in s if r["verdict"] is not None]
    ptpr = ratio(sum(1 for r in pv if r["verdict"] == "VULNERABLE"), len(pv))
    ptnr = ratio(sum(1 for r in ps if r["verdict"] == "SAFE"), len(ps))
    pfpr = ratio(sum(1 for r in ps if r["verdict"] == "VULNERABLE"), len(ps))
    pj = None if ptpr is None or ptnr is None else ptpr + ptnr - 1
    pbalacc = None if ptpr is None or ptnr is None else (ptpr + ptnr) / 2
    unparsed = sum(1 for r in rows if r["verdict"] is None)
    return {
        "n_vulnerable": len(v), "n_safe": len(s),
        "verdict_coverage": rounded(ratio(len(rows) - unparsed, len(rows))),
        "tpr_recall": rounded(tpr),
        "tnr_specificity": rounded(tnr),
        "fpr_explicit_vulnerable": rounded(explicit_fpr),
        "safe_abstention_rate": rounded(safe_abstention_rate),
        "safe_error_rate": rounded(safe_error_rate),
        "youden_j": rounded(j),
        "balanced_accuracy": rounded(balacc),
        "parsed_only": {
            "n_vulnerable": len(pv), "n_safe": len(ps),
            "tpr_recall": rounded(ptpr), "tnr_specificity": rounded(ptnr),
            "fpr_explicit_vulnerable": rounded(pfpr), "youden_j": rounded(pj),
            "balanced_accuracy": rounded(pbalacc),
        },
        "always_vulnerable_would_score": {
            "tpr_recall": 1.0, "fpr_explicit_vulnerable": 1.0,
            "safe_abstention_rate": 0.0, "safe_error_rate": 1.0,
            "youden_j": 0.0, "balanced_accuracy": 0.5,
        },
        "unparsed_verdicts": unparsed,
        "mean_words": round(sum(len(r["completion"].split()) for r in rows) / max(len(rows), 1), 1),
    }


def cmd_pairs(a) -> None:
    """Matched-pair discrimination eval across every variant."""
    official, model, tok, args, rank, world_size = build(
        a.ckpt_path, a.config, a.code_dir, a.max_seq_len, 1, a.temperature)
    model_identity = _model_identity(a.ckpt_path, a.config, a.code_dir, world_size)
    sys.path.insert(0, str(Path(a.code_dir).parent / "encoding"))
    from encoding_dsv4 import encode_messages, parse_message_from_completion_text
    sys.path.insert(0, a.code_dir)
    from generate import generate

    dev = torch.device("cuda")
    prof = Profiler(args.n_layers, args.n_routed_experts, dev)
    prof.gate_topk = int(args.n_activated_experts)
    patch(official, prof)
    out = Path(a.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    def say(m):
        if rank == 0:
            print(f"[easyep] {m}", flush=True)

    st = _require_score_artifact(
        torch.load(a.scores_in, map_location="cpu"), a.scores_in,
        n_hash_layers=args.n_hash_layers, model_identity=model_identity)
    expected = (args.n_layers, args.n_routed_experts)
    if tuple(st["score"].shape) != expected:
        raise ValueError(f"score artifact shape must match model {expected}")
    if st.get("gate_topk") not in (None, int(args.n_activated_experts)):
        raise ValueError("score artifact gate top-k does not match model config")
    _validate_keep(a.keep, args.n_routed_experts, args.n_activated_experts)
    _validate_score_state_observations(st, args.n_hash_layers)
    variants = all_variants(st["score"], st["score_no_simibr"],
                            st["counts"], st["gate_sums"],
                            a.keep, args.n_hash_layers,
                            args.n_layers, args.n_routed_experts,
                            a.seed, not a.no_controls,
                            st.get("score_mhc"), st.get("score_reduced_legacy"))
    input_limit = a.max_seq_len - a.max_new_tokens
    if a.max_new_tokens < 1 or input_limit < 1:
        raise ValueError("max_seq_len leaves no room for matched-pair generation")

    def pair_fits(candidate):
        for _, _, path, code in _pair_eval_items([candidate]):
            ids = tok.encode(encode_messages(
                [{"role": "user", "content": VERDICT_PROMPT.format(path=path, code=code)}],
                thinking_mode=THINKING_MODE))
            if len(ids) > input_limit:
                return False
        return True

    selection = {}
    calibration = _require_evaluation_calibration(
        st, a.scores_in, need_corpus_paths=True)
    calibration_paths = {
        str(item["path"]) for item in calibration["selected_sources"]
    }
    pairs = load_pairs(
        Path(a.pairs_manifest), Path(a.calib_dir), a.n_pairs, a.seed,
        accept=pair_fits, stats=selection,
        exclude_relative_paths=calibration_paths)
    say(f"{len(pairs)} matched pairs -> {2*len(pairs)} items per variant")
    say(f"pair selection examined {selection['rows_examined']} rows; skipped "
        f"{selection['skipped_by_acceptance']} over-context and "
        f"{selection['skipped_calibration_overlap']} calibration-overlap pairs")
    say(f"variants: {', '.join(t for t, _ in variants)}")

    items = _pair_eval_items(pairs)
    prepared_items = []
    prompt_provenance: dict[int, dict[str, dict]] = {}
    for pid, truth, path, code in items:
        prompt = VERDICT_PROMPT.format(path=path, code=code)
        ids = tok.encode(encode_messages(
            [{"role": "user", "content": prompt}],
            thinking_mode=THINKING_MODE))
        if len(ids) > input_limit:
            raise ValueError(
                f"matched pair {pid} prompt has {len(ids)} tokens but only "
                f"{input_limit} fit with the response reservation; refusing to truncate")
        source_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        prepared_items.append({
            "pair_id": pid, "truth": truth, "path": path,
            "snippet": code, "prompt": prompt, "prompt_ids": ids,
            "source_sha256": source_sha256, "prompt_sha256": prompt_sha256,
        })
        prompt_provenance.setdefault(pid, {})[truth.lower()] = {
            "display_path": path,
            "snippet": code,
            "source_sha256": source_sha256,
            "prompt": prompt,
            "prompt_sha256": prompt_sha256,
            "prompt_tokens": len(ids),
            "prompt_ids_sha256": _ids_sha256(ids),
        }
    if rank == 0:
        provenance = {
            "seed": a.seed, "requested": a.n_pairs, "input_token_limit": input_limit,
            "decoding": {"temperature": args.temperature,
                         "max_new_tokens": a.max_new_tokens,
                         "max_seq_len": a.max_seq_len,
                         "paired_seed_rule": "seed + pair_id"},
            "score_artifact": {
                "score_schema_version": st["score_schema_version"],
                "score_semantics": st["score_semantics"],
                "model_identity_sha256": st["model_identity_sha256"],
                "calibration": st["calibration"],
            },
            "selection": selection,
            "pairs": [{**{key: pair.get(key) for key in
                           ("pair_id", "vuln_path", "safe_path", "original_sha256",
                            "secure_sha256", "queries", "alert_locations")},
                       "prompts": prompt_provenance[pair["pair_id"]]}
                      for pair in pairs],
        }
        (out / "pairs_used.json").write_text(json.dumps(provenance, indent=1))

    summary = {}
    for tag, m in variants:
        set_mask(model, m, args.n_hash_layers, dev)
        say(f"variant={tag}: {len(items)} items")
        rows = []
        for i, item in enumerate(prepared_items):
            pid, ids = item["pair_id"], item["prompt_ids"]
            # Both members of a matched pair receive identical decoder noise.
            torch.manual_seed(a.seed + pid)
            torch.cuda.manual_seed_all(a.seed + pid)
            t1 = time.time()
            with torch.inference_mode():
                gen = generate(model, [ids], a.max_new_tokens, tok.eos_token_id)
            completion, n_completion = _decode_completion(tok, gen, ids)
            answer = final_message(parse_message_from_completion_text, completion)
            if rank == 0:
                rows.append({"pair_id": pid, "truth": item["truth"],
                             "path": item["path"],
                             "snippet": item["snippet"],
                             "source_sha256": item["source_sha256"],
                             "prompt": item["prompt"],
                             "prompt_sha256": item["prompt_sha256"],
                             "prompt_ids_sha256": _ids_sha256(ids),
                             "verdict": parse_verdict(answer),
                             "answer": answer,
                             "completion": completion,
                             "seed": a.seed + pid,
                             "temperature": args.temperature,
                             "prompt_tokens": len(ids),
                             "completion_tokens": n_completion,
                             "seconds": round(time.time() - t1, 2)})
                if (i + 1) % 10 == 0:
                    say(f"  {tag} {i+1}/{len(items)}")
        if rank == 0:
            with (out / f"pairs_{tag}.jsonl").open("w", encoding="utf-8") as fp:
                for r in rows:
                    fp.write(json.dumps(r) + "\n")
            summary[tag] = discrimination_stats(rows)

    if rank == 0:
        summary["_decoding"] = {
            "temperature": args.temperature,
            "max_new_tokens": a.max_new_tokens,
            "max_seq_len": a.max_seq_len,
            "paired_seeding": True,
            "seed_base": a.seed,
            "note": "both members and every variant use torch.manual_seed(seed+pair_id)",
        }
        summary["_score_artifact"] = {
            "score_schema_version": st["score_schema_version"],
            "score_semantics": st["score_semantics"],
            "model_identity_sha256": st["model_identity_sha256"],
        }
        (out / "pairs_summary.json").write_text(json.dumps(summary, indent=1))
        say("=" * 72)
        say("DISCRIMINATION  (matched vulnerable/secure pairs)")
        say(f"{'variant':<20}{'TPR':>8}{'safeerr':>8}{'J':>8}{'balacc':>9}{'words':>8}")
        for tag, _ in variants:
            d = summary[tag]
            say(f"{tag:<20}{d['tpr_recall']:>8.3f}{d['safe_error_rate']:>8.3f}"
                f"{d['youden_j']:>8.3f}{d['balanced_accuracy']:>9.3f}{d['mean_words']:>8.1f}")
        say("always-VULNERABLE baseline: TPR 1.000  safeerr 1.000  J 0.000  balacc 0.500")
        say("=" * 72)
    if world_size > 1:
        dist.destroy_process_group()


def cmd_blind(a) -> None:
    """Emit an anonymised judging file plus a sealed key.

    Whoever grades these -- a person or a model -- must not see which variant
    produced an answer, or the grading is worthless. Items are shuffled and the
    variant tags replaced with opaque labels; the mapping goes in a separate key
    file that is not opened until the grades are in.
    """
    import hashlib, hmac, random, secrets
    src, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pattern = "pairs_*.jsonl" if a.pairs else "answers_*.jsonl"
    prefix = "pairs_" if a.pairs else "answers_"

    # Prefer the self-contained completion row.  The question file remains a
    # source-of-truth join for legacy answer rows and for reference reasoning.
    qmap = {}
    if a.questions:
        qs = json.loads(Path(a.questions).read_text(encoding="utf-8"))
        if isinstance(qs, dict):
            qs = qs.get("questions", [])
        for q in qs:
            if isinstance(q, dict) and q.get("id") is not None:
                qmap[str(q["id"])] = q

    records, variants = [], []
    for f in sorted(src.glob(pattern)):
        tag = f.stem[len(prefix):]
        variants.append(tag)
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r["_variant"] = tag
                records.append(r)
    if not records:
        raise SystemExit(f"no {pattern} under {src}")

    rng = random.Random(a.seed)
    # The label assignment is the thing blinding exists to hide, so it must not
    # be derivable. random.Random(a.seed) made it a pure function of a default
    # seed and the published variant list -- three lines of public information
    # reproduced the whole mapping, and the sealed key was decorative. a.seed
    # still orders the items, which reveals nothing on its own.
    labels = [chr(ord("A") + i) for i in range(len(variants))]
    if len(variants) > 26:
        raise SystemExit("blinding supports at most 26 variants")
    secrets.SystemRandom().shuffle(labels)
    tag2label = dict(zip(sorted(variants), labels))
    uid_salt = secrets.token_bytes(32)
    item_plaintext: dict[str, str] = {}
    item_truth: dict[str, str] = {}

    items = []
    for r in records:
        record_id = r.get("id")
        if record_id is None:
            record_id = r.get("pair_id")
        if record_id is None:
            raise ValueError("completion record has neither an id nor a pair_id")
        item_key = str(record_id) + "|" + str(r.get("truth", ""))
        q = qmap.get(str(r.get("id")))
        judge_input = {}
        reference = {}
        if a.pairs:
            missing = [name for name in ("snippet", "prompt")
                       if not isinstance(r.get(name), str) or not r[name]]
            if missing:
                raise ValueError(
                    "paired completion record is missing judge input field(s) "
                    f"{', '.join(missing)}; rerun the pair evaluation")
            judge_input = {
                "snippet": r["snippet"], "prompt": r["prompt"],
                "source_sha256": r.get("source_sha256"),
                "prompt_sha256": r.get("prompt_sha256"),
                "prompt_ids_sha256": r.get("prompt_ids_sha256"),
            }
        else:
            snippet = r.get("snippet") or (q.get("snippet") if q else None)
            prompt = r.get("prompt") or (question_text(q) if q else None)
            missing = [name for name, value in (("snippet", snippet), ("prompt", prompt))
                       if not isinstance(value, str) or not value]
            if missing:
                raise ValueError(
                    "question completion record is missing judge input field(s) "
                    f"{', '.join(missing)}; supply --questions or rerun the evaluation")
            judge_input = {
                "language": r.get("language") or (q.get("language") if q else None),
                "snippet": snippet,
                "prompt": prompt,
                "prompt_sha256": r.get("prompt_sha256")
                or hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_ids_sha256": r.get("prompt_ids_sha256"),
            }
            reference_source = q or r
            reference = {
                "reference": {
                    "vulnerability": reference_source.get("vulnerability"),
                    "cwe": reference_source.get("cwe"),
                    "expected_reasoning": reference_source.get("expected_reasoning"),
                }
            }
        # item_key embeds the matched-pair ground truth, so publish only an
        # opaque, salted form of it: graders still group the same item across
        # systems, but cannot read the answer off the bundle. The plaintext
        # mapping lives in the withheld key alongside the variant labels.
        item_id = hmac.new(uid_salt, ("item\0" + item_key).encode(),
                           hashlib.sha256).hexdigest()[:16]
        item_plaintext[item_id] = item_key
        items.append({
            "uid": hmac.new(
                uid_salt, (item_key + "\0" + r["_variant"]).encode(),
                hashlib.sha256).hexdigest()[:16],
            "item": item_id,
            "system": tag2label[r["_variant"]],
            "completion": r["completion"],
            # what was asked, so the answer can actually be graded
            **judge_input,
            # reference answer, withheld from a blind-quality pass but needed for
            # a correctness pass -- keep it in a separate field the grader can drop
            **reference,
            # ``truth`` is the CodeQL label being tested; it is withheld with the
            # key. ``path`` is already the label-neutral display path.
            **({"path": r.get("path")} if "truth" in r else {}),
            **({"cwe": r.get("cwe")} if "cwe" in r and not q else {}),
        })
        if "truth" in r:
            item_truth[item_id] = r["truth"]
    rng.shuffle(items)

    (out / "to_judge.jsonl").write_text(
        "\n".join(json.dumps(i) for i in items) + "\n", encoding="utf-8")
    key_path = out / "KEY_do_not_open_until_graded.json"
    key_payload = json.dumps({
        "label_to_variant": {v: k for k, v in tag2label.items()},
        "uid_hmac_salt_hex": uid_salt.hex(),
        "item_to_plaintext": item_plaintext,
        "item_to_truth": item_truth,
        "n_items": len(items), "seed": a.seed,
    }, indent=1)
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fp:
        os.fchmod(fp.fileno(), 0o600)
        fp.write(key_payload)
    print(f"[easyep] {len(items)} items from {len(variants)} variants -> {out/'to_judge.jsonl'}")
    print(f"[easyep] systems anonymised as {sorted(labels)}; key withheld in "
          f"{out/'KEY_do_not_open_until_graded.json'}")


PARITY_PROMPTS = [
    "What is 2+2? Answer with the number only.",
    "Review this snippet for security issues:\n\napp.get('/x', (req,res)=>res.send(req.query.q))",
    "Name three sorting algorithms.",
]


def _logits_for(model, tok, encode_messages, prompts, max_seq_len, dev):
    """Prefill only, returning the final-position logits. No sampling, so this is
    free of the decoder RNG entirely."""
    outs = []
    for text in prompts:
        ids = tok.encode(encode_messages([{"role": "user", "content": text}],
                                         thinking_mode=THINKING_MODE))
        if not ids or len(ids) > max_seq_len:
            raise ValueError(
                f"parity prompt has {len(ids)} tokens but max_seq_len is {max_seq_len}; "
                "refusing an empty or truncated correctness check")
        with torch.inference_mode():
            _, logits, _ = model.forward(torch.tensor([ids], device=dev), 0)
        outs.append(logits.detach().float().cpu())
    return outs


def _cmp(a_list, b_list):
    if not a_list or not b_list:
        raise ValueError("logit comparison requires non-empty outputs")
    if len(a_list) != len(b_list):
        raise ValueError("logit comparison output lengths differ")
    md = mr = 0.0
    disagree = 0
    for index, (a_, b_) in enumerate(zip(a_list, b_list)):
        if not isinstance(a_, torch.Tensor) or not isinstance(b_, torch.Tensor):
            raise ValueError(f"logit comparison item {index} is not a tensor")
        if a_.shape != b_.shape or a_.dtype != b_.dtype:
            raise ValueError(f"logit comparison item {index} shape/dtype differs")
        if a_.numel() == 0:
            raise ValueError(f"logit comparison item {index} is empty")
        if not bool(torch.isfinite(a_).all().item()) or not bool(torch.isfinite(b_).all().item()):
            raise ValueError(f"logit comparison item {index} contains NaN or infinity")
        d = (a_ - b_).abs()
        md = max(md, d.max().item())
        denom = a_.abs().max().item() or 1.0
        mr = max(mr, d.max().item() / denom)
        disagree += int((a_.argmax(-1) != b_.argmax(-1)).sum().item())
    return {"max_abs": md, "max_rel": mr, "argmax_disagreements": disagree}


def _global_error_tensor(profiler: Profiler, world_size: int) -> torch.Tensor:
    """All-gather layer-balanced, bounded norm errors from every expert shard."""
    local = profiler.err_tensor().float()
    if world_size <= 1:
        return local
    size = torch.tensor([local.numel()], dtype=torch.long, device=local.device)
    sizes = [torch.zeros_like(size) for _ in range(world_size)]
    dist.all_gather(sizes, size)
    counts = [int(value.item()) for value in sizes]
    width = max(counts)
    if width == 0:
        return local
    padded = torch.zeros(width, dtype=local.dtype, device=local.device)
    if local.numel():
        padded[:local.numel()] = local
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    return torch.cat([tensor[:count] for tensor, count in zip(gathered, counts)])


def _global_error_sample_counts(profiler: Profiler, world_size: int) -> list[int]:
    """Return globally summed sample counts for each layer."""
    counts = torch.tensor(
        profiler._err_n_by_layer, dtype=torch.int64, device=profiler.score.device)
    if world_size > 1:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return [int(value) for value in counts.cpu().tolist()]


def _quantiles(err: torch.Tensor, qs=(0.5, 0.9, 0.99, 1.0)) -> list[float]:
    if err.numel() == 0:
        return [float("nan")] * len(qs)
    if not bool(torch.isfinite(err).all().item()):
        raise ValueError("norm-recovery errors contain NaN or infinity")
    # new_tensor inherits err's dtype and device even after build() changes both
    # global defaults to BF16/CUDA.
    q = err.new_tensor(qs)
    return torch.quantile(err, q).cpu().tolist()


def cmd_parity(a) -> None:
    """Assert the patched-but-inactive model is numerically the official model.

    The instrumentation rewrites Gate/MoE/Block forwards. If any rewrite changes
    the maths, every downstream number is contaminated and the change would look
    exactly like a pruning effect. This runs fixed prompts through the untouched
    model and the patched one and compares logits -- against a noise floor
    measured by running the untouched model twice, since expert accumulation and
    all-reduce are not bitwise reproducible.
    """
    if (not math.isfinite(a.tol) or a.tol < 0
            or not math.isfinite(a.floor_mult) or a.floor_mult < 0):
        raise ValueError("parity tolerances must be finite and non-negative")
    official, model, tok, args, rank, world_size = build(
        a.ckpt_path, a.config, a.code_dir, a.max_seq_len, 1)
    sys.path.insert(0, str(Path(a.code_dir).parent / "encoding"))
    from encoding_dsv4 import encode_messages

    dev = torch.device("cuda")

    def say(m):
        if rank == 0:
            print(f"[easyep] {m}", flush=True)

    P = PARITY_PROMPTS
    say(f"parity: {len(P)} fixed prompts, prefill logits only")

    # Compressor carries persistent kv_state/score_state buffers, so the first
    # forward can differ from later ones. Discard it, then measure the floor from
    # two settled runs and compare everything against the most recent of them.
    _logits_for(model, tok, encode_messages, P, a.max_seq_len, dev)   # warmup
    base1 = _logits_for(model, tok, encode_messages, P, a.max_seq_len, dev)
    base2 = _logits_for(model, tok, encode_messages, P, a.max_seq_len, dev)
    floor = _cmp(base1, base2)
    say(f"  noise floor (official vs official): max_abs {floor['max_abs']:.3e}  "
        f"argmax disagreements {floor['argmax_disagreements']}")

    prof = Profiler(args.n_layers, args.n_routed_experts, dev)
    originals = patch(official, prof)
    set_mask(model, None, args.n_hash_layers, dev)

    prof.enabled = False
    patched_off = _logits_for(model, tok, encode_messages, P, a.max_seq_len, dev)
    c_off = _cmp(base2, patched_off)
    say(f"  patched, profiling OFF:            max_abs {c_off['max_abs']:.3e}  "
        f"argmax disagreements {c_off['argmax_disagreements']}")

    # Bisect: enable one observation at a time so a perturbation is attributed to
    # the exact step that causes it, rather than to "profiling" as a whole.
    stages = [("enabled, no observations", False, False, False),
              ("norms only              ", True, False, False),
              ("norms + mHC            ", True, False, True),
              ("norms + mHC + accumulate", True, True, True)]
    stage_reports = {}
    c_on = None
    for label, dn, da, dm in stages:
        prof.enabled, prof.do_norms, prof.do_accum, prof.do_mhc = True, dn, da, dm
        got = _logits_for(model, tok, encode_messages, P, a.max_seq_len, dev)
        prof.enabled = False
        c = _cmp(base2, got)
        stage_reports[label.strip()] = c
        say(f"  profiling: {label}      max_abs {c['max_abs']:.3e}  "
            f"argmax disagreements {c['argmax_disagreements']}")
        c_on = c
    prof.do_norms = prof.do_accum = prof.do_mhc = True

    unpatch(official, originals)
    restored = _logits_for(model, tok, encode_messages, P, a.max_seq_len, dev)
    c_res = _cmp(base2, restored)
    say(f"  after unpatch:                     max_abs {c_res['max_abs']:.3e}")

    tol = max(a.tol, floor["max_abs"] * a.floor_mult)
    if not math.isfinite(tol):
        raise ValueError("derived parity tolerance is not finite")
    report = {"noise_floor": floor, "patched_profiling_off": c_off,
              "patched_profiling_on": c_on, "bisect": stage_reports,
              "after_unpatch": c_res,
              "tolerance_used": tol, "abs_tol": a.tol, "floor_multiplier": a.floor_mult}
    ok = True
    gated = [("profiling OFF", c_off)]
    gated.extend((f"profiling {name}", result)
                 for name, result in stage_reports.items())
    gated.append(("after unpatch", c_res))
    for name, c in gated:
        if c["max_abs"] > tol or c["argmax_disagreements"] > floor["argmax_disagreements"]:
            ok = False
            say(f"  FAIL: patched {name} differs from official beyond tolerance "
                f"({c['max_abs']:.3e} > {tol:.3e})")
    report["pass"] = ok
    if rank == 0:
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        (out / "parity.json").write_text(json.dumps(report, indent=1))
    say("=" * 64)
    say(f"PARITY {'PASS' if ok else 'FAIL'}  (tolerance {tol:.3e}, "
        f"noise floor {floor['max_abs']:.3e})")
    say("=" * 64)
    if world_size > 1:
        dist.destroy_process_group()
    if not ok:
        raise SystemExit("parity check failed - instrumentation changes the model")


def cmd_validate(a) -> None:
    """Test the unweighted-norm recovery against explicit unweighted forwards.

    ||w*out||/w == ||out|| is exact only in exact arithmetic. V4 applies the
    routing weight BEFORE w2, and w2 runs act_quant + fp4_gemm, so scaling the
    activation changes quantisation bins. This measures how far that pushes the
    recovered norms, and -- what actually matters -- whether it changes which
    experts the top-k selects.
    """
    official, model, tok, args, rank, world_size = build(
        a.ckpt_path, a.config, a.code_dir, a.max_seq_len, 1,
        getattr(a, "temperature", None))
    sys.path.insert(0, str(Path(a.code_dir).parent / "encoding"))
    from encoding_dsv4 import encode_messages
    sys.path.insert(0, a.code_dir)
    from generate import generate

    dev = torch.device("cuda")
    prof = Profiler(args.n_layers, args.n_routed_experts, dev)
    prof.gate_topk = int(args.n_activated_experts)
    patch(official, prof)
    set_mask(model, None, args.n_hash_layers, dev)
    _validate_keep(a.keep, args.n_routed_experts, args.n_activated_experts)
    if not 0.0 <= a.fail_under <= 1.0:
        raise ValueError("fail_under must be in [0, 1]")
    out = Path(a.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    def say(m):
        if rank == 0:
            print(f"[easyep] {m}", flush=True)

    files = sample_calibration_files(Path(a.calib_dir), a.n_calib, a.seed)
    inputs = _calibration_chunks(
        files, tok, encode_messages, a.max_seq_len, a.max_new_tokens, a.max_chunks)
    say(f"validating norm recovery on {len(files)} files / {len(inputs)} chunks "
        f"(each routed expert is run weighted and unweighted)")
    prof.validate = True
    try:
        cal = _run_calibration(
            model, inputs, generate, tok.eos_token_id, prof, a.max_seq_len,
            a.max_new_tokens, a.seed, dev, say)
    finally:
        prof.enabled = prof.validate = False
    _validate_profile_observations(prof, args.n_hash_layers, require_true=True)

    err = _global_error_tensor(prof, world_size)
    error_samples_per_layer = _global_error_sample_counts(prof, world_size)
    if err.numel() < 1:
        raise ValueError("validation recorded no distributed norm-error observations")
    rec, tru = prof.score.cpu(), prof.score_true.cpu()
    pct = _quantiles(err)

    n_hash, n_layers = args.n_hash_layers, args.n_layers
    missing_error_layers = [
        lid for lid in range(n_hash, n_layers) if error_samples_per_layer[lid] < 1
    ]
    if missing_error_layers:
        raise ValueError(
            "validation recorded no norm-error samples for prunable layer(s) "
            + ", ".join(map(str, missing_error_layers)))
    if sum(error_samples_per_layer) != err.numel():
        raise RuntimeError("distributed norm-error samples/counts are inconsistent")
    rows, ov_all, sp_all = [], [], []
    for lid in range(n_hash, n_layers):
        a_top = set(torch.argsort(rec[lid], descending=True)[: a.keep].tolist())
        b_top = set(torch.argsort(tru[lid], descending=True)[: a.keep].tolist())
        ov = len(a_top & b_top)
        ra = torch.argsort(torch.argsort(rec[lid], descending=True)).float()
        rb = torch.argsort(torch.argsort(tru[lid], descending=True)).float()
        sp = torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
        rows.append({"layer": lid, "overlap": ov, "overlap_frac": round(ov / a.keep, 4),
                     "spearman": round(sp, 6),
                     "norm_error_samples": error_samples_per_layer[lid]})
        ov_all.append(ov); sp_all.append(sp)

    exact_fracs = [overlap / a.keep for overlap in ov_all]
    min_frac = min(exact_fracs)
    ok = all(frac >= a.fail_under for frac in exact_fracs)
    report = {
        "score_schema_version": SCORE_SCHEMA_VERSION,
        "keep": a.keep, "n_error_samples": int(err.numel()),
        "calibration": {"files": len(files), "chunks": len(inputs), **cal},
        "norm_error_sampling": {
            "method": "first observations within an equal per-layer/per-rank cap",
            "cap_per_layer_per_rank": MAX_ERROR_SAMPLES_PER_LAYER_PER_RANK,
            "global_samples_per_layer": error_samples_per_layer,
            "note": "full expert-ranking score accumulation is uncapped",
        },
        "norm_rel_err": {"median": pct[0], "p90": pct[1], "p99": pct[2], "max": pct[3],
                         "mean": err.mean().item() if err.numel() else None},
        "topk_overlap": {"mean": sum(ov_all) / len(ov_all),
                         "min": min(ov_all), "max": max(ov_all),
                         "mean_frac": round(sum(ov_all) / len(ov_all) / a.keep, 4),
                         "n_layers_identical": sum(1 for o in ov_all if o == a.keep)},
        "spearman_full_ranking": {"mean": sum(sp_all) / len(sp_all), "min": min(sp_all)},
        "per_layer": rows,
        "pass_rule": "every prunable layer must meet fail_under",
        "pass": ok,
        "fail_under": a.fail_under,
    }
    if rank == 0:
        (out / "norm_validation.json").write_text(json.dumps(report, indent=1))
        torch.save({"score_schema_version": SCORE_SCHEMA_VERSION,
                    "score_recovered": rec, "score_true": tru},
                   out / "validation_scores.pt")

    say("=" * 68)
    say("NORM RECOVERY VALIDATION   ||w*out||/w   vs   explicit ||out||")
    say(f"  relative norm error   median {pct[0]:.3e}  p90 {pct[1]:.3e}  "
        f"p99 {pct[2]:.3e}  max {pct[3]:.3e}   (n={err.numel()})")
    say(f"  top-{a.keep} overlap      mean {report['topk_overlap']['mean']:.2f}/{a.keep} "
        f"({report['topk_overlap']['mean_frac']:.2%})  min {min(ov_all)}  "
        f"identical on {report['topk_overlap']['n_layers_identical']}/{len(ov_all)} layers")
    say(f"  full-ranking spearman mean {report['spearman_full_ranking']['mean']:.6f}  "
        f"min {report['spearman_full_ranking']['min']:.6f}")
    say("=" * 68)

    say(f"NORM VALIDATION {'PASS' if ok else 'FAIL'}  "
        f"(worst-layer top-{a.keep} overlap {min_frac:.4f}, "
        f"required >= {a.fail_under})")
    if world_size > 1:
        dist.destroy_process_group()
    if not ok:
        raise SystemExit(
            f"norm recovery changes the top-{a.keep} selection "
            f"(worst layer {min_frac:.4f} < {a.fail_under}); "
            f"masks built from ||w*out||/w would not "
            f"match masks built from explicit unweighted forwards")


def cmd_pipeline(a) -> None:
    """profile -> mask -> eval(full) -> eval(pruned), on ONE model load."""
    official, model, tok, args, rank, world_size = build(
        a.ckpt_path, a.config, a.code_dir, a.max_seq_len, 1, a.temperature
    )
    model_identity = _model_identity(a.ckpt_path, a.config, a.code_dir, world_size)
    sys.path.insert(0, str(Path(a.code_dir).parent / "encoding"))
    from encoding_dsv4 import encode_messages
    sys.path.insert(0, a.code_dir)
    from generate import generate

    dev = torch.device("cuda")
    prof = Profiler(args.n_layers, args.n_routed_experts, dev)
    prof.gate_topk = int(args.n_activated_experts)
    patch(official, prof)
    out = Path(a.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    def say(m):
        if rank == 0:
            print(f"[easyep] {m}", flush=True)

    # ---------------- phase 1: calibration ----------------
    set_mask(model, None, args.n_hash_layers, dev)
    _validate_keep(a.keep, args.n_routed_experts, args.n_activated_experts)
    if a.scores_in:
        # Reuse the provenance-bound score artifact rather than spend ~10 min of
        # 4xH100 recomputing it.  The artifact attests deterministic score
        # reductions and exact calibration-token hashes; all masks are rebuilt
        # here from one consistent set of statistics.
        st = _require_score_artifact(
            torch.load(a.scores_in, map_location="cpu"), a.scores_in,
            n_hash_layers=args.n_hash_layers, model_identity=model_identity)
        expected = (args.n_layers, args.n_routed_experts)
        if tuple(st["score"].shape) != expected:
            raise ValueError(f"score artifact shape must match model {expected}")
        if st.get("gate_topk") not in (None, int(args.n_activated_experts)):
            raise ValueError("score artifact gate top-k does not match model config")
        _validate_score_state_observations(st, args.n_hash_layers)
        # Every mask below is built from these scores, so the calibration
        # distribution has to match this evaluation.
        _require_evaluation_calibration(st, a.scores_in)
        variants = all_variants(st["score"], st["score_no_simibr"],
                                st["counts"], st["gate_sums"],
                                a.keep, args.n_hash_layers,
                                args.n_layers, args.n_routed_experts,
                                a.seed, not a.no_controls,
                                st.get("score_mhc"), st.get("score_reduced_legacy"))
        say(f"phases 1-2 skipped; masks rebuilt from {a.scores_in}")
        say(f"variants: {', '.join(t for t, _ in variants)}")
        if rank == 0:
            for tag, m in variants:
                if m is not None:
                    (out / f"mask_{tag}.json").write_text(json.dumps(m, indent=1))
        return _evaluate(a, model, tok, args, rank, world_size, dev, out, say,
                         encode_messages, generate, variants)
    files = sample_calibration_files(Path(a.calib_dir), a.n_calib, a.seed)
    inputs = _calibration_chunks(
        files, tok, encode_messages, a.max_seq_len, a.max_new_tokens, a.max_chunks)
    say(f"phase 1: profiling {len(files)} calibration files / {len(inputs)} chunks")
    t0 = time.time()
    cal = _run_calibration(
        model, inputs, generate, tok.eos_token_id, prof, a.max_seq_len,
        a.max_new_tokens, a.seed, dev, say)
    _validate_profile_observations(prof, args.n_hash_layers)
    say(f"  {cal['forwards']} full prompt+response passes over {len(files)} files; "
        f"{cal['response_tokens']} generated response tokens")
    say(f"phase 1 done in {time.time()-t0:.0f}s, {prof.tokens_seen} token-layer records")

    if rank == 0:
        state = prof.state(args.n_hash_layers, model_identity)
        state["calibration"] = _calibration_provenance(
            files, inputs, cal, seed=a.seed, temperature=args.temperature,
            max_seq_len=a.max_seq_len, max_new_tokens=a.max_new_tokens,
            max_chunks=a.max_chunks,
            # real corpus-relative paths, profiled through the same
            # security-review prompt the evaluation uses
            source_kind="corpus_files", security_prompt=True)
        torch.save(state, out / "expert_scores.pt")

    # ---------------- phase 2: mask ----------------
    scores = prof.score.cpu()
    scores_alt = prof.score_no_simibr.cpu()
    mask = build_mask_from_scores(scores, a.keep, args.n_hash_layers)
    mask_alt = build_mask_from_scores(scores_alt, a.keep, args.n_hash_layers)
    variants = all_variants(
        scores, scores_alt, prof.counts.cpu(), prof.gate_sums.cpu(), a.keep,
        args.n_hash_layers, args.n_layers, args.n_routed_experts,
        a.seed, not a.no_controls, prof.score_mhc.cpu(),
        prof.score_reduced_legacy.cpu())
    if rank == 0:
        (out / ("mask_keep%d.json" % a.keep)).write_text(json.dumps(mask, indent=1))
        (out / ("mask_keep%d_no_simibr.json" % a.keep)).write_text(json.dumps(mask_alt, indent=1))
        for tag, variant_mask in variants:
            if variant_mask is not None:
                (out / f"mask_{tag}.json").write_text(json.dumps(variant_mask, indent=1))
        cmp_rows = compare_scorings(scores, scores_alt, mask, mask_alt,
                                    a.keep, args.n_hash_layers, args.n_layers)
        # The frequency control ranks by an integer count, so its cutoff ties
        # constantly and topk resolves those by index. Report its margins beside
        # the scored ones: "the paper's score beats frequency" means little if
        # the frequency cut was index order rather than signal.
        cmp_rows["frequency_control_cutoff"] = score_cutoff_diagnostics(
            prof.counts.cpu(), a.keep, args.n_hash_layers)
        (out / "score_comparison.json").write_text(json.dumps(cmp_rows, indent=1))
        ov = [r["overlap"] for r in cmp_rows["per_layer"]]
        say(f"phase 2: mask keeps {a.keep}/{args.n_routed_experts} per layer on "
            f"layers {args.n_hash_layers}..{args.n_layers-1}")
        say(f"         top-{a.keep} overlap vs no-simibr: "
            f"mean {cmp_rows['overlap_mean']}/{a.keep} "
            f"({cmp_rows['overlap_mean_frac']:.1%}), "
            f"min {min(ov)} (layer {cmp_rows['min_overlap_layer']}), max {max(ov)}")
        cutoff_margin = cmp_rows["paper_cutoff_minimum_relative_margin"]
        if cutoff_margin is not None:
            say(f"         smallest paper-score rank-{a.keep}/rank-{a.keep + 1} "
                f"relative margin: {cutoff_margin:.3e} "
                f"(layer {cmp_rows['paper_cutoff_minimum_margin_layer']})")
        say(f"         per-layer detail -> score_comparison.json")
        never = int((scores[args.n_hash_layers:] == 0).sum())
        say(f"         experts never activated during calibration: {never}")

    if a.profile_only:
        say("--profile-only: stopping after scoring; masks and scores written")
        if world_size > 1:
            dist.destroy_process_group()
        return

    # ---------------- phase 3+4: evaluate ----------------
    # Both scoring rules are evaluated, not just the paper's. The structural
    # overlap in score_comparison.json shows the rules DISAGREE; only running
    # both shows which one selects better experts.
    say(f"variants: {', '.join(t for t, _ in variants)}")
    return _evaluate(a, model, tok, args, rank, world_size, dev, out, say,
                     encode_messages, generate, variants)


def _evaluate(a, model, tok, args, rank, world_size, dev, out, say,
              encode_messages, generate, variants) -> None:
    """variants: list of (tag, mask_or_None), all decoded under identical per-question seeds."""
    questions = json.loads(Path(a.questions).read_text(encoding="utf-8"))
    if isinstance(questions, dict):
        questions = questions.get("questions", [])
    if a.limit:
        questions = questions[: a.limit]
    if not questions:
        raise ValueError("pipeline evaluation question set is empty")
    if any(not isinstance(question, dict) for question in questions):
        raise ValueError("pipeline questions must be JSON objects")
    input_limit = a.max_seq_len - a.max_new_tokens
    if a.max_new_tokens < 1 or input_limit < 1:
        raise ValueError("max_seq_len leaves no room for pipeline response generation")
    prepared = []
    for i, q in enumerate(questions):
        prompt = question_text(q)
        ids = tok.encode(encode_messages(
            [{"role": "user", "content": prompt}], thinking_mode=THINKING_MODE))
        if len(ids) > input_limit:
            raise ValueError(
                f"question {q.get('id', i)} has {len(ids)} prompt tokens but only "
                f"{input_limit} fit with the response reservation; refusing to truncate")
        prepared.append((i, q, prompt, ids))

    summary = {}
    for tag, m in variants:
        set_mask(model, m, args.n_hash_layers, dev)
        say(f"phase 3: evaluating variant={tag} on {len(questions)} questions")
        rows = []
        for i, q, prompt, ids in prepared:
            # Paired comparison: question i must see the SAME sampling noise under
            # both variants, otherwise the full/pruned delta mixes the pruning effect
            # with decoder randomness. Reseed per question, identically on every rank.
            torch.manual_seed(a.seed + i)
            torch.cuda.manual_seed_all(a.seed + i)
            t1 = time.time()
            with torch.inference_mode():
                gen = generate(model, [ids], a.max_new_tokens, tok.eos_token_id)
            completion, n_completion = _decode_completion(tok, gen, ids)
            if rank == 0:
                rows.append({
                    "id": q.get("id"),
                    "cwe": q.get("cwe"),
                    "language": q.get("language"),
                    "snippet": q.get("snippet"),
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "prompt_ids_sha256": _ids_sha256(ids),
                    "vulnerability": q.get("vulnerability"),
                    "expected_reasoning": q.get("expected_reasoning"),
                    "completion": completion,
                    "seed": a.seed + i,
                    "temperature": args.temperature,
                    "prompt_tokens": len(ids),
                    "completion_tokens": n_completion,
                    "seconds": round(time.time() - t1, 2),
                })
                say(f"  {tag} {i+1}/{len(questions)}  {time.time()-t1:.1f}s")
        if rank == 0:
            with (out / f"answers_{tag}.jsonl").open("w", encoding="utf-8") as fp:
                for r in rows:
                    fp.write(json.dumps(r) + "\n")
            summary[tag] = {"n": len(rows),
                            "mean_seconds": round(sum(r["seconds"] for r in rows) / max(len(rows), 1), 2),
                            "mean_words": round(sum(len(r["completion"].split()) for r in rows)
                                                / max(len(rows), 1), 1)}

    if rank == 0:
        summary["_decoding"] = {
            "temperature": args.temperature,
            "paired_seeding": True,
            "seed_base": a.seed,
            "note": ("question i is decoded under torch.manual_seed(seed+i) in BOTH "
                     "variants, so the full/pruned delta is not confounded by "
                     "sampling noise"),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        say("=" * 60)
        say(f"GENERATED  keep={a.keep}/{args.n_routed_experts} experts "
            f"on layers {args.n_hash_layers}..{args.n_layers-1}")
        for tag in [t for t, _ in variants]:
            st_ = summary.get(tag, {})
            say(f"  {tag:17s} n={st_.get('n')}  mean {st_.get('mean_seconds')}s  "
                f"{st_.get('mean_words')} words -> answers_{tag}.jsonl")
        say("  no scoring applied; grade with `blind` + your own judge")
        say("=" * 60)
    if world_size > 1:
        dist.destroy_process_group()


def main() -> None:
    p = ArgumentParser(description="EASY-EP for DeepSeek-V4-Flash")
    sub = p.add_subparsers(dest="mode", required=True)

    def common(sp):
        sp.add_argument("--ckpt-path", required=True)
        sp.add_argument("--config", required=True)
        sp.add_argument("--code-dir", required=True, help="the model's inference/ directory")
        sp.add_argument("--max-seq-len", type=int, default=16384,
                        help=f"context window; at most {MAX_SUPPORTED_SEQ_LEN} "
                             "because prefill memory is quadratic and unblocked")
        sp.add_argument("--limit", type=int, default=0)

    sp = sub.add_parser("profile", help="calibration -> expert scores")
    common(sp)
    sp.add_argument("--calib", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--max-new-tokens", type=int, default=256,
                    help="response tokens generated before profiling prompt+response")
    sp.add_argument("--max-chunks", type=int, default=0,
                    help="maximum token-exact chunks per sample; 0 is unlimited")
    sp.add_argument("--seed", type=int, default=965)
    sp.add_argument("--temperature", type=float, default=None,
                    help="calibration-response temperature; default keeps model configuration")

    sp = sub.add_parser("mask", help="scores -> mask")
    sp.add_argument("--scores", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--keep", type=int, default=192)
    sp.add_argument("--n-hash-layers", type=int, default=3)
    mask_score = sp.add_mutually_exclusive_group()
    mask_score.add_argument(
        "--no-simibr", action="store_true",
        help="rank by weight*norm only, i.e. EASY-EP without token contribution")
    mask_score.add_argument(
        "--gating-score", action="store_true",
        help="rank by total activated gate weight, the paper's gating-score baseline")

    sp = sub.add_parser("eval", help="generate with or without a mask")
    common(sp)
    sp.add_argument("--questions", required=True)
    sp.add_argument("--mask", default="")
    sp.add_argument("--out", required=True,
                    help="output DIRECTORY; writes answers_<tag>.jsonl and "
                         "summary.json, the same layout the pipeline emits")
    sp.add_argument("--max-new-tokens", type=int, default=256)
    sp.add_argument("--seed", type=int, default=965,
                    help="item i is decoded with manual_seed(seed+i)")
    sp.add_argument("--temperature", type=float, default=None,
                    help="decoding temperature; default keeps the model configuration")

    sp = sub.add_parser("pipeline", help="profile -> mask -> eval(full) -> eval(pruned), one load")
    common(sp)
    sp.add_argument("--calib-dir", required=True)
    sp.add_argument("--questions", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--n-calib", type=int, default=25)
    sp.add_argument("--keep", type=int, default=192)
    sp.add_argument("--max-new-tokens", type=int, default=256)
    sp.add_argument("--scores-in", default="",
                    help="reuse expert_scores.pt from an earlier run; skips profiling "
                         "and rebuilds every mask from those statistics")
    sp.add_argument("--no-controls", action="store_true",
                    help="skip the gating-score, frequency, and random baselines")
    sp.add_argument("--max-chunks", type=int, default=0,
                    help="maximum token-exact chunks profiled per file; 0 is unlimited; "
                         "an exceeded positive cap fails rather than dropping source")
    sp.add_argument("--profile-only", action="store_true",
                    help="stop after producing scores and masks (no generation)")
    sp.add_argument("--seed", type=int, default=965,
                    help="question i is decoded with manual_seed(seed+i) in both variants")
    sp.add_argument("--temperature", type=float, default=None,
                    help="decoding temperature; 0 = greedy (removes sampling noise entirely). "
                         "Default None keeps the model's configured 1.0")

    sp = sub.add_parser("pairs", help="matched vulnerable/secure discrimination eval")
    common(sp)
    sp.add_argument("--scores-in", required=True)
    sp.add_argument("--pairs-manifest", required=True)
    sp.add_argument("--calib-dir", required=True, help="root of vulnerable-js-files")
    sp.add_argument("--out", required=True)
    sp.add_argument("--n-pairs", type=int, default=25)
    sp.add_argument("--keep", type=int, default=128)
    sp.add_argument("--max-new-tokens", type=int, default=128)
    sp.add_argument("--seed", type=int, default=965)
    sp.add_argument("--temperature", type=float, default=None)
    sp.add_argument("--no-controls", action="store_true",
                    help="skip the gating-score, frequency, and random baselines")

    sp = sub.add_parser("blind", help="anonymise completions for unbiased judging")
    sp.add_argument("--results", required=True, help="dir holding answers_*.jsonl / pairs_*.jsonl")
    sp.add_argument("--out", required=True)
    sp.add_argument("--pairs", action="store_true", help="blind pairs_*.jsonl instead of answers_*.jsonl")
    sp.add_argument("--questions", default="",
                    help="questions_used.json, joined by id so the judge sees the snippet "
                         "and the reference answer alongside each completion")
    sp.add_argument("--seed", type=int, default=7)

    sp = sub.add_parser("parity", help="assert the patched model matches the official one")
    common(sp)
    sp.add_argument("--out", required=True)
    sp.add_argument("--tol", type=float, default=1e-3, help="absolute logit tolerance")
    sp.add_argument("--floor-mult", type=float, default=4.0,
                    help="also allow this multiple of the measured run-to-run noise floor")

    sp = sub.add_parser("validate", help="check unweighted-norm recovery against explicit forwards")
    common(sp)
    sp.add_argument("--calib-dir", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--n-calib", type=int, default=25)
    sp.add_argument("--keep", type=int, default=128)
    sp.add_argument("--max-new-tokens", type=int, default=256)
    sp.add_argument("--max-chunks", type=int, default=0,
                    help="maximum token-exact chunks per file; 0 is unlimited")
    sp.add_argument("--seed", type=int, default=965)
    sp.add_argument("--temperature", type=float, default=None)
    sp.add_argument("--fail-under", type=float, default=0.98,
                    help="fail if the recovered-norm top-k overlap with explicit "
                         "unweighted norms falls below this fraction")

    a = p.parse_args()
    {"profile": cmd_profile, "mask": cmd_mask, "eval": cmd_eval,
     "pipeline": cmd_pipeline, "pairs": cmd_pairs, "blind": cmd_blind,
     "validate": cmd_validate, "parity": cmd_parity}[a.mode](a)


if __name__ == "__main__":
    main()
