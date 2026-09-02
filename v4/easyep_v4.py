"""EASY-EP (arXiv:2504.06792) ported to DeepSeek-V4-Flash.

Faithful to the paper's score, which is a sum of per-token PRODUCTS
(expert_selection.py:38 in RUCAIBox/EASYEP):

    score[layer, e] += weight_t[e] * simibr_t * norm_t[e]

    weight   gating score for expert e on token t
    norm     || unweighted expert output ||_2
    simibr   max(1 - cos(h, h + y_routed), 0)   "token contribution"

Three V4-specific adaptations, each forced by the architecture:

1. V4 folds the gating weight inside Expert.forward -- expert(x, w) computes
   w2(w * silu(gate)*up). w2 is linear, so the output is exactly w * unweighted.
   The unweighted norm is recovered by dividing, rather than paying a second
   forward pass.

2. V4's MoE returns routed+shared summed. simibr needs the routed part alone,
   so the routed sum is stashed on the module for Block to read.

3. Layers 0..n_hash_layers-1 route by token ID (Gate.tid2eid), not by content.
   Pruning them would disable specific vocabulary items outright, so they are
   protected and keep all experts.

Hyper-connections need no special handling: Block.hc_pre reduces the hc_mult
copies to a single vector before the FFN and hc_post re-expands after, so within
the FFN sub-block the stream is one vector per token, exactly as in R1.

Modes
-----
  profile   calibration forward passes -> per-layer/expert scores
  mask      scores -> keep/prune mask
  eval      generate answers with the mask applied at the gate (or without)
"""
from __future__ import annotations

import json
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import re

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_model
from transformers import AutoTokenizer

EPS = 1e-6

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
        # float64: this is a long-running sum of many small products
        self.score = torch.zeros((n_layers, n_experts), dtype=torch.float64, device=device)
        # diagnostics, so the paper's score can be compared against the
        # simpler weight*norm variant without a second calibration run
        self.score_no_simibr = torch.zeros_like(self.score)
        self.counts = torch.zeros((n_layers, n_experts), dtype=torch.int64, device=device)
        self.gate_sums = torch.zeros_like(self.score)
        self.tokens_seen = 0
        # --- norm-recovery validation (only populated in validate mode) ---
        self.validate = False
        # bisect switches: each adds one observation on top of the previous
        self.do_norms = True       # per-expert unweighted-norm recovery + all_gather
        self.do_accum = True       # index_add_ into the score tensors
        self.do_mhc = True         # the mHC simibr, which touches hc_post inputs
        # mHC-aware simibr: measured after hc_post remixes the FFN output into the
        # hc_mult residual copies, i.e. where the contribution actually lands.
        self.score_mhc = torch.zeros_like(self.score)
        self.score_true = torch.zeros_like(self.score)   # from explicit unweighted forwards
        self._err = []                                   # relative error samples
        self._err_n = 0

    def err_tensor(self):
        return torch.cat(self._err) if self._err else torch.zeros(0)

    def state(self) -> dict[str, Any]:
        return {
            "score_mhc": self.score_mhc.cpu(),
            "score_true": self.score_true.cpu(),
            "norm_rel_err": self.err_tensor().cpu(),
            "score": self.score.cpu(),
            "score_no_simibr": self.score_no_simibr.cpu(),
            "counts": self.counts.cpu(),
            "gate_sums": self.gate_sums.cpu(),
            "tokens_seen": self.tokens_seen,
            "n_layers": self.n_layers,
            "n_experts": self.n_experts,
        }


# ------------------------------------------------------------------ patching


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
                    if profiler._err_n < 400_000:
                        rel = ((wn - tn).abs() / tn.clamp_min(EPS)).flatten()
                        profiler._err.append(rel.detach())
                        profiler._err_n += rel.numel()

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

    def moe_accumulate(self, h_flat: torch.Tensor, simibr_mhc: torch.Tensor = None) -> None:
        """Form the per-token product and add it to the running score.

        h_flat is the pre-norm reduced residual entering the FFN sub-block --
        the V4 analogue of EASY-EP's x_before_moe, measured in the single-vector
        space the FFN actually sees.

        simibr_mhc is the same quantity measured after hc_post has remixed the
        contribution into the hc_mult residual copies. Both are accumulated so the
        two definitions can be compared without a second calibration pass.
        """
        y_r = self._ep_y_routed
        simibr = (1.0 - F.cosine_similarity(h_flat.float(), (h_flat + y_r).float(), dim=-1)).clamp_min(0.0)

        w, idx, nrm = self._ep_weights, self._ep_indices, self._ep_norms
        lid = self.layer_id
        for slot in range(idx.size(1)):
            e = idx[:, slot]
            profiler.score[lid].index_add_(0, e, (w[:, slot] * simibr * nrm[:, slot]).double())
            if simibr_mhc is not None:
                profiler.score_mhc[lid].index_add_(
                    0, e, (w[:, slot] * simibr_mhc * nrm[:, slot]).double())
            profiler.score_no_simibr[lid].index_add_(0, e, (w[:, slot] * nrm[:, slot]).double())
            if profiler.validate:
                profiler.score_true[lid].index_add_(
                    0, e, (w[:, slot] * simibr * self._ep_norms_true[:, slot]).double())
            profiler.gate_sums[lid].index_add_(0, e, w[:, slot].double())
            profiler.counts[lid].index_add_(0, e, torch.ones_like(e, dtype=torch.int64))
        profiler.tokens_seen += idx.size(0)

        # release: these hold [n_tok, *] tensors per layer and would otherwise stay
        # resident through the generation phases, eating KV-cache headroom
        self._ep_norms = self._ep_weights = self._ep_indices = self._ep_y_routed = None
        self._ep_norms_true = None

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
        x = self.ffn(x, input_ids)
        if (profiler.enabled
                and not self.ffn.gate.hash
                and self.layer_id < profiler.n_layers):   # excludes the MTP/DSpark stages
            simibr_mhc = None
            y_r = getattr(self.ffn, "_ep_y_routed", None)
            if y_r is not None and profiler.do_mhc:
                # hc_post(0, ...) is the residual carried with no FFN contribution;
                # hc_post(y_routed, ...) adds only the routed experts. Comparing the
                # two isolates the routed contribution in the space it lands in.
                y_r_bsd = y_r.view(h.shape)
                # hc_post(x, ...) = post*x + sum(comb*residual); at x=0 only the
                # residual-mixing term survives, so build it directly.
                base_hc = torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
                routed_hc = post.unsqueeze(-1) * y_r_bsd.unsqueeze(-2) + base_hc
                simibr_mhc = (1.0 - F.cosine_similarity(
                    base_hc.flatten(2).float(), routed_hc.flatten(2).float(), dim=-1)
                ).clamp_min(0.0).view(-1)
            if profiler.do_accum and getattr(self.ffn, "_ep_norms", None) is not None:
                self.ffn.ep_accumulate(h.view(-1, h.size(-1)), simibr_mhc)
        x = self.hc_post(x, residual, post, comb)
        return x

    originals = {"Gate.forward": Gate.forward, "MoE.forward": MoE.forward,
                 "Block.forward": Block.forward}
    Gate.forward = gate_forward
    MoE.forward = moe_forward
    MoE.ep_accumulate = moe_accumulate
    Block.forward = block_forward
    return originals


def unpatch(official: Any, originals: dict) -> None:
    official.Gate.forward = originals["Gate.forward"]
    official.MoE.forward = originals["MoE.forward"]
    official.Block.forward = originals["Block.forward"]


def set_mask(model, mask: dict | None, n_hash_layers: int, device) -> None:
    for lid, layer in enumerate(model.layers):
        if mask is None or lid < n_hash_layers:
            layer.ffn.gate.ep_keep = None
            continue
        kept = mask["layers"][str(lid)]["kept"]
        keep = torch.zeros(layer.ffn.n_routed_experts, dtype=torch.bool, device=device)
        keep[torch.tensor(kept, device=device)] = True
        layer.ffn.gate.ep_keep = keep


# --------------------------------------------------------------------- model


def build(ckpt_path: str, config_path: str, code_dir: str, max_seq_len: int, max_bs: int,
          temperature: float | None = None):
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
    if path.suffix == ".jsonl":
        rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
        return [r.get("text") or r.get("prompt") or r.get("content") or json.dumps(r) for r in rows]
    data = json.loads(raw)
    if isinstance(data, list):
        return [d if isinstance(d, str) else (d.get("text") or d.get("prompt") or json.dumps(d)) for d in data]
    raise SystemExit(f"unsupported calibration file: {path}")


# --------------------------------------------------------------------- modes


def cmd_profile(a) -> None:
    official, model, tok, args, rank, world_size = build(
        a.ckpt_path, a.config, a.code_dir, a.max_seq_len, 1
    )
    sys.path.insert(0, str(Path(a.code_dir).parent / "encoding"))
    from encoding_dsv4 import encode_messages

    prof = Profiler(args.n_layers, args.n_routed_experts, torch.device("cuda"))
    patch(official, prof)
    set_mask(model, None, args.n_hash_layers, torch.device("cuda"))

    samples = load_calibration(Path(a.calib))
    if a.limit:
        samples = samples[: a.limit]
    if rank == 0:
        print(f"[easyep] {len(samples)} calibration samples", flush=True)

    prof.enabled = True
    used = 0
    for i, text in enumerate(samples):
        prompt = encode_messages([{"role": "user", "content": text}], thinking_mode="chat")
        ids = tok.encode(prompt)
        if len(ids) > a.max_seq_len:
            ids = ids[: a.max_seq_len]
        if len(ids) < 8:
            continue
        toks = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            model.forward(toks, 0)
        used += 1
        if rank == 0:
            print(f"[easyep] {i+1}/{len(samples)}  {len(ids)} tok  "
                  f"(cum {prof.tokens_seen} tok-layers)", flush=True)
    prof.enabled = False

    if rank == 0:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        st = prof.state()
        st["samples_used"] = used
        st["source"] = str(a.calib)
        torch.save(st, out)
        print(f"[easyep] wrote {out}  ({used} samples)", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


def cmd_mask(a) -> None:
    st = torch.load(a.scores, map_location="cpu")
    key = "score_no_simibr" if a.no_simibr else "score"
    scores = st[key].float()
    n_layers, n_experts = scores.shape
    keep_n = a.keep
    mask = {
        "keep_per_layer": keep_n,
        "n_experts": n_experts,
        "protected_hash_layers": list(range(a.n_hash_layers)),
        "score_key": key,
        "layers": {},
    }
    for lid in range(n_layers):
        if lid < a.n_hash_layers:
            mask["layers"][str(lid)] = {"kept": list(range(n_experts)), "pruned": []}
            continue
        order = torch.argsort(scores[lid], descending=True)
        kept = sorted(order[:keep_n].tolist())
        mask["layers"][str(lid)] = {
            "kept": kept,
            "pruned": sorted(order[keep_n:].tolist()),
        }
    Path(a.out).write_text(json.dumps(mask, indent=1), encoding="utf-8")

    dead = int((scores[a.n_hash_layers:] == 0).sum())
    print(f"[easyep] mask -> {a.out}")
    print(f"[easyep] keep {keep_n}/{n_experts} per layer on layers "
          f"{a.n_hash_layers}..{n_layers-1}; hash layers kept whole")
    print(f"[easyep] experts never activated during calibration: {dead}")


def cmd_eval(a) -> None:
    official, model, tok, args, rank, world_size = build(
        a.ckpt_path, a.config, a.code_dir, a.max_seq_len, 1
    )
    sys.path.insert(0, str(Path(a.code_dir).parent / "encoding"))
    from encoding_dsv4 import encode_messages, parse_message_from_completion_text
    sys.path.insert(0, a.code_dir)
    from generate import generate

    prof = Profiler(args.n_layers, args.n_routed_experts, torch.device("cuda"))
    patch(official, prof)

    mask = json.loads(Path(a.mask).read_text()) if a.mask else None
    set_mask(model, mask, args.n_hash_layers, torch.device("cuda"))
    tag = "pruned" if mask else "full"

    questions = json.loads(Path(a.questions).read_text(encoding="utf-8"))
    if isinstance(questions, dict):
        questions = questions.get("questions", [])
    if a.limit:
        questions = questions[: a.limit]

    out_path = Path(a.out)
    if rank == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fp = out_path.open("a", encoding="utf-8")

    for i, q in enumerate(questions):
        text = q if isinstance(q, str) else (q.get("question") or q.get("prompt") or json.dumps(q))
        qid = None if isinstance(q, str) else q.get("id")
        prompt = encode_messages([{"role": "user", "content": text}], thinking_mode="chat")
        ids = tok.encode(prompt)
        t0 = time.time()
        with torch.inference_mode():
            out = generate(model, [ids], a.max_new_tokens, tok.eos_token_id)
        completion = tok.decode(out[0])
        if rank == 0:
            fp.write(json.dumps({
                "variant": tag,
                "index": i,
                "id": qid,
                "question": text,
                "completion": completion,
                "message": parse_message_from_completion_text(completion, thinking_mode="chat"),
                "seconds": round(time.time() - t0, 2),
            }) + "\n")
            fp.flush()
            print(f"[easyep:{tag}] {i+1}/{len(questions)}  {time.time()-t0:.1f}s", flush=True)

    if rank == 0:
        fp.close()
        print(f"[easyep] wrote {out_path}", flush=True)
    if world_size > 1:
        dist.destroy_process_group()


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
    by_cwe: dict[str, list[Path]] = {}
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix in {".js", ".ts", ".jsx", ".tsx"}:
            cwe = next((part for part in f.relative_to(root).parts if part.startswith("CWE-")), "other")
            by_cwe.setdefault(cwe, []).append(f)
    rng = random.Random(seed)
    for v in by_cwe.values():
        rng.shuffle(v)
    picked: list[Path] = []
    cwes = sorted(by_cwe)
    i = 0
    while len(picked) < n and any(by_cwe[c] for c in cwes):
        c = cwes[i % len(cwes)]
        if by_cwe[c]:
            picked.append(by_cwe[c].pop())
        i += 1
    return [(str(f.relative_to(root)), f.read_text(encoding="utf-8", errors="ignore")) for f in picked]



def build_mask_from_scores(scores: torch.Tensor, keep_n: int, n_hash: int) -> dict:
    n_layers, n_experts = scores.shape
    mask = {"keep_per_layer": keep_n, "n_experts": n_experts,
            "protected_hash_layers": list(range(n_hash)), "layers": {}}
    for lid in range(n_layers):
        if lid < n_hash:
            mask["layers"][str(lid)] = {"kept": list(range(n_experts)), "pruned": []}
            continue
        order = torch.argsort(scores[lid], descending=True)
        mask["layers"][str(lid)] = {"kept": sorted(order[:keep_n].tolist()),
                                    "pruned": sorted(order[keep_n:].tolist())}
    return mask


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
        "per_layer": rows,
    }


def build_frequency_mask(counts: torch.Tensor, keep_n: int, n_hash: int) -> dict:
    """Baseline: keep the most-often-selected experts. The naive heuristic that
    any sensible person tries first -- if the paper's score cannot beat this,
    the extra machinery is not earning its place."""
    return build_mask_from_scores(counts.float(), keep_n, n_hash)


def build_random_mask(n_layers: int, n_experts: int, keep_n: int, n_hash: int, seed: int) -> dict:
    """Floor: a seeded random subset. If this matches the scored masks, the
    scoring carries no signal and the model is simply robust to expert removal."""
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


def all_variants(scores, scores_alt, counts, keep_n, n_hash, n_layers, n_experts,
                 seed: int, controls: bool, scores_mhc=None):
    """The variant set, in a fixed order so runs stay comparable.

    pruned_paper and pruned_mhc differ only in where simibr is measured: the
    single-vector space the FFN sees, versus the hc_mult residual space the
    contribution is mixed back into.
    """
    v = [("full", None),
         ("pruned_paper", build_mask_from_scores(scores, keep_n, n_hash))]
    if scores_mhc is not None:
        v.append(("pruned_mhc", build_mask_from_scores(scores_mhc, keep_n, n_hash)))
    v.append(("pruned_no_simibr", build_mask_from_scores(scores_alt, keep_n, n_hash)))
    if controls:
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


def load_pairs(manifest: Path, root: Path, n: int, seed: int = 42) -> list[dict]:
    """Matched vulnerable/secure file pairs from the CodeQL manifest.

    The secure file is the SAME file with the flagged expression neutralised and
    re-scanned clean, so a model that simply calls everything vulnerable scores
    100% on one half and 0% on the other. Recall-only metrics cannot see that;
    this can.
    """
    import random
    rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    base = root.parent
    out = []
    for r in rows:
        # Ground truth here is "CodeQL raised alerts on this file, and the
        # neutralised copy rescans clean" -- not a human exploitability judgement.
        # Require actual alerts so the vulnerable label means something.
        if int(r.get("alert_locations", 0)) < 1:
            continue
        vp, sp = base / r["original_file"], base / r["secure_file"]
        if not (vp.is_file() and sp.is_file()):
            continue
        try:
            vc, sc = vp.read_text(errors="ignore"), sp.read_text(errors="ignore")
        except OSError:
            continue
        if not vc.strip() or not sc.strip() or vc == sc:
            continue
        out.append({"pair_id": len(out), "queries": r.get("queries", []),
                    "alert_locations": int(r.get("alert_locations", 0)),
                    "vuln_path": r["original_file"], "vuln_code": vc,
                    "safe_path": r["secure_file"], "safe_code": sc})
        if len(out) >= n:
            break
    return out


def parse_verdict(text: str) -> str | None:
    for line in text.splitlines():
        t = line.strip().lower()
        if t.startswith("verdict:"):
            v = t.split(":", 1)[1].strip()
            if v.startswith("vulnerable"):
                return "VULNERABLE"
            if v.startswith("safe"):
                return "SAFE"
            return None
    return None


def discrimination_stats(rows: list[dict]) -> dict:
    """TPR / FPR / Youden's J. J=0 means the model is not discriminating at all,
    however confidently it words its answers."""
    v = [r for r in rows if r["truth"] == "VULNERABLE"]
    s = [r for r in rows if r["truth"] == "SAFE"]
    tpr = sum(1 for r in v if r["verdict"] == "VULNERABLE") / max(len(v), 1)
    fpr = sum(1 for r in s if r["verdict"] == "VULNERABLE") / max(len(s), 1)
    unparsed = sum(1 for r in rows if r["verdict"] is None)
    return {
        "n_vulnerable": len(v), "n_safe": len(s),
        "tpr_recall": round(tpr, 4),
        "fpr_false_alarm": round(fpr, 4),
        "youden_j": round(tpr - fpr, 4),
        "balanced_accuracy": round((tpr + (1 - fpr)) / 2, 4),
        "always_vulnerable_would_score": {"tpr_recall": 1.0, "fpr_false_alarm": 1.0,
                                          "youden_j": 0.0, "balanced_accuracy": 0.5},
        "unparsed_verdicts": unparsed,
        "mean_words": round(sum(len(r["completion"].split()) for r in rows) / max(len(rows), 1), 1),
    }


def cmd_pairs(a) -> None:
    """Matched-pair discrimination eval across every variant."""
    official, model, tok, args, rank, world_size = build(
        a.ckpt_path, a.config, a.code_dir, a.max_seq_len, 1, a.temperature)
    sys.path.insert(0, str(Path(a.code_dir).parent / "encoding"))
    from encoding_dsv4 import encode_messages
    sys.path.insert(0, a.code_dir)
    from generate import generate

    dev = torch.device("cuda")
    prof = Profiler(args.n_layers, args.n_routed_experts, dev)
    patch(official, prof)
    out = Path(a.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    def say(m):
        if rank == 0:
            print(f"[easyep] {m}", flush=True)

    st = torch.load(a.scores_in, map_location="cpu")
    smhc = st.get("score_mhc")
    variants = all_variants(st["score"].float(), st["score_no_simibr"].float(),
                            st["counts"], a.keep, args.n_hash_layers,
                            args.n_layers, args.n_routed_experts,
                            a.seed, not a.no_controls,
                            smhc.float() if smhc is not None else None)
    pairs = load_pairs(Path(a.pairs_manifest), Path(a.calib_dir), a.n_pairs)
    say(f"{len(pairs)} matched pairs -> {2*len(pairs)} items per variant")
    say(f"variants: {', '.join(t for t, _ in variants)}")

    items = []
    for p_ in pairs:
        items.append((p_["pair_id"], "VULNERABLE", p_["vuln_path"], p_["vuln_code"]))
        items.append((p_["pair_id"], "SAFE", p_["safe_path"], p_["safe_code"]))

    summary = {}
    for tag, m in variants:
        set_mask(model, m, args.n_hash_layers, dev)
        say(f"variant={tag}: {len(items)} items")
        rows = []
        for i, (pid, truth, path, code) in enumerate(items):
            ids = tok.encode(encode_messages(
                [{"role": "user", "content": VERDICT_PROMPT.format(path=path, code=code)}],
                thinking_mode="chat"))
            if len(ids) > a.max_seq_len:
                ids = ids[: a.max_seq_len]
            torch.manual_seed(a.seed + i)
            torch.cuda.manual_seed_all(a.seed + i)
            t1 = time.time()
            with torch.inference_mode():
                gen = generate(model, [ids], a.max_new_tokens, tok.eos_token_id)
            completion = tok.decode(gen[0])
            if rank == 0:
                rows.append({"pair_id": pid, "truth": truth, "path": path,
                             "verdict": parse_verdict(completion),
                             "completion": completion,
                             "seconds": round(time.time() - t1, 2)})
                if (i + 1) % 10 == 0:
                    say(f"  {tag} {i+1}/{len(items)}")
        if rank == 0:
            with (out / f"pairs_{tag}.jsonl").open("w", encoding="utf-8") as fp:
                for r in rows:
                    fp.write(json.dumps(r) + "\n")
            summary[tag] = discrimination_stats(rows)

    if rank == 0:
        (out / "pairs_summary.json").write_text(json.dumps(summary, indent=1))
        say("=" * 72)
        say("DISCRIMINATION  (matched vulnerable/secure pairs)")
        say(f"{'variant':<20}{'TPR':>8}{'FPR':>8}{'J':>8}{'balacc':>9}{'words':>8}")
        for tag, _ in variants:
            d = summary[tag]
            say(f"{tag:<20}{d['tpr_recall']:>8.3f}{d['fpr_false_alarm']:>8.3f}"
                f"{d['youden_j']:>8.3f}{d['balanced_accuracy']:>9.3f}{d['mean_words']:>8.1f}")
        say("always-VULNERABLE baseline: TPR 1.000  FPR 1.000  J 0.000  balacc 0.500")
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
    import random, hashlib
    src, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pattern = "pairs_*.jsonl" if a.pairs else "answers_*.jsonl"
    prefix = "pairs_" if a.pairs else "answers_"

    # A judge cannot grade an answer without the question. answers_*.jsonl stores
    # only the id, so join back to the question file to recover the snippet and
    # the reference reasoning.
    qmap = {}
    if a.questions:
        qs = json.loads(Path(a.questions).read_text(encoding="utf-8"))
        if isinstance(qs, dict):
            qs = qs.get("questions", [])
        for q in qs:
            if isinstance(q, dict) and q.get("id"):
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
    labels = [chr(ord("A") + i) for i in range(len(variants))]
    rng.shuffle(labels)
    tag2label = dict(zip(sorted(variants), labels))

    items = []
    for r in records:
        item_key = str(r.get("id") or r.get("pair_id")) + "|" + str(r.get("truth", ""))
        q = qmap.get(str(r.get("id")))
        items.append({
            "uid": hashlib.sha1((item_key + r["_variant"]).encode()).hexdigest()[:12],
            "item": item_key,
            "system": tag2label[r["_variant"]],
            "completion": r["completion"],
            # what was asked, so the answer can actually be graded
            **({"language": q.get("language"), "snippet": q.get("snippet")} if q else {}),
            # reference answer, withheld from a blind-quality pass but needed for
            # a correctness pass -- keep it in a separate field the grader can drop
            **({"reference": {"vulnerability": q.get("vulnerability"),
                              "cwe": q.get("cwe"),
                              "expected_reasoning": q.get("expected_reasoning")}} if q else {}),
            **({"truth": r["truth"], "path": r.get("path")} if "truth" in r else {}),
            **({"cwe": r.get("cwe")} if "cwe" in r and not q else {}),
        })
    rng.shuffle(items)

    (out / "to_judge.jsonl").write_text(
        "\n".join(json.dumps(i) for i in items) + "\n", encoding="utf-8")
    (out / "KEY_do_not_open_until_graded.json").write_text(
        json.dumps({"label_to_variant": {v: k for k, v in tag2label.items()},
                    "n_items": len(items), "seed": a.seed}, indent=1), encoding="utf-8")
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
                                         thinking_mode="chat"))[:max_seq_len]
        with torch.inference_mode():
            _, logits, _ = model.forward(torch.tensor([ids], device=dev), 0)
        outs.append(logits.detach().float().cpu())
    return outs


def _cmp(a_list, b_list):
    md = mr = 0.0
    disagree = 0
    for a_, b_ in zip(a_list, b_list):
        d = (a_ - b_).abs()
        md = max(md, d.max().item())
        denom = a_.abs().max().item() or 1.0
        mr = max(mr, d.max().item() / denom)
        disagree += int((a_.argmax(-1) != b_.argmax(-1)).sum().item())
    return {"max_abs": md, "max_rel": mr, "argmax_disagreements": disagree}


def cmd_parity(a) -> None:
    """Assert the patched-but-inactive model is numerically the official model.

    The instrumentation rewrites Gate/MoE/Block forwards. If any rewrite changes
    the maths, every downstream number is contaminated and the change would look
    exactly like a pruning effect. This runs fixed prompts through the untouched
    model and the patched one and compares logits -- against a noise floor
    measured by running the untouched model twice, since expert accumulation and
    all-reduce are not bitwise reproducible.
    """
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
              ("norms + accumulate     ", True, True, False),
              ("norms + accum + mHC    ", True, True, True)]
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
    report = {"noise_floor": floor, "patched_profiling_off": c_off,
              "patched_profiling_on": c_on, "bisect": stage_reports,
              "after_unpatch": c_res,
              "tolerance_used": tol, "abs_tol": a.tol, "floor_multiplier": a.floor_mult}
    ok = True
    for name, c in (("profiling OFF", c_off), ("profiling ON", c_on),
                    ("after unpatch", c_res)):
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
        a.ckpt_path, a.config, a.code_dir, a.max_seq_len, 1)
    sys.path.insert(0, str(Path(a.code_dir).parent / "encoding"))
    from encoding_dsv4 import encode_messages

    dev = torch.device("cuda")
    prof = Profiler(args.n_layers, args.n_routed_experts, dev)
    patch(official, prof)
    set_mask(model, None, args.n_hash_layers, dev)
    out = Path(a.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    def say(m):
        if rank == 0:
            print(f"[easyep] {m}", flush=True)

    files = sample_calibration_files(Path(a.calib_dir), a.n_calib)
    say(f"validating norm recovery on {len(files)} files "
        f"(each expert is run twice: weighted and unweighted)")
    prof.enabled = prof.validate = True
    for i, (rel, code) in enumerate(files):
        ids = tok.encode(encode_messages(
            [{"role": "user", "content": security_review_text(rel, code)}],
            thinking_mode="chat"))
        if len(ids) > a.max_seq_len:
            ids = ids[: a.max_seq_len]
        if len(ids) < 32:
            continue
        with torch.inference_mode():
            model.forward(torch.tensor([ids], device=dev), 0)
        say(f"  {i+1}/{len(files)}  {len(ids)} tok")
    prof.enabled = prof.validate = False

    if rank != 0:
        if world_size > 1:
            dist.destroy_process_group()
        return

    err = prof.err_tensor().float()
    rec, tru = prof.score.float().cpu(), prof.score_true.float().cpu()
    q = torch.tensor([0.5, 0.9, 0.99, 1.0])
    pct = torch.quantile(err, q.to(err.device)).cpu().tolist() if err.numel() else [float("nan")] * 4

    n_hash, n_layers = args.n_hash_layers, args.n_layers
    rows, ov_all, sp_all = [], [], []
    for lid in range(n_hash, n_layers):
        a_top = set(torch.argsort(rec[lid], descending=True)[: a.keep].tolist())
        b_top = set(torch.argsort(tru[lid], descending=True)[: a.keep].tolist())
        ov = len(a_top & b_top)
        ra = torch.argsort(torch.argsort(rec[lid], descending=True)).float()
        rb = torch.argsort(torch.argsort(tru[lid], descending=True)).float()
        sp = torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
        rows.append({"layer": lid, "overlap": ov, "overlap_frac": round(ov / a.keep, 4),
                     "spearman": round(sp, 6)})
        ov_all.append(ov); sp_all.append(sp)

    report = {
        "keep": a.keep, "n_error_samples": int(err.numel()),
        "norm_rel_err": {"median": pct[0], "p90": pct[1], "p99": pct[2], "max": pct[3],
                         "mean": err.mean().item() if err.numel() else None},
        "topk_overlap": {"mean": sum(ov_all) / len(ov_all),
                         "min": min(ov_all), "max": max(ov_all),
                         "mean_frac": round(sum(ov_all) / len(ov_all) / a.keep, 4),
                         "n_layers_identical": sum(1 for o in ov_all if o == a.keep)},
        "spearman_full_ranking": {"mean": sum(sp_all) / len(sp_all), "min": min(sp_all)},
        "per_layer": rows,
    }
    (out / "norm_validation.json").write_text(json.dumps(report, indent=1))
    torch.save({"score_recovered": rec, "score_true": tru}, out / "validation_scores.pt")

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
    if world_size > 1:
        dist.destroy_process_group()


def cmd_pipeline(a) -> None:
    """profile -> mask -> eval(full) -> eval(pruned), on ONE model load."""
    official, model, tok, args, rank, world_size = build(
        a.ckpt_path, a.config, a.code_dir, a.max_seq_len, 1, a.temperature
    )
    sys.path.insert(0, str(Path(a.code_dir).parent / "encoding"))
    from encoding_dsv4 import encode_messages
    sys.path.insert(0, a.code_dir)
    from generate import generate

    dev = torch.device("cuda")
    prof = Profiler(args.n_layers, args.n_routed_experts, dev)
    patch(official, prof)
    out = Path(a.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    def say(m):
        if rank == 0:
            print(f"[easyep] {m}", flush=True)

    # ---------------- phase 1: calibration ----------------
    set_mask(model, None, args.n_hash_layers, dev)
    if a.scores_in:
        # Profiling is deterministic, so reuse an earlier run's scores rather than
        # spend ~10 min of 4xH100 recomputing them. All masks are rebuilt here, so
        # every variant comes from one consistent set of statistics.
        st = torch.load(a.scores_in, map_location="cpu")
        smhc = st.get("score_mhc")
        variants = all_variants(st["score"].float(), st["score_no_simibr"].float(),
                                st["counts"], a.keep, args.n_hash_layers,
                                args.n_layers, args.n_routed_experts,
                                a.seed, not a.no_controls,
                                smhc.float() if smhc is not None else None)
        say(f"phases 1-2 skipped; masks rebuilt from {a.scores_in}")
        if smhc is None:
            say("  NOTE: no score_mhc in this scores file; skipping the mHC variant")
        say(f"variants: {', '.join(t for t, _ in variants)}")
        if rank == 0:
            for tag, m in variants:
                if m is not None:
                    (out / f"mask_{tag}.json").write_text(json.dumps(m, indent=1))
        return _evaluate(a, model, tok, args, rank, world_size, dev, out, say,
                         encode_messages, generate, variants)
    files = sample_calibration_files(Path(a.calib_dir), a.n_calib)
    say(f"phase 1: profiling on {len(files)} calibration files")
    prof.enabled = True
    t0 = time.time()
    for i, (rel, code) in enumerate(files):
        ids = tok.encode(encode_messages(
            [{"role": "user", "content": security_review_text(rel, code)}],
            thinking_mode="chat"))
        if len(ids) > a.max_seq_len:
            ids = ids[: a.max_seq_len]
        if len(ids) < 32:
            continue
        with torch.inference_mode():
            model.forward(torch.tensor([ids], device=dev), 0)
        say(f"  calib {i+1}/{len(files)}  {len(ids)} tok  {rel}")
    prof.enabled = False
    say(f"phase 1 done in {time.time()-t0:.0f}s, {prof.tokens_seen} token-layer records")

    if rank == 0:
        torch.save(prof.state(), out / "expert_scores.pt")

    # ---------------- phase 2: mask ----------------
    scores = prof.score.float().cpu()
    scores_alt = prof.score_no_simibr.float().cpu()
    mask = build_mask_from_scores(scores, a.keep, args.n_hash_layers)
    mask_alt = build_mask_from_scores(scores_alt, a.keep, args.n_hash_layers)
    if rank == 0:
        (out / ("mask_keep%d.json" % a.keep)).write_text(json.dumps(mask, indent=1))
        (out / ("mask_keep%d_no_simibr.json" % a.keep)).write_text(json.dumps(mask_alt, indent=1))
        cmp_rows = compare_scorings(scores, scores_alt, mask, mask_alt,
                                    a.keep, args.n_hash_layers, args.n_layers)
        (out / "score_comparison.json").write_text(json.dumps(cmp_rows, indent=1))
        ov = [r["overlap"] for r in cmp_rows["per_layer"]]
        say(f"phase 2: mask keeps {a.keep}/{args.n_routed_experts} per layer on "
            f"layers {args.n_hash_layers}..{args.n_layers-1}")
        say(f"         top-{a.keep} overlap vs no-simibr: "
            f"mean {cmp_rows['overlap_mean']}/{a.keep} "
            f"({cmp_rows['overlap_mean_frac']:.1%}), "
            f"min {min(ov)} (layer {cmp_rows['min_overlap_layer']}), max {max(ov)}")
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
    variants = all_variants(scores, scores_alt, prof.counts.cpu(), a.keep,
                            args.n_hash_layers, args.n_layers, args.n_routed_experts,
                            a.seed, not a.no_controls, prof.score_mhc.float().cpu())
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

    summary = {}
    for tag, m in variants:
        set_mask(model, m, args.n_hash_layers, dev)
        say(f"phase 3: evaluating variant={tag} on {len(questions)} questions")
        rows = []
        for i, q in enumerate(questions):
            ids = tok.encode(encode_messages(
                [{"role": "user", "content": question_text(q)}], thinking_mode="chat"))
            # Paired comparison: question i must see the SAME sampling noise under
            # both variants, otherwise the full/pruned delta mixes the pruning effect
            # with decoder randomness. Reseed per question, identically on every rank.
            torch.manual_seed(a.seed + i)
            torch.cuda.manual_seed_all(a.seed + i)
            t1 = time.time()
            with torch.inference_mode():
                gen = generate(model, [ids], a.max_new_tokens, tok.eos_token_id)
            completion = tok.decode(gen[0])
            if rank == 0:
                rows.append({
                    "id": q.get("id"),
                    "cwe": q.get("cwe"),
                    "language": q.get("language"),
                    "snippet": q.get("snippet"),
                    "vulnerability": q.get("vulnerability"),
                    "expected_reasoning": q.get("expected_reasoning"),
                    "completion": completion,
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
        sp.add_argument("--max-seq-len", type=int, default=8192)
        sp.add_argument("--limit", type=int, default=0)

    sp = sub.add_parser("profile", help="calibration -> expert scores")
    common(sp)
    sp.add_argument("--calib", required=True)
    sp.add_argument("--out", required=True)

    sp = sub.add_parser("mask", help="scores -> mask")
    sp.add_argument("--scores", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--keep", type=int, default=192)
    sp.add_argument("--n-hash-layers", type=int, default=3)
    sp.add_argument("--no-simibr", action="store_true",
                    help="rank by weight*norm only, i.e. EASY-EP without token contribution")

    sp = sub.add_parser("eval", help="generate with or without a mask")
    common(sp)
    sp.add_argument("--questions", required=True)
    sp.add_argument("--mask", default="")
    sp.add_argument("--out", required=True)
    sp.add_argument("--max-new-tokens", type=int, default=256)

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
                    help="skip the frequency and random baselines")
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
    sp.add_argument("--no-controls", action="store_true")

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
    sp.add_argument("--n-calib", type=int, default=8)
    sp.add_argument("--keep", type=int, default=128)

    a = p.parse_args()
    {"profile": cmd_profile, "mask": cmd_mask, "eval": cmd_eval,
     "pipeline": cmd_pipeline, "pairs": cmd_pairs, "blind": cmd_blind,
     "validate": cmd_validate, "parity": cmd_parity}[a.mode](a)


if __name__ == "__main__":
    main()
