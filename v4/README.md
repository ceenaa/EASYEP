# EASY-EP for DeepSeek-V4-Flash

Adaptation of EASY-EP ([arXiv:2504.06792](https://arxiv.org/abs/2504.06792)) from
DeepSeek-R1 to **DeepSeek-V4-Flash-0731** (284B MoE, 13B active), running on a
single 4×H100 node.

This is an *adaptation*, not a port: V4-Flash differs from R1 in four ways that
each touch the scoring path. Nothing under `pruning/`, `sglang/` or `configs/` is
modified — the R1 pipeline is left intact and all V4 work is confined to `v4/`.

## Scope of this first version

| | |
|---|---|
| Layers 0–2 | **unchanged**, all 256 experts kept |
| Layers 3–42 | pruned 256 → 128 experts |
| MTP / DSpark layers 43–45 | **excluded** from profiling and masking (see *Open items*) |
| Validation | **masking only** — expert weights are never modified |

## The score

The paper's score is a sum of per-token **products**
(`pruning/expert_selection.py:38` upstream):

```python
score[layer, e] += weight_t[e] * simibr_t * norm_t[e]
```

| term | meaning |
|---|---|
| `weight` | gating score for expert `e` on token `t` |
| `norm`   | ‖unweighted expert output‖₂ |
| `simibr` | `max(1 − cos(h, h + y_routed), 0)` — token contribution |

Implemented in `easyep_v4.py`:

- `moe_forward` — collects per-token `weights`, `indices`, and unweighted `norms`
- `moe_accumulate` — forms the product and adds it to the running score
- `block_forward` — supplies `h`, the pre-FFN residual, so `simibr` is computable

The accumulation must happen per token. A previous attempt accumulated `Σweight`
and `Σnorm` into separate marginal buffers and ranked by the latter; a product-sum
cannot be recovered from marginals, and that variant drops `simibr` entirely.
For comparison, `score_no_simibr` (= `Σ weight·norm`) is recorded in the **same
calibration pass**, so the effect of the token-contribution term is measurable
without a second run and without calibration variance between the two.

`score_comparison.json` reports, for every layer 3–42:

| field | meaning |
|---|---|
| `overlap` | how many of the top-128 experts the two rules agree on |
| `jaccard` | overlap over union |
| `spearman_full_ranking` | rank correlation over all 256 experts |
| `only_in_paper_score` / `only_in_no_simibr` | the experts that actually differ |
| `n_never_activated` | experts with zero calibration traffic in that layer |

plus `overlap_mean`, `overlap_min`, `overlap_max` and `min_overlap_layer` so the
layers where the token-contribution term matters most are immediately visible.

## The four V4 divergences

### 1. `sqrtsoftplus` gate

R1 uses `sigmoid`; V4 uses `F.softplus(scores).sqrt()` with a `noaux_tc` bias that
shifts top-k selection but not the returned routing weights. `gate_forward` mirrors
upstream V4 exactly and adds only an optional keep-mask applied to the scores
before `topk`. The returned `weights` are unchanged in meaning, so the score
formula carries over as-is.

### 2. mHC residuals

V4 maintains `hc_mult = 4` copies of the hidden state. This turns out **not** to
need an aggregation choice: `Block.hc_pre` reduces the 4 copies to a single vector
*before* the FFN and `hc_post` re-expands *after*, so inside the FFN sub-block the
stream is one vector per token, exactly as in R1.

```python
residual = x                              # [b,s,hc,d]
x, post, comb = self.hc_pre(x, ...)       # [b,s,d]   <- h, the analogue of x_before_moe
x = self.ffn_norm(x)
x = self.ffn(x, input_ids)                # [b,s,d]
x = self.hc_post(x, residual, post, comb) # [b,s,hc,d]
```

`h` is the pre-norm reduced residual, matching upstream's use of the pre-norm
residual rather than the normed FFN input.

### 3. Hash-routed layers 0–2

`Gate` routes by **token ID** (`tid2eid`) for `layer_id < n_hash_layers`, not by
content. An expert pruned there is unreachable for the specific vocabulary items
mapped to it, so these layers are excluded from both profiling and masking and
keep all 256 experts. (Upstream also skips its first 3 layers, but because they
are dense MLPs — different reason, same range.)

### 4. FP4 expert format

Experts are stored `float4_e2m1fn_x2` — two values per byte, with a per-32
`float8_e8m0fnu` block scale. This does **not** affect masking, which is why this
version validates by masking first. It will matter for actual deletion: removing
an expert means repacking rather than slicing, since expert boundaries need not
land on byte boundaries.

Two consequences already handled:

- V4 folds the gating weight *inside* `Expert.forward`, computing
  `w2(w · silu(gate)·up)`. Since `w2` is linear, the output is exactly
  `w × unweighted`, so the unweighted norm is recovered by dividing by `w`
  rather than paying a second forward pass.
- V4's `MoE.forward` returns routed + shared summed. `simibr` needs the routed
  part alone, so it is stashed on the module for `Block` to read.

## Running it

```bash
sbatch v4/easyep.sbatch      # profile -> mask -> eval(full) -> eval(pruned), one model load
```

The four phases share a single model load; on Rorqual that load is ~15 minutes,
so splitting them into separate jobs would triple the dominant cost.

Outputs, under `--out`:

| file | contents |
|---|---|
| `expert_scores.pt` | per-layer/expert `score`, `score_no_simibr`, `counts`, `gate_sums` |
| `mask_keep128.json` | kept/pruned expert ids per layer |
| `mask_keep128_no_simibr.json` | same, ranked without the token-contribution term |
| `score_comparison.json` | **per-layer** top-128 overlap between the two scorings |
| `answers_full.jsonl` | 50 questions, unmasked |
| `answers_pruned.jsonl` | 50 questions, masked |
| `summary.json` | term-overlap rubric per variant |

Standalone modes (`profile`, `mask`, `eval`) exist for iterating on one stage.

## Evaluated variants

Three, all decoded under identical per-question seeds:

| tag | mask |
|---|---|
| `full` | none - all 256 experts |
| `pruned_paper` | top-128 by `weight x simibr x norm` (the paper's rule) |
| `pruned_no_simibr` | top-128 by `weight x norm` (the earlier port's rule) |
| `pruned_frequency` | top-128 by selection count - the naive heuristic |
| `pruned_random` | seeded random 128 - the floor |

The two controls answer different questions. `pruned_frequency` asks whether the
paper's machinery beats the obvious "keep what gets picked most". `pruned_random`
asks whether the scoring carries any signal at all -- if it ties the scored masks,
the model is simply robust to expert removal and no scoring rule is doing work.
Measured mask overlap against `pruned_paper`: no_simibr 94.5%, frequency 88.6%,
random 49.7% (chance, as it should be).

`score_comparison.json` shows the two rules *disagree*; running both shows which
one selects better experts. Use `--skip-alt-eval` to drop the third variant.

## Correctness gates

Run before anything else; the later jobs are Slurm-gated on the first.

| gate | what it rules out |
|---|---|
| `parity` | the instrumentation itself changing the model. Fixed prompts through the untouched official forwards vs the patched-but-inactive ones, comparing prefill logits against a noise floor measured by running the official model twice (expert accumulation and all-reduce are not bitwise reproducible). Fails the job on a mismatch. |
| strict checkpoint load | a partially loaded model masquerading as a pruning effect. `load_model(strict=False)` returns `(missing, unexpected)` and callers normally discard both. Now every unexpected key, and every missing key outside `ALLOWED_MISSING`, aborts the run. |
| `validate` | the unweighted-norm recovery. `||w*out||/w == ||out||` is exact only in exact arithmetic; V4 applies the routing weight *before* `w2`, whose FP4 path runs `act_quant`, so quantisation can break proportionality. Runs each expert twice and reports norm error plus top-k ranking overlap. |

`ALLOWED_MISSING` covers only `mtp.N.embed.*` and `mtp.N.head.*`, which are
references to the trunk embedding and head (`Transformer.__init__` assigns
`mtp[i].embed = self.embed`) and are stored once, not per stage.

## Open items

- **Paired decoding.** `config.json` sets no `temperature`, so `ModelArgs`
  defaults to 1.0 and `Transformer.forward` samples every token via Gumbel-max,
  which consumes RNG. Seeding once at load would decode `full` and `pruned` under
  different noise and confound the comparison. Question *i* is therefore decoded
  under `manual_seed(seed + i)` in **both** variants; `--temperature 0` gives
  greedy decoding if an entirely noise-free A/B is wanted. The regime used is
  recorded in `summary.json._decoding`.


- **MTP layers are excluded, not handled.** `DSparkBlock.forward` delegates to
  `Block.forward` when `start_pos > 0` with `layer_id` 43–45; the accumulator is
  bounds-guarded against that. They have their own experts under the `mtp.*`
  namespace and warrant their own profiling and pruning decision.
- **No weight deletion.** Masking only, by design.
- **Calibration size.** 25 files stratified across CWE directories, matching the
  paper's 25 samples. Not yet swept.
- **`simibr` reference point.** `h` is the pre-`ffn_norm` reduced residual. The
  alternative — comparing in the post-`hc_post` `[b,s,hc,d]` space, with and
  without the routed term — is closer to V4's true residual dynamics but compares
  in a different space. Worth a look if scores appear insensitive to `simibr`.
