# EASY-EP on DeepSeek-V4.1-Flash: required changes

Assessment date: 2026-09-10. Target interpreted as the released `deepseek-ai/DeepSeek-V4.1-Flash`. Official checkpoint/source revision examined: `dba1be0a40aa45a94ad051997016db3960a90277`. This assessment answers what would need to change; it does not modify the working V4 engine or claim that V4.1 inference/pruning has run.

**Conclusion:** adapting EASY-EP is plausible, but V4.1 requires a new model integration and a different memory plan. Changing the model name or mask dimensions is insufficient. The existing RTX 5090 plus approximately 250 GB host RAM setup cannot hold the full model using our current resident host-bank design. Even halving the routed experts does not make that design fit when Engram tables remain resident.

## Evidence inspected

- [Official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash): release architecture and encoding/inference references.
- [Official technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/DeepSeek_V41_Tech_Report.pdf): architecture Figure 3/page 7; CED section 2.2/page 9; Single-Pass mHC and Engram sections 2.4.1-2.4.2/pages 12-13; deployment sections 3.2.1-3.2.2; model setup section 4.2.1/pages 21-22. Relevant architecture, CED, mHC and model-setup pages were also rendered and visually inspected.
- [Pinned inference source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py), configuration, conversion code, kernel code and encoding documentation. These are essential because the minimal reference does not implement every production optimization described in the paper.
- [EASY-EP paper](https://arxiv.org/html/2504.06792v2): expert importance combines routing weight, output norm and token representation change. Its experiments concern R1 and V3-0324, not V4.1; preservation of security-review ability must be measured anew.
- Current local and live upstream FreeToken registry at `46d27439a4fccf457e1a9a858841168ec7a1c65d`: registers `DeepseekV4ForCausalLM`, not `DeepseekV41ForCausalLM`.

Reference files, hashes, model metadata, the complete 48-shard size inventory, memory calculations and the adapter rejection check are saved under `research/reference-configs/v4.1-flash/`. No weight payloads were downloaded.

## Architectural differences that affect this project

| Item | Current V4-Flash-0731 adapter | V4.1-Flash requirement |
|---|---|---|
| Model identity | `deepseek_v4` | `deepseek_v41`; nested `text_config` |
| Backbone | 43 layers | 40 layers, organized as 20 encoder + 20 decoder |
| Routed experts | 256 per layer, top-6 | 384 per backbone layer, top-6 |
| Hidden/expert dimensions | 4096 / 2048 | 5120 / 2304 |
| First three MoE layers | Token-ID hash routing, preserved | Learned routers; the V4 hash exemption does not apply |
| Residual connections | V4 mHC | Single-Pass mHC: shifted input-mixing coefficients passed between sublayers |
| Attention/cache | V4 cache/compression logic | CSA2 shared KV/index state, Full/Reindex/Reuse modes, hierarchical sparse indexing, revised compression and FP4 main KV quantization |
| Conditional memory | No Engram path in our adapter | Engram modules at layers 1 and 14, including n-gram history and large embedding tables |
| Quantization | V4 FP8 blocks/activation groups of 128 in current paths | FP8 weight blocks and activation groups of 32; FP4 expert scales remain per 32 values |
| Routing bias | Text bias | Text bias plus image-token bias; preserve original selection/weight semantics |
| Draft layers | Excluded | Three DSpark layers have 128 experts/top-3; keep distinct from the 40 x 384 backbone |
| Prompt encoding | V4 encoder discovery | V4.1 `encoding.py`, numeric reasoning effort 1-100 and revised protocol tags |

These differences are verified in the [configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/config.json), [model implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py) and [encoding notes](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/encoding/README.md).

## Memory and hardware

The official Hugging Face tree lists **510,296,708,312 bytes = 510.30 GB = 475.25 GiB** across 48 safetensors files. All files referenced by the weight index are accounted for. This is storage, not measured peak runtime memory.

Using the released dimensions, the 40-layer routed expert bank contains:

`40 * 384 * 3 * 5120 * 2304 = 543,581,798,400 parameters`.

At packed FP4 plus one scale byte per 32 values, that is **288.78 GB / 268.95 GiB** for base routed weights/scales alone. One complete expert layer is about **6.72 GiB**, so two staging layers would consume about **13.45 GiB** of VRAM before dense weights, KV and workspace.

Engram embedding payloads account for about **183.11 GiB**, plus roughly **5.72 GiB** of embedding scales, from the released table row counts and 256-wide rows. These tables are separate from the routed experts and are not removed by an ordinary expert mask.

Even after physical removal of half the routed experts, the remaining routed bank plus Engram embedding tables/scales totals approximately **347.15 GB / 323.31 GiB**, before shared/dense weights, Engram projections, vision, DSpark, caches, activations and allocator overhead. Masking alone removes no stored weights.

Consequences:

- More host RAM alone would not make the official reference fit on one 5090: its model is GPU-resident and tensor-parallel. The [reference instructions](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/README.md) demonstrate conversion/inference with `MP=8`; that example is not a guarantee for arbitrary eight-GPU types or context sizes.
- Keeping one 5090 would require a V4.1-compatible offload engine and either substantially more host RAM, storage-backed expert/Engram access, or a separately validated smaller quantization. The existing FreeToken adapter provides none of those V4.1 paths.
- Calibration needs the unpruned model first. A future compressed checkpoint cannot solve the initial calibration residency problem by itself.
- API calls can return reviews but do not supply internal expert activations or editable expert weights, so API access alone is not an EASY-EP calibration backend.

## Required implementation changes

### 1. Establish a correct V4.1 inference baseline

Add a separate `deepseek_v41` model integration, configuration reader, weight loader and registry entry, or initially instrument the official reference on suitable hardware. Implement Engram lookup/history, Single-Pass mHC, CSA2's shared state and the released quantization semantics. Preserve the `1e-20` normalization constants and router temperature/normalization behavior rather than inheriting V4 defaults.

The reference applies the routing weight to the intermediate expert activation before the down-projection's activation quantization. Our current FreeToken FP4 path applies the routed weight later. Moving a gate across quantization is not generally numerically equivalent. Compare per-expert outputs against the V4.1 reference before reusing the old kernels or treating divided weighted-output norms as exact unweighted norms.

Relevant local implementation areas are `models/register.py`, a new `models/deepseek_v41/`, quantization/FP4 kernels, expert-bank loading, and the attention/KV cache interfaces. This work is a model port, not just pruning instrumentation.

### 2. Replace the V4-specific mask/provenance contract

Use a new model/metric schema and a **40 x 384** backbone mask. Preserve original expert IDs and at least six eligible experts in every row. Apply exclusion with negative infinity before top-k, then gather and normalize the original unbiased scores. The first three backbone layers are no longer hash-routed; preserve them only if an explicit experimental policy chooses to do so.

Keep shared experts, Engram and DSpark outside the initial expert-pruning scope. Do not put DSpark's 128-expert/top-3 rows into the backbone matrix. Record the pinned checkpoint, tokenizer/encoder, quantization, runtime implementation and calibration identities; reject old V4 masks and scores. The current adapter already rejects this official V4.1 config, which was verified locally.

### 3. Re-derive the output-aware score at the correct residual boundary

Observe expert outputs without changing the model's forward arithmetic. For a proposed residual-space V4.1 extension, let `p` be the FFN output-expansion coefficients, `B` the residual mixing map, `X` the residual before the FFN and `r` the routed-only FFN sum. Compare `B X` with `B X + p outer r`, excluding the shared-expert contribution. Express each expert contribution's norm in that same residual space, including the `norm(p)` factor.

Single-Pass mHC changes which input-mixing coefficients feed the FFN; hooks must follow that shift. Keep the observed, already weighted contribution and any explicit unweighted-reference diagnostic separate, since quantization can break the algebraic gate-weight equivalence. Version the chosen score definition and test it against small tensor/reference cases. This is an experimental extension of EASY-EP, not a result demonstrated by its paper.

### 4. Make calibration coverage depend on the execution path

The paper's optimized CED prefill runs the encoder over the prompt and rebuilds only recent decoder SWA states. Therefore, an optimized runtime cannot use our current invariant that every layer sees every prefill token exactly once. Track layer, token position, encoder/decoder phase and replay purpose; distinguish absent computation from inactive experts and prevent replay double-counting.

**Reference-code qualification:** the released minimal `Transformer.forward` currently loops through all 40 backbone layers for the supplied tokens; it does not implement the paper's encoder-only prefill shortcut. A full teacher-forced reference pass is a useful first baseline and can cover all layers. If moving to optimized CED later, validate that calibration samples the actual response-generating decoder work; do not infer that a prompt-only encoder trace measures decoder importance.

Generate complete security reviews with the unmasked model, replay the exact prompt/response token IDs where possible, and keep warm-up and speculative/draft work outside the calibration totals. The minimal reference's later-position path expects one-token decode in several places, so our existing arbitrary continuation-chunk replay cannot simply be copied into it.

### 5. Update encoding and defensive evaluation

Use the V4.1 encoder and record numeric reasoning effort. Match the same prompts, token budgets and sampling between baseline and masks. Retain the custom source-manifest and human review workflow, but use neutral display paths to avoid labels leaking through filenames. Supply representative security-review code, with separated calibration, validation and held-out test projects; a small activation sample does not establish that all cybersecurity ability is retained.

Add frequency, gate-weight and seeded-random mask controls, score cutoff/stability diagnostics, and blinded review. Check recall, precision, clean-case false positives and critical misses. Text-only code review does not require calibrating a new vision-pruning policy; preserve the multimodal components and make no claim about their retained capability.

### 6. Validate before physical pruning

Required gates are: unmodified-reference parity; all-ones-mask parity; masked-expert exclusion; quantized expert-output/norm checks; Single-Pass mHC reference checks; correct encoder/decoder/replay coverage; and held-out quality evaluation. Measure real peak host/GPU memory and throughput.

Only then add an exporter and compact loader that remove selected expert tensors/scales, update routers and checkpoint indexes consistently, and preserve Engram/shared/vision/draft components. Verify compact-model outputs against the masked model. Existing V4 test results do not establish any V4.1 compatibility.

## Suggested order

First choose a viable unpruned calibration backend and memory layout. Then establish V4.1 inference correctness, add the new profiler/mask contract, run a small defensive-code pilot, and evaluate retention levels before export. The official tensor-parallel reference is the most direct correctness starting point; making the same experiment economical on one 5090 is additional offload/storage engineering.

The V4 deployment bundle remains a V4-only artifact. It should not be used as a ready-to-run V4.1 package. No engine changes, GPU execution, model downloads or paid-server actions were performed for this assessment.
