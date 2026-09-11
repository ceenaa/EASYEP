# DeepSeek expert masking with FreeToken and EASY-EP

Research date: 2026-09-10. Confirmed target: **DeepSeek-V4-Flash-0731**, on a Vast.ai **RTX 5090** instance with approximately 250 GB system RAM, quoted at $0.80/hour. The earlier RTX 5060 description was a typo. The user wants to preserve defensive cybersecurity code review and vulnerability detection. The previous FreeToken launch command/version is unavailable; use the pinned checkout below as the proposed baseline and verify it on the rental. No instance has been rented or benchmarked in this workspace.

**Decision:** an RTX 5090 with 32 GB VRAM and approximately 250 GB host RAM is a plausible platform for a FreeToken-based EASY-EP experiment on the confirmed V4-Flash-0731 checkpoint. The original EASY-EP stack still cannot run unchanged: its collector targets CUDA-resident R1/V3, so it needs a FreeToken/V4 port. The earlier 8 GB GPU rejection does not apply to the corrected hardware. Calibration memory, runtime, and security-review quality must be measured. See [the concrete security-review experiment](SECURITY_EXPERIMENT.md). Research and local implementation are complete. An experimental FreeToken/V4 adapter and deployment workflow are available in [RUN_ON_VAST.md](RUN_ON_VAST.md); [VALIDATION.md](VALIDATION.md) records the local checks. Full-model CUDA execution and a physically pruned model remain pending.

## What FreeToken is

[FlashML's FreeToken](https://github.com/FlashML-org/FreeToken) is an inference engine for models whose expert weights exceed GPU memory. It keeps expert banks in host RAM, caches experts on the GPU, streams weights during prefill, and supports CPU/GPU execution of decode misses. Its API is compatible with common model clients. It is not an expert-pruning method. This matches the user's description much better than the unrelated service using the freetoken.ai name. The [FreeToken paper](https://arxiv.org/html/2608.16157v1) describes bandwidth-adaptive scheduling and caching and reports DeepSeek-V4-Flash inference on an RTX 5090. That supports the hardware choice for inference, but does not benchmark EASY-EP instrumentation or this specific rental.

The cloned model registry has a DeepSeek-V4 implementation and no DeepSeek-V2/V3/R1 implementation. Current [supported-model documentation](https://github.com/FlashML-org/FreeToken/blob/46d27439a4fccf457e1a9a858841168ec7a1c65d/docs/models.md) lists DeepSeek-V4-Flash-0731, matching the user's confirmed model.

## What EASY-EP does

The [paper, arXiv v2](https://arxiv.org/html/2504.06792v2), scores routed experts using a forward pass on domain demonstrations, including model-generated responses. Its score combines routing weight, expert-output magnitude, and token sensitivity:

\[
I_{l,e}=\sum_t g_{l,t,e}\|E_{l,e}(x_{l,t})\|_2\left[1-\cos(h_{l,t},h_{l,t}+r_{l,t})\right].
\]

Here `x` is the expert input, `h` the residual before routed-expert addition, and `r` the sum of weighted routed outputs. The implementation clamps negative token contributions to zero. Experts with the highest scores are retained per layer. Shared experts remain intact. Mixed-domain selection combines scores normalized within each domain/layer.

Experiments use 25 demonstrations per domain on DeepSeek-R1 and V3-0324. The claimed 2.99x throughput is a particular equal-memory multi-GPU comparison, not a prediction for this rental. V4 was not evaluated. A port to V4 needs new quality measurements and an explicit treatment of its different residual structure.

Collect scores with the **unmasked baseline**, then apply the selected mask. Collecting only after arbitrary masking changes the model being measured and removes evidence about excluded experts. Counting router selections alone is a useful diagnostic, but does not implement EASY-EP's output-aware score.

## Feasibility on the proposed rental

The standard [RTX 5090 has 32 GB VRAM](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/). Its approximately 250 GB host RAM is a separate memory pool.

| Candidate checkpoint | Verified weight-file storage | Assessment |
|---|---:|---|
| Official DeepSeek-R1 | 688.59 GB / 641.30 GiB | Full checkpoint cannot reside in 250 GB RAM. EASY-EP's CUDA-resident collector also cannot fit on a single 32 GB GPU. |
| Official DeepSeek-V4-Flash-0731 | 166.89 GB / 155.43 GiB | Plausible with FreeToken offloading on 32 GB VRAM and 250 GB host RAM. Requires the V4/FreeToken EASY-EP port and validation of the exact build, prompt lengths, and allocation. |
| A separately quantized R1 or another DeepSeek model | Unknown until identified | Format, architecture, and runtime support must be checked. Distilled dense DeepSeek models do not have routed MoE experts to prune. |

Storage values are sums of actual `.safetensors` file sizes from Hugging Face's public repository-tree API, not runtime memory estimates: [R1 files](https://huggingface.co/deepseek-ai/DeepSeek-R1/tree/main), [V4-Flash-0731 files](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main). Retrieved inventories are in `research/reference-configs/*/storage-summary.json`. R1's index metadata reports a different total, so relying on `metadata.total_size` alone would be misleading.

For perspective, an idealized 671B model at exactly four bits per parameter is approximately 335.5 GB before quantization scales and runtime overhead. Ordinary full-model four-bit R1 is therefore not a solution to the 250 GB limit. Smaller quantizations or disk/layer streaming are different implementation choices; neither is provided by EASY-EP's stock collector.

For the V4 base transformer, the [configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/config.json) has 43 MoE layers, 256 routed experts each, hidden size 4096, and expert intermediate size 2048. Three matrices per expert at four bits plus one scale byte per 32 weights give approximately **137.06 GiB for routed expert weights/scales**, or **3.1875 GiB per whole layer**. This is a shape-based calculation excluding shared/dense weights, additional prediction modules, allocator overhead, and runtime state.

FreeToken's [prefill cache](https://github.com/FlashML-org/FreeToken/blob/46d27439a4fccf457e1a9a858841168ec7a1c65d/python/freetoken/moe/offload_cache.py) can use two full expert layers for overlap: about 6.375 GiB in this calculation, before other GPU allocations. Disabling overlap removes one staging layer but does not solve dense-weight residency. Use a separate environment for this build; the previous launch command is unavailable.

### Verified GPU lower bound for the current V4 configuration

Read-only HTTP range requests retrieved three small safetensors headers (approximately 169 KiB total), including the embedding, output head, and layer-0 projections. No weight payloads were read. The [saved header metadata](research/reference-configs/v4-flash-0731/selected-weight-headers.json) confirms BF16 embedding/head weights and FP8 attention/shared-expert matrices. The [attention implementation](FreeToken/python/freetoken/models/deepseek_v4/attention.py) constructs the same projection shapes in all 43 base layers. The loader expands `wo_a` to BF16, and the [engine](FreeToken/python/freetoken/engine/engine.py) materializes these tensors on the GPU.

| Persistent tensor subset | Runtime size |
|---|---:|
| Embedding and output head, BF16 | 1.97266 GiB |
| `wq_a`, `wq_b`, `wkv`, `wo_b`, FP8, all 43 layers | 2.93945 GiB |
| `wo_a`, BF16, all 43 layers | 2.68750 GiB |
| Shared experts, FP8, all 43 layers | 1.00781 GiB |
| **Subtotal** | **8.60742 GiB** |

The [calculation record](research/reference-configs/v4-flash-0731/gpu-lower-bound.json) contains every shape, dtype, multiplicity, and byte total. This is a conservative lower bound, not measured peak VRAM: it excludes scales, routers, hash tables, norms, compressors, indexers, hyper-connections, routed-expert caches, KV state, activation buffers, and CUDA overhead. This subset fits within the corrected 32 GB GPU capacity. Adding 6.375 GiB of expert staging gives a subtotal of approximately 14.98 GiB, still excluding the other allocations; this is not a complete peak-memory estimate. The calculation's `exceeds_8_GiB` field documents the earlier comparison and is not a rejection of the RTX 5090.

Vast.ai runs Linux containers. Its [resource-allocation documentation](https://docs.vast.ai/guides/instances/docker-environment) says CPU and RAM baselines are proportional to allocated GPUs; host-wide free RAM is not the same as the rental's guaranteed allowance. Check the offer's allocation, CPU/NUMA placement, PCIe bandwidth, and disk size. The supplied $0.80/hour quote is $4 for 5 hours or $19.20 for 24 hours, before any separate storage/network charges. These are arithmetic examples, not estimates of calibration duration.

The corrected RTX 5090 is in the hardware class used for V4 inference in the FreeToken paper. The rental's CPU and PCIe bandwidth will affect collection speed; no runtime estimate or marketplace price comparison has been established.

## Masking versus removing weights

**First implement routing masks while retaining original expert IDs and weights.** This makes quality comparison and rollback straightforward. Keep the active top-k unchanged: R1 uses 8; the inspected V4 uses 6. Retaining half the expert pool does not halve the active expert count per token.

Masking alone does not shrink the checkpoint or host expert-bank allocations. In [FreeToken's MoE execution](https://github.com/FlashML-org/FreeToken/blob/46d27439a4fccf457e1a9a858841168ec7a1c65d/python/freetoken/layers/moe.py), long-prefill paths materialize full layers. Those transfers may remain unchanged after routing masks. Decode cache locality could improve, but speedup must be measured. Actual storage/RAM savings require a later exporter and loader/bank changes.

## Code changes required

| Area | Evidence and required change |
|---|---|
| Architecture | EASY-EP assumes MoE layers 3–60 and 256 experts. Derive layer mappings and dimensions from the target config. V4 has MoE layers 0–42, including three hash layers. |
| GPU residency on the 5090 | Start with FreeToken's existing offload path and measure peak memory before adding statistics. Stream score accumulation and tune cache/prefill sizes if needed. The corrected 32 GB GPU does not require a new non-routed-weight offload implementation merely to overcome the earlier 8 GB limitation. |
| Mask application | Add an explicit mask option and checkpoint provenance validation to FreeToken. Reject missing/invalid files. Apply exclusion to selection scores before top-k using negative infinity, while preserving unbiased weights, normalization, and route scaling. Retain original IDs during the masking stage. |
| V4 hash routing | `Gate.forward` uses a token-to-expert table in layers 0–2, bypassing score top-k. For the initial experiment retain all experts there. Pruning those layers requires a separately evaluated hash-routing policy, not a score-mask edit. |
| Expert statistics | Instrument expert computation before weighted reduction to obtain each activated expert's output norm. The public API and a hook on the final aggregate cannot recover individual norms. FreeToken uses custom operators and fused kernels, so ordinary PyTorch module hooks are insufficient. |
| V4 residual sensitivity | V4 has four hyper-connected residual streams. Define the before/after comparison at that boundary; do not blindly substitute `cos(x, x + routed)` using the normalized FFN input. A possible extension compares the hyper-connection residual-only output to that output plus the routed contribution, keeping shared contribution separate. This is a proposed adaptation requiring validation. |
| Offloading and IDs | Collect statistics for both prefill and decode and for CPU/GPU paths actually used. Copy logical expert IDs before FreeToken remaps them to cache slots. Avoid double-counting CPU/GPU contributions. |
| Bounded memory | Accumulate per-layer/per-expert scores online in FP32, with small token chunks. Avoid storing every hidden state or per-token expert trace. For 43×256 experts one FP32 score accumulator is only 43 KiB. Preserve sequence/position semantics across chunks. |
| Reproducibility | Calibrate with the exact checkpoint, quantization, tokenizer, sample IDs, and original model responses; use separate held-out evaluations. Disable or explicitly handle graph warm-up, padding, and prefix reuse during collection so token coverage is correct. |
| Physical pruning later | Rewrite expert weights AND scale tensors, router mappings, bias/hash metadata as applicable, indexes, config, and packed banks. Verify masked and physically pruned logits agree before judging memory/performance. Variable counts for preserved hash layers need loader/bank support. |

Relevant FreeToken files: [router/MoE](FreeToken/python/freetoken/models/deepseek_v4/moe.py), [block and hyper-connections](FreeToken/python/freetoken/models/deepseek_v4/model.py), [expert execution](FreeToken/python/freetoken/layers/moe.py), [MXFP4 kernels](FreeToken/python/freetoken/moe/fused_ds_fp4.py), [weight loader](FreeToken/python/freetoken/models/deepseek_v4/weight.py). These sites now contain the experimental local adapter. GPU validation remains pending.

## Issues found in the EASY-EP snapshot

These findings are from the pinned source checkout, not GPU reproductions:

- [inf_new.py](EASYEP/pruning/inf_new.py) constructs the model on CUDA; adding host RAM does not enable offloading. It hardcodes layer bounds. Its overlong-input skip occurs after hook registration, so skipped samples leave hooks attached.
- [model_new.py](EASYEP/pruning/model_new.py) calls `dist.all_gather` for expert norms unconditionally, although the runner initializes distributed state only when world size exceeds one. Single-process collection needs that fixed as well as memory changes.
- [expert_selection.py](EASYEP/pruning/expert_selection.py) hardcodes 58×256 and reads the entire JSONL. It shuffles, then sorts by serialized line length and takes 25; that is not simple random sampling of demonstrations.
- The supplied [SGLang model patch](EASYEP/sglang/sglang_full/sglang/srt/models/deepseek_v2.py) defaults to an absolute-context-dependent 64-expert mask path and silently uses all experts if the file is absent.
- [biased_grouped_topk](EASYEP/sglang/sglang_full/sglang/srt/layers/moe/topk.py) excludes candidates by multiplying selection scores by zero. With correction bias, allowed scores can be negative, allowing excluded zero-score experts into top-k. A robust implementation must use negative infinity and check sufficient eligible experts after group selection. Changing group-selection semantics requires explicit evaluation.
- [model_prune.py](EASYEP/pruning/model_prune.py) writes a filtered index, then copies original non-safetensors files, including the original index, over the output. That overwrites the filtered index. Its names and layer bounds target R1, not V4. It must be corrected before actual export; it is not needed for the first masking experiment.

The [EASY-EP README](https://github.com/RUCAIBox/EASYEP) describes an 8×H200 conversion/collection setup and a vendored SGLang 0.4.3 workflow. Do not install that old environment over the working FreeToken environment. Current [FreeToken installation instructions](https://github.com/FlashML-org/FreeToken/blob/46d27439a4fccf457e1a9a858841168ec7a1c65d/docs/install.md) require Linux x86_64, driver r580+, and CUDA 13 for JIT builds. Its pinned [package definition](FreeToken/pyproject.toml) uses Torch 2.11, Triton 3.6, and Transformers 5.5+.

## Next experiment and acceptance criteria

1. Use the confirmed model ID `deepseek-ai/DeepSeek-V4-Flash-0731`. Record its resolved checkpoint revision and quantization at download, the pinned FreeToken commit, and the rental's actual resource allocation. The old launch command is unavailable and is not required to design the new experiment.
2. Establish an unmodified baseline on that checkpoint: short and representative long prompts, peak RAM/VRAM, prefill time, decode rate, and output correctness. No EasyEP inference has yet been run here.
3. Implement and check an all-ones mask: outputs and routing must agree with the baseline. Then verify that no excluded expert is selected, at least top-k candidates remain, and shared/hash behavior follows the chosen policy.
4. Validate output norms and aggregate scores against a small unfused reference using the same quantized expert arithmetic. Check chunked collection against unchunked collection on a small case, including token coverage.
5. Collect the selected domain's calibration data with all experts available. Compare retained sets, then evaluate several retention levels on held-out prompts before settling on 50%. For V4 this tests an extension of EASY-EP, not a reproduction of its R1 results.
6. Measure speed and memory for the masked model. Only after quality is acceptable, implement physical pruning and verify equivalence to the masked version.

## Delivered and verified locally

- `EASYEP/`: cloned from `https://github.com/RUCAIBox/EASYEP.git`, commit `04946c9562ddee772b3c3e1fb2c7552c9b604a25`.
- `FreeToken/`: cloned from `https://github.com/FlashML-org/FreeToken.git`, commit `46d27439a4fccf457e1a9a858841168ec7a1c65d`.
- EASYEP remains unmodified. FreeToken contains the experimental V4 adapter and tests; it has not been committed or pushed. No model weights were downloaded locally.
- [scripts/inspect_target.py](scripts/inspect_target.py): standard-library, read-only host/config inventory and mask structural validation. It does not load a model or prove inference feasibility.
- Public reference configs and file-size inventories are in `research/reference-configs/`. They are comparison data, not the user's checkpoint.
- Actual selected safetensors headers plus source inspection establish a static 8.6074 GiB GPU lower bound for official V4-Flash-0731 with the pinned FreeToken build. This is stronger than the initial host-RAM-only assessment, but is not a GPU execution result.
- [Local checks](research/checks/summary.json): the supplied R1 128-expert mask passes structural checks; V4 config parses; applying that R1 mask to V4 is rejected because it needs 43 rows instead of 58. No CUDA tests, quality evaluations, or performance benchmarks have been run.

Run the inspector on the actual rental after copying this script there:

```bash
python3 scripts/inspect_target.py --model-dir /path/to/your/checkpoint > target.json
```

Structural checking can additionally use `--mask /path/to/mask.json`. The inspector rejects changes to V4 hash layers. New mask envelopes also verify checkpoint configuration/index identity; old raw matrices receive structural checks only. Neither check establishes quality or execution correctness.
