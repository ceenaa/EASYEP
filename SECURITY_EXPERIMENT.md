# Security-review expert masking experiment

Target: `deepseek-ai/DeepSeek-V4-Flash-0731`, RTX 5090 (32 GB VRAM), approximately 250 GB host RAM. Proposed engine: the local FreeToken checkout at `46d27439a4fccf457e1a9a858841168ec7a1c65d`. The local adapter and workflow are implemented and CPU-tested. Deployment and full-model CUDA validation remain pending. Use [RUN_ON_VAST.md](RUN_ON_VAST.md) for executable commands and [VALIDATION.md](VALIDATION.md) for the verification record.

## Objective and limits

Preserve the model's ability to review source code for security defects, explain the evidence, and recommend corrections while reducing its routed-expert pool. Optimize selection for this workload rather than overall benchmark averages.

Experts are not cleanly labeled by capability. Code comprehension, reasoning, and language experts can be necessary for a security review. An expert that never fires on a small sample may still matter for a rare defect. Consequently, activation frequency is a diagnostic, not sufficient justification to delete an expert. The experiment cannot guarantee that all security ability is preserved or that unrelated abilities disappear.

## Data to provide

The user-supplied PrimeVul ZIP is now imported as 100 final test cases (50 pairs); see [PRIMEVUL_EVALUATION.md](PRIMEVUL_EVALUATION.md). Its ground-truth labels and original repository metadata are missing. Separate calibration and validation cases are still required; these held-out cases must not be used to rank experts or select retention.

Supply local paths to repositories or source snapshots plus the languages that matter most. Include enough surrounding code to establish calls, data flow, trust boundaries, and authorization behavior. A vulnerable function without its callers or validation helpers can be misleading.

Prefer cases with a reviewed finding and a corresponding fix. Include clean cases and fixed versions to measure false positives. Cover several projects and defect families relevant to the deployment: for example, memory safety, authorization, injection prevention, path handling, unsafe deserialization, secret handling, and security logic. These are static code-review cases; the workflow does not execute submitted code or generate exploit payloads.

For each case, keep these fields in a JSONL manifest. Source paths refer to supplied files; they are not fetched automatically:

```json
{
  "id": "project-a-review-001",
  "repository": "project-a",
  "revision": "SOURCE_COMMIT",
  "language": "python",
  "files": ["src/component.py", "src/caller.py"],
  "label": "vulnerable",
  "expected_findings": [
    {
      "id": "finding-1",
      "category": "authorization",
      "location": "src/component.py:FUNCTION_OR_LINE",
      "explanation": "Reviewer-verified defect and relevant preconditions",
      "fix_reference": "FIX_COMMIT_OR_PATH"
    }
  ],
  "split": "calibration"
}
```

For clean/fixed cases, use `label: "clean"` and an empty findings list. Group a vulnerable version and its fix in the same split. Keep repositories, related revisions, and near-duplicate functions out of different splits. Calibration, retention-selection validation, and final held-out evaluation serve different purposes.

Start with a small pilot to validate instrumentation. Then increase calibration coverage until expert rankings and held-out results stabilize across subsets. The paper's 25 demonstrations per domain are a reference point, not evidence that 25 files cover this task.

## Review task and calibration

Use the same review instruction for baseline and every masked run:

> Review the supplied source code for security defects. For each supported finding, identify its location, relevant data flow or trust boundary, impact, and a suggested correction. Distinguish confirmed evidence from missing context. Do not invent a finding when the evidence is insufficient. Do not execute the source code.

Generate reviews with the full unmasked model and retain the exact prompt, response, tokenizer, and sampling settings. Score the prompt and review tokens together using consistent chat formatting. Keep held-out labels out of the prompts. Preserve correct reviews, false positives, misses, and failures in the evaluation record rather than selecting only favorable outputs.

Collect three per-layer/per-expert statistics on the calibration set: activation count, sum of routing weights, and EASY-EP-style output-aware importance. Rank and select with the output-aware score; use the first two statistics to diagnose coverage and compare simpler baselines.

For V4, the token-sensitivity component needs an explicit adaptation for hyper-connections. A candidate compares the residual-only hyper-connection output with the residual-plus-routed output, excluding the shared contribution. Validate this against a small reference implementation. Do not label a router-count mask as EASY-EP or claim the V4 adaptation reproduces the R1 paper.

## First masking experiment

- Keep all 256 experts in hash-routed layers 0–2 and all shared experts.
- Rank routed experts separately in learned-routing layers 3–42.
- Preserve six active experts per token and the original expert IDs, routing-weight normalization, and route scale.
- Exclude masked experts before top-k selection. Validate the complete mask and checkpoint provenance before loading it.
- Run an all-ones mask first and establish baseline equivalence before testing smaller masks.
- Evaluate the following candidate budgets. They are experiments, not promised quality levels.

| Retained experts per learned-routing layer | Total retained routed experts across 43 layers | Fraction retained |
|---:|---:|---:|
| 256 | 11,008 | 100.00% |
| 224 | 9,728 | 88.37% |
| 192 | 8,448 | 76.74% |
| 160 | 7,168 | 65.12% |
| 128 | 5,888 | 53.49% |

Masking retains the full checkpoint and host banks. It tests quality and routing behavior. Memory savings require physical pruning later; long-prefill transfers can remain unchanged until the loader/cache supports loading only retained experts. The retained fraction above is an expert-count ratio, not a total-model memory or speed ratio.

## Implemented FreeToken changes

1. **Mask configuration and router:** add validated checkpoint-bound mask loading and apply it in `models/deepseek_v4/moe.py`. Keep mask tensors out of checkpoint state and initialize them on the correct device. Preserve hash routing initially.
2. **Per-expert observations:** `moe/fused_ds_fp4.py` already materializes a per-route `down` tensor before summation. It contains gate-weighted expert outputs; their norms correspond to the output-aware term up to the existing finite-precision rounding. An observer here can avoid storing full hidden-state traces or modifying CUDA kernel arithmetic. Record logical IDs before cache-slot remapping.
3. **V4 token sensitivity:** instrument `models/deepseek_v4/model.py` around the FFN hyper-connection boundary. Retain the routed contribution separately from the shared expert for this calculation. The corrected `easyep_v4_hc_weighted_v2` multiplies weighted expert norms by `norm(post)` to express them in residual space; V1 statistics require recollection.
4. **Collection control:** distinguish real requests from warm-up and graph capture; ensure every intended token is counted once despite chunking and prefix reuse. For a first implementation, use a dedicated eager GPU-offload collection mode and reject unsupported CPU/hybrid collection explicitly. CPU/GPU serving masks can be evaluated separately once validated.
5. **Aggregation and provenance:** accumulate scores online and save sample IDs, token coverage, checkpoint revision, engine commit, collection mode, and metric definition with them. Produce ranked expert tables and mask files. The existing 58×256 EasyEP masks are incompatible with this 43-layer model.

These changes are implemented in the local FreeToken checkout. The CPU suite checks scoring, masks, provenance, token accounting, corpus preparation and workflow behavior. Four real-kernel tests are included for the rented GPU. CUDA/full-model correctness and quality are not established by the local checks.

## Evaluation and selection

Evaluate findings against reviewer-confirmed defects by location and cause. Measure recall, precision, false positives on clean/fixed cases, per-category recall, explanation/fix quality, and invalid or truncated responses. Inspect critical misses separately; an overall average can hide losses in an important defect family. Use the same prompts, token budget, and decoding settings for baseline and masked candidates.

Choose acceptable quality loss before selecting a mask. A conservative initial rule is no observed increase in critical misses, with separately agreed recall and false-positive tolerances; uncertainty still depends on the number and diversity of cases. Repeat stochastic evaluations as needed and report uncertainty. Select a retention level on the validation set, then evaluate it once on the held-out test set.

Measure peak RAM/VRAM, prompt-processing time, and decode throughput independently. At the quoted rental rate, compute cost is `0.80 * elapsed_hours`, with storage/network charges recorded separately. Collection and evaluation runtime cannot be estimated credibly until a representative pilot runs.

Only after a mask meets the quality criteria should an exporter remove weights and quantization scales. That exporter must preserve the chosen hash-layer policy, update routing/index/config/bank mappings, and show numerical equivalence to the masked version. No physical pruning should be inferred from merely writing a mask file.

## Current state

Both repositories are cloned; FreeToken contains the corrected experimental V4 scoring and masking integration. The launcher, calibration/review client, ranked-mask builder, paired human evaluation and deployment bundle are prepared for GPU validation. `scripts/inspect_target.py` accepts both old raw matrices and new checkpoint-bound mask envelopes. The supplied 100 PrimeVul functions are imported for final evaluation. Ground truth, separate calibration/validation data and the instance connection remain needed. No real calibration scores, full-model validation or physically pruned checkpoint have been produced.
