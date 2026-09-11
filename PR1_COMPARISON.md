# ceenaa/EASYEP PR 1 compared with this workspace

Reviewed 2026-09-10. This is a focused source comparison of activation profiling, calibration data and evaluation safeguards, not a full audit or a GPU execution result.

Follow-up on the same date: the local V4 adapter now includes the residual-norm and neutral-filename fixes identified below. The supplied PrimeVul ZIP has been imported as 100 held-out cases. See [VALIDATION.md](VALIDATION.md) for the 55-pass/4-skip local result and [PRIMEVUL_EVALUATION.md](PRIMEVUL_EVALUATION.md) for the dataset's missing labels/provenance and execution instructions.

The live [GitHub PR API](https://api.github.com/repos/ceenaa/EASYEP/pulls/1) reported **35 commits and 16 changed files**, head `b6b5ab3b65d002a06e6142b4539ef514b88c0272`, base `04946c9562ddee772b3c3e1fb2c7552c9b604a25`. The cached commits page still displayed 21 commits; the API supplied the revision used for this review. Metadata, the file list, source snapshots and content hashes are saved in `research/ceenaa-pr1/`.

## Do we have its files?

**None of the 16 PR paths exist in the working `EASYEP/` checkout.** That checkout is the upstream base commit, not the author's PR branch. Reference copies were downloaded into `research/ceenaa-pr1/source/` for inspection; they are not merged into the engine and are not in the previously built deployment bundle.

We do have an independent activation observer in [FreeToken's pruning runtime](FreeToken/python/freetoken/pruning/runtime.py), with V4 gate, FFN and FP4 hooks. It records `activation_counts`, `gate_sums` and output-aware `scores` for each layer/expert. These are accumulated statistics saved to JSON, not a live graphical viewer or stored full per-token activation tensors. Raw expert-output tensors are observed transiently and discarded after aggregation.

## What are the cybersecurity files?

The PR's [run script](https://github.com/ceenaa/EASYEP/blob/b6b5ab3b65d002a06e6142b4539ef514b88c0272/v4/run_experiment.sbatch#L85) sets `N_CALIB=25`: that is a default number of source files to sample, not a bundled set of 25 security examples.

It expects externally supplied data:

- `vulnerable-js-files/` with `CODEQL_SECURE_MANIFEST.jsonl`: JavaScript/TypeScript source pairs based on CodeQL alerts and corresponding modified/rescanned versions.
- `primevul-c-files/` with `PRIMEVUL_PAIRED_MANIFEST.jsonl`: C/C++ pairs converted from externally obtained PrimeVul paired splits. The PR adds a [conversion/verification helper](https://github.com/ceenaa/EASYEP/blob/b6b5ab3b65d002a06e6142b4539ef514b88c0272/v4/primevul_dataset.py), not the dataset itself.
- `questions_used.json`: an external question-evaluation input.

The PR's `.gitignore` excludes its local input directory, downloaded PrimeVul JSONL and converted PrimeVul corpus. Those PR-specific data files were not supplied by fetching the PR. The user subsequently supplied an aligned PrimeVul ZIP, now imported into the local manifest format by `import-primevul`; this importer does not parse the PR-specific manifests or upstream PrimeVul JSONL. A fixed function is not necessarily free of all other vulnerabilities, so binary dataset labels need care when assessing false positives.

## Activation and evaluation comparison

| Capability | PR at the reviewed head | Local FreeToken adapter |
|---|---|---|
| Expert selection counts and gate-weight sums | Yes | Yes |
| Per-route output norms used in importance | Yes, recovered from the official weighted expert output | Yes, observes FreeToken's already weighted per-route FP4 output |
| Hash layers and shared experts preserved | Yes | Yes |
| Prompt plus generated review calibration | Exact generated token IDs replayed | Returned review text retokenized and replayed; replay token hashes are checked |
| V4 residual sensitivity | FP32 residual-space calculation | Uses existing HC kernel outputs, then FP32 cosine |
| Norm in V4's expanded residual space | Includes the token's `norm(post)` multiplier | Includes it in corrected metric `easyep_v4_hc_weighted_v2` |
| Alternative scoring rules and control masks | No-sensitivity, legacy reduced-space, gating-only, frequency and seeded random | One importance rule; raw counts/gate sums are available but controls are not implemented |
| Ranking diagnostics | Overlap, Jaccard, rank correlation, cutoff margins and never-active counts | Ranked expert IDs and score/count arrays |
| Score reduction reproducibility | Scoped deterministic CUDA reductions into FP64 accumulators | FP32 CUDA scatter-add followed by FP64 CPU totals; deterministic accumulation is not guaranteed |
| Security dataset tools | CodeQL pair loader and PrimeVul converter | Custom source manifest, aligned ZIP importer and repository/exact-content separation checks within a manifest |
| Label-neutral source paths | Yes | Neutral display paths; original mappings retained outside prompts |
| Blinded evaluation | Shuffled opaque variant labels with a separate key | Human grading explicitly shows baseline/candidate |
| Quality scoring | Binary paired discrimination and question judging | Human-matched finding recall, precision and clean-case false positives |
| Execution platform | Official converted checkpoint; supplied launcher requests 4 H100s and 450 GB RAM | FreeToken offloading; prepared profile targets one RTX 5090 and about 250 GB RAM |

Sources: PR [profiler and observers](https://github.com/ceenaa/EASYEP/blob/b6b5ab3b65d002a06e6142b4539ef514b88c0272/v4/easyep_v4.py#L175), [score accumulation](https://github.com/ceenaa/EASYEP/blob/b6b5ab3b65d002a06e6142b4539ef514b88c0272/v4/easyep_v4.py#L573), [neutral source names](https://github.com/ceenaa/EASYEP/blob/b6b5ab3b65d002a06e6142b4539ef514b88c0272/v4/easyep_v4.py#L1451), [blinded judging](https://github.com/ceenaa/EASYEP/blob/b6b5ab3b65d002a06e6142b4539ef514b88c0272/v4/easyep_v4.py#L3103), and local [workflow](FreeToken/python/freetoken/pruning/workflow.py).

## Findings and follow-up

1. **Residual expansion fixed.** The PR multiplies each weighted expert-output norm by `norm(post)` to express it in the same four-stream residual space used for cosine sensitivity. Algebraically, the norm of `post` outer-product `expert_output` is the product of their norms. Local metric `easyep_v4_hc_weighted_v2` now includes that factor, tested against the explicit outer product. V1 statistics and masks tagged with V1 are rejected. The existing HC kernel output precision and CUDA aggregation still differ from the PR, so this is not a claim of identical rankings.
2. **Filename leakage fixed.** Preparation now uses stable per-case names such as `review_target_001.cpp`, preserving real paths and hashes in metadata. Case/repository/pair IDs and labels are not inserted into prompts. The regression covers label-bearing names and multiple source files; legacy prepared artifacts must be regenerated. Source contents remain unchanged.
3. **We lack the PR's controls and blind judging.** Counts alone do not show that output-aware ranking improves on frequency, gate weights or random selection. These controls, cutoff diagnostics and blinded review would strengthen the experiment.
4. **Evaluation source is available; ground truth and calibration remain missing.** The user's ZIP contains 100 C/C++ functions in 50 pairs, all reserved for final evaluation. It has no labels or original repository/revision mapping. The importer leaves those unknown and records source hashes; separate calibration/validation data is still required.

The supplied PR runner creates the official model on CUDA and loads converted `model{rank}-mp{world_size}.safetensors` shards. It is not a FreeToken offload runner and cannot simply replace our launch command for the single 5090. Useful profiler and evaluation ideas need adaptation to the existing engine hooks.

The initial read-only review left the engine unchanged. This follow-up applies the two local fixes and imports the supplied evaluation archive, without merging or executing the external PR. The updated deployment bundle contains the local fixes and importer; source datasets remain separate. No rented GPU jobs or full-model evaluations have run.
