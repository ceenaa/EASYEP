# Local implementation and validation

Prepared on 2026-09-10 for `deepseek-ai/DeepSeek-V4-Flash-0731`, RTX 5090 / approximately 250 GB host RAM. No rented server was started and no checkpoint weights were downloaded locally.

## Implemented behavior

- `FreeToken/python/freetoken/pruning/`: versioned checkpoint-bound mask/statistics artifacts, online per-expert aggregation, source preparation, baseline reviews, calibration replay, ranked mask generation and human-graded evaluation.
- V4 gate: masks biased selection scores before top-k using negative infinity, preserves original unbiased gate weights and expert IDs, and leaves hash layers intact.
- V4 FFN/FP4 path: records the already weighted per-route outputs before summation; multiplies their norms by the per-token `norm(post)` residual expansion and routed-only hyper-connection sensitivity. Shared experts are excluded from scoring and preserved in execution.
- Prompt preparation uses neutral display paths and retains the original path mapping outside model input. Legacy prepared prompts are rejected. The PrimeVul importer preserves bytes, hashes and pair membership, leaves missing labels unknown and reserves every supplied case for the test split.
- Engine: collects only inside real eager prefill forwards, outside warm-up; rejects incompatible collection settings and checks contiguous token positions, complete layers and replay token hashes/counts.
- CLI: `ft serve --expert-mask` and `--expert-stats`; `python -m freetoken.pruning` exposes the workflow. `/v1/pruning/config` supplies checkpoint/mode metadata so clients can reject the wrong server or a restart during a run.
- `scripts/start_pruning_server.py`: consistent baseline/statistics/mask launch profiles, artifact validation, missing-shard checks and a dry-run option. The server binds to localhost.
- `scripts/build_pruning_bundle.py`: packages tracked engine source plus the adapter, tests, launcher and docs; excludes datasets, weights, environments and Git metadata.

The metric is explicitly `easyep_v4_hc_weighted_v2`, an EASYEP-style V4 extension. Fresh statistics are required; V1 statistics and old-metric masks are rejected. The HC sensitivity still uses the existing kernel's outputs cast to FP32, so the fix does not establish numerical identity with the external PR's FP32 implementation. It is not a claim that EASYEP's R1/V3 results transfer to V4. The first implementation performs masking; physical expert-bank/checkpoint compaction is not implemented.

## Checks actually run

Local environment: macOS, Python 3.12, Torch 2.11.0, NumPy 2.4.6, pytest 9.1.1. Only a lightweight local test environment was installed; the Linux CUDA engine dependencies were not installed on macOS.

```bash
PYTHONPATH=FreeToken/python .venv-pruning/bin/python -m pytest --confcutdir FreeToken/tests/pruning FreeToken/tests/pruning -q
```

Result after the V4 fixes and dataset import: **55 passed, 4 skipped**. The skipped tests require NVIDIA CUDA. `--confcutdir` avoids FreeToken's top-level test fixtures that import its CUDA stack; it does not bypass any pruning test.

Passing checks cover known-value score math (including avoiding a second gate multiplication), logical IDs before cache remapping, tensor-level chunk aggregation, all-ones masking and exclusion of negative biased scores, invalid masks, checkpoint/index changes, incomplete/contaminated replay, per-layer activation coverage, atomic batch commits, duplicate/omitted tokens, source confinement, cross-split contamination, label exclusion, mock review/replay clients, human evaluation metrics, comparison consistency and launch modes.

The two bug regressions failed before the fixes and passed afterward. Additional checks cover the explicit residual outer-product norm (including token-varying and zero post factors), old-metric rejection, neutral multi-file names and private path mapping, legacy prompt rejection, ZIP traversal/symlink/duplicate/size/encoding rejection, imported source integrity and held-out split protection. All 100 supplied source files passed manifest loading and byte-hash checks; they form 50 pairs with no identical complete files. Labels and original repository metadata are absent. Native prompt tokenization and model evaluation remain pending; see [PRIMEVUL_EVALUATION.md](PRIMEVUL_EVALUATION.md).

The pruning CLI and launch-script help commands were exercised. Changed/new Python files were syntax-compiled, and `git diff --check` passed. The deployment archive was inspected against its file manifest. These checks do not execute a model.

## Checks to run on the rented GPU

Run the same suite first. Its four CUDA tests should execute and pass:

1. FP4 short-prefill route observation preserves the original output and sees all weighted routes.
2. FP4 grouped-prefill route observation preserves the original output and sees all weighted routes.
3. Hyper-connection output agrees with a tensor reference using the actual mixing orientation.
4. The actual V4 gate has all-ones parity and never selects masked experts.

Then follow [RUN_ON_VAST.md](RUN_ON_VAST.md): full-model baseline, all-ones completion parity, small calibration replay with exact token coverage, ranked masks and held-out evaluation. Measure peak GPU/host memory and wall time during the pilot. Tensor-level aggregation was tested locally; equivalence of full-model inference across different prefill chunk sizes was not tested and may differ numerically because different FP4 kernel paths are used. Keep the chunk size fixed throughout the first comparison.

## Limits of the evidence

No full-model CUDA tests, quality evaluations, performance benchmarks or actual pruning runs have been performed. No speedup, memory saving, or retained cybersecurity quality is established. The checkpoint fingerprint hashes configuration/index files rather than every weight byte; keep the pinned checkpoint immutable and retain its download revision. The clients replay returned review text through the tokenizer, not original generation token IDs. Corpus checks catch repository and exact-file leakage, not all near duplicates. Human judgments and representative data are required to assess security-review quality.
