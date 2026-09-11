# FreeToken + EASYEP

Local source snapshots for the DeepSeek-V4-Flash-0731 expert-monitoring and pruning experiment.

## Layout

- [`FreeToken/`](FreeToken/): FreeToken with the local V4 integration, expert statistics collector, pruning workflow, and regression tests.
- [`EASYEP/`](EASYEP/): upstream EASYEP source, including its bundled examples and SGLang adaptations. This snapshot has no local source modifications; generated Python bytecode was omitted.
- [`scripts/`](scripts/): launch, evaluation, live expert smoke-test, analysis, and deployment helpers.
- [`deployment/`](deployment/) and [`setup_vast_5090.sh`](setup_vast_5090.sh): the existing Vast RTX 5090 recreation workflow.
- [`launcher/`](launcher/): Slurm launcher for an H100 40 GB slice and 250G host RAM, shared server launcher, checks, and [instructions](launcher/README.md).
- [`UPSTREAM.json`](UPSTREAM.json): source repositories and base revisions. Both projects are ordinary folders in this repository, not Git submodules.

## Current status

The V4 changes are in FreeToken. They include the V4 residual-space scoring correction, label-neutral review paths, routing and expert-output hooks, and opt-in tracing during prefill and generation. The analysis scripts distinguish source reading, generated reasoning, final answers, and repeated continuation tokens.

The previous RTX 5090 smoke test used the full unmasked model with CPU offload and native reasoning. It captured all 43 layers with six routed selections per token per layer. The GPU regression suite passed 74 tests, including five CUDA tests. The two-file run verified monitoring, not detection accuracy: the safe case was a false positive, and the vulnerable case's predicted CWE differed from the dataset. No expert mask was applied, and the full evaluation remains paused.

The external `ceenaa/EASYEP` V4 pull request was reviewed as a reference; it is not merged into `EASYEP/`. See the dated [comparison](PR1_COMPARISON.md). Older experiment notes describe earlier stages; this README reflects the verified smoke-test status.

## Inputs and deployment

Model weights, local environments, run artifacts, credentials, and the user-supplied PrimeVul dataset are not stored here. The prompt is [`v03_extended.txt`](v03_extended.txt). To use the smoke-test or Vast recreation scripts, supply these original inputs at the repository root:

```text
metadata_full.csv
primevul_aligned_100_samples 2.zip
```

The smoke-test runner additionally uses the saved two-case selection under `runs/easyep-smoke/input/`; restore that input folder from the original local workspace before running it.

See [REBUILD_5090.md](REBUILD_5090.md) for the existing Vast workflow. Its source transfer and dependency-resolution checks passed, but a complete fresh-instance rebuild has not yet been tested. Those scripts target a root-owned Ubuntu Vast instance. The separate [Rorqual launcher](launcher/README.md) uses modules and Slurm; full-model validation on that configuration is pending.

Upstream notices and licenses are preserved within each source tree.
