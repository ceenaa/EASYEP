# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Two independent codebases share this checkout:

1. **Upstream EASY-EP for DeepSeek-R1** (`pruning/`, `sglang/`, `evaluation/`, `configs/`,
   `expert_statistics/`, `dataset/`) — the reference implementation from
   [arXiv:2504.06792](https://arxiv.org/abs/2504.06792). Documented in `README.md`.
   Torch 2.4 / Transformers 4.46 / sglang 0.4.3 (`requirements.txt`).

2. **The V4-Flash adaptation** (`v4/`) — all active work, on branch `v4-flash-adaptation`.
   Adapts EASY-EP to DeepSeek-V4-Flash-0731 (284B MoE, 13B active) on one 4×H100 node
   under Slurm (Compute Canada / Rorqual). Torch 2.10 / Transformers 5.0 / tilelang
   (`v4/requirements-v4.txt`) — a **separate venv**; the two environments are mutually
   incompatible and must not be merged.

**Hard boundary: `v4/` never modifies `pruning/`, `sglang/`, `configs/` or `evaluation/`.**
The R1 pipeline is deliberately left byte-identical so the adaptation stays auditable
against it. `v4/` is self-contained: four Python files, four shell/sbatch launchers, one
config, two markdown docs.

Read `v4/README.md` before changing anything in `v4/`; it is the design record, and
`v4/COMPUTE_CANADA_RUNBOOK.md` is the operational one.

## Commands

### Tests (V4)

No GPU, no model, seconds. 63 `t_*` functions run by a hand-rolled runner in
`v4/test_easyep_v4.py` — **not pytest**, despite the stale `.pytest_cache/`.

```bash
"$EASYEP_VENV/bin/python" v4/test_easyep_v4.py
```

Add a test by defining a module-level `def t_<name>()`; `main()` picks up every `t_`-prefixed
global automatically. There is no name-filter flag, so run one test by calling it directly:

```bash
cd v4 && "$EASYEP_VENV/bin/python" -c \
  "import test_easyep_v4 as t; t.check('one', t.t_mask_hash_layers_untouched)"
```

The tests need `torch`, so they do not run on a bare macOS checkout — use the V4 venv.
`must_raise()` is the local stand-in for `pytest.raises`.

### Running the V4 experiment

The documented entrypoint is Slurm. `v4/easyep.sbatch` is a thin wrapper that `exec`s
`v4/run_experiment.sbatch`, which owns the whole gated workflow:

```text
parity (gate) -> validate (gate) -> profile -> question eval -> pair eval -> blind
```

```bash
# all paths must be absolute and exported; see v4/README.md for the full list
export EASYEP_REPO="$PWD" EASYEP_VENV=... EASYEP_CODE_DIR=... EASYEP_CHECKPOINT=... \
       EASYEP_DATA_ROOT=... EASYEP_RESULTS_ROOT=... EASYEP_CONFIG="$PWD/v4/config_v4_flash.json"

env -u RUN_ID -u RESUME sbatch --account=YOUR_ALLOCATION \
  --export=ALL,RUN_ID=dev_001,RESUME=0,N_CALIB=1,N_PAIRS=1,LIMIT=1 v4/easyep.sbatch
```

`N_CALIB=1 N_PAIRS=1 LIMIT=1` is the cheap end-to-end rehearsal — same gates, same artifact
contracts. Always clear inherited `RUN_ID`/`RESUME` with `env -u`; a stale `RESUME=1` has
already caused a job to attach to an incompatible manifest and die.

Inside an existing allocation (`v4/dev_hold.sbatch` + `srun --overlap --pty bash -l`), use
`bash v4/run_in_allocation.sh` — it stages and re-verifies the 159 GiB checkpoint under
`$SLURM_TMPDIR` before delegating to the same `run_experiment.sbatch`.

### Standalone V4 CLI

`v4/easyep_v4.py` has eight modes: `profile`, `mask`, `eval`, `pipeline`, `pairs`, `blind`,
`parity`, `validate`. All GPU modes take `--ckpt-path --config --code-dir --max-seq-len`
and run under `torchrun --nproc-per-node=4`. These exist for development; the Slurm
entrypoint is the reproducible path and always runs both gates. `blind` is CPU-only:

```bash
"$EASYEP_VENV/bin/python" v4/easyep_v4.py blind \
  --results "$RESULTS/run_ID/questions" --questions "$DATA/questions_used.json" \
  --out "$RESULTS/run_ID/judge/questions"
```

### R1 pipeline

See `README.md`. `torchrun --nproc_per_node=8 pruning/inf_new.py` →
`pruning/expert_selection.py` → mask JSON → either sglang gate-masking (copy
`sglang/sglang_full/sglang` over the installed package) or `pruning/model_prune.py` for
real weight deletion. `sglang_full` vs `sglang_pruned` differ only in
`srt/models/deepseek_v2.py` and `srt/layers/moe/topk.py`.

## V4 architecture

### The score

`score[layer, e] += weight_t[e] * simibr_mhc_t * residual_norm_t[e]`, accumulated **per
token** in float64. A product-sum cannot be recovered from marginals — an earlier variant
that summed `weight` and `norm` separately was wrong and is gone. Three scores plus two
control statistics come out of the *same* calibration pass so scoring rules can be compared
without extra calibration variance: `score` (primary, mHC-aware), `score_no_simibr`,
`score_reduced_legacy`, and `counts`/`gate_sums` for the frequency and gating baselines.
`score_mhc` is an alias of `score`, not a fourth rule.

### How instrumentation works

`build()` inserts `--code-dir` (the official HF snapshot's `inference/`) on `sys.path` and
imports the official `model` module; `patch()` then replaces `Gate.forward`, `MoE.forward`
and `Block.forward` with instrumented equivalents. **Weights and module structure are never
touched** — this is masking-only validation, no expert deletion. The `parity` gate exists
precisely to prove the patched-but-inactive forwards match the official ones within a
measured run-to-run noise floor.

Four V4 divergences from R1 drive the design (each detailed in `v4/README.md`):
`sqrtsoftplus` gating, `hc_mult=4` hyper-connection residuals (scoring must happen in the
`hc_post`-mapped residual space, not the reduced stream), token-ID hash routing in layers
0–2 (protected, never masked), and FP4 expert storage (why `validate` checks recovered
norms against explicit unweighted forwards). Layers 43–45 (MTP/DSpark) are excluded, not
handled.

### Provenance, which is most of the code

Three cooperating layers — treat these as load-bearing, not boilerplate:

- `v4/checkpoint_provenance.py` — `create` fully hashes the converted MP4 shards once and
  binds them to the pinned HF commit `9e165c30…`; `verify` re-checks shard metadata and the
  upstream inference sources cheaply on every job.
- `v4/run_provenance.py` — `init` / `preflight` / `mark-stage` / `verify-stage` / `finalize`.
  `RUN_MANIFEST.json` is written **once** and never rewritten; resume rebuilds a candidate
  in memory and demands exact equality. Mutable state lives in `RUN_STATUS.json`.
- Stage checkpoints under `run_<RUN_ID>/.stages/*.done` bind the manifest hash, the exact
  command, upstream artifacts and complete output trees. A resume rehashes all of them.

Everything downstream **fails closed**. Artifacts are rejected — not warned about — when
`score_schema_version != 2`, when `accumulation_mode` is absent, when the model identity
digest differs, when the calibration distribution (`source_kind`, `security_prompt`,
`thinking_mode`) does not match what the consumer needs, or when a stage's inputs/outputs
have changed. If you change a score's meaning, bump `SCORE_SCHEMA_VERSION`/`SCORE_SEMANTICS`
rather than making an old artifact silently loadable.

### Constants in `v4/easyep_v4.py` that will bite

- `MAX_SUPPORTED_SEQ_LEN = 24576` — prefill memory is quadratic and unblocked (~`16*T²`
  bytes) against ~35 GiB headroom. 65536 OOMs on 4×H100. Enforced in `build()` *and* the
  launcher, because standalone invocations bypass the launcher.
- `THINKING_MODE = "thinking"` — the single definition every prompt encodes through.
  `encoding_dsv4.render_message` asserts `["chat", "thinking"]`; `"reasoning"` fails on the
  first encode. Artifacts profiled under another mode are refused, not reused.
- `ACCUMULATION_MODE = "torch-deterministic-index-add-v1"` — deterministic `index_add` is
  scoped to the duplicate-index reductions only, so it does not perturb the official
  FP4/TileLang kernels, and the process-global setting is restored afterward.
- `ALLOWED_MISSING` — only `mtp.N.embed.*` / `mtp.N.head.*`. Every other missing or
  unexpected checkpoint key aborts the run.

### Evaluation design

Seven variants (`full`, `pruned_paper`, `pruned_reduced_legacy`, `pruned_no_simibr`,
`pruned_gating`, `pruned_frequency`, `pruned_random`) decoded under identical per-item
seeds, greedy by default (`TEMPERATURE=0`). The three controls are the point: gating,
frequency and random tell you whether the scoring carries signal at all.

There is **no built-in question-quality metric** by design — runs generate completions, and
`blind` produces an anonymised bundle plus a mode-0600 withheld key (labels drawn from the
system CSPRNG, public UIDs HMAC-salted) for an external judge. The one precomputed result
is matched-pair discrimination against CodeQL ground truth, parsed from the answer only,
because reasoning traces can contain a discarded intermediate verdict.

Calibration and matched-pair sets are disjoint, both sides use the same outcome-neutral
`VULNERABLE or SAFE` prompt with a neutralised display path, and no evaluation prompt ever
carries the reference answer. Tests exist for each of these; keep them passing.

## Working conventions

- The launcher refuses to start a new run from a checkout with tracked or staged changes.
  **Commit intentional changes before submitting**; never bypass that check.
- `deepseek_easy_ep_inputs/` is the local experiment corpus and is gitignored. Its content
  digest is recorded in the run manifest; the data itself must not be committed.
- Never use `RESUME=1` to change temperature, sample counts, token limits, `KEEP`, code or
  data — create a new `RUN_ID`.
- Comments here explain *why* a constraint exists (an OOM measured, a determinism claim, an
  attack on the blinding). Match that: when you tighten or relax a guard, say what it rules
  out.
- A one-sample rehearsal is not an accuracy experiment. Report calibration count, question
  count, pair count, seed, temperature, token limits, git revision and result path with any
  number.
