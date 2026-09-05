# EASY-EP for DeepSeek-V4-Flash

Adaptation of EASY-EP ([arXiv:2504.06792](https://arxiv.org/abs/2504.06792)) from
DeepSeek-R1 to **DeepSeek-V4-Flash-0731** (284B MoE, 13B active), running on a
single 4×H100 node.

This is an *adaptation*, not a port: V4-Flash differs from R1 in four ways that
each touch the scoring path. Nothing under `pruning/`, `sglang/` or `configs/` is
modified — the R1 pipeline is left intact and all V4 work is confined to `v4/`.

## Reproducible setup

The V4 path does **not** use the R1 environment documented in the repository-level
README. In particular, Torch 2.4 and Transformers 4.x cannot load the official
V4 FP4 inference implementation. The pins in `v4/requirements-v4.txt` satisfy
the requirements shipped with the official model snapshot.

The reference implementation and configuration are pinned to
[`deepseek-ai/DeepSeek-V4-Flash-0731` revision
`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/9e165c30e2704aec5d9d593cce3eebd58bbef1cb).
`v4/config_v4_flash.json` is the checked-in copy of that revision's
`inference/config.json`; the launchers use the checked-in copy.

On Compute Canada, create a separate virtual environment with an available
Python 3.10+ module and CUDA 13.2. Set `PYTHON_MODULE` to the exact versioned
module at your site when possible, and reuse the same value for submission:

```bash
export PYTHON_MODULE=python
export CUDA_MODULE=cuda/13.2
module load "$PYTHON_MODULE" "$CUDA_MODULE"
python -m venv "$PROJECT/easyep-v4-venv"
"$PROJECT/easyep-v4-venv/bin/python" -m pip install --upgrade pip
"$PROJECT/easyep-v4-venv/bin/python" -m pip install -r v4/requirements-v4.txt
```

Download the pinned Hugging Face snapshot and convert it to four model-parallel
shards with the conversion script from the same snapshot:

```bash
export MODEL_REVISION=9e165c30e2704aec5d9d593cce3eebd58bbef1cb
export HF_MODEL="$PROJECT/DeepSeek-V4-Flash-0731-hf"
export EASYEP_CHECKPOINT="$PROJECT/DeepSeek-V4-Flash-0731-mp4"

hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision "$MODEL_REVISION" --local-dir "$HF_MODEL"
"$PROJECT/easyep-v4-venv/bin/python" "$HF_MODEL/inference/convert.py" \
  --hf-ckpt-path "$HF_MODEL" \
  --save-path "$EASYEP_CHECKPOINT" \
  --n-experts 256 --model-parallel 4 --expert-dtype fp4

# Run once after conversion. This fully hashes the large converted shards and
# verifies that the Hugging Face local-dir metadata names the pinned commit.
"$PROJECT/easyep-v4-venv/bin/python" v4/checkpoint_provenance.py create \
  --checkpoint "$EASYEP_CHECKPOINT" --source-snapshot "$HF_MODEL" \
  --model-id deepseek-ai/DeepSeek-V4-Flash-0731 \
  --model-revision "$MODEL_REVISION" --inference-revision "$MODEL_REVISION" \
  --nproc 4
```

Create the provenance record in a compute allocation with enough wall time for
one complete read of the converted checkpoint. Normal experiment jobs validate
the record and exact shard metadata without rehashing hundreds of GiB. For a
fresh byte-for-byte audit (which deliberately tolerates harmless mtime changes),
run:

```bash
"$PROJECT/easyep-v4-venv/bin/python" v4/checkpoint_provenance.py verify \
  --checkpoint "$EASYEP_CHECKPOINT" --full \
  --config v4/config_v4_flash.json --source-snapshot "$HF_MODEL" \
  --model-id deepseek-ai/DeepSeek-V4-Flash-0731 \
  --model-revision "$MODEL_REVISION" --inference-revision "$MODEL_REVISION" \
  --nproc 4
```

After `create` succeeds, the original Hugging Face weight shards may be removed
to save storage. Keep the snapshot's `inference/` and `encoding/` directories
and their `.cache/huggingface/download` metadata because every job revalidates
the exact upstream sources it executes.

The experiment data is prepared separately. Its root must have this layout:

```text
inputs/
├── questions_used.json
└── vulnerable-js-files/
    ├── CODEQL_SECURE_MANIFEST.jsonl
    └── ... source files named by the manifest ...
```

The run manifest hashes every file in `vulnerable-js-files/`, so a run remains
auditable even when the dataset is stored outside Git.

## Scope of this first version

| | |
|---|---|
| Layers 0–2 | **unchanged**, all 256 experts kept |
| Layers 3–42 | pruned 256 → 128 experts |
| MTP / DSpark layers 43–45 | **excluded** from profiling and masking (see *Open items*) |
| Validation | **masking only** — expert weights are never modified |

## The score

The paper's score is a sum of per-token **products**
(`pruning/expert_selection.py:38` upstream). V4's residual state has four mHC
copies, so the architecture-faithful adaptation measures both the token
contribution and expert-output norm in that residual space:

```python
score[layer, e] += weight_t[e] * simibr_mhc_t * residual_norm_t[e]
```

| term | meaning |
|---|---|
| `weight` | gating score for expert `e` on token `t` |
| `residual_norm` | ‖unweighted expert output‖₂ after `hc_post` maps it into the four-copy residual |
| `simibr_mhc` | `max(1 − cos(h_mhc, h_mhc + y_routed_mhc), 0)` — token contribution in the actual V4 residual state |

Implemented in `easyep_v4.py`:

- `moe_forward` — collects per-token `weights`, `indices`, and unweighted `norms`
- `moe_accumulate` — forms the product and adds it to the running score
- `block_forward` — supplies the `hc_pre`/`hc_post` coefficients needed to map
  the routed contribution back into mHC residual space

The accumulation must happen per token. A previous attempt accumulated `Σweight`
and `Σnorm` into separate marginal buffers and ranked by the latter; a product-sum
cannot be recovered from marginals, and that variant drops `simibr` entirely.
For comparison, `score_no_simibr` (= `Σ weight·residual_norm`) and the old
reduced-vector approximation (`score_reduced_legacy`) are recorded in the **same
calibration pass**. This isolates the scoring definition without adding
calibration variance. `score_mhc` is a compatibility alias of the primary
schema-v2 `score`, not a separate score.

Calibration follows the paper's prompt-plus-response trajectory rather than
profiling prompt prefills alone. For each seeded, CWE-stratified source file,
the implementation first generates a response with profiling disabled, then
teacher-forces the exact prompt and generated response once with profiling
enabled. Token-exact chunking preserves the whole source; `MAX_CHUNKS=0` is
unlimited, while exceeding a positive cap fails instead of silently dropping a
file tail. A source that does not fit therefore becomes multiple independent
prompt-plus-response trajectories: `N_CALIB=25` means 25 source files, not
necessarily 25 forwards. The score artifact records the seed, decoding limits,
exact selected source hashes, prompt-token hashes, generated-response token
hashes, and the resulting file, chunk, forward, and generated-token counts.
Later stages reject the artifact unless all schema-v2, model, implementation,
and calibration provenance checks pass.

All floating-point expert accumulators use PyTorch's strict deterministic
CUDA `index_add` path. The setting is scoped to the duplicate-index reductions
so it does not alter the official FP4/TileLang forward kernels, and the previous
process-global determinism setting is restored afterward. The score artifact
records `accumulation_mode=torch-deterministic-index-add-v1`; artifacts without
that attestation are rejected and must be re-profiled.

`score_comparison.json` reports, for every layer 3–42:

| field | meaning |
|---|---|
| `overlap` | how many of the top-128 experts the two rules agree on |
| `jaccard` | overlap over union |
| `spearman_full_ranking` | rank correlation over all 256 experts |
| `only_in_paper_score` / `only_in_no_simibr` | the experts that actually differ |
| `n_never_activated` | experts with zero calibration traffic in that layer |
| `frequency_control_cutoff` | the frequency control's cutoff margins, and how many experts tie exactly on its boundary |
| `paper_cutoff` / `no_simibr_cutoff` | rank-128, rank-129, absolute-margin, and scale-normalized relative-margin diagnostics |

plus `overlap_mean`, `overlap_min`, `overlap_max` and `min_overlap_layer` so the
layers where the token-contribution term matters most are immediately visible.

## The four V4 divergences

### 1. `sqrtsoftplus` gate

R1 uses `sigmoid`; V4 uses `F.softplus(scores).sqrt()` with a `noaux_tc` bias that
shifts top-k selection but not the returned routing weights. `gate_forward` mirrors
upstream V4 exactly and adds only an optional keep-mask applied to the scores
before `topk`. The returned `weights` are unchanged in meaning, so the score's
multiplicative gate-weight term carries over.

### 2. mHC residuals

V4 maintains `hc_mult = 4` copies of the hidden state. `Block.hc_pre` reduces the
four copies to a single vector before the FFN, but `hc_post` distributes the FFN
output back into the four-copy residual using token-dependent coefficients:

```python
residual = x                              # [b,s,hc,d]
x, post, comb = self.hc_pre(x, ...)       # [b,s,d]   <- h, the analogue of x_before_moe
x = self.ffn_norm(x)
x = self.ffn(x, input_ids)                # [b,s,d]
x = self.hc_post(x, residual, post, comb) # [b,s,hc,d]
```

The expert output is computed in the reduced stream, but its contribution to the
model state is the `hc_post`-mapped vector. The primary score therefore computes
`simibr` and the corresponding output norm after that mapping. The literal
reduced-stream `h -> h + y_routed` score remains available only as
`score_reduced_legacy` for comparison.

### 3. Hash-routed layers 0–2

`Gate` routes by **token ID** (`tid2eid`) for `layer_id < n_hash_layers`, not by
content. An expert pruned there is unreachable for the specific vocabulary items
mapped to it, so these layers are excluded from both profiling and masking and
keep all 256 experts. (Upstream also skips its first 3 layers, but because they
are dense MLPs — different reason, same range.)

### 4. FP4 expert format

Experts are stored `float4_e2m1fn_x2` — two values per byte, with a per-32
`float8_e8m0fnu` block scale. This does **not** affect masking, which is why this
version validates by masking first. Packing is internal to each expert tensor, so
removing a whole expert does not require bit-level repacking across expert
boundaries. Actual deletion still requires selecting and renumbering expert
tensors, re-sharding the checkpoint, and updating router/config mappings; that is
not implemented here.

Two consequences already handled:

- V4 folds the gating weight *inside* `Expert.forward`, computing
  `w2(w · silu(gate)·up)`. In exact arithmetic, linearity would make this
  `w × unweighted`, permitting norm recovery by division. FP4 activation
  quantisation can break exact proportionality, so the mandatory `validate`
  gate checks the shortcut against explicit unweighted forwards.
- V4's `MoE.forward` returns routed + shared summed. `simibr` needs the routed
  part alone, so it is stashed on the module for `Block` to read.

## Running it

From the repository root, export absolute site-specific paths. `EASYEP_CODE_DIR` must
point to the `inference/` directory from the pinned Hugging Face snapshot, while
`EASYEP_CHECKPOINT` must point to its four-way converted checkpoint:

```bash
export EASYEP_REPO="$PWD"
export EASYEP_VENV="$PROJECT/easyep-v4-venv"
export HF_MODEL="$PROJECT/DeepSeek-V4-Flash-0731-hf"
export EASYEP_CODE_DIR="$HF_MODEL/inference"
export EASYEP_CHECKPOINT="$PROJECT/DeepSeek-V4-Flash-0731-mp4"
export EASYEP_DATA_ROOT="$SCRATCH/deepseek_easy_ep/inputs"
export EASYEP_RESULTS_ROOT="$SCRATCH/deepseek_easy_ep/results"
export MODEL_ID=deepseek-ai/DeepSeek-V4-Flash-0731
export MODEL_REVISION=9e165c30e2704aec5d9d593cce3eebd58bbef1cb
export PYTHON_MODULE=python       # replace with the exact setup module when available
export CUDA_MODULE=cuda/13.2

# Replace YOUR_ALLOCATION with the Compute Canada allocation for this run.
sbatch --account=YOUR_ALLOCATION --export=ALL v4/easyep.sbatch
```

`easyep.sbatch` is a compatibility wrapper around `run_experiment.sbatch`; both
execute the same complete workflow:

```text
parity (gate) -> validate (gate) -> profile -> question eval -> pair eval -> blind
```

Each GPU stage uses a fresh `torchrun` process. The profile is performed once and
the resulting `expert_scores.pt` is reused for both evaluation suites. Either gate
failing stops the job before profiling or evaluation. Slurm writes
`easyep-easyep_v4-JOB_ID.out` in the submission directory unless `sbatch --output`
overrides it.

The main environment overrides are:

| variable | default | meaning |
|---|---:|---|
| `KEEP` | `128` | experts retained in each prunable layer |
| `N_CALIB` | `25` | calibration source files; token-exact chunking may produce more profiling trajectories |
| `N_PAIRS` | `25` | matched vulnerable/secure pairs |
| `MAX_SEQ_LEN` | `16384` | context window used by each stage. Not the config's `original_seq_len=65536`: prefill memory is quadratic and unblocked (~`16*T^2` bytes) against ~35 GiB of headroom, so 65536 OOMs and 24576 is the ceiling, enforced by both the launcher and `build()`. Token-exact chunking preserves whole sources, so a smaller window costs no coverage |
| `MAX_NEW_TOKENS` | `2048` | question-evaluation response limit. Sized for reasoning mode, where the chain of thought precedes the answer; too small truncates before the answer is reached |
| `PAIR_MAX_NEW_TOKENS` | `1024` | matched-pair response limit, same reasoning-mode sizing |
| `MAX_CHUNKS` | `0` | calibration chunks per file; `0` means unlimited and a positive value is a fail-fast cap |
| `LIMIT` | `0` | questions evaluated per variant; `0` is all of them. `N_CALIB` and `N_PAIRS` shrink the other stages, so this is the knob for a cheap end-to-end rehearsal |
| `SEED` | `965` | calibration ordering, controls, and paired decoding seed |
| `TEMPERATURE` | `0` | greedy decoding for deterministic full-vs-pruned comparisons; override with a positive value only for an explicitly repeated-seed robustness study |
| `RUN_ID` | Slurm job id | suffix of the unique output directory |
| `RESUME` | `0` | `1` reattaches only when the complete immutable provenance and every skipped stage checkpoint still match |
| `STRICT_PINS` | `0` | `1` aborts when the active environment does not match `requirements-v4.txt` exactly; otherwise divergences are logged |
| `MASTER_PORT_BASE` | job-derived | first of the per-stage rendezvous ports |
| `PYTHON_MODULE` | `python` | module used to create and run the V4 virtual environment |
| `CUDA_MODULE` | `cuda/13.2` | CUDA module loaded in the job |

### Fast development loop

Use a small, complete batch rehearsal before a full experiment. It follows the
same gates and artifact contracts while evaluating one calibration source, one
matched pair, and one question per variant:

```bash
env -u RUN_ID -u RESUME sbatch --account=YOUR_ALLOCATION --time=03:00:00 \
  --export=ALL,RUN_ID=dev_001,RESUME=0,TEMPERATURE=0,N_CALIB=1,N_PAIRS=1,LIMIT=1 \
  v4/easyep.sbatch
```

Always set both `RUN_ID` and `RESUME` on submission. Clearing inherited values
first prevents stale shell state from silently selecting an old run.

When several hands-on iterations are planned for the same work block, one
short-lived allocation avoids returning to the queue between attempts:

```bash
sbatch --account=YOUR_ALLOCATION v4/dev_hold.sbatch
squeue -u "$USER"                         # wait for easyep_dev to run
srun --jobid=JOB_ID --overlap --pty bash -l
```

`EASYEP_CODE_DIR` must name the attested Hugging Face snapshot used to create
the checkpoint, including its `.cache/huggingface/download/*.metadata` files. A
copied inference directory without that revision evidence is intentionally
rejected even when its visible Python files match.

Inside the allocation, export every site-specific path plus an explicit new
`RUN_ID` and `RESUME=0`. The helper stages the checkpoint once under
`$SLURM_TMPDIR`, verifies the staged copy against its provenance sidecar, and
then launches the normal reproducible workflow:

```bash
cd /path/to/EASYEP
export EASYEP_REPO="$PWD"
export EASYEP_VENV=/absolute/path/to/venv
export EASYEP_CODE_DIR=/absolute/path/to/attested-snapshot/inference
export EASYEP_CHECKPOINT=/absolute/path/to/mp4-checkpoint
export EASYEP_DATA_ROOT=/absolute/path/to/inputs
export EASYEP_RESULTS_ROOT=/absolute/path/to/results
export EASYEP_CONFIG="$PWD/v4/config_v4_flash.json"
export RUN_ID=dev_002 RESUME=0
export N_CALIB=1 N_PAIRS=1 LIMIT=1
export TEMPERATURE=0

bash v4/run_in_allocation.sh
```

The allocation survives an SSH disconnect; a command tied to the attached PTY
may not. Use a normal `sbatch` submission for any unattended run. Cancel the
development allocation when finished:

```bash
scancel JOB_ID
```

Holding the node removes repeated queue waits, but it does not keep model state
warm: every GPU stage below is still a separate `torchrun` process. Node-local
staging makes those reloads substantially cheaper; eliminating them entirely
requires a persistent multi-stage Python worker and is a separate change to the
execution and checkpoint model.

For the complete Rorqual workflow, site-specific paths, monitoring commands,
resume rules, and failure recovery, see
[`COMPUTE_CANADA_RUNBOOK.md`](COMPUTE_CANADA_RUNBOOK.md).

A stage that exits zero atomically records a JSON checkpoint under
`run_RUN_ID/.stages/`. Each checkpoint binds the immutable run-provenance hash,
exact stage command, direct upstream artifacts, and complete output trees. A
resume rehashes all of them before skipping; an empty legacy marker, missing or
modified artifact, changed command, or changed upstream stage aborts instead of
splicing results. A per-`RUN_ID` lock rejects overlapping submissions.

The first launch publishes the immutable `RUN_MANIFEST.json` once.
On resume, the launcher builds a candidate in memory and requires exact equality
across source, model, environment, parameters, and input data. The original
manifest is never rewritten; attempt and completion data live in
`RUN_STATUS.json`. Schema-v2 manifests and empty legacy markers are
intentionally not resumable; use a new `RUN_ID`. Without `RESUME=1` the launcher
refuses to overwrite an existing output directory and, for a Git checkout,
refuses tracked or staged changes. The compact provenance record content-hashes
the executing EASY-EP files, official inference/encoding trees, configuration,
tokenizers, questions, pair manifest, and complete input data tree. The required
checkpoint provenance sidecar records full SHA-256 hashes for
every converted shard, the pinned commit recorded by the Hugging Face local-dir
metadata, and content-derived identifiers for the critical upstream source
files. Each job verifies its model/revision declarations, exact shard metadata,
and the currently executed inference sources against that record, then records
the checkpoint's portable content identity in `RUN_MANIFEST.json`. This avoids
rereading the entire checkpoint on every launch without reducing provenance to
a path or timestamp claim. Immediately before every model process, the launcher
rechecks the sidecar, runtime-file hashes, official source trees, and input-tree
hashes against the run manifest; any mid-job change aborts the workflow.

Outputs are grouped under `$EASYEP_RESULTS_ROOT/run_RUN_ID/`:

| path | contents |
|---|---|
| `RUN_MANIFEST.json` | immutable provenance and its canonical hash; written exactly once |
| `RUN_STATUS.json` | mutable attempt history, exit status, completed stages, and artifact paths |
| `parity/parity.json` | instrumentation parity gate report |
| `norm_validation/norm_validation.json` | recovered-vs-explicit norm gate report |
| `norm_validation/validation_scores.pt` | validation score tensors |
| `scores/expert_scores.pt` | all score tensors, counts, gate sums, and score-schema metadata |
| `scores/mask_keep{KEEP}.json` | primary mask from profiling |
| `scores/mask_keep{KEEP}_no_simibr.json` | no-`simibr` mask from profiling |
| `scores/mask_pruned_*.json` | every evaluated scored/control mask, including gating-score, frequency, and random |
| `scores/score_comparison.json` | per-layer overlap and cutoff margins for primary and no-`simibr` rankings |
| `scores/mask_*.json` | the mask actually used for each evaluated variant |
| `questions/answers_*.jsonl` | question completions for every variant, including the exact prompt, snippet, references, decoding seed/settings, token counts, and prompt hashes |
| `questions/mask_*.json` | masks rebuilt from the saved score artifact |
| `questions/summary.json` | question-evaluation summary and decoding settings |
| `pairs/pairs_*.jsonl` | matched-pair completions for every variant, including the exact neutral display path, source snippet, review prompt, and their hashes |
| `pairs/pairs_used.json` | exact selected pair identities, paths, labels, source/prompt text and hashes, prompt-token hashes/counts, decoding settings, and score/calibration provenance |
| `pairs/pairs_summary.json` | coverage-aware discrimination metrics, decoding settings, and score provenance |
| `judge/{questions,pairs}/` | anonymised judging bundle and separate withheld key |

Standalone modes remain available for development, but the documented Slurm
entrypoint always runs both correctness gates and the full workflow. `eval`
delegates to the pipeline's own evaluation routine, so it shares the prompt
construction, per-question seeding and row schema rather than keeping a second
copy that can drift; its `--out` is a directory and it writes the same
`answers_<tag>.jsonl` plus `summary.json` layout.

## Evaluated variants

All variants are decoded under identical per-item seeds:

| tag | mask |
|---|---|
| `full` | none - all 256 experts |
| `pruned_paper` | top-`KEEP` by the mHC-aware V4 adaptation of `weight x simibr x norm` |
| `pruned_reduced_legacy` | top-`KEEP` by the old reduced-vector approximation, when present in the score artifact |
| `pruned_no_simibr` | top-`KEEP` by `weight x residual_norm` |
| `pruned_gating` | top-`KEEP` by total activated gate weight - the paper's gating-score baseline |
| `pruned_frequency` | top-`KEEP` by selection count - the naive heuristic |
| `pruned_random` | seeded random `KEEP` - the floor |

These are the paper's three pruning controls. `pruned_gating` tests whether output
norms and token contribution improve on router weights alone. `pruned_frequency`
asks whether the paper's machinery beats the obvious "keep what gets picked most".
`pruned_random` asks whether the scoring carries any signal at all -- if it ties
the scored masks, the model is simply robust to expert removal and no scoring rule
is doing work.
Schema-v1 overlap numbers are not comparable to the mHC-aware primary score and
must be regenerated from a schema-v2 run.

Calibration uses a near-equal, seeded sample of manifest-labelled vulnerable and
secure files, never both sides of one matched pair. Both labels receive the same
outcome-neutral `VULNERABLE or SAFE` prompt and a neutral display path, so CWE,
GHSA, `Unsafe*`, and `_Code` names cannot steer routing. The prompt requires
concrete evidence without assuming either outcome and treats delimited source as
data rather than instructions. Calibration and matched-pair evaluation share
this exact prompt because generated calibration responses are included in the
profiled trajectory.

Matched-pair selection excludes every source file selected for calibration, so
the discrimination evaluation cannot reuse a profiled program. The selected
paths and their content/token identities are persisted in `pairs_used.json`.
That exclusion is only as good as the paths the score artifact recorded, so the
score schema also records *which* calibration distribution produced it
(`source_kind`, `security_prompt`, `thinking_mode`). `pairs` refuses an artifact profiled from
sample texts, whose synthetic names match nothing in the corpus and would make
the exclusion silently vacuous, or one profiled without the security-review
prompt, whose routing statistics come from a different distribution. The same
applies across thinking modes: every prompt this pipeline encodes goes through
the single `THINKING_MODE` constant (currently `thinking`), and an artifact
profiled under a different mode is refused rather than silently reused, because
the mode changes what the model emits and therefore which experts route.

Because reasoning-mode completions carry the chain of thought ahead of the
answer, the matched-pair eval parses its verdict from the answer alone. A trace
that considers `Verdict: SAFE` before settling on `Verdict: VULNERABLE` would
otherwise read as two conflicting verdicts and score as an abstention. The full
completion is still stored and still goes to the judge; `answer` is recorded
beside it.

`score_comparison.json` shows where the primary and no-`simibr` rules disagree;
running both measures whether the token-contribution term helps. The standalone
`pipeline` and `pairs` modes accept `--no-controls` to omit the gating-score,
frequency, and random controls.

## Throughput

`generate()` issues one prefill forward (`start_pos == 0`, T = prompt length) and
then one forward per decoded token (T == 1). `completion_tokens / seconds` over
the whole call is therefore neither rate, and the first call additionally pays
TileLang kernel compilation. `ForwardTimer` times each forward and reports them
apart, so the headline number is the **warmed decode rate**:

| field | meaning |
|---|---|
| `decode_tokens_per_second` | warmed decode rate - the number to report |
| `decode_ms_per_token_median` | per-step latency, robust to a stalled step |
| `prefill_tokens_per_second` | prefill rate, measured separately |
| `cold_first_prefill_seconds` / `cold_first_decode_seconds` | the excluded compilation cost, kept visible |
| `warmed` | **false** means the run was too small to drop the warm-up, so the numbers still carry compilation cost |

`WARMUP_ITEMS = 1` whole generate() call and `WARMUP_DECODE_STEPS = 2` steps per
item are excluded. The exclusions are identical across variants, so their rates
are comparable; a run smaller than the budget relaxes them and sets
`warmed: false` rather than reporting a cold number as warm. Every
`answers_*.jsonl` / `pairs_*.jsonl` row carries its own `decode` block, and each
variant's `summary.json` entry carries the aggregate under `throughput`.

For runs recorded before this instrumentation, `analyze_throughput.py` estimates
the split by least squares over the rows:

```bash
python v4/analyze_throughput.py --run "$EASYEP_RESULTS_ROOT/run_ID"
```

It reports the measured block directly when present and only falls back to the
fit otherwise. The fit is unreliable when prompt and completion lengths are
correlated, and it says so.

Measured on the September rehearsal: decode is **~3.8 tok/s** and prefill
~244 tok/s, so prefill was ~2s of a ~35s call and the low rate is entirely in
the decode loop. The reference `inference/generate.py` decodes one token per
forward at batch 1, and upstream `MoE.forward` forces a device sync per layer
(`bincount(...).tolist()`) plus an all-reduce per layer, so a decoded token
costs ~43 syncs and ~86 collectives. Masking does not change this: a pruned
variant still activates the same top-`k` experts, which is why the pruned
variants run only ~5-8% faster than `full`. Raising this materially means
evaluating under an optimised runtime, the way the R1 pipeline uses sglang.

## Tests

```bash
"$EASYEP_VENV/bin/python" v4/test_easyep_v4.py  # 62 tests, seconds, no GPU
```

Covers the parts a reviewer would otherwise have to check by reading: mask
construction and partitioning, hash-layer protection, the gating-score,
frequency, and random baselines, schema rejection, deterministic score-reduction
state restoration, cutoff-margin reporting, balanced seeded calibration sampling,
token-exact chunking,
prompt-plus-response profiling, prefill/decode timing separation and its
warm-up exclusion, the discrimination metric (including that a
constant "VULNERABLE" answerer scores J=0), label-neutral paired inputs,
calibration/matched-pair disjointness, neutral prompt/path construction,
path-traversal rejection, immutable resume comparison, content-bound stage
checkpoint tamper detection, portable checkpoint provenance and full-hash
verification, parity-threshold validation,
blinding (including that the matched-pair label is withheld), that no evaluation
prompt carries the reference answer or grading rubric, the context-window
ceiling, the checkpoint allowlist, and the inlined `hc_post` algebra used by
the primary mHC-aware score.

## Judging

The question runs generate; they do not grade. There is no built-in question-quality metric --
`answers_*.jsonl` holds the question, the snippet, the reference answer and the
completion, and nothing that scores it.

```bash
"$EASYEP_VENV/bin/python" v4/easyep_v4.py blind \
  --results "$EASYEP_RESULTS_ROOT/run_JOB_ID/questions" \
  --questions "$EASYEP_DATA_ROOT/questions_used.json" \
  --out "$EASYEP_RESULTS_ROOT/run_JOB_ID/judge/questions"
```

`to_judge.jsonl` is one line per (item, system): snippet, exact prompt,
completion, the `answer` with the reasoning trace stripped, an opaque system label, and (for the question set) a separate
`reference` block. Paired judging refuses legacy completion rows that omit the
snippet or prompt, rather than emitting an answer that a reasoning judge cannot
assess. The variant mapping is withheld
in the mode-0600 `KEY_do_not_open_until_graded.json`. Public UIDs are keyed with
a random HMAC salt stored only in that withheld file, so the published variant
names cannot be brute-forced from a judging bundle. The label assignment itself
is drawn from the system CSPRNG rather than a seeded shuffle: seeded, it was a
pure function of the default `--seed` and the variant list published above, and
the whole mapping could be reproduced in three lines without the key. `--seed`
still orders the items, which reveals nothing on its own.

The published `item` is likewise an opaque salted id, not the join key. For the
matched pairs the plaintext key embeds the CodeQL label, so publishing it would
hand the grader the answer; `item_to_plaintext` and `item_to_truth` live in the
withheld key instead. Grading still groups systems per item, because the same
item yields the same opaque id across variants. Grade it with whatever judge you
like, then join on `uid` and open the key.

The one result that is still computed, because it needs no judge: the matched-pair
eval parses the `Verdict:` line and scores it against CodeQL ground truth. The
headline TPR, TNR, balanced accuracy, and Youden's J penalise unparsed verdicts.
`fpr_explicit_vulnerable` counts only explicit false alarms,
`safe_abstention_rate` reports rejected SAFE items separately, and
`safe_error_rate = 1 - TNR` is the rate used with TPR in Youden's J. The raw
completions are saved either way, so a judge can override that reading. Pass
`--no-controls` or ignore `pairs_summary.json` if you would rather have nothing
precomputed at all.

## Correctness gates

`run_experiment.sbatch` runs these sequentially in one Slurm job. Shell fail-fast
handling prevents profiling and evaluation from starting if either gate fails.

| gate | what it rules out |
|---|---|
| `parity` | the instrumentation itself changing the model. Fixed prompts through the untouched official forwards vs the patched-but-inactive ones, comparing prefill logits against a noise floor measured by running the official model twice (expert accumulation and all-reduce are not bitwise reproducible). Fails the job on a mismatch. |
| strict checkpoint load | a partially loaded model masquerading as a pruning effect. `load_model(strict=False)` returns `(missing, unexpected)` and callers normally discard both. Now every unexpected key, and every missing key outside `ALLOWED_MISSING`, aborts the run. |
| `validate` | the unweighted-norm recovery. `||w*out||/w == ||out||` is exact only in exact arithmetic; V4 applies the routing weight *before* `w2`, whose FP4 path runs `act_quant`, so quantisation can break proportionality. Runs each expert twice and reports norm error plus top-k ranking overlap. |

`ALLOWED_MISSING` covers only `mtp.N.embed.*` and `mtp.N.head.*`, which are
references to the trunk embedding and head (`Transformer.__init__` assigns
`mtp[i].embed = self.embed`) and are stored once, not per stage.

## Open items

- **Decoding control.** The experiment launcher and direct CLI commands default
  to `TEMPERATURE=0`, making the primary full-vs-pruned comparison greedy and
  deterministic. Question *i* is still decoded under `manual_seed(seed + i)` in
  every variant for reproducibility when a positive temperature is explicitly
  requested, but matching seeds alone do not remove sampling variance once
  pruning changes the token distribution. The regime used is recorded in
  `questions/summary.json` under `_decoding`.
- **MTP layers are excluded, not handled.** `DSparkBlock.forward` delegates to
  `Block.forward` when `start_pos > 0` with `layer_id` 43–45; the accumulator is
  bounds-guarded against that. They have their own experts under the `mtp.*`
  namespace and warrant their own profiling and pruning decision.
- **No weight deletion.** Masking only, by design.
- **Calibration size.** The default is 25 seeded, shuffled source files stratified
  across CWE directories, matching the paper's source-sample count. Token-exact
  chunking can expand those files into more than 25 profiled trajectories; both
  counts are recorded and must be reported for a literal comparison. The size
  has not yet been swept.
