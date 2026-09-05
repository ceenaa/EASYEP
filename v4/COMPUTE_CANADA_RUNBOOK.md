# EASY-EP V4 Compute Canada runbook

This is the operational checklist for running the DeepSeek V4 EASY-EP pipeline
on Rorqual. It records the working paths and the failures encountered during the
September 2026 rehearsal so the next run does not repeat them.

## What the workflow runs

The reproducible launcher executes:

```text
parity -> norm validation -> calibration/profile -> question evaluation
       -> vulnerable/safe pair evaluation -> blinded judging artifacts
```

The parity and norm-validation stages are mandatory gates. Every successful
stage writes an authenticated JSON checkpoint under `run_RUN_ID/.stages/`.

## Current Rorqual paths

```bash
export EASYEP_REPO=/home/sinam/EASYEP-v4
export EASYEP_VENV=/project/6084786/sinam/dsv4/venv
export EASYEP_CODE_DIR=/scratch/sinam/models/DeepSeek-V4-Flash-0731-9e165c30/inference
export EASYEP_CHECKPOINT=/project/6114328/sinam/dsv4/checkpoint
export EASYEP_DATA_ROOT=/scratch/sinam/deepseek_easy_ep/inputs
export EASYEP_RESULTS_ROOT=/scratch/sinam/deepseek_easy_ep/results
export EASYEP_CONFIG=/home/sinam/EASYEP-v4/v4/config_v4_flash.json
```

`EASYEP_CODE_DIR` must be the attested Hugging Face snapshot above. A copied
`inference/` directory without the snapshot's
`.cache/huggingface/download/*.metadata` files fails provenance verification.

Before every run:

```bash
cd "$EASYEP_REPO"
git status --short
git rev-parse HEAD
test -f "$EASYEP_CHECKPOINT/EASYEP_CHECKPOINT_PROVENANCE.json"
test -f "$EASYEP_DATA_ROOT/questions_used.json"
```

The launcher rejects tracked or staged source changes for a new run. Commit the
intended implementation first; do not bypass this protection.

## Experimental defaults

The main launcher defaults are:

```bash
export TEMPERATURE=0          # greedy and deterministic
export KEEP=128               # retain 128/256 experts in layers 3-42
export MAX_SEQ_LEN=16384
export MAX_NEW_TOKENS=2048
export PAIR_MAX_NEW_TOKENS=1024
export MAX_CHUNKS=0            # preserve every calibration chunk
export SEED=965
```

Greedy decoding is the primary comparison setting. Using the same RNG seed at
temperature 1 does not eliminate sampling variance after pruning changes the
token distribution. Run positive-temperature experiments separately over
multiple seeds and report their mean and variance.

Do not reduce response limits to 256/128 for quality evaluation: reasoning can
consume the entire budget before the structured answer. The September smoke run
used those small limits only to debug the pipeline.

## Choose the right submission mode

Use a normal batch job for unattended or production work. Use a three-hour held
allocation only for concentrated debugging where several iterations are likely.
Both modes enter the same queue; the held allocation saves later queue waits but
charges all four H100s while idle.

### Small batch rehearsal

Run one calibration source, one question, and one vulnerable/safe pair before a
large experiment:

```bash
cd "$EASYEP_REPO"
env -u RUN_ID -u RESUME sbatch \
  --account=rrg-tayebi_gpu \
  --time=03:00:00 \
  --export=ALL,RUN_ID=dev_YYYYMMDD_01,RESUME=0,TEMPERATURE=0,N_CALIB=1,N_PAIRS=1,LIMIT=1 \
  v4/easyep.sbatch
```

Always clear inherited `RUN_ID` and `RESUME`, then set both explicitly. A stale
`RESUME=1` caused job `20146663` to attach to an incompatible old manifest and
fail immediately.

### Interactive debugging block

```bash
cd "$EASYEP_REPO"
sbatch --account=rrg-tayebi_gpu v4/dev_hold.sbatch
squeue -u "$USER"
srun --jobid=JOB_ID --overlap --pty bash -l
```

Inside the compute-node shell, export the paths and explicit experiment values:

```bash
export EASYEP_REPO=/home/sinam/EASYEP-v4
export EASYEP_VENV=/project/6084786/sinam/dsv4/venv
export EASYEP_CODE_DIR=/scratch/sinam/models/DeepSeek-V4-Flash-0731-9e165c30/inference
export EASYEP_CHECKPOINT=/project/6114328/sinam/dsv4/checkpoint
export EASYEP_DATA_ROOT=/scratch/sinam/deepseek_easy_ep/inputs
export EASYEP_RESULTS_ROOT=/scratch/sinam/deepseek_easy_ep/results
export EASYEP_CONFIG=/home/sinam/EASYEP-v4/v4/config_v4_flash.json
export RUN_ID=dev_YYYYMMDD_01 RESUME=0 TEMPERATURE=0
export N_CALIB=1 N_PAIRS=1 LIMIT=1

cd "$EASYEP_REPO"
bash v4/run_in_allocation.sh
```

The helper copies the 159 GiB checkpoint to `$SLURM_TMPDIR` once, verifies it,
and reuses that local copy. In the rehearsal this reduced the first load to about
74 seconds and later loads to roughly 10-12 seconds; reading directly from
`/project` had stalled for more than three minutes.

The allocation survives an SSH disconnect, but the attached `srun` command may
not. Reconnect and attach again when necessary. Release the allocation as soon
as debugging ends:

```bash
scancel JOB_ID
```

### Production batch

Choose sample counts deliberately and record them with the result. `LIMIT=0`
evaluates all standalone questions.

```bash
cd "$EASYEP_REPO"
export TEMPERATURE=0 KEEP=128
export N_CALIB=CALIBRATION_SAMPLE_COUNT
export N_PAIRS=PAIR_COUNT
export LIMIT=0
export STRICT_PINS=1

env -u RUN_ID -u RESUME sbatch \
  --account=rrg-tayebi_gpu \
  --time=24:00:00 \
  --export=ALL,RUN_ID=prod_YYYYMMDD_01,RESUME=0 \
  v4/easyep.sbatch
```

Do not describe a one-sample rehearsal as an accuracy experiment. Record the
calibration count, standalone-question count, pair count, seed, temperature,
token limits, Git revision, and result path in every report.

## Monitor a job

```bash
squeue -j JOB_ID -o '%.18i %.9T %.10M %.10l %.6D %R'
scontrol show job JOB_ID
sacct -j JOB_ID --format=JobID,State,Elapsed,Submit,Start,End,ExitCode,NodeList
tail -F easyep-easyep_v4-JOB_ID.out
```

Queue wait is `Start - Submit`; `Elapsed` is runtime after resources were
allocated. Trust only terminal scheduler states such as `COMPLETED`, `FAILED`,
`TIMEOUT`, `CANCELLED`, `NODE_FAIL`, or `OUT_OF_MEMORY`.

After completion, confirm both scheduler and application state:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
python3 -m json.tool "$EASYEP_RESULTS_ROOT/run_RUN_ID/RUN_STATUS.json"
find "$EASYEP_RESULTS_ROOT/run_RUN_ID/.stages" -maxdepth 1 -type f -print
```

## Resume safely

Resume only the exact same immutable run after an interrupted stage:

```bash
cd "$EASYEP_REPO"
env -u RUN_ID -u RESUME sbatch \
  --account=rrg-tayebi_gpu \
  --export=ALL,RUN_ID=THE_ORIGINAL_RUN_ID,RESUME=1 \
  v4/easyep.sbatch
```

All original parameters, code, model provenance, inputs, and completed-stage
artifacts must still match. Never use `RESUME=1` to change temperature, sample
counts, token limits, pruning budget, code, or data; create a new `RUN_ID`.

## Frequent failures and their meaning

- **Old manifest or unexpected output directory:** stale `RUN_ID`/`RESUME` was
  exported in the login shell; resubmit with `env -u RUN_ID -u RESUME`.
- **Missing Hugging Face metadata:** `EASYEP_CODE_DIR` points to a copied code
  tree instead of the attested snapshot.
- **Long checkpoint reads:** use `run_in_allocation.sh` so the checkpoint is
  verified and staged on node-local storage.
- **No parsed verdicts:** inspect `completion_tokens`; the reasoning output
  probably exhausted `MAX_NEW_TOKENS` or `PAIR_MAX_NEW_TOKENS`.
- **Pin mismatch:** rebuild/use the environment recorded in
  `requirements-v4.txt`; production runs should use `STRICT_PINS=1`.
- **Dirty-checkout rejection:** commit intentional tracked changes before
  submitting a new reproducible run.
- **SSH disconnected:** the held batch allocation still exists; reconnect and
  attach with a new `srun --jobid=... --overlap --pty bash -l`.

## Interpreting results

Raw wall time is misleading when variants emit different token counts, and so
is `completion_tokens / seconds`: that spans prefill and decode together. Report
`throughput.decode_tokens_per_second` from `summary.json`, and check
`throughput.warmed` is true before quoting it. For older runs use
`python v4/analyze_throughput.py --run <dir>`. Compare quality only on
sufficiently large identical datasets. The blinded files under `judge/` should be graded before
opening their keys. Report uncertainty or repeated-seed variance where relevant.

The September smoke run at
`/scratch/sinam/deepseek_easy_ep/results/run_dev_20157423_smoke3` proved the
pipeline and provenance gates, but it used only one calibration source, one
standalone question, and one vulnerable/safe pair; it is not scientific evidence
for model quality.
