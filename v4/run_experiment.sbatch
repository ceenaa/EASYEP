#!/bin/bash
# The experiment entrypoint. Runs the full chain in order and stops the moment a
# correctness gate fails, so no scores, masks or completions can be produced by a
# model that is not the model being evaluated.
#
#   parity -> profile -> validate -> eval(questions) -> pairs -> blind
#            ^gate                 ^gate
#
# Everything runs the checked-in v4/easyep_v4.py from a git checkout, and the
# commit SHA plus content hashes of every input are written to RUN_MANIFEST.json
# beside the results, so any output can be traced back to the exact code.
#
#   sbatch v4/run_experiment.sbatch
#   sbatch --export=ALL,KEEP=192,N_CALIB=50 v4/run_experiment.sbatch
#
#SBATCH --job-name=easyep_run
#SBATCH --account=rrg-tayebi_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=450G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/sinam/deepseek_easy_ep/easyep-run-%j.out

set -euo pipefail

# ---------------------------------------------------------------- inputs
REPO=${EASYEP_REPO:-/project/6084786/sinam/dsv4/EASYEP}
BIG=${BIG:-/project/6114328/sinam/dsv4}
SMALL=${SMALL:-/project/6084786/sinam/dsv4}
DATA=${DATA:-/scratch/sinam/deepseek_easy_ep/inputs}
RESULTS_ROOT=${RESULTS_ROOT:-/scratch/sinam/deepseek_easy_ep/results}

SCRIPT="$REPO/v4/easyep_v4.py"
CONFIG="$BIG/code/inference/config.json"
CKPT="$BIG/checkpoint"
CODE="$BIG/code/inference"
VENV="$SMALL/venv"

KEEP=${KEEP:-128}
N_CALIB=${N_CALIB:-25}
N_PAIRS=${N_PAIRS:-25}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
SEED=${SEED:-965}

OUT="$RESULTS_ROOT/run_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT"

module load cuda/13.2
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-$HOME/.tilelang/cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$HOME/.triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$HOME/.inductor}

# ------------------------------------------------------------ provenance
# Written before anything runs, so even a failed run is traceable.
sha() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }

GIT_SHA=$(cat "$REPO/GIT_SHA" 2>/dev/null || echo "unknown")
if [ -d "$REPO/.git" ]; then
  GIT_SHA=$(git -C "$REPO" rev-parse HEAD)
  GIT_DIRTY=$(git -C "$REPO" status --porcelain | wc -l)
else
  GIT_DIRTY="n/a (archive export)"
fi

cat > "$OUT/RUN_MANIFEST.json" <<MANIFEST
{
  "slurm_job_id": "${SLURM_JOB_ID:-manual}",
  "node": "$(hostname)",
  "started_utc": "$(date -u +%FT%TZ)",
  "git_sha": "$GIT_SHA",
  "git_dirty_files": "$GIT_DIRTY",
  "script": "$SCRIPT",
  "script_sha256": "$(sha "$SCRIPT")",
  "entrypoint_sha256": "$(sha "${BASH_SOURCE[0]}")",
  "model_config": "$CONFIG",
  "model_config_sha256": "$(sha "$CONFIG")",
  "model_py_sha256": "$(sha "$CODE/model.py")",
  "kernel_py_sha256": "$(sha "$CODE/kernel.py")",
  "questions": "$DATA/questions_used.json",
  "questions_sha256": "$(sha "$DATA/questions_used.json")",
  "pairs_manifest_sha256": "$(sha "$DATA/vulnerable-js-files/CODEQL_SECURE_MANIFEST.jsonl")",
  "checkpoint": "$CKPT",
  "checkpoint_shards": "$(cd "$CKPT" && ls -la model*-mp4.safetensors | awk '{print $9":"$5}' | paste -sd, -)",
  "tokenizer_sha256": "$(sha "$CKPT/tokenizer.json")",
  "params": {
    "keep": $KEEP, "n_calib": $N_CALIB, "n_pairs": $N_PAIRS,
    "max_seq_len": $MAX_SEQ_LEN, "max_new_tokens": $MAX_NEW_TOKENS, "seed": $SEED
  }
}
MANIFEST

echo "================================================================"
echo " easyep run  job=${SLURM_JOB_ID:-manual}  node=$(hostname)"
echo " git       $GIT_SHA  (dirty files: $GIT_DIRTY)"
echo " script    $(sha "$SCRIPT")"
echo " out       $OUT"
echo "================================================================"

if [ "$GIT_DIRTY" != "0" ] && [ "$GIT_DIRTY" != "n/a (archive export)" ]; then
  echo "ABORT: the checkout has uncommitted changes; results would not be traceable."
  exit 2
fi

run() {                       # run() <stage> <args...>
  local stage=$1; shift
  echo; echo "---------- stage: $stage ----------"
  cd "$CODE"
  "$VENV/bin/torchrun" --nnodes=1 --nproc-per-node=4 \
    --master-addr=127.0.0.1 --master-port=$((29520 + RANDOM % 200)) \
    "$SCRIPT" "$@"
}

COMMON=(--ckpt-path "$CKPT" --config "$CONFIG" --code-dir "$CODE" --max-seq-len "$MAX_SEQ_LEN")

# ---- gate 1: the instrumentation must not change the model -------------
run parity parity "${COMMON[@]}" --out "$OUT/parity" \
  || { echo "ABORT: parity failed - instrumentation changes the model."; exit 1; }

# ---- gate 2: the unweighted-norm shortcut must not change the ranking ---
run validate validate "${COMMON[@]}" --calib-dir "$DATA/vulnerable-js-files" \
  --out "$OUT/norm_validation" --n-calib 8 --keep "$KEEP" \
  || { echo "ABORT: norm validation failed - the ||w*out||/w shortcut is unsound."; exit 1; }

# ---- profile: one calibration pass, all scores -------------------------
run profile pipeline "${COMMON[@]}" \
  --calib-dir "$DATA/vulnerable-js-files" --questions "$DATA/questions_used.json" \
  --out "$OUT/scores" --n-calib "$N_CALIB" --keep "$KEEP" --seed "$SEED" --profile-only

# ---- eval: every variant on the question set ---------------------------
run eval pipeline "${COMMON[@]}" \
  --calib-dir "$DATA/vulnerable-js-files" --questions "$DATA/questions_used.json" \
  --scores-in "$OUT/scores/expert_scores.pt" \
  --out "$OUT/questions" --keep "$KEEP" --seed "$SEED" \
  --max-new-tokens "$MAX_NEW_TOKENS"

# ---- pairs: matched vulnerable/secure discrimination -------------------
run pairs pairs "${COMMON[@]}" \
  --scores-in "$OUT/scores/expert_scores.pt" \
  --pairs-manifest "$DATA/vulnerable-js-files/CODEQL_SECURE_MANIFEST.jsonl" \
  --calib-dir "$DATA/vulnerable-js-files" \
  --out "$OUT/pairs" --n-pairs "$N_PAIRS" --keep "$KEEP" --seed "$SEED" \
  --max-new-tokens 128

# ---- blind: anonymised bundles for an external judge -------------------
"$VENV/bin/python" "$SCRIPT" blind --results "$OUT/questions" \
  --questions "$DATA/questions_used.json" --out "$OUT/judge/questions"
"$VENV/bin/python" "$SCRIPT" blind --results "$OUT/pairs" --pairs \
  --out "$OUT/judge/pairs"

python3 - "$OUT" <<'PY'
import json, sys, datetime
from pathlib import Path
out = Path(sys.argv[1])
m = json.loads((out / "RUN_MANIFEST.json").read_text())
m["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
m["stages_completed"] = ["parity", "validate", "profile", "eval", "pairs", "blind"]
m["artifacts"] = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
(out / "RUN_MANIFEST.json").write_text(json.dumps(m, indent=1))
print(f"\nmanifest: {out/'RUN_MANIFEST.json'}  ({len(m['artifacts'])} artifacts)")
PY

echo
echo "================================================================"
echo " DONE  $OUT"
echo " grade: $OUT/judge/questions/to_judge.jsonl  (key withheld alongside)"
echo "================================================================"
