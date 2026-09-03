#!/bin/bash
# Reproducible Compute Canada entrypoint for the complete V4 experiment:
#
#   parity -> validate -> profile -> eval(questions) -> pairs -> blind
#      ^gate      ^gate
#
# Submit from the repository root, or export EASYEP_REPO explicitly:
#
#   sbatch --account=<allocation> v4/run_experiment.sbatch
#   sbatch --account=<allocation> --export=ALL,KEEP=192,N_CALIB=50 \
#     v4/run_experiment.sbatch
#
#SBATCH --job-name=easyep_v4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=450G
#SBATCH --time=24:00:00
#SBATCH --output=easyep-%x-%j.out

set -euo pipefail
umask 077

# Slurm executes a spooled copy, so BASH_SOURCE is not necessarily inside the
# checkout. Resolve the common submission locations before falling back to it.
resolve_repo() {
  if [[ -n ${EASYEP_REPO:-} ]]; then
    printf '%s\n' "$EASYEP_REPO"
  elif [[ -n ${SLURM_SUBMIT_DIR:-} && -f "$SLURM_SUBMIT_DIR/v4/easyep_v4.py" ]]; then
    printf '%s\n' "$SLURM_SUBMIT_DIR"
  elif [[ -n ${SLURM_SUBMIT_DIR:-} && -f "$SLURM_SUBMIT_DIR/easyep_v4.py" && \
          $(basename "$SLURM_SUBMIT_DIR") == v4 ]]; then
    (cd "$SLURM_SUBMIT_DIR/.." && pwd -P)
  else
    local script_dir
    script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
    (cd "$script_dir/.." && pwd -P)
  fi
}

repo_candidate=$(resolve_repo)
if [[ ! -d $repo_candidate ]]; then
  echo "ABORT: repository directory not found: $repo_candidate" >&2
  exit 2
fi
REPO=$(cd "$repo_candidate" && pwd -P)
SCRIPT="$REPO/v4/easyep_v4.py"
CONFIG=${EASYEP_CONFIG:-$REPO/v4/config_v4_flash.json}

# EASYEP_* variables are the public interface. BIG/SMALL/DATA remain optional
# fallback aliases for older submissions; there are no user-specific path defaults.
MODEL_ROOT=${EASYEP_MODEL_ROOT:-${BIG:-}}
CODE=${EASYEP_CODE_DIR:-${MODEL_ROOT:+$MODEL_ROOT/code/inference}}
CKPT=${EASYEP_CHECKPOINT:-${MODEL_ROOT:+$MODEL_ROOT/checkpoint}}
CHECKPOINT_PROVENANCE=${CKPT:+$CKPT/EASYEP_CHECKPOINT_PROVENANCE.json}
VENV=${EASYEP_VENV:-${SMALL:+$SMALL/venv}}
scratch_default=${SCRATCH:-/scratch/${USER:-unknown}}
DATA_ROOT=${EASYEP_DATA_ROOT:-${DATA:-$scratch_default/deepseek_easy_ep/inputs}}
RESULTS_ROOT=${EASYEP_RESULTS_ROOT:-$scratch_default/deepseek_easy_ep/results}

# This instrumentation was reviewed against one immutable official snapshot.
PINNED_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash-0731
PINNED_REVISION=9e165c30e2704aec5d9d593cce3eebd58bbef1cb
MODEL_ID=${MODEL_ID:-$PINNED_MODEL_ID}
MODEL_REVISION=${MODEL_REVISION:-$PINNED_REVISION}
INFERENCE_REVISION=${INFERENCE_REVISION:-$MODEL_REVISION}
if [[ $MODEL_ID != "$PINNED_MODEL_ID" || $MODEL_REVISION != "$PINNED_REVISION" || \
      $INFERENCE_REVISION != "$PINNED_REVISION" ]]; then
  echo "ABORT: this V4 implementation supports only $PINNED_MODEL_ID@$PINNED_REVISION." >&2
  exit 2
fi

KEEP=${KEEP:-128}
N_CALIB=${N_CALIB:-25}
N_PAIRS=${N_PAIRS:-25}
# NOT the config's original_seq_len=65536.  That is the model's trained context,
# not a context one 4xH100 node can prefill: Indexer.forward scores the whole
# prefill unblocked and materialises [1,T,16,T/4] bf16 twice, so the transient
# is ~16*T^2 bytes against ~35 GiB of headroom (weights are 43.5 GiB/rank).
# T=16384 costs 4.9 GiB; T=24576 costs 11.0 GiB; T=65280 costs 77.4 GiB and OOMs.
# Chunking keeps whole sources, so a smaller window is not a loss of coverage --
# raising this makes each chunk the OOM case instead of avoiding chunking.
MAX_SEQ_LEN=${MAX_SEQ_LEN:-16384}
# Sized for reasoning mode, not chat. The model emits a chain of thought before
# the answer, so the old 256/128 budgets truncated mid-deliberation: the pair
# eval would never reach its "Verdict:" line and every row would score as an
# abstention. These are estimates -- measure completion_tokens in the first run
# and tighten them, because they dominate the wall time.
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
PAIR_MAX_NEW_TOKENS=${PAIR_MAX_NEW_TOKENS:-1024}
MAX_CHUNKS=${MAX_CHUNKS:-0}
# Rehearsal knob. N_CALIB and N_PAIRS already shrink profiling and the pair
# eval; the question eval had no equivalent, and it is the largest single
# consumer of wall time. 0 evaluates every question.
LIMIT=${LIMIT:-0}
SEED=${SEED:-965}
TEMPERATURE=${TEMPERATURE:-}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}

if [[ -n ${RUN_ID:-} ]]; then
  run_id=$RUN_ID
elif [[ -n ${SLURM_JOB_ID:-} ]]; then
  run_id=$SLURM_JOB_ID
  [[ -z ${SLURM_ARRAY_TASK_ID:-} ]] || run_id="${run_id}_${SLURM_ARRAY_TASK_ID}"
else
  run_id="manual_$(date -u +%Y%m%dT%H%M%SZ)_$$"
fi
if [[ ! $run_id =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ABORT: RUN_ID may contain only letters, digits, dot, underscore, and dash." >&2
  exit 2
fi
OUT="$RESULTS_ROOT/run_$run_id"

require_file() {
  [[ -f $1 ]] || { echo "ABORT: required file not found: $1" >&2; exit 2; }
}
require_dir() {
  [[ -d $1 ]] || { echo "ABORT: required directory not found: $1" >&2; exit 2; }
}
require_value() {
  [[ -n $2 ]] || { echo "ABORT: export $1 before submitting this job." >&2; exit 2; }
}
require_absolute() {
  [[ $2 == /* ]] || { echo "ABORT: $1 must be an absolute path: $2" >&2; exit 2; }
}
require_uint() {
  local name=$1 value=$2 minimum=$3 maximum=${4:-}
  if [[ ! $value =~ ^[0-9]+$ || ${#value} -gt 9 ]]; then
    echo "ABORT: $name must be a non-negative decimal integer: $value" >&2
    exit 2
  fi
  local decimal=$((10#$value))
  if (( decimal < minimum )) || { [[ -n $maximum ]] && (( decimal > maximum )); }; then
    echo "ABORT: $name is outside the supported range [$minimum, ${maximum:-unbounded}]: $value" >&2
    exit 2
  fi
  printf -v "$name" '%d' "$decimal"
}

require_uint KEEP "$KEEP" 6 256
require_uint N_CALIB "$N_CALIB" 1
require_uint N_PAIRS "$N_PAIRS" 1
# Ceiling, not taste: the unblocked ~16*T^2 prefill transient reaches 11.0 GiB
# at 24576 against ~35 GiB of headroom, and OOMs well before 65536.
require_uint MAX_SEQ_LEN "$MAX_SEQ_LEN" 2 24576
require_uint MAX_NEW_TOKENS "$MAX_NEW_TOKENS" 1
require_uint PAIR_MAX_NEW_TOKENS "$PAIR_MAX_NEW_TOKENS" 1
require_uint MAX_CHUNKS "$MAX_CHUNKS" 0
require_uint LIMIT "$LIMIT" 0
require_uint SEED "$SEED" 0
require_uint NPROC_PER_NODE "$NPROC_PER_NODE" 4 4
if (( MAX_NEW_TOKENS >= MAX_SEQ_LEN || PAIR_MAX_NEW_TOKENS >= MAX_SEQ_LEN )); then
  echo "ABORT: response-token limits must be smaller than MAX_SEQ_LEN." >&2
  exit 2
fi
if [[ -n $TEMPERATURE && ! $TEMPERATURE =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
  echo "ABORT: TEMPERATURE must be a non-negative finite number: $TEMPERATURE" >&2
  exit 2
fi

require_value EASYEP_CODE_DIR "$CODE"
require_value EASYEP_CHECKPOINT "$CKPT"
require_value EASYEP_VENV "$VENV"
require_absolute EASYEP_CODE_DIR "$CODE"
require_absolute EASYEP_CHECKPOINT "$CKPT"
require_absolute "checkpoint provenance sidecar" "$CHECKPOINT_PROVENANCE"
require_absolute EASYEP_VENV "$VENV"
require_absolute EASYEP_DATA_ROOT "$DATA_ROOT"
require_absolute EASYEP_RESULTS_ROOT "$RESULTS_ROOT"
require_absolute EASYEP_CONFIG "$CONFIG"
require_dir "$REPO"
require_file "$SCRIPT"
require_file "$CONFIG"
require_file "$REPO/v4/easyep.sbatch"
require_file "$REPO/v4/requirements-v4.txt"
require_file "$REPO/v4/checkpoint_provenance.py"
require_file "$REPO/v4/run_provenance.py"
require_dir "$CODE"
CODE=$(cd -P "$CODE" && pwd -P)
SOURCE_SNAPSHOT=$(cd -P "$CODE/.." && pwd -P)
require_file "$CODE/model.py"
require_file "$CODE/kernel.py"
require_file "$CODE/generate.py"
require_file "$SOURCE_SNAPSHOT/encoding/encoding_dsv4.py"
require_dir "$CKPT"
require_file "$CKPT/tokenizer.json"
require_file "$CKPT/tokenizer_config.json"
require_file "$CHECKPOINT_PROVENANCE"
require_file "$VENV/bin/python"
require_file "$VENV/bin/torchrun"
require_dir "$DATA_ROOT/vulnerable-js-files"
require_file "$DATA_ROOT/questions_used.json"
require_file "$DATA_ROOT/vulnerable-js-files/CODEQL_SECURE_MANIFEST.jsonl"

shopt -s nullglob
checkpoint_shards=("$CKPT"/model*-mp"$NPROC_PER_NODE".safetensors)
shopt -u nullglob
if (( ${#checkpoint_shards[@]} != NPROC_PER_NODE )); then
  echo "ABORT: expected $NPROC_PER_NODE model*-mp${NPROC_PER_NODE}.safetensors shards" \
       "under $CKPT; found ${#checkpoint_shards[@]}." >&2
  exit 2
fi
GIT_SHA=unknown
GIT_TRACKED_DIRTY=n/a
if git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_SHA=$(git -C "$REPO" rev-parse HEAD)
  if git -C "$REPO" diff --quiet && git -C "$REPO" diff --cached --quiet; then
    GIT_TRACKED_DIRTY=0
  else
    GIT_TRACKED_DIRTY=1
  fi
fi
if [[ $GIT_TRACKED_DIRTY == 1 ]]; then
  echo "ABORT: the checkout has tracked or staged changes; commit them before a run." >&2
  exit 2
fi

export PYTHON_MODULE=${PYTHON_MODULE:-python}
export CUDA_MODULE=${CUDA_MODULE:-cuda/13.2}
module load "$PYTHON_MODULE" "$CUDA_MODULE"

# The pinned requirements are hashed into RUN_MANIFEST.json, which records what
# the run *should* have used. Compare them against what is actually importable
# so a divergence is visible in the log instead of only inferable afterwards.
# Not fatal by default: site wheelhouse builds (e.g. 2.13.0+cc) are legitimate
# and intended. Set STRICT_PINS=1 to require an exact match.
STRICT_PINS=${STRICT_PINS:-0}
export STRICT_PINS
if ! "$VENV/bin/python" - "$REPO/v4/requirements-v4.txt" "$STRICT_PINS" <<'PYPIN'; then
import importlib.metadata as md, re, sys
requirements, strict = sys.argv[1], sys.argv[2] == "1"
diverged = []
for line in open(requirements, encoding="utf-8"):
    line = line.split("#", 1)[0].strip()
    match = re.fullmatch(r"([A-Za-z0-9._-]+)==([^\s]+)", line)
    if not match:
        continue
    name, pinned = match.group(1), match.group(2)
    try:
        installed = md.version(name)
    except md.PackageNotFoundError:
        installed = "MISSING"
    if installed != pinned:
        diverged.append((name, pinned, installed))
        print(f"  PIN MISMATCH  {name}: pinned {pinned}, installed {installed}")
if not diverged:
    print("  all pins match the active environment")
sys.exit(1 if (diverged and strict) else 0)
PYPIN
  echo "ABORT: STRICT_PINS=1 and the environment does not match v4/requirements-v4.txt." >&2
  exit 2
fi


# The shell regex above rejects malformed values, but a syntactically valid
# exponent such as 1e999 overflows Python's float to infinity.  Reject that
# before writing non-standard JSON or starting a multi-GPU model process.
if [[ -n $TEMPERATURE ]] && ! "$VENV/bin/python" -c \
  'import math, sys; value = float(sys.argv[1]); raise SystemExit(not (math.isfinite(value) and value >= 0.0))' \
  "$TEMPERATURE"; then
  echo "ABORT: TEMPERATURE must be a non-negative finite number: $TEMPERATURE" >&2
  exit 2
fi

# The provenance sidecar contains full SHA-256 hashes made once after conversion.
# Routine launches compare its declarations and exact shard metadata; use the
# utility's `verify --full` mode whenever a fresh byte-for-byte audit is desired.
"$VENV/bin/python" "$REPO/v4/checkpoint_provenance.py" verify \
  --checkpoint "$CKPT" --manifest "$CHECKPOINT_PROVENANCE" \
  --config "$CONFIG" --source-snapshot "$SOURCE_SNAPSHOT" \
  --model-id "$MODEL_ID" --model-revision "$MODEL_REVISION" \
  --inference-revision "$INFERENCE_REVISION" --nproc "$NPROC_PER_NODE"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=${PYTHONHASHSEED:-$SEED}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
# The profiler enables strict deterministic algorithms around its score
# reductions. CUDA >= 10.2 requires this workspace setting for cuBLAS to be
# deterministic, and raises rather than silently degrading without it.
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

# Job-local caches avoid races between concurrent compilations. A persistent
# cache can still be selected explicitly with EASYEP_CACHE_ROOT.
CACHE_ROOT=${EASYEP_CACHE_ROOT:-${SLURM_TMPDIR:-$scratch_default/easyep-cache/$run_id}}
export TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-$CACHE_ROOT/tilelang}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/inductor}
mkdir -p "$TILELANG_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Derive a per-job range instead of choosing a collision-prone random port.
job_number=${SLURM_JOB_ID:-$$}
job_number=${job_number%%_*}
[[ $job_number =~ ^[0-9]+$ ]] || job_number=$$
MASTER_PORT_BASE=${MASTER_PORT_BASE:-$((20000 + job_number % 20000))}
require_uint MASTER_PORT_BASE "$MASTER_PORT_BASE" 1024 65527
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

mkdir -p "$RESULTS_ROOT"
# Hold one advisory lock for the lifetime of the job. Duplicate resubmissions of
# the same RUN_ID must never write the same manifest or artifact directories.
command -v flock >/dev/null || {
  echo "ABORT: flock is required for exclusive run ownership." >&2
  exit 2
}
exec {RUN_LOCK_FD}>"$RESULTS_ROOT/.run_${run_id}.lock"
if ! flock -n "$RUN_LOCK_FD"; then
  echo "ABORT: another job already owns RUN_ID=$run_id." >&2
  exit 2
fi
ATTEMPT_STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
# RESUME=1 reattaches to an existing run directory and skips stages that already
# recorded a verified completion checkpoint. Without it the launcher refuses to touch
# an existing directory, so a resume is always a deliberate act.
RESUME=${RESUME:-0}
if [[ $RESUME != 0 && $RESUME != 1 ]]; then
  echo "ABORT: RESUME must be 0 or 1." >&2
  exit 2
fi
if [[ $RESUME == 1 ]]; then
  require_dir "$OUT"
  # A run that died before stage setup has no .stages; that means nothing
  # completed, not that the resume is invalid, so create it and re-run all.
  mkdir -p "$OUT/.stages"
  require_file "$OUT/RUN_MANIFEST.json"
  echo "RESUME: validating all immutable provenance before reattaching to $OUT."
elif ! mkdir "$OUT"; then
  echo "ABORT: output already exists or could not be created (choose a new RUN_ID," \
       "or resubmit the same RUN_ID with RESUME=1): $OUT" >&2
  exit 2
fi

# Build the current immutable record in memory. On resume the helper requires an
# exact match and leaves the old record untouched; on a new run it publishes the
# record atomically. Volatile attempt/status fields live outside provenance.
MANIFEST_ARGS=(init --manifest "$OUT/RUN_MANIFEST.json" --repo "$REPO"
  --script "$SCRIPT" --entrypoint "${BASH_SOURCE[0]}" --config "$CONFIG"
  --code "$CODE" --checkpoint "$CKPT"
  --checkpoint-provenance "$CHECKPOINT_PROVENANCE" --data-root "$DATA_ROOT"
  --venv "$VENV" --git-sha "$GIT_SHA" --git-tracked-dirty "$GIT_TRACKED_DIRTY"
  --model-id "$MODEL_ID" --model-revision "$MODEL_REVISION"
  --inference-revision "$INFERENCE_REVISION" --keep "$KEEP" --n-calib "$N_CALIB"
  --n-pairs "$N_PAIRS" --max-seq-len "$MAX_SEQ_LEN"
  --max-new-tokens "$MAX_NEW_TOKENS" --pair-max-new-tokens "$PAIR_MAX_NEW_TOKENS"
  --max-chunks "$MAX_CHUNKS" --question-limit "$LIMIT" --seed "$SEED"
  --nproc-per-node "$NPROC_PER_NODE" --attempt-started-utc "$ATTEMPT_STARTED_UTC")
[[ -z $TEMPERATURE ]] || MANIFEST_ARGS+=(--temperature "$TEMPERATURE")
[[ $RESUME == 1 ]] && MANIFEST_ARGS+=(--resume)
"$VENV/bin/python" "$REPO/v4/run_provenance.py" "${MANIFEST_ARGS[@]}"

STAGES_COMPLETED=()
finalize_manifest() {
  local rc=$?
  trap - EXIT
  set +e
  local finalize_args=(finalize --manifest "$OUT/RUN_MANIFEST.json"
    --status "$OUT/RUN_STATUS.json"
    --out-dir "$OUT" --return-code "$rc"
    --attempt-started-utc "$ATTEMPT_STARTED_UTC")
  [[ $RESUME == 1 ]] && finalize_args+=(--resumed)
  local stage
  for stage in "${STAGES_COMPLETED[@]:-}"; do
    [[ -z $stage ]] || finalize_args+=(--stage "$stage")
  done
  "$VENV/bin/python" "$REPO/v4/run_provenance.py" "${finalize_args[@]}"
  local status_rc=$?
  if (( rc == 0 && status_rc != 0 )); then
    echo "ABORT: run succeeded but status finalization failed." >&2
    rc=$status_rc
  fi
  exit "$rc"
}
trap finalize_manifest EXIT

preflight_integrity() {
  # Recheck immediately before each process launch. This narrows the verification
  # boundary to the bytes the next stage is about to import and the inputs it is
  # about to consume, rather than trusting a job-start-only observation.
  "$VENV/bin/python" "$REPO/v4/checkpoint_provenance.py" verify \
    --checkpoint "$CKPT" --manifest "$CHECKPOINT_PROVENANCE" \
    --config "$CONFIG" --source-snapshot "$SOURCE_SNAPSHOT" \
    --model-id "$MODEL_ID" --model-revision "$MODEL_REVISION" \
    --inference-revision "$INFERENCE_REVISION" --nproc "$NPROC_PER_NODE"
  "$VENV/bin/python" "$REPO/v4/run_provenance.py" preflight \
    --manifest "$OUT/RUN_MANIFEST.json" --entrypoint "${BASH_SOURCE[0]}"
}

echo "================================================================"
echo " easyep V4 run  job=${SLURM_JOB_ID:-manual}  node=$(hostname)"
echo " git        $GIT_SHA"
echo " model      $MODEL_ID @ $MODEL_REVISION"
echo " config     $CONFIG"
echo " code       $CODE"
echo " checkpoint $CKPT"
echo " data       $DATA_ROOT"
echo " out        $OUT"
echo "================================================================"

stage_index=0
# A stage is complete only if its atomic JSON marker still authenticates the
# immutable run provenance, exact command, direct inputs, and complete outputs.
mkdir -p "$OUT/.stages"
stage_done() { [[ -f "$OUT/.stages/$1.done" ]]; }

stage_record_args() {          # stage_record_args <name> <output> <inputs...> -- <command...>
  local stage=$1
  local output=$2
  shift 2
  STAGE_RECORD_ARGS=(--manifest "$OUT/RUN_MANIFEST.json"
    --marker "$OUT/.stages/$stage.done" --stage "$stage")
  while [[ $1 != -- ]]; do
    STAGE_RECORD_ARGS+=(--input="$1")
    shift
  done
  shift
  local arg
  for arg in "$@"; do
    STAGE_RECORD_ARGS+=(--command-arg="$arg")
  done
  STAGE_RECORD_ARGS+=(--output="$output")
}

run_stage() {                  # run_stage <name> <output> <inputs...> -- <mode> <args...>
  local stage=$1 output=$2
  shift 2
  local inputs=()
  while [[ $1 != -- ]]; do
    inputs+=("$1")
    shift
  done
  shift
  local command=("$@")
  local port=$((MASTER_PORT_BASE + stage_index))
  stage_index=$((stage_index + 1))
  stage_record_args "$stage" "$output" "${inputs[@]}" -- \
    torchrun "$NPROC_PER_NODE" "$SCRIPT" "${command[@]}"
  # This check happens before marker verification, including for skipped stages.
  # It therefore compares current bytes to the original immutable provenance.
  preflight_integrity
  if [[ $RESUME == 1 ]] && stage_done "$stage"; then
    "$VENV/bin/python" "$REPO/v4/run_provenance.py" verify-stage \
      "${STAGE_RECORD_ARGS[@]}"
    echo
    echo "---------- stage: $stage  SKIPPED (inputs and outputs verified) ----------"
    STAGES_COMPLETED+=("$stage")
    return 0
  fi
  echo
  echo "---------- stage: $stage (master $MASTER_ADDR:$port) ----------"
  (
    cd "$CODE"
    "$VENV/bin/torchrun" --nnodes=1 --nproc-per-node="$NPROC_PER_NODE" \
      --master-addr="$MASTER_ADDR" --master-port="$port" \
      "$SCRIPT" "${command[@]}"
  )
  "$VENV/bin/python" "$REPO/v4/run_provenance.py" mark-stage \
    "${STAGE_RECORD_ARGS[@]}"
  STAGES_COMPLETED+=("$stage")
}

run_local_stage() {            # same checkpoint contract for non-torchrun stages
  local stage=$1 output=$2 input=$3
  shift 3
  local command=("$@")
  stage_record_args "$stage" "$output" "$input" -- "${command[@]}"
  preflight_integrity
  if [[ $RESUME == 1 ]] && stage_done "$stage"; then
    "$VENV/bin/python" "$REPO/v4/run_provenance.py" verify-stage \
      "${STAGE_RECORD_ARGS[@]}"
    echo "---------- stage: $stage  SKIPPED (inputs and outputs verified) ----------"
  else
    echo "---------- stage: $stage ----------"
    "${command[@]}"
    "$VENV/bin/python" "$REPO/v4/run_provenance.py" mark-stage \
      "${STAGE_RECORD_ARGS[@]}"
  fi
  STAGES_COMPLETED+=("$stage")
}

COMMON=(--ckpt-path "$CKPT" --config "$CONFIG" --code-dir "$CODE"
        --max-seq-len "$MAX_SEQ_LEN")
TEMPERATURE_ARGS=()
[[ -z $TEMPERATURE ]] || TEMPERATURE_ARGS=(--temperature "$TEMPERATURE")

# Both correctness gates are mandatory and stop the job on failure.
run_stage parity "$OUT/parity" -- parity "${COMMON[@]}" --out "$OUT/parity"
run_stage validate "$OUT/norm_validation" "$OUT/parity" -- validate "${COMMON[@]}" \
  "${TEMPERATURE_ARGS[@]}" \
  --calib-dir "$DATA_ROOT/vulnerable-js-files" \
  --out "$OUT/norm_validation" --n-calib "$N_CALIB" --keep "$KEEP" \
  --max-new-tokens "$MAX_NEW_TOKENS" --max-chunks "$MAX_CHUNKS" \
  --seed "$SEED"

# Profile once and reuse exactly that score tensor for both evaluation suites.
run_stage profile "$OUT/scores" "$OUT/parity" "$OUT/norm_validation" -- \
  pipeline "${COMMON[@]}" "${TEMPERATURE_ARGS[@]}" \
  --calib-dir "$DATA_ROOT/vulnerable-js-files" \
  --questions "$DATA_ROOT/questions_used.json" \
  --out "$OUT/scores" --n-calib "$N_CALIB" --keep "$KEEP" \
  --max-new-tokens "$MAX_NEW_TOKENS" --max-chunks "$MAX_CHUNKS" \
  --seed "$SEED" --profile-only

run_stage eval "$OUT/questions" "$OUT/scores/expert_scores.pt" -- \
  pipeline "${COMMON[@]}" "${TEMPERATURE_ARGS[@]}" \
  --calib-dir "$DATA_ROOT/vulnerable-js-files" \
  --questions "$DATA_ROOT/questions_used.json" \
  --scores-in "$OUT/scores/expert_scores.pt" \
  --out "$OUT/questions" --keep "$KEEP" --seed "$SEED" \
  --limit "$LIMIT" \
  --max-new-tokens "$MAX_NEW_TOKENS"

run_stage pairs "$OUT/pairs" "$OUT/scores/expert_scores.pt" -- \
  pairs "${COMMON[@]}" "${TEMPERATURE_ARGS[@]}" \
  --scores-in "$OUT/scores/expert_scores.pt" \
  --pairs-manifest "$DATA_ROOT/vulnerable-js-files/CODEQL_SECURE_MANIFEST.jsonl" \
  --calib-dir "$DATA_ROOT/vulnerable-js-files" \
  --out "$OUT/pairs" --n-pairs "$N_PAIRS" --keep "$KEEP" --seed "$SEED" \
  --max-new-tokens "$PAIR_MAX_NEW_TOKENS"

# Re-running blind draws a fresh label assignment, which would invalidate the
# key any in-flight grading is holding. Content-bound markers make skipping safe.
run_local_stage blind_questions "$OUT/judge/questions" "$OUT/questions" \
  "$VENV/bin/python" "$SCRIPT" blind --results "$OUT/questions" \
  --questions "$DATA_ROOT/questions_used.json" --out "$OUT/judge/questions"
run_local_stage blind_pairs "$OUT/judge/pairs" "$OUT/pairs" \
  "$VENV/bin/python" "$SCRIPT" blind --results "$OUT/pairs" --pairs \
  --out "$OUT/judge/pairs"

echo
echo "================================================================"
echo " DONE  $OUT"
echo " grade: $OUT/judge/questions/to_judge.jsonl"
echo " key:   $OUT/judge/questions/KEY_do_not_open_until_graded.json"
echo "================================================================"
