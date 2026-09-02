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
#SBATCH --time=08:00:00
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
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
PAIR_MAX_NEW_TOKENS=${PAIR_MAX_NEW_TOKENS:-128}
MAX_CHUNKS=${MAX_CHUNKS:-0}
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
require_uint MAX_SEQ_LEN "$MAX_SEQ_LEN" 2
require_uint MAX_NEW_TOKENS "$MAX_NEW_TOKENS" 1
require_uint PAIR_MAX_NEW_TOKENS "$PAIR_MAX_NEW_TOKENS" 1
require_uint MAX_CHUNKS "$MAX_CHUNKS" 0
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
if ! mkdir "$OUT"; then
  echo "ABORT: output already exists or could not be created (choose a new RUN_ID): $OUT" >&2
  exit 2
fi

# Write valid JSON with compact provenance. Code and input-tree digests commit to
# every relative path and byte without expanding thousands of per-file records.
# Checkpoint shards use a separately verified, full-content provenance record so
# jobs do not reread hundreds of GB at every launch.
"$VENV/bin/python" - \
  "$OUT/RUN_MANIFEST.json" "$REPO" "$SCRIPT" "${BASH_SOURCE[0]}" \
  "$CONFIG" "$CODE" "$CKPT" "$CHECKPOINT_PROVENANCE" "$DATA_ROOT" "$VENV" \
  "$GIT_SHA" "$GIT_TRACKED_DIRTY" \
  "$MODEL_ID" "$MODEL_REVISION" "$INFERENCE_REVISION" \
  "$KEEP" "$N_CALIB" "$N_PAIRS" "$MAX_SEQ_LEN" "$MAX_NEW_TOKENS" \
  "$PAIR_MAX_NEW_TOKENS" "$MAX_CHUNKS" "$SEED" "$TEMPERATURE" \
  "$NPROC_PER_NODE" <<'PY'
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys

(
    manifest_path, repo, script, entrypoint, config, code, ckpt,
    checkpoint_provenance, data_root, venv,
    git_sha, git_tracked_dirty, model_id, model_revision,
    inference_revision, keep, n_calib, n_pairs, max_seq_len, max_new_tokens,
    pair_max_new_tokens, max_chunks, seed, temperature, nproc,
) = sys.argv[1:]

def digest(path, capture=False):
    path = Path(path)
    before = path.stat()
    identity = lambda stat: (
        stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
    )
    h = hashlib.sha256()
    payload = bytearray() if capture else None
    with path.open("rb") as fp:
        if identity(os.fstat(fp.fileno())) != identity(before):
            raise RuntimeError(f"file changed while opening for hashing: {path}")
        for block in iter(lambda: fp.read(16 * 1024 * 1024), b""):
            h.update(block)
            if payload is not None:
                payload.extend(block)
        finished = os.fstat(fp.fileno())
    after = path.stat()
    if identity(before) != identity(finished) or identity(before) != identity(after):
        raise RuntimeError(f"file changed while hashing: {path}")
    record = {"path": str(path.resolve()), "size": finished.st_size,
              "sha256": h.hexdigest()}
    return (record, bytes(payload)) if payload is not None else record

def tree_digest(root):
    root = Path(root)
    aggregate = hashlib.sha256()
    count = total = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        item = digest(path)
        relative = str(path.relative_to(root))
        aggregate.update(relative.encode("utf-8") + b"\0")
        aggregate.update(item["sha256"].encode("ascii") + b"\n")
        count += 1
        total += item["size"]
    return {"root": str(root.resolve()), "file_count": count,
            "total_bytes": total, "tree_sha256": aggregate.hexdigest()}

def metadata(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}

def layout_digest(paths):
    records = [metadata(path) for path in sorted(paths)]
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {"files": records, "layout_sha256": hashlib.sha256(encoded).hexdigest()}

repo, code, ckpt, data_root = (
    Path(value).resolve() for value in (repo, code, ckpt, data_root)
)
runtime_paths = [
    Path(script), Path(entrypoint), Path(config), repo / "v4/easyep.sbatch",
    repo / "v4/requirements-v4.txt", repo / "v4/checkpoint_provenance.py",
    Path(checkpoint_provenance), ckpt / "tokenizer.json",
    ckpt / "tokenizer_config.json",
]
packages = {}
for name in ("torch", "transformers", "safetensors", "tilelang",
             "fast-hadamard-transform"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None

checkpoint_sidecar_digest, checkpoint_sidecar_bytes = digest(
    checkpoint_provenance, capture=True)
checkpoint_record = json.loads(checkpoint_sidecar_bytes)
checkpoint_content_identity = checkpoint_record.get("content_identity_sha256")
if (not isinstance(checkpoint_content_identity, str)
        or re.fullmatch(r"[0-9a-f]{64}", checkpoint_content_identity) is None):
    raise RuntimeError("verified checkpoint sidecar lacks a portable content identity")

manifest = {
    "schema_version": 2,
    "status": "running",
    "slurm": {"job_id": os.environ.get("SLURM_JOB_ID", "manual"),
              "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
              "node": platform.node()},
    "started_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "source": {"git_sha": git_sha, "git_tracked_dirty": git_tracked_dirty,
               "runtime_files": [
                   checkpoint_sidecar_digest
                   if Path(path).resolve() == Path(checkpoint_provenance).resolve()
                   else digest(path)
                   for path in runtime_paths
               ],
               "official_inference_tree": tree_digest(code),
               "official_encoding_tree": tree_digest(code.parent / "encoding")},
    "model": {"declared_id": model_id, "declared_revision": model_revision,
              "declared_inference_revision": inference_revision,
              "checkpoint_root": str(ckpt.resolve()),
              "checkpoint_content_identity_sha256": checkpoint_content_identity,
              "checkpoint_provenance": checkpoint_sidecar_digest,
              "checkpoint_layout": layout_digest(
                  ckpt.glob(f"model*-mp{nproc}.safetensors"))},
    "inputs": {
        "questions": digest(data_root / "questions_used.json"),
        "pair_manifest": digest(data_root / "vulnerable-js-files/CODEQL_SECURE_MANIFEST.jsonl"),
        "calibration_and_pair_tree": tree_digest(data_root / "vulnerable-js-files"),
    },
    "environment": {"python": sys.version, "platform": platform.platform(),
                    "packages": packages,
                    "python_module": os.environ.get("PYTHON_MODULE"),
                    "cuda_module": os.environ.get("CUDA_MODULE"),
                    "loaded_modules": os.environ.get("LOADEDMODULES", "").split(":"),
                    "venv": str(Path(venv).resolve())},
    "parameters": {"keep": int(keep), "n_calib": int(n_calib),
                   "n_pairs": int(n_pairs), "max_seq_len": int(max_seq_len),
                   "max_new_tokens": int(max_new_tokens),
                   "pair_max_new_tokens": int(pair_max_new_tokens),
                   "validate_n_calib": int(n_calib), "max_chunks": int(max_chunks),
                   "seed": int(seed),
                   "temperature": None if temperature == "" else float(temperature),
                   "nproc_per_node": int(nproc)},
}
Path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"manifest initialized: {manifest_path}")
PY

STAGES_COMPLETED=()
finalize_manifest() {
  local rc=$?
  trap - EXIT
  set +e
  local stages
  stages=$(IFS=,; printf '%s' "${STAGES_COMPLETED[*]:-}")
  "$VENV/bin/python" - "$OUT/RUN_MANIFEST.json" "$OUT" "$rc" "$stages" <<'PY'
import datetime
import json
from pathlib import Path
import sys

manifest_path, out_dir, return_code, stages = sys.argv[1:]
path = Path(manifest_path)
if path.is_file():
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update({
        "status": "complete" if int(return_code) == 0 else "failed",
        "exit_code": int(return_code),
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stages_completed": [stage for stage in stages.split(",") if stage],
        "artifacts": sorted(str(item.relative_to(out_dir)) for item in
                            Path(out_dir).rglob("*") if item.is_file() and item != path),
    })
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY
  local manifest_rc=$?
  if (( rc == 0 && manifest_rc != 0 )); then
    echo "ABORT: run succeeded but manifest finalization failed." >&2
    rc=$manifest_rc
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
  "$VENV/bin/python" - "$OUT/RUN_MANIFEST.json" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

def digest(path, capture=False):
    path = Path(path)
    before = path.stat()
    identity = lambda stat: (
        stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
    )
    h = hashlib.sha256()
    payload = bytearray() if capture else None
    with path.open("rb") as fp:
        if identity(os.fstat(fp.fileno())) != identity(before):
            raise RuntimeError(f"file changed while opening for hashing: {path}")
        for block in iter(lambda: fp.read(16 * 1024 * 1024), b""):
            h.update(block)
            if payload is not None:
                payload.extend(block)
        finished = os.fstat(fp.fileno())
    after = path.stat()
    if identity(before) != identity(finished) or identity(before) != identity(after):
        raise RuntimeError(f"file changed while hashing: {path}")
    record = {"path": str(path.resolve()), "size": finished.st_size,
              "sha256": h.hexdigest()}
    return (record, bytes(payload)) if payload is not None else record

def tree_digest(root):
    root = Path(root)
    aggregate = hashlib.sha256()
    count = total = 0
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        item = digest(path)
        relative = str(path.relative_to(root))
        aggregate.update(relative.encode("utf-8") + b"\0")
        aggregate.update(item["sha256"].encode("ascii") + b"\n")
        count += 1
        total += item["size"]
    return {"root": str(root.resolve()), "file_count": count,
            "total_bytes": total, "tree_sha256": aggregate.hexdigest()}

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for expected in manifest["source"]["runtime_files"]:
    if digest(expected["path"]) != expected:
        raise RuntimeError(f"runtime file changed after manifest creation: {expected['path']}")
for key in ("official_inference_tree", "official_encoding_tree"):
    expected = manifest["source"][key]
    if tree_digest(expected["root"]) != expected:
        raise RuntimeError(f"official source tree changed after manifest creation: {key}")
for key in ("questions", "pair_manifest"):
    expected = manifest["inputs"][key]
    if digest(expected["path"]) != expected:
        raise RuntimeError(f"experiment input changed after manifest creation: {key}")
expected_tree = manifest["inputs"]["calibration_and_pair_tree"]
if tree_digest(expected_tree["root"]) != expected_tree:
    raise RuntimeError("calibration/pair input tree changed after manifest creation")
sidecar_expected = manifest["model"]["checkpoint_provenance"]
sidecar_digest, sidecar_bytes = digest(sidecar_expected["path"], capture=True)
if sidecar_digest != sidecar_expected:
    raise RuntimeError("checkpoint provenance sidecar changed after manifest creation")
sidecar = json.loads(sidecar_bytes)
if sidecar.get("content_identity_sha256") != manifest["model"].get(
        "checkpoint_content_identity_sha256"):
    raise RuntimeError("checkpoint content identity changed after manifest creation")
print("run inputs still match RUN_MANIFEST.json")
PY
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
run_stage() {                  # run_stage <name> <mode> <args...>
  local stage=$1
  shift
  local port=$((MASTER_PORT_BASE + stage_index))
  stage_index=$((stage_index + 1))
  echo
  echo "---------- stage: $stage (master $MASTER_ADDR:$port) ----------"
  preflight_integrity
  (
    cd "$CODE"
    "$VENV/bin/torchrun" --nnodes=1 --nproc-per-node="$NPROC_PER_NODE" \
      --master-addr="$MASTER_ADDR" --master-port="$port" \
      "$SCRIPT" "$@"
  )
  STAGES_COMPLETED+=("$stage")
}

COMMON=(--ckpt-path "$CKPT" --config "$CONFIG" --code-dir "$CODE"
        --max-seq-len "$MAX_SEQ_LEN")
TEMPERATURE_ARGS=()
[[ -z $TEMPERATURE ]] || TEMPERATURE_ARGS=(--temperature "$TEMPERATURE")

# Both correctness gates are mandatory and stop the job on failure.
run_stage parity parity "${COMMON[@]}" --out "$OUT/parity"
run_stage validate validate "${COMMON[@]}" \
  "${TEMPERATURE_ARGS[@]}" \
  --calib-dir "$DATA_ROOT/vulnerable-js-files" \
  --out "$OUT/norm_validation" --n-calib "$N_CALIB" --keep "$KEEP" \
  --max-new-tokens "$MAX_NEW_TOKENS" --max-chunks "$MAX_CHUNKS" \
  --seed "$SEED"

# Profile once and reuse exactly that score tensor for both evaluation suites.
run_stage profile pipeline "${COMMON[@]}" "${TEMPERATURE_ARGS[@]}" \
  --calib-dir "$DATA_ROOT/vulnerable-js-files" \
  --questions "$DATA_ROOT/questions_used.json" \
  --out "$OUT/scores" --n-calib "$N_CALIB" --keep "$KEEP" \
  --max-new-tokens "$MAX_NEW_TOKENS" --max-chunks "$MAX_CHUNKS" \
  --seed "$SEED" --profile-only

run_stage eval pipeline "${COMMON[@]}" "${TEMPERATURE_ARGS[@]}" \
  --calib-dir "$DATA_ROOT/vulnerable-js-files" \
  --questions "$DATA_ROOT/questions_used.json" \
  --scores-in "$OUT/scores/expert_scores.pt" \
  --out "$OUT/questions" --keep "$KEEP" --seed "$SEED" \
  --max-new-tokens "$MAX_NEW_TOKENS"

run_stage pairs pairs "${COMMON[@]}" "${TEMPERATURE_ARGS[@]}" \
  --scores-in "$OUT/scores/expert_scores.pt" \
  --pairs-manifest "$DATA_ROOT/vulnerable-js-files/CODEQL_SECURE_MANIFEST.jsonl" \
  --calib-dir "$DATA_ROOT/vulnerable-js-files" \
  --out "$OUT/pairs" --n-pairs "$N_PAIRS" --keep "$KEEP" --seed "$SEED" \
  --max-new-tokens "$PAIR_MAX_NEW_TOKENS"

preflight_integrity
"$VENV/bin/python" "$SCRIPT" blind --results "$OUT/questions" \
  --questions "$DATA_ROOT/questions_used.json" --out "$OUT/judge/questions"
STAGES_COMPLETED+=(blind_questions)
"$VENV/bin/python" "$SCRIPT" blind --results "$OUT/pairs" --pairs \
  --out "$OUT/judge/pairs"
STAGES_COMPLETED+=(blind_pairs)

echo
echo "================================================================"
echo " DONE  $OUT"
echo " grade: $OUT/judge/questions/to_judge.jsonl"
echo " key:   $OUT/judge/questions/KEY_do_not_open_until_graded.json"
echo "================================================================"
