#!/bin/bash -l
#SBATCH --job-name=v4-experts
#SBATCH --account=rrg-tayebi_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.log

set -eo pipefail
mode="${1:-serve}"
case "$mode" in
    prepare|serve|check) ;;
    *) echo 'Usage: bash launcher/run_rorqual.sh [prepare|serve|check]' >&2; exit 2 ;;
esac

# sbatch runs a spool copy; interactive use can resolve the original script.
if [[ -z "${V4_ROOT:-}" ]]; then
    V4_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
    if [[ ! -f "$V4_ROOT/launcher/start_pruning_server.py" ]]; then
        V4_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
    fi
fi
cd "$V4_ROOT"
[[ -f FreeToken/pyproject.toml && -f deployment/config.json && -f launcher/start_pruning_server.py ]] || {
    echo 'Run from the repository root, or set V4_ROOT to its absolute path.' >&2; exit 2;
}
export V4_ROOT="$PWD"
export V4_WORK="${V4_WORK:-${SCRATCH:-$HOME/scratch}/freetoken-easyep}"
export V4_ENV="${V4_ENV:-$V4_WORK/venv}"
export V4_MODEL_DIR="${V4_MODEL_DIR:-$V4_WORK/models/DeepSeek-V4-Flash-0731}"
export V4_PORT="${V4_PORT:-1919}"

# Set V4_MODULES=loaded to use a compatible environment already loaded by you.
if [[ "${V4_MODULES:-}" != loaded ]]; then
    command -v module >/dev/null || { echo 'Load the cluster module environment first.' >&2; exit 2; }
    read -r -a v4_modules <<< "${V4_MODULES:-StdEnv/2023 cuda/13.0 python/3.12}"
    module load "${v4_modules[@]}" || {
        echo 'CUDA 13/Python 3.12 modules are required. Check module spider cuda, then set V4_MODULES.' >&2; exit 2;
    }
fi
set -u
command -v nvcc >/dev/null || { echo 'CUDA 13 nvcc is required.' >&2; exit 2; }
nvcc --version | grep -q 'release 13\.' || { echo 'This FreeToken build requires CUDA 13.' >&2; exit 2; }
export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export UV_CACHE_DIR="$V4_WORK/uv-cache"
mkdir -p "$V4_WORK"

if [[ "$mode" == prepare ]]; then
    # Run on a node with Internet access before starting the inference job.
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    mkdir -p "$V4_WORK/tools" "$V4_MODEL_DIR"
    if [[ ! -x "$V4_WORK/tools/uv" ]]; then
        curl --fail --location --retry 3 https://astral.sh/uv/0.12.10/install.sh -o "$V4_WORK/install-uv.sh"
        UV_INSTALL_DIR="$V4_WORK/tools" UV_NO_MODIFY_PATH=1 sh "$V4_WORK/install-uv.sh"
    fi
    [[ -x "$V4_ENV/bin/python" ]] || "$V4_WORK/tools/uv" venv --python python3.12 "$V4_ENV"
    "$V4_WORK/tools/uv" pip install --python "$V4_ENV/bin/python" \
        --extra-index-url https://docs.sglang.io/whl/cu130 --index-strategy unsafe-best-match \
        -r deployment/requirements-5090.txt
    "$V4_WORK/tools/uv" pip install --python "$V4_ENV/bin/python" --no-build-isolation --no-deps -e './FreeToken[accel]'
    "$V4_WORK/tools/uv" pip check --python "$V4_ENV/bin/python"
    "$V4_ENV/bin/python" - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import snapshot_download
config = json.loads((Path(os.environ['V4_ROOT']) / 'deployment/config.json').read_text())
snapshot_download(repo_id=config['model_repo'], revision=config['model_revision'],
                  local_dir=os.environ['V4_MODEL_DIR'], max_workers=4)
print('Prepared pinned checkpoint:', config['model_revision'])
PY
    echo 'Preparation finished. Run bash launcher/run_rorqual.sh inside your allocation, or sbatch launcher/run_rorqual.sh.'
    exit 0
fi

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch, or run this inside an existing Slurm allocation.' >&2; exit 2; }
[[ -x "$V4_ENV/bin/python" ]] || { echo 'Run bash launcher/run_rorqual.sh prepare first, or set V4_ENV.' >&2; exit 2; }
export PATH="$V4_ENV/bin:$PATH"
export PYTHONPATH="$V4_ROOT/FreeToken/python${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$V4_WORK/cache"
export TRITON_CACHE_DIR="$V4_WORK/cache/triton-h100"
export V4_RUN_DIR="${V4_RUN_DIR:-$V4_WORK/runs/${SLURM_JOB_ID}-$(date +%Y%m%d-%H%M%S)-$$}"
mkdir -p "$V4_RUN_DIR"
echo "Job $SLURM_JOB_ID; output: $V4_RUN_DIR"

# srun enters the allocation from a login shell; an existing compute step can run directly.
# Never replace CUDA_VISIBLE_DEVICES: Slurm owns the MIG assignment.
run_in_allocation() {
    case "${SLURM_STEP_ID:-${SLURM_STEPID:-}}" in
        ''|batch|extern) exec srun --ntasks=1 "$@" ;;
        *) exec "$@" ;;
    esac
}
if [[ "$mode" == check ]]; then
    run_in_allocation "$V4_ENV/bin/python" "$V4_ROOT/launcher/check_slurm.py"
fi
run_in_allocation bash -c '
    set -e
    "$V4_ENV/bin/python" "$V4_ROOT/launcher/check_slurm.py"
    exec "$V4_ENV/bin/python" -u "$V4_ROOT/launcher/start_pruning_server.py" statistics \
        --model-dir "$V4_MODEL_DIR" --artifact "$V4_RUN_DIR/statistics.json" \
        --trace-decode --inherit-gpu --max-seq-len 16384 --chunk-tokens 256 \
        --cache-slots 512 --port "$V4_PORT"
'
