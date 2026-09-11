#!/bin/bash
# Run EASY-EP inside an existing Slurm allocation, staging the checkpoint once
# on node-local storage so every torchrun stage does not reread it from /project.

set -euo pipefail
umask 077

abort() {
  echo "ABORT: $*" >&2
  exit 2
}

[[ -n ${SLURM_JOB_ID:-} ]] || abort "run this helper inside a Slurm allocation"
[[ -n ${SLURM_TMPDIR:-} ]] || abort "SLURM_TMPDIR is unavailable"

for name in EASYEP_REPO EASYEP_VENV EASYEP_CODE_DIR EASYEP_CHECKPOINT \
            EASYEP_DATA_ROOT EASYEP_RESULTS_ROOT EASYEP_CONFIG RUN_ID RESUME; do
  [[ -n ${!name:-} ]] || abort "export $name explicitly"
done
[[ $RESUME == 0 || $RESUME == 1 ]] || abort "RESUME must be exactly 0 or 1"

repo=$(cd -P "$EASYEP_REPO" && pwd -P)
source_checkpoint=$(cd -P "$EASYEP_CHECKPOINT" && pwd -P)
source_snapshot=$(cd -P "$EASYEP_CODE_DIR/.." && pwd -P)
local_checkpoint=${EASYEP_LOCAL_CHECKPOINT:-$SLURM_TMPDIR/easyep-checkpoint}

[[ $local_checkpoint == /* ]] || abort "EASYEP_LOCAL_CHECKPOINT must be absolute"
case "$local_checkpoint/" in
  "$SLURM_TMPDIR"/*) ;;
  *) abort "local checkpoint must be under SLURM_TMPDIR: $SLURM_TMPDIR" ;;
esac

require_file() {
  [[ -f $1 ]] || abort "required file not found: $1"
}
require_file "$repo/v4/run_experiment.sbatch"
require_file "$repo/v4/checkpoint_provenance.py"
require_file "$EASYEP_VENV/bin/python"
require_file "$source_checkpoint/EASYEP_CHECKPOINT_PROVENANCE.json"

verify_checkpoint() {
  local checkpoint=$1
  "$EASYEP_VENV/bin/python" "$repo/v4/checkpoint_provenance.py" verify \
    --checkpoint "$checkpoint" \
    --manifest "$checkpoint/EASYEP_CHECKPOINT_PROVENANCE.json" \
    --config "$EASYEP_CONFIG" --source-snapshot "$source_snapshot" \
    --model-id "${MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash-0731}" \
    --model-revision "${MODEL_REVISION:-9e165c30e2704aec5d9d593cce3eebd58bbef1cb}" \
    --inference-revision "${INFERENCE_REVISION:-9e165c30e2704aec5d9d593cce3eebd58bbef1cb}" \
    --nproc "${NPROC_PER_NODE:-4}"
}

command -v flock >/dev/null || abort "flock is required"
exec {stage_lock_fd}>"$SLURM_TMPDIR/easyep-checkpoint-stage.lock"
flock "$stage_lock_fd"

if [[ $source_checkpoint == "$local_checkpoint" ]]; then
  echo "checkpoint already uses node-local storage: $local_checkpoint"
  verify_checkpoint "$local_checkpoint"
elif [[ -d $local_checkpoint ]]; then
  echo "verifying existing node-local checkpoint: $local_checkpoint"
  verify_checkpoint "$local_checkpoint"
else
  partial_checkpoint="${local_checkpoint}.partial.$$"
  [[ ! -e $partial_checkpoint ]] || abort "partial staging path exists: $partial_checkpoint"
  cleanup_partial_checkpoint() {
    [[ ! -d $partial_checkpoint ]] || rm -rf -- "$partial_checkpoint"
  }
  trap cleanup_partial_checkpoint EXIT
  echo "staging checkpoint to node-local storage: $local_checkpoint"
  mkdir "$partial_checkpoint"
  cp -a "$source_checkpoint/." "$partial_checkpoint/"
  verify_checkpoint "$partial_checkpoint"
  mv "$partial_checkpoint" "$local_checkpoint"
  trap - EXIT
fi

export EASYEP_CHECKPOINT=$local_checkpoint
echo "launching RUN_ID=$RUN_ID with checkpoint $EASYEP_CHECKPOINT"
exec bash "$repo/v4/run_experiment.sbatch"
