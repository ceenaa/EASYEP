#!/bin/bash
# Compatibility entrypoint. The complete, gated implementation lives in
# run_experiment.sbatch; keep this short name for existing submission commands.
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

if [[ -n ${EASYEP_REPO:-} ]]; then
  repo=$EASYEP_REPO
elif [[ -n ${SLURM_SUBMIT_DIR:-} && -f "$SLURM_SUBMIT_DIR/v4/run_experiment.sbatch" ]]; then
  repo=$SLURM_SUBMIT_DIR
elif [[ -n ${SLURM_SUBMIT_DIR:-} && -f "$SLURM_SUBMIT_DIR/run_experiment.sbatch" && \
        $(basename "$SLURM_SUBMIT_DIR") == v4 ]]; then
  repo=$(cd "$SLURM_SUBMIT_DIR/.." && pwd -P)
else
  script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
  repo=$(cd "$script_dir/.." && pwd -P)
fi

export EASYEP_REPO=$repo
exec bash "$repo/v4/run_experiment.sbatch"
