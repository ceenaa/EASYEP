#!/bin/bash
# Short-lived allocation for concentrated EASY-EP debugging.
#
# Submit:
#   sbatch --account=YOUR_ALLOCATION v4/dev_hold.sbatch
#
# Attach after it starts:
#   srun --jobid=JOB_ID --overlap --pty bash -l
#
# The allocation survives an SSH disconnect, but an interactive command running
# in the attached PTY may not. Submit unattended/production work as a normal
# batch job instead. Always scancel the allocation when the debugging block ends.
#SBATCH --job-name=easyep_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=450G
#SBATCH --time=03:00:00
#SBATCH --output=easyep-dev-%j.out

set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "ABORT: submit this file with sbatch; do not execute it directly." >&2
  exit 2
fi

echo "EASY-EP development allocation is ready."
echo "job:  $SLURM_JOB_ID"
echo "node: $(hostname)"
echo
echo "Attach from a Rorqual login node with:"
echo "  srun --jobid=$SLURM_JOB_ID --overlap --pty bash -l"
echo "Or launch a durable rehearsal with v4/run_in_allocation.sh (see README)."
echo
echo "Release it as soon as the debugging block ends:"
echo "  scancel $SLURM_JOB_ID"

# Keep the batch allocation alive independently of any attached SSH/PTTY step.
# A bounded walltime is the final safeguard if the user forgets to cancel it.
while true; do
  sleep 300
done
