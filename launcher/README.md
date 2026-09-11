# Run FreeToken expert monitoring on Rorqual

`launcher/run_rorqual.sh` targets one H100 `3g.40gb` MIG slice, eight CPU cores, 250G host RAM, and an eight-hour job under `rrg-tayebi_gpu`. Slurm must grant this resource combination; 40 GB refers to GPU memory, separately from the host RAM request.

The script uses our existing CPU-offload configuration and records live routed experts during both input processing and generation. It launches a server; it does not automatically run evaluation or apply a mask. The MIG launcher inherits Slurm's device visibility without passing FreeToken's physical `--gpu` selector.

## Prepare once on the cluster

Keep this folder inside the full repository: it uses `FreeToken/`, `scripts/`, and `deployment/` alongside it. Place the repository on Rorqual and run from its root, on a node with Internet access:

```bash
bash launcher/run_rorqual.sh prepare
```

This uses user-owned storage under `$SCRATCH/freetoken-easyep`, installs the pinned Python packages from the successful 5090 setup into a new environment, builds this FreeToken checkout, and downloads the pinned checkpoint (approximately 167 GB). Allow at least 240 GB of free storage and sufficient scratch quota. It does not install system packages or drivers.

Defaults are the module names `StdEnv/2023 cuda/13.0 python/3.12`. Their availability has not been checked in this cluster session. If needed, use `module spider cuda` and load compatible compiler/CUDA 13/Python 3.12 modules, then set `V4_MODULES=loaded`. The driver must be 580 or newer. A CUDA 12-only environment cannot use the pinned CUDA 13 build as-is.

To reuse an existing compatible environment and checkpoint, skip preparation and set their absolute paths:

```bash
export V4_ENV=/absolute/path/to/venv
export V4_MODEL_DIR=/absolute/path/to/DeepSeek-V4-Flash-0731
```

## Use the allocation you already have

From a shell belonging to that allocation:

```bash
bash launcher/run_rorqual.sh check
bash launcher/run_rorqual.sh
```

The script uses `srun` to enter the allocated compute resources, or runs directly if you are already inside an interactive compute step. The existing allocation's memory and time limits apply; the `#SBATCH` lines do not enlarge an existing job.

For a new batch allocation, run from the repository root:

```bash
sbatch launcher/run_rorqual.sh
```

Override the account or time through normal `sbatch` flags. For another 40 GB MIG profile, override `--gres` with the exact name shown by `sinfo -o '%G'`; the checked Rorqual configuration exposed `nvidia_h100_80gb_hbm3_3g.40gb`.

## Output and access

The script prints its output directory under `$SCRATCH/freetoken-easyep/runs/`. It stores the hardware report, `statistics.json`, and the corresponding token trace there; these survive the job's end. The batch console log is `slurm-v4-experts-JOB_ID.log` in the submission directory. Server readiness appears in that log. `V4_PORT` defaults to 1919, and the server binds to the compute node's loopback interface.

Run clients inside the same allocation using `http://127.0.0.1:1919`. Two-file evaluation still requires the original ZIP, CSV, and saved selection described in the root README. The existing `scripts/run_expert_smoke.py` and analyzer can then use this server. Jobs stop at their Slurm time limit.

Local checks passed: 70 pruning tests and three Slurm tests, including MIG device selection, login/compute-shell dispatch, and allocation/cgroup RAM limits. Five CUDA tests were skipped on the Mac. The 40 GB MIG / 250 GB RAM configuration has not yet been tested with the full checkpoint. The earlier successful 5090 instance had about 513 GiB of host RAM, so the smaller host allocation remains a configuration to validate.

GPU profile reference: [SHARCNET's Rorqual MIG examples](https://helpwiki.sharcnet.ca/wiki/images/a/ac/Migration_webinar_2025.pdf).
