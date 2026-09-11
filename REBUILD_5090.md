# Recreate the V4 expert-monitoring setup

Run this on your Mac from the project directory:

```sh
./setup_vast_5090.sh NEW_SSH_HOST NEW_SSH_PORT
```

Use the SSH host and mapped SSH port shown by Vast. Add your existing public SSH key to the new rental first. The script uses `~/.ssh/id_ed25519`; use `--key /path/to/key` for a different key. The private key stays on your Mac.

Keep this project directory, including `FreeToken/`, `EASYEP/`, `scripts/`, `deployment/`, the dataset ZIP/CSV, the smoke selection and the prompt. The `.sh` file is the entry point; its supporting scripts live in this directory.

## Rental requirements

- RTX 5090 with approximately 32 GB VRAM, Linux x86_64, root SSH access.
- Ubuntu **24.04 recommended**; Ubuntu 22.04 is also accepted by the installer. The existing setup was tested on 24.04.
- NVIDIA host driver **580 or newer**. The script installs the CUDA toolkit, not the host driver.
- At least **240 GB allocated host RAM**; **384–512 GB recommended**. The working instance had approximately 513 GiB allocated. Smaller supported allocations have not been tested end to end.
- At least **240 GB free disk** on a fresh instance; provision 300 GB or more for room to work. The checkpoint download is approximately 167 GB.
- An idle GPU and at least 190 GB available RAM before loading. More CPU capacity improves offload speed.

The script checks these values, including visible cgroup limits, before installation. Hardware eligibility is not a guarantee of identical performance on another host.

## Default: complete reasoning smoke test

One command performs these steps:

1. Check the target and save its Vast agent guide and hardware report locally.
2. Build a fresh archive of both repositories, including local code changes, supporting scripts, exact prompt, selected inputs and dataset ZIP/CSV. Git internals, model weights, virtual environments, caches and previous results are excluded.
3. Transfer and verify every source file by SHA-256. Existing remote edits are never silently overwritten.
4. Create a supervised setup job, install CUDA 13.0 and `uv`, and build the transferred FreeToken source in a dedicated Python 3.12 environment. The Python package versions come from the successful RTX 5090 setup.
5. Run the pruning tests. All tests must pass, with no skipped CUDA tests.
6. Download the official `deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint at revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`. Partial downloads can resume; revision and file sizes are checked.
7. Load the full model, enable live expert tracing and native `low` reasoning, and review sample_009 and sample_010 with `v03_extended.txt`. The initial output budget is 14,336 tokens, larger than the first successful manual smoke run's budget.
8. Verify input/reasoning/answer traces, create the heatmap, and download the evidence archive to `runs/vast-rebuild/HOST-PORT/` on your Mac. A false-positive review is reported honestly; label accuracy is not a condition for passing the instrumentation test.

If an eligible second-file response reaches its output limit, the runner attempts an exact-token continuation within the remaining context and records duplicated prefix work separately. Other inference failures are preserved and reported; they are not marked successful.

The model is left loaded on `127.0.0.1:1919`. The script prints an SSH forwarding command for private access. It does not resume the 100-file evaluation, apply a pruning mask, create another rental, or destroy an instance.

## Useful options

```sh
# Read-only eligibility check.
./setup_vast_5090.sh HOST PORT --check-only

# Install, download and run GPU tests, then finish without loading the model.
./setup_vast_5090.sh HOST PORT --mode prepare

# Install and load a normal baseline server without running the two reviews.
./setup_vast_5090.sh HOST PORT --mode serve

# Let supervisor continue remotely while the local command returns.
./setup_vast_5090.sh HOST PORT --detach

# Download completed evidence later without deploying again.
./setup_vast_5090.sh HOST PORT --fetch-only

# Use another GPU, API port, SSH key or clean remote directory.
./setup_vast_5090.sh HOST PORT --gpu 1 --server-port 1920 --key ~/.ssh/another_key --remote-dir /workspace/prunungdsk4-v2
```

Re-running the same command reconnects to an active setup or retrieves a completed run. SSH disconnection does not kill the supervised job. After a reported failure, correct the cause and rerun; previous smoke attempts are kept in separate directories. If you change local source after uploading, choose a new `--remote-dir` or reconcile the existing remote copy first. The script refuses to overwrite a different source version.

`--bundle-only` builds and verifies the local transfer archive without connecting. `--upload-only` checks hardware and transfers source without installing dependencies or starting a model. These are useful when checking a new setup script against an already occupied server.

Remote progress is in `<remote-dir>/.rebuild/status.json` and `worker.log`. Each new project path gets its own supervisor job names. Only those jobs are stopped or restarted; existing Vast management services remain running.

## Source reconciliation

The 2026-09-11 comparison checked 613 source files on the working server: 611 were identical locally; the remaining two (`build_pruning_bundle.py` and `analyze_expert_smoke.py`) were older on the server. All FreeToken runtime changes were already local. The newer local versions were retained. The inventory and comparison are in [runs/server-sync-20260911](runs/server-sync-20260911/).

This deployment packages the current working tree instead of reusing the earlier incomplete `dist/deepseek-v4-easyep-ready.tar.gz` bundle.

The installer follows the [NVIDIA CUDA 13 installation guide](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-installation-guide-linux/index.html) and [uv's Python installation workflow](https://docs.astral.sh/uv/guides/install-python/). Exact CUDA and Python package pins are in `deployment/config.json` and `deployment/requirements-5090.txt`.
