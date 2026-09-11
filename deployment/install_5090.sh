#!/bin/bash
# Invoked by the supervised rebuild worker from the transferred project root.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export MAX_JOBS="${V4_BUILD_JOBS:-12}"
mkdir -p .rebuild .tools
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl git build-essential python3-dev pkg-config libnuma-dev ninja-build unzip
if ! apt-cache show cuda-toolkit-13-0 >/dev/null 2>&1; then
    source /etc/os-release
    case "$ID:$VERSION_ID" in
        ubuntu:22.04) cuda_distro=ubuntu2204 ;;
        ubuntu:24.04) cuda_distro=ubuntu2404 ;;
        *) echo 'Use an Ubuntu 22.04 or 24.04 image for this CUDA 13 profile.' >&2; exit 2 ;;
    esac
    curl --fail --location --retry 3 "https://developer.download.nvidia.com/compute/cuda/repos/$cuda_distro/x86_64/cuda-keyring_1.1-1_all.deb" -o .rebuild/cuda-keyring.deb
    dpkg -i .rebuild/cuda-keyring.deb
    apt-get update
fi
cuda_package="$(python3 -c 'import json; print(json.load(open("deployment/config.json"))["cuda_package"])')"
apt-get --simulate install --no-install-recommends "$cuda_package" > .rebuild/cuda-install-plan.txt
python3 - <<'PY'
from pathlib import Path
names=[line.split()[1] for line in Path('.rebuild/cuda-install-plan.txt').read_text().splitlines() if line.startswith(('Inst ', 'Remv '))]
blocked=[name for name in names if name.startswith(('cuda-drivers', 'nvidia-driver-', 'libcuda', 'nvidia-dkms', 'nvidia-kernel', 'cuda-compat'))]
if blocked: raise SystemExit('Refusing host-driver changes: '+', '.join(blocked))
PY
apt-get install -y --no-install-recommends "$cuda_package"
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$PWD/.tools:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
uv_version="$(python3 -c 'import json; print(json.load(open("deployment/config.json"))["uv_version"])')"
if [[ ! -x .tools/uv ]] || [[ "$(.tools/uv --version)" != "uv $uv_version"* ]]; then
    curl --fail --location --retry 3 "https://astral.sh/uv/$uv_version/install.sh" -o .rebuild/install-uv.sh
    UV_INSTALL_DIR="$PWD/.tools" UV_NO_MODIFY_PATH=1 sh .rebuild/install-uv.sh
fi
if [[ ! -x .venv/bin/python ]]; then
    .tools/uv venv --python 3.12 .venv
fi
.tools/uv pip install --python .venv/bin/python -c deployment/requirements-5090.txt torch setuptools wheel ninja
.tools/uv pip install --python .venv/bin/python --extra-index-url https://docs.sglang.io/whl/cu130 --index-strategy unsafe-best-match -r deployment/requirements-5090.txt
.tools/uv pip install --python .venv/bin/python --no-build-isolation --no-deps -e './FreeToken[accel]'
.tools/uv pip check --python .venv/bin/python
.tools/uv pip freeze --python .venv/bin/python > .rebuild/installed-packages.txt
nvcc --version
