#!/bin/bash
set -euo pipefail
cd /workspace/prunungdsk4
export DEBIAN_FRONTEND=noninteractive
export UV_CACHE_DIR=/workspace/prunungdsk4/.uv-cache
export MAX_JOBS=12
apt-get update
apt-get --simulate install --no-install-recommends cuda-toolkit-13-0 > results/cuda-install-plan.txt
python3 - <<'PY'
from pathlib import Path
names = [line.split()[1] for line in Path('results/cuda-install-plan.txt').read_text().splitlines() if line.startswith('Inst ')]
assert not any(name.startswith(('cuda-drivers', 'nvidia-driver-', 'libcuda')) for name in names), names
PY
apt-get install -y --no-install-recommends cuda-toolkit-13-0
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version
if [ ! -x .venv/bin/python ]; then
    uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python 'torch>=2.11,<2.12' 'setuptools>=77' wheel ninja pytest 'huggingface_hub>=1.5,<2'
uv pip install --python .venv/bin/python --no-build-isolation -e './FreeToken[accel]'
uv pip freeze --python .venv/bin/python > results/installed-packages.txt
PYTHONPATH=FreeToken/python .venv/bin/python -m pytest --confcutdir FreeToken/tests/pruning FreeToken/tests/pruning -q | tee results/gpu-pruning-tests.txt
.venv/bin/python - <<'PY'
import json, torch
from pathlib import Path
x = torch.ones(256, device='cuda')
assert x.sum().item() == 256
Path('results/setup-complete.json').write_text(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'cuda_smoke_passed': True}, indent=2) + '\n')
PY
