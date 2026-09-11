#!/usr/bin/env python3
"""Check the allocated CUDA device and RAM before loading the offloaded V4 model."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from vast_remote import cgroup_paths


def memory_budget(environ, meminfo, groups):
    limit, available = meminfo['MemTotal'], meminfo['MemAvailable']
    per_node = int(environ.get('SLURM_MEM_PER_NODE') or 0)
    per_cpu = int(environ.get('SLURM_MEM_PER_CPU') or 0)
    if per_node:
        limit = min(limit, per_node * 1024**2)
    elif per_cpu:
        limit = min(limit, per_cpu * int(environ.get('SLURM_CPUS_PER_TASK') or 1) * 1024**2)
    for group in groups:
        for name, used_name in (('memory.max', 'memory.current'), ('memory.limit_in_bytes', 'memory.usage_in_bytes')):
            path = group / name
            if path.exists() and (value := path.read_text().strip()).isdigit():
                limit = min(limit, int(value))
                used = int((group / used_name).read_text()) if (group / used_name).exists() else 0
                available = min(available, max(0, int(value) - used))
    return limit, min(limit, available)


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('A Slurm allocation is required.')
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit('Expected exactly one CUDA-visible GPU/MIG device from Slurm.')
    device = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    if 'H100' not in device.name or total < 35 * 1024**3 or free < 30 * 1024**3:
        raise SystemExit(f'Need an available H100 40 GB slice or larger; got {device.name}, {free / 1024**3:.1f} GiB free.')
    meminfo = {key: int(value.split()[0]) * 1024 for key, value in
               (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
               if key in ('MemTotal', 'MemAvailable')}
    limit, available = memory_budget(os.environ, meminfo, cgroup_paths())
    config = json.loads((Path(os.environ['V4_ROOT']) / 'deployment/config.json').read_text())
    if limit < config['min_ram_bytes'] or available < config['min_available_ram_bytes']:
        raise SystemExit(f'Insufficient host RAM: limit {limit / 1e9:.1f} GB, available {available / 1e9:.1f} GB. '
                         'This profile needs at least 240 GB allocated and 190 GB available.')
    driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).splitlines()[0]
    if int(driver.split('.')[0]) < 580:
        raise SystemExit('This CUDA 13 environment requires an NVIDIA driver of 580 or newer.')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', int(os.environ['V4_PORT'])))
    report = {'job_id': os.environ['SLURM_JOB_ID'], 'hostname': socket.gethostname(),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'gpu': device.name, 'vram_bytes': total, 'free_vram_bytes': free,
              'ram_limit_bytes': limit, 'ram_available_bytes': available,
              'driver': driver, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda}
    (Path(os.environ['V4_RUN_DIR']) / 'hardware.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
