#!/usr/bin/env python3
"""Server-side rebuild worker. Importable with only the system Python library."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import traceback
from urllib.request import urlopen


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def output(command):
    return subprocess.check_output(command, text=True).strip()


def run(command, **kwargs):
    print('+ ' + shlex.join(map(str, command)), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def cgroup_paths():
    roots = {Path('/sys/fs/cgroup'), Path('/sys/fs/cgroup/memory')}
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, relative = line.split(':', 2)
        base = Path('/sys/fs/cgroup/memory') if 'memory' in controllers.split(',') else Path('/sys/fs/cgroup')
        candidate = (base / relative.lstrip('/')).resolve()
        if candidate.is_relative_to(base) and candidate.exists():
            roots.add(candidate)
            roots.update(p for p in candidate.parents if p.is_relative_to(base))
    return sorted(roots)


def probe(config, directory, gpu=0, idle=False, port=None):
    errors = []
    os_release = {}
    for line in Path('/etc/os-release').read_text().splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            os_release[key] = value.strip('"')
    if os_release.get('ID') != 'ubuntu' or os_release.get('VERSION_ID') not in ('22.04', '24.04'):
        errors.append('Use an Ubuntu 22.04 or 24.04 image.')
    if platform.machine() != 'x86_64' or os.geteuid() != 0:
        errors.append('This profile requires Linux x86_64 and root SSH access.')
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable'):
            memory[key] = int(value.split()[0]) * 1024
    limit, available = memory['MemTotal'], memory['MemAvailable']
    cpus = float(len(os.sched_getaffinity(0)))
    limits = {}
    for root in cgroup_paths():
        for name, used_name in (('memory.max', 'memory.current'), ('memory.limit_in_bytes', 'memory.usage_in_bytes')):
            path = root / name
            if path.exists():
                value = path.read_text().strip()
                limits[str(path)] = value
                if value.isdigit() and int(value) < 2**60:
                    limit = min(limit, int(value))
                    used = int((root / used_name).read_text()) if (root / used_name).exists() else 0
                    available = min(available, max(0, int(value) - used))
        cpu = root / 'cpu.max'
        if cpu.exists():
            quota, period = cpu.read_text().split()
            if quota != 'max':
                cpus = min(cpus, int(quota) / int(period))
    rows = list(csv.reader(output(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version', '--format=csv,noheader,nounits']).splitlines()))
    if not 0 <= gpu < len(rows):
        raise ValueError('Selected GPU index is not present')
    name, total, free, driver = [v.strip() for v in rows[gpu]]
    if 'RTX 5090' not in name or int(total) < 31000:
        errors.append('Selected GPU must be an RTX 5090 with approximately 32 GB VRAM.')
    if int(driver.split('.')[0]) < 580:
        errors.append('Host driver must be r580 or newer for this CUDA 13 build; choose another host.')
    if limit < config['min_ram_bytes']:
        errors.append(f'Allocated RAM is below {config["min_ram_bytes"] / 1e9:g} GB.')
    existing = Path(directory)
    while not existing.exists():
        existing = existing.parent
    disk = shutil.disk_usage(existing).free
    weights_exist = (Path(directory) / 'models' / config['model_directory'] / 'model.safetensors.index.json').exists()
    needed_disk = 12_000_000_000 if weights_exist else config['fresh_disk_bytes']
    if disk < needed_disk:
        errors.append(f'Need at least {needed_disk / 1e9:g} GB free disk for this stage.')
    if idle and (int(free) < 29000 or available < config['min_available_ram_bytes']):
        errors.append('Insufficient free GPU/RAM; another model may be loaded. Stop that workload before provisioning.')
    if idle and port:
        try:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', port))
        except OSError:
            errors.append(f'Local server port {port} is already in use.')
    return {'ok': not errors, 'errors': errors, 'os': os_release.get('PRETTY_NAME'),
            'gpu': {'index': gpu, 'name': name, 'vram_mib': int(total), 'free_mib': int(free), 'driver': driver},
            'ram_limit_bytes': limit, 'ram_available_bytes': available, 'cpu_quota': cpus,
            'disk_free_bytes': disk, 'cgroup_limits': limits}


def verify_bundle(root):
    manifest = json.loads((root / 'deployment-bundle.json').read_text())
    for name, record in manifest['files'].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or digest(path) != record['sha256']:
            raise ValueError('Transferred source differs from manifest: ' + name)
    return manifest


def job_names(root):
    suffix = hashlib.sha256(str(root).encode()).hexdigest()[:10]
    return 'v4_rebuild_' + suffix, 'v4_model_' + suffix


def install_job(root, name, command, log, env=None):
    wrappers = Path('/opt/supervisor-scripts')
    wrappers.mkdir(parents=True, exist_ok=True)
    lines = ['#!/bin/bash',
             'if [[ -f /opt/supervisor-scripts/utils/logging.sh ]]; then source /opt/supervisor-scripts/utils/logging.sh; fi',
             'if [[ -f /opt/supervisor-scripts/utils/environment.sh ]]; then source /opt/supervisor-scripts/utils/environment.sh; fi',
             'set -euo pipefail', 'cd ' + shlex.quote(str(root))]
    for key, value in (env or {}).items():
        lines.append('export ' + key + '=' + shlex.quote(str(value)))
    invocation = shlex.join(map(str, command))
    lines += ['if command -v pty >/dev/null 2>&1; then',
              '  pty ' + invocation + ' 2>&1 | tee -a ' + shlex.quote(str(log)),
              'else', '  ' + invocation + ' 2>&1 | tee -a ' + shlex.quote(str(log)), 'fi']
    wrapper = wrappers / (name + '.sh')
    wrapper.write_text('\n'.join(lines) + '\n')
    wrapper.chmod(0o755)
    conf = Path('/etc/supervisor/conf.d')
    conf.mkdir(parents=True, exist_ok=True)
    (conf / (name + '.conf')).write_text(f'''[program:{name}]
command={wrapper}
autostart=false
autorestart=false
startsecs=0
startretries=0
stopasgroup=true
killasgroup=true
stopwaitsecs=60
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
''')
    run(['supervisorctl', 'reread'])
    run(['supervisorctl', 'update', name])


def stop_owned_job(name):
    result = subprocess.run(['supervisorctl', 'status', name], capture_output=True, text=True)
    if any(state in result.stdout for state in ('RUNNING', 'STARTING', 'BACKOFF')):
        run(['supervisorctl', 'stop', name])


def register(root, mode, gpu, port):
    manifest = verify_bundle(root)
    if not shutil.which('supervisorctl'):
        run(['apt-get', 'update'])
        run(['apt-get', 'install', '-y', '--no-install-recommends', 'supervisor'])
    if subprocess.run(['supervisorctl', 'pid'], capture_output=True).returncode:
        run(['supervisord', '-c', '/etc/supervisor/supervisord.conf'])
    setup_job, model_job = job_names(root)
    request = {'mode': mode, 'gpu': gpu, 'port': port, 'bundle_id': manifest['bundle_id'],
               'setup_job': setup_job, 'model_job': model_job}
    req_path, state_path = root / '.rebuild/request.json', root / '.rebuild/status.json'
    current = subprocess.run(['supervisorctl', 'status', setup_job], capture_output=True, text=True).stdout
    if req_path.exists() and 'RUNNING' in current:
        if json.loads(req_path.read_text()) != request:
            raise RuntimeError('Another rebuild is already running with different settings.')
        print('Reconnecting to the existing supervised rebuild.', flush=True)
        return
    if state_path.exists() and req_path.exists():
        state = json.loads(state_path.read_text())
        if state.get('status') == 'complete' and json.loads(req_path.read_text()) == request:
            if mode == 'prepare' or 'RUNNING' in subprocess.run(['supervisorctl', 'status', model_job], capture_output=True, text=True).stdout:
                print('This exact deployment is already complete.', flush=True)
                return
    save(req_path, request)
    save(state_path, {'status': 'queued', 'stage': 'registration', 'request': request})
    install_job(root, setup_job, ['/usr/bin/python3', '-u', root / 'scripts/vast_remote.py', 'worker', '--root', root], root / '.rebuild/worker.log')
    run(['supervisorctl', 'start', setup_job])


def download(root):
    from huggingface_hub import HfApi, snapshot_download
    config = json.loads((root / 'deployment/config.json').read_text())
    repo, revision = config['model_repo'], config['model_revision']
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError('Checkpoint revision mismatch')
    files = [{'name': f.rfilename, 'size': f.size} for f in info.siblings]
    destination = root / 'models' / config['model_directory']
    missing_bytes = sum(f['size'] or 0 for f in files if not (destination / f['name']).is_file()
                        or (destination / f['name']).stat().st_size != f['size'])
    if shutil.disk_usage(root).free < missing_bytes + 12_000_000_000:
        raise RuntimeError(f'Need {missing_bytes / 1e9:.1f} GB for missing checkpoint files plus 12 GB headroom')
    record_path = root / '.rebuild/checkpoint-revision.json'
    if record_path.exists() and json.loads(record_path.read_text())['revision'] != revision:
        raise RuntimeError('Existing model revision differs from the pinned checkpoint')
    save(record_path, {'repo': repo, 'revision': revision, 'files': files})
    snapshot_download(repo_id=repo, revision=revision, local_dir=destination, max_workers=8)
    for record in files:
        path = destination / record['name']
        if not path.is_file() or record['size'] is not None and path.stat().st_size != record['size']:
            raise RuntimeError('Incomplete checkpoint: ' + record['name'])
    save(root / '.rebuild/download-complete.json', {'revision': revision, 'file_sizes_verified': len(files)})


def make_evidence(root, out, attempt):
    path = root / 'results' / f'rebuild-evidence-{attempt:03d}.tar.gz'
    while path.exists():
        attempt += 1
        path = root / 'results' / f'rebuild-evidence-{attempt:03d}.tar.gz'
    with tarfile.open(path, 'w:gz') as archive:
        if out is not None:
            archive.add(out, arcname='run', filter=lambda m: None if '.matplotlib-cache' in m.name else m)
        for item in (root / '.rebuild').iterdir():
            if item.is_file() and item.suffix in ('.json', '.txt', '.log', '.xml'):
                archive.add(item, arcname='provisioning/' + item.name)
        archive.add(root / 'deployment-bundle.json', arcname='deployment-bundle.json')
        source_bundle = root / '.rebuild/source.tar.gz'
        if source_bundle.exists():
            archive.add(source_bundle, arcname='source-bundle.tar.gz')
        config = json.loads((root / 'deployment/config.json').read_text())
        model = root / 'models' / config['model_directory']
        for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                     'generation_config.json', 'model.safetensors.index.json', 'inference/config.json', 'encoding'):
            if (model / name).exists():
                archive.add(model / name, arcname='tokenizer/' + name, filter=lambda m: None if '__pycache__' in m.name else m)
        for folder in ('deployment', 'scripts', 'launcher'):
            archive.add(root / folder, arcname='code/' + folder, filter=lambda m: None if '__pycache__' in m.name else m)
    return path


def worker(root):
    import fcntl
    import xml.etree.ElementTree as ET
    root = root.resolve()
    os.chdir(root)
    lock = (root / '.rebuild/worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads((root / 'deployment/config.json').read_text())
    request = json.loads((root / '.rebuild/request.json').read_text())
    state = {'status': 'running', 'stage': 'preflight', 'request': request, 'started_at': time.time()}
    state_path = root / '.rebuild/status.json'
    def stage(name):
        state.update(stage=name, updated_at=time.time())
        save(state_path, state)
        print('STAGE: ' + name, flush=True)
    try:
        stage('preflight')
        verify_bundle(root)
        stop_owned_job(request['model_job'])
        hardware = probe(config, root, request['gpu'], idle=True, port=request['port'])
        save(root / '.rebuild/hardware.json', hardware)
        if not hardware['ok']:
            raise RuntimeError('; '.join(hardware['errors']))
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=str(request['gpu']), CUDA_HOME='/usr/local/cuda-13.0',
                   PATH=f'{root}/.venv/bin:{root}/.tools:/usr/local/cuda-13.0/bin:' + os.environ['PATH'],
                   LD_LIBRARY_PATH='/usr/local/cuda-13.0/lib64:' + os.environ.get('LD_LIBRARY_PATH', ''),
                   PYTHONPATH=str(root / 'FreeToken/python'), OMP_NUM_THREADS=str(max(1, min(16, int(hardware['cpu_quota'])))),
                   V4_BUILD_JOBS=str(max(1, min(12, int(hardware['cpu_quota'])))),
                   HF_XET_HIGH_PERFORMANCE='1', MPLCONFIGDIR=str(root / '.rebuild/matplotlib'),
                   XDG_CACHE_HOME=str(root / '.rebuild/cache'))
        stage('dependencies')
        environment_stamp = root / '.rebuild/environment-complete.json'
        wanted = digest(root / 'deployment/requirements-5090.txt')
        python = str(root / '.venv/bin/python')
        installed_ok = False
        if Path(python).exists():
            check = '''import importlib.metadata as m,sys
from pathlib import Path
for line in Path(sys.argv[1]).read_text().splitlines():
    if line and not line.startswith('#'):
        name,version=line.split('==',1)
        if m.version(name)!=version:raise SystemExit(1)
'''
            installed_ok = subprocess.run([python, '-c', check, 'deployment/requirements-5090.txt'], capture_output=True).returncode == 0
        if not installed_ok or not environment_stamp.exists() or json.loads(environment_stamp.read_text()).get('requirements_sha256') != wanted:
            run(['bash', 'deployment/install_5090.sh'], env=env)
            save(environment_stamp, {'requirements_sha256': wanted})
        run([str(root / '.tools/uv'), 'pip', 'check', '--python', python], env=env)
        stage('gpu_tests')
        tests_xml = root / '.rebuild/gpu-tests.xml'
        with (root / '.rebuild/gpu-tests.log').open('w') as log:
            run([python, '-m', 'pytest', '--confcutdir', 'FreeToken/tests/pruning', 'FreeToken/tests/pruning', '-q', f'--junitxml={tests_xml}'], env=env, stdout=log, stderr=subprocess.STDOUT)
        tests = ET.parse(tests_xml).findall('.//testcase')
        if len(tests) < 74 or any(t.find('skipped') is not None or t.find('failure') is not None or t.find('error') is not None for t in tests):
            raise RuntimeError('GPU tests must pass without skips; inspect .rebuild/gpu-tests.log')
        if sum('test_cuda' in t.get('classname', '') for t in tests) < 5:
            raise RuntimeError('The five CUDA pruning tests did not execute')
        stage('checkpoint_download')
        run([python, '-u', 'scripts/vast_remote.py', 'download', '--root', root], env=env)
        (root / 'results').mkdir(exist_ok=True)
        attempt = max([0] + [int(p.name.rsplit('-', 1)[1]) for p in (root / 'results').glob('rebuild-smoke-*') if p.is_dir() and p.name.rsplit('-', 1)[1].isdigit()]) + 1
        out = None
        if request['mode'] != 'prepare':
            stage('model_load')
            out = root / 'results' / f'rebuild-smoke-{attempt:03d}'
            out.mkdir()
            model = root / 'models' / config['model_directory']
            phase = 'statistics' if request['mode'] == 'smoke' else 'baseline'
            command = [python, '-u', 'scripts/start_pruning_server.py', phase, '--model-dir', model,
                       '--port', str(request['port']), '--max-seq-len', str(config['max_seq_len']), '--chunk-tokens', '256']
            if phase == 'statistics':
                command += ['--artifact', out / 'statistics.json', '--trace-decode']
            install_job(root, request['model_job'], command, out / 'server.log', {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'OMP_NUM_THREADS')})
            run(['supervisorctl', 'start', request['model_job']])
            endpoint = 'http://127.0.0.1:' + str(request['port'])
            for _ in range(720):
                try:
                    with urlopen(endpoint + '/health', timeout=5) as response:
                        health = json.load(response)
                    if health.get('status') == 'ok':
                        break
                    if health.get('status') == 'error':
                        raise RuntimeError(str(health))
                except OSError:
                    pass
                job = subprocess.run(['supervisorctl', 'status', request['model_job']], capture_output=True, text=True).stdout
                if any(s in job for s in ('EXITED', 'FATAL', 'STOPPED')):
                    raise RuntimeError('Model process stopped; inspect server.log')
                time.sleep(5)
            else:
                raise TimeoutError('Model load did not finish within one hour')
            if request['mode'] == 'smoke':
                stage('reasoning_smoke')
                command = [python, '-u', 'scripts/run_expert_smoke.py', '--model-dir', model,
                    '--output-dir', out, '--statistics', out / 'statistics.json', '--server', endpoint,
                    '--reasoning-effort', config['reasoning_effort'], '--max-tokens', str(config['max_output_tokens'])]
                result = subprocess.run(command, env=env)
                if result.returncode:
                    reviews = json.loads((out / 'reviews.json').read_text())
                    failed = [c for c in reviews['cases'] if c['status'] == 'error']
                    if len(reviews['cases']) != 2 or len(failed) != 1 or failed[0].get('finish_reason') != 'length':
                        raise RuntimeError('Smoke inference failed; original attempt preserved')
                    remaining = config['max_seq_len'] - failed[0]['prompt_tokens'] - failed[0]['usage']['completion_tokens'] - 1
                    if remaining < 256:
                        raise RuntimeError('Review exhausted the context; original attempt preserved')
                    run([python, '-u', 'scripts/continue_expert_smoke.py', '--run-dir', out, '--model-dir', model,
                         '--server', endpoint, '--max-tokens', str(min(7000, remaining))], env=env)
                stage('trace_verification')
                run([python, 'scripts/analyze_expert_smoke.py', '--run-dir', out, '--model-dir', model], env=env)
                run([python, 'scripts/plot_expert_smoke.py', '--run-dir', out], env=env)
                if not json.loads((out / 'verification.json').read_text())['verified']:
                    raise RuntimeError('Trace verification did not pass')
        stage('evidence_archive')
        archive = make_evidence(root, out, attempt)
        state.update(status='complete', stage='complete', finished_at=time.time(),
                     evidence=str(archive.relative_to(root)), evidence_sha256=digest(archive),
                     output=str(out.relative_to(root)) if out else None)
        save(state_path, state)
        print('REBUILD COMPLETE', flush=True)
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}', finished_at=time.time())
        save(state_path, state)
        traceback.print_exc()
        stop_owned_job(request['model_job'])
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('probe', 'register', 'worker', 'download'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=('prepare', 'serve', 'smoke'), default='smoke')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--port', type=int, default=1919)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'register':
        register(root, args.mode, args.gpu, args.port)
    elif args.action == 'worker':
        worker(root)
    elif args.action == 'download':
        download(root)
    else:
        report = probe(json.loads((root / 'deployment/config.json').read_text()), root, args.gpu)
        print(json.dumps(report, indent=2))
        return 0 if report['ok'] else 2


if __name__ == '__main__':
    sys.exit(main())
