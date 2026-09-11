#!/usr/bin/env python3
"""Copy finalized V03 inference evidence, audit it locally, and refresh results.md."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/workspace/prunungdsk4'
HOST = 'root@141.0.85.220'
OPTIONS = ['-i', str(Path.home() / '.ssh/id_ed25519'), '-o', 'BatchMode=yes',
           '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes',
           '-o', 'UserKnownHostsFile=/private/tmp/prunungdsk4-vast-known-hosts']


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = ROOT / 'runs/full-model-v03/snapshots' / stamp
    destination.mkdir(parents=True)
    remote_archive = f'{REMOTE}/results/v03-snapshot-{stamp}.tar.gz'
    # Finalized case records are atomic writes made after their streams close.
    # Read records once; never include a still-growing stream in the audited set.
    code = r'''
from datetime import datetime, timezone
import hashlib, io, json
from pathlib import Path
import subprocess, sys, tarfile

base = Path('/workspace/prunungdsk4')
run = base / 'results/full-baseline-v03'
files = {'run/run.json': (run / 'run.json').read_bytes()}
records = []
for path in sorted((run / 'cases').glob('*.json')):
    data = path.read_bytes()
    record = json.loads(data)
    records.append(record)
    files['run/cases/' + path.name] = data
    events = (run / record['events_file']).resolve()
    assert events.is_relative_to(run.resolve())
    if events.exists():
        files['run/' + record['events_file']] = events.read_bytes()
for source, target in [
    ('results/primevul-v03-prepared.json', 'prepared.json'),
    ('v03_extended.txt', 'v03_extended.txt'),
    ('scripts/evaluate_full_model.py', 'evaluate_full_model.py'),
    ('scripts/run_full_baseline.sh', 'run_full_baseline.sh'),
    ('results/evaluation-v03.log', 'evaluation-v03.log'),
    ('results/baseline-server.log', 'baseline-server.log'),
    ('results/full-model-smoke-v03.json', 'full-model-smoke-v03.json'),
]:
    files[target] = (base / source).read_bytes()
for name in ('v4_baseline', 'v4_evaluate'):
    files[name + '.conf'] = Path('/etc/supervisor/conf.d/' + name + '.conf').read_bytes()
    files[name + '.sh'] = Path('/opt/supervisor-scripts/' + name + '.sh').read_bytes()
status = subprocess.run(['supervisorctl', 'status', 'v4_baseline', 'v4_evaluate'], capture_output=True, text=True)
manifest = {
    'snapshot_at': datetime.now(timezone.utc).isoformat(),
    'finalized_records': len(records),
    'completed_records': sum(r['status'] == 'complete' for r in records),
    'supervisor_status': status.stdout.strip(),
    'supervisor_returncode': status.returncode,
    'sha256': {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
}
files['snapshot.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
with tarfile.open(sys.argv[1], 'w:gz') as archive:
    for name, data in files.items():
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mode = 0o600
        archive.addfile(info, io.BytesIO(data))
print(json.dumps({k: v for k, v in manifest.items() if k != 'sha256'}))
'''
    subprocess.run(['ssh', '-p', '40579', *OPTIONS, HOST,
                    'python3 -c ' + shlex.quote(code) + ' ' + shlex.quote(remote_archive)], check=True)
    archive_path = destination / 'evidence.tar.gz'
    subprocess.run(['scp', '-P', '40579', *OPTIONS, HOST + ':' + remote_archive, str(archive_path)], check=True)
    with tarfile.open(archive_path) as archive:
        archive.extractall(destination, filter='data')
    import hashlib
    manifest = json.loads((destination / 'snapshot.json').read_text())
    for name, digest in manifest['sha256'].items():
        path = (destination / name).resolve()
        assert path.is_relative_to(destination.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    run = destination / 'run'
    (run / 'cases').mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_full_model.py'), 'report',
                    '--prepared', str(destination / 'prepared.json'), '--output-dir', str(run),
                    '--output', str(run / 'results.md')], check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/audit_full_evaluation.py'),
                    '--run-dir', str(run), '--prepared', str(destination / 'prepared.json'),
                    '--prompt', str(ROOT / 'v03_extended.txt'),
                    '--archive', str(ROOT / 'primevul_aligned_100_samples 2.zip'),
                    '--metadata', str(ROOT / 'metadata_full.csv'),
                    '--runner', str(destination / 'evaluate_full_model.py'),
                    '--output', str(destination / 'verification.json'), '--allow-partial'], check=True)
    relative = destination.relative_to(ROOT).as_posix()
    report = (run / 'results.md').read_text()
    report += ('\n## Local verification snapshot\n\n'
               f'Captured {manifest["snapshot_at"]}. Only finalized case records are included; '
               'the current in-flight request is counted as pending.\n\n'
               f'- [Independent verification]({relative}/verification.json).\n'
               f'- [Snapshot manifest and job status]({relative}/snapshot.json).\n'
               f'- [Run settings]({relative}/run/run.json).\n'
               f'- [Exact rendered prompts]({relative}/prepared.json).\n'
               f'- [Server log]({relative}/baseline-server.log).\n'
               f'- Finalized records and streams: `{relative}/run/cases/` and `{relative}/run/streams/`.\n')
    temporary = ROOT / 'results.md.tmp'
    temporary.write_text(report)
    temporary.replace(ROOT / 'results.md')
    (ROOT / 'runs/full-model-v03/latest-snapshot.json').write_text(json.dumps({
        'snapshot': str(destination), 'snapshot_at': manifest['snapshot_at'],
        'verification': json.loads((destination / 'verification.json').read_text())
    }, indent=2) + '\n')
    print('Locally verified snapshot: ' + str(destination))


if __name__ == '__main__':
    main()
