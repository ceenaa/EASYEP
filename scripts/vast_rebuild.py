#!/usr/bin/env python3
"""Deploy the current local project to an authorized rented RTX 5090 over SSH."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '__pycache__', '.pytest_cache', '.ruff_cache', '.venv', 'build', 'dist', 'node_modules'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_files(root=ROOT):
    paths, commits = set(), {}
    for name in ('FreeToken', 'EASYEP'):
        repo = root / name
        if not repo.is_dir():
            raise ValueError('Missing local repository: ' + str(repo))
        result = subprocess.run(['git', '-C', str(repo), 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], capture_output=True)
        if result.returncode == 0:
            paths.update(repo / p for p in result.stdout.decode().split('\0') if p)
            commits[name] = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        else:
            for parent, dirs, names in os.walk(repo):
                dirs[:] = [d for d in dirs if d not in EXCLUDED and not d.endswith('.egg-info')]
                paths.update(Path(parent) / p for p in names)
            commits[name] = None
    for folder in ('scripts', 'deployment', 'launcher', 'runs/easyep-smoke/input'):
        paths.update(p for p in (root / folder).rglob('*') if p.is_file())
    paths.update(root.glob('*.md'))
    paths.update(root / name for name in ('setup_vast_5090.sh', 'v03_extended.txt', 'metadata_full.csv', 'primevul_aligned_100_samples 2.zip'))
    result = []
    for p in paths:
        relative = p.relative_to(root)
        if any(part in EXCLUDED or part.endswith('.egg-info') for part in relative.parts):
            continue
        if p.suffix in ('.pyc', '.so', '.o', '.a') or p.name in ('.DS_Store', '.env'):
            continue
        if p.is_symlink():
            raise ValueError('Source bundle will not follow a symlink: ' + str(relative))
        if p.is_file():
            result.append(p)
    for name in ('v03_extended.txt', 'metadata_full.csv', 'primevul_aligned_100_samples 2.zip', 'deployment/config.json'):
        if root / name not in result:
            raise ValueError('Required input is missing: ' + name)
    return sorted(result), commits


def build_bundle(root=ROOT):
    paths, commits = source_files(root)
    files = {str(p.relative_to(root)): {'sha256': digest(p), 'mode': p.stat().st_mode & 0o777} for p in paths}
    identity = {'files': files, 'commits': commits}
    bundle_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    folder = root / 'dist/vast-rebuild' / bundle_id
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {**identity, 'bundle_id': bundle_id}
    manifest_path = folder / 'deployment-bundle.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    archive = folder / 'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as handle:
        for path in paths:
            handle.add(path, arcname=str(path.relative_to(root)), recursive=False)
        handle.add(manifest_path, arcname='deployment-bundle.json', recursive=False)
    return archive, manifest


def valid_remote_path(value):
    if not re.fullmatch(r'/workspace/[A-Za-z0-9_./-]+', value) or '..' in PurePosixPath(value).parts:
        raise argparse.ArgumentTypeError('Use a project path under /workspace with letters, digits, _, -, / or .')
    if str(PurePosixPath(value)) == '/workspace':
        raise argparse.ArgumentTypeError('Choose a project subdirectory, not /workspace itself')
    return str(PurePosixPath(value))


def checked_members(handle, destination):
    destination = Path(destination).resolve()
    members = handle.getmembers()
    for member in members:
        name = PurePosixPath(member.name)
        if name.is_absolute() or '..' in name.parts or not (member.isfile() or member.isdir()):
            raise ValueError('Unsafe archive member: ' + member.name)
        if not (destination / member.name).resolve().is_relative_to(destination):
            raise ValueError('Archive member escapes its destination')
    return members


class Connection:
    def __init__(self, args):
        self.args = args
        self.target = 'root@' + args.host
        known = str(args.known_hosts.resolve()).replace('\\', '\\\\').replace('"', '\\"')
        self.options = ['-i', str(args.key.expanduser().resolve()), '-o', 'BatchMode=yes',
                        '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4',
                        '-o', 'StrictHostKeyChecking=accept-new', '-o', f'UserKnownHostsFile="{known}"']

    def ssh(self, command, source=None, capture=False):
        return subprocess.run(['ssh', '-p', str(self.args.ssh_port), *self.options, self.target, command],
                              input=source, text=True, check=True, capture_output=capture)

    def python(self, source):
        result = self.ssh('python3 -', source, capture=True)
        lines = [line[len('V4_JSON:'):] for line in result.stdout.splitlines() if line.startswith('V4_JSON:')]
        if not lines:
            raise RuntimeError('No structured reply from remote helper: ' + result.stdout[-1000:])
        return json.loads(lines[-1])

    def copy(self, source, destination):
        subprocess.run(['scp', '-P', str(self.args.ssh_port), *self.options, str(source), str(destination)], check=True)


def check_remote(connection, directory, gpu):
    config = json.loads((ROOT / 'deployment/config.json').read_text())
    module = (ROOT / 'scripts/vast_remote.py').read_text()
    source = f'''import json
from pathlib import Path
namespace={{'__name__':'rebuild_probe'}}
exec({module!r},namespace)
report=namespace['probe']({config!r},{directory!r},{gpu!r})
guide=Path('/etc/vast-agents-guide.md')
report['agent_guide']=guide.read_text() if guide.exists() else None
print('V4_JSON:'+json.dumps(report))
'''
    return connection.python(source)


def upload(connection, archive, manifest, directory):
    remote_archive = '/tmp/prunungdsk4-' + manifest['bundle_id'] + '.tar.gz'
    connection.copy(archive, connection.target + ':' + remote_archive)
    source = f'''import hashlib,json,tarfile,tempfile,shutil
from pathlib import Path,PurePosixPath
archive=Path({remote_archive!r});root=Path({directory!r})
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={digest(archive)!r},'Upload checksum mismatch'
root.parent.mkdir(parents=True,exist_ok=True)
stage=Path(tempfile.mkdtemp(prefix='.v4-upload-',dir=root.parent))
try:
    with tarfile.open(archive) as handle:
        members=handle.getmembers()
        for m in members:
            assert not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts
            assert m.isfile() or m.isdir(),'No archive links or special files'
        handle.extractall(stage,members=members)
    doc=json.loads((stage/'deployment-bundle.json').read_text())
    assert doc['bundle_id']=={manifest['bundle_id']!r}
    for name,record in doc['files'].items():
        assert hashlib.sha256((stage/name).read_bytes()).hexdigest()==record['sha256'],name
    if root.exists():
        old=root/'deployment-bundle.json'
        if not old.exists() or json.loads(old.read_text())['bundle_id']!=doc['bundle_id']:
            raise RuntimeError('Remote project already contains different code. Choose another --remote-dir; nothing was overwritten.')
        for name,record in doc['files'].items():
            p=root/name
            if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:
                raise RuntimeError('Remote edits detected: '+name+'. Save them locally before choosing a fresh directory.')
    else:
        (stage/'.rebuild').mkdir()
        shutil.copy2(archive,stage/'.rebuild/source.tar.gz')
        stage.rename(root)
    print('V4_JSON:'+json.dumps({{'uploaded':True,'files':len(doc['files']),'bundle_id':doc['bundle_id'],'root':str(root)}}))
finally:
    if stage.exists():shutil.rmtree(stage)
'''
    return connection.python(source)


def remote_status(connection, directory):
    return connection.python(f'''import json
from pathlib import Path
p=Path({directory!r})/'.rebuild/status.json'
d=json.loads(p.read_text()) if p.exists() else {{'status':'missing'}}
print('V4_JSON:'+json.dumps(d))
''')


def fetch(connection, directory, status, destination):
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'server-status.json').write_text(json.dumps(status, indent=2) + '\n')
    if status.get('status') != 'complete':
        try:
            connection.copy(connection.target + ':' + directory + '/.rebuild/worker.log', destination / 'worker.log')
        except subprocess.CalledProcessError:
            pass
        raise RuntimeError('Rebuild is not complete: ' + str(status.get('error') or status.get('status')))
    relative = PurePosixPath(status['evidence'])
    if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'results':
        raise ValueError('Invalid evidence path returned by server')
    tag = status['evidence_sha256'][:12]
    archive = destination / (tag + '-' + relative.name)
    if not archive.exists() or digest(archive) != status['evidence_sha256']:
        connection.copy(connection.target + ':' + directory + '/' + str(relative), archive)
    if digest(archive) != status['evidence_sha256']:
        raise RuntimeError('Evidence download checksum mismatch')
    extracted = destination / ('evidence-' + tag)
    if extracted.exists() and not (extracted / '.extraction-complete').exists():
        raise RuntimeError('An incomplete extraction exists at ' + str(extracted) + '; preserve it and choose a clean destination.')
    if not extracted.exists():
        with tempfile.TemporaryDirectory(prefix='.extract-', dir=destination) as temporary:
            stage = Path(temporary) / 'evidence'
            stage.mkdir()
            with tarfile.open(archive) as handle:
                members = checked_members(handle, stage)
                handle.extractall(stage, members=members)
            (stage / '.extraction-complete').write_text(status['evidence_sha256'] + '\n')
            stage.rename(extracted)
    verification = extracted / 'run/verification.json'
    if status['request']['mode'] == 'smoke':
        if not verification.exists() or not json.loads(verification.read_text()).get('verified'):
            raise RuntimeError('Downloaded evidence lacks successful smoke verification')
    print('Evidence saved: ' + str(extracted), flush=True)
    if (extracted / 'run/SMOKE_TEST.md').exists():
        print('Report: ' + str(extracted / 'run/SMOKE_TEST.md'), flush=True)
    return extracted


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('host', nargs='?', help='Vast SSH IP/hostname (not the web/API URL)')
    p.add_argument('ssh_port', nargs='?', type=int, help='Vast mapped SSH port')
    p.add_argument('--key', type=Path, default=Path.home() / '.ssh/id_ed25519')
    p.add_argument('--known-hosts', type=Path, default=ROOT / '.rebuild-local/known_hosts')
    p.add_argument('--remote-dir', type=valid_remote_path, default='/workspace/prunungdsk4')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--server-port', type=int, default=1919)
    p.add_argument('--mode', choices=('smoke', 'serve', 'prepare'), default='smoke')
    choices = p.add_mutually_exclusive_group()
    choices.add_argument('--check-only', action='store_true', help='Read-only hardware/OS check; no installation or upload')
    choices.add_argument('--bundle-only', action='store_true', help='Build the local transfer archive; no network access')
    choices.add_argument('--upload-only', action='store_true', help='Verify hardware and transfer source; no installation or model jobs')
    choices.add_argument('--fetch-only', action='store_true', help='Download the completed deployment evidence without deploying')
    p.add_argument('--detach', action='store_true', help='Start supervised work and return; rerun the command later to reconnect')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.bundle_only:
        archive, manifest = build_bundle()
        print(json.dumps({'archive': str(archive), 'bundle_id': manifest['bundle_id'], 'files': len(manifest['files']), 'bytes': archive.stat().st_size}, indent=2))
        return 0
    if not args.host or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', args.host):
        raise ValueError('Supply the SSH IP/hostname and SSH port shown by Vast.')
    if args.ssh_port is None or not 1 <= args.ssh_port <= 65535 or not 1 <= args.server_port <= 65535 or args.gpu < 0:
        raise ValueError('Invalid SSH port, server port or GPU index')
    if not args.key.expanduser().is_file():
        raise ValueError('SSH private key not found: ' + str(args.key))
    args.known_hosts.parent.mkdir(parents=True, exist_ok=True)
    connection = Connection(args)
    local = ROOT / 'runs/vast-rebuild' / f'{args.host}-{args.ssh_port}'
    local.mkdir(parents=True, exist_ok=True)
    if args.fetch_only:
        fetch(connection, args.remote_dir, remote_status(connection, args.remote_dir), local)
        return 0
    print('Checking the target instance...', flush=True)
    report = check_remote(connection, args.remote_dir, args.gpu)
    guide = report.pop('agent_guide')
    if guide:
        (local / 'vast-agents-guide.md').write_text(guide)
    (local / 'hardware.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if not report['ok']:
        raise RuntimeError('; '.join(report['errors']))
    if args.check_only:
        return 0
    archive, manifest = build_bundle()
    print(f'Transferring {len(manifest["files"])} current local files ({archive.stat().st_size / 1e6:.1f} MB)...', flush=True)
    print(json.dumps(upload(connection, archive, manifest, args.remote_dir)), flush=True)
    if args.upload_only:
        return 0
    connection.ssh(shlex.join(['python3', args.remote_dir + '/scripts/vast_remote.py', 'register', '--root', args.remote_dir,
                             '--mode', args.mode, '--gpu', str(args.gpu), '--port', str(args.server_port)]))
    print('Setup runs under supervisor and survives an SSH disconnect. Rerun the same command to reconnect.', flush=True)
    if args.detach:
        return 0
    last = None
    while True:
        status = remote_status(connection, args.remote_dir)
        change = (status.get('status'), status.get('stage'))
        if change != last:
            print('Remote status: ' + ' / '.join(str(v) for v in change), flush=True)
            last = change
        if status.get('status') in ('complete', 'failed'):
            fetch(connection, args.remote_dir, status, local)
            break
        if status.get('status') == 'missing':
            raise RuntimeError('Remote status file disappeared; inspect the instance.')
        time.sleep(15)
    print(f'Private API tunnel: ssh -i {shlex.quote(str(args.key))} -p {args.ssh_port} -N -L 8080:127.0.0.1:{args.server_port} {connection.target}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nDisconnected locally; supervised remote work continues. Rerun to reconnect.', file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        print('Existing remote work and evidence are preserved. Correct the reported issue, then rerun.', file=sys.stderr)
        sys.exit(1)
