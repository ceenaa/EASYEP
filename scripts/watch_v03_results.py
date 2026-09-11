#!/usr/bin/env python3
"""Monitor the authorized V03 job and keep verified local result snapshots."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from snapshot_v03_results import HOST, OPTIONS, ROOT


RUN = ROOT / 'runs/full-model-v03'
REMOTE_STATUS = r'''
from collections import Counter
from datetime import datetime, timezone
import json, subprocess
from pathlib import Path
root = Path('/workspace/prunungdsk4/results/full-baseline-v03')
records = [json.loads(p.read_text()) for p in (root / 'cases').glob('*.json')]
jobs = {}
for name in ('v4_baseline', 'v4_evaluate'):
    response = subprocess.run(['supervisorctl', 'status', name], capture_output=True, text=True)
    fields = response.stdout.split()
    jobs[name] = {'state': fields[1] if len(fields) >= 2 and fields[0] == name else 'UNKNOWN',
                  'description': response.stdout.strip(), 'returncode': response.returncode}
log = root.parent / 'baseline-server.log'
with log.open('rb') as handle:
    handle.seek(max(0, log.stat().st_size - 4096))
    lines = handle.read().decode(errors='replace').splitlines()
print(json.dumps({'at': datetime.now(timezone.utc).isoformat(), 'jobs': jobs,
    'finalized': len(records), 'counts': dict(Counter(r['status'] for r in records)),
    'last_server_event': lines[-1] if lines else None,
    'latest_case': max(({'id': r['id'], 'status': r['status'], 'finished_at': r['finished_at'],
                        'error': r.get('error')} for r in records),
                       key=lambda r: r['finished_at'], default=None)}))
'''


def emit(event, **fields):
    text = json.dumps({'at': datetime.now(timezone.utc).isoformat(), 'event': event, **fields})
    print(text, flush=True)
    with (RUN / 'watcher.log').open('a') as handle:
        handle.write(text + '\n')


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    # A process lock prevents two watchers from duplicating snapshots.
    import fcntl
    lock = (RUN / 'watcher.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('A V03 results watcher is already running')
    last_snapshot = 0
    latest = RUN / 'latest-snapshot.json'
    if latest.exists():
        previous = json.loads(latest.read_text())['verification']
        last_snapshot = previous['verified_finished_cases'] + len(previous['errors'])
    emit('watcher_started', pid=os.getpid(), snapshot_every=10,
         last_snapshot_finalized=last_snapshot)
    terminal_states = {'STOPPED', 'EXITED', 'FATAL'}
    while True:
        try:
            response = subprocess.run(['ssh', '-p', '40579', *OPTIONS,
                '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2', HOST,
                'python3 -c ' + shlex.quote(REMOTE_STATUS)],
                check=True, capture_output=True, text=True, timeout=50)
            status = json.loads(response.stdout)
            emit('remote_status', **status)
            state = {'watcher_pid': os.getpid(), 'remote': status,
                     'last_snapshot_finalized': last_snapshot}
            temporary = RUN / 'watcher-state.json.tmp'
            temporary.write_text(json.dumps(state, indent=2) + '\n')
            temporary.replace(RUN / 'watcher-state.json')
            count = status['finalized']
            stopped = status['jobs']['v4_evaluate']['state'] in terminal_states
            needs_snapshot = count > last_snapshot and (
                last_snapshot == 0 or count // 10 > last_snapshot // 10 or count == 100 or stopped)
            if needs_snapshot:
                emit('snapshot_started', finalized=count)
                result = subprocess.run([sys.executable, str(ROOT / 'scripts/snapshot_v03_results.py')],
                                        capture_output=True, text=True)
                if result.returncode:
                    emit('snapshot_failed', returncode=result.returncode,
                         stdout=result.stdout, stderr=result.stderr)
                else:
                    verification = json.loads(latest.read_text())['verification']
                    last_snapshot = verification['verified_finished_cases'] + len(verification['errors'])
                    emit('snapshot_verified', finalized=last_snapshot, verification=verification)
            if stopped and count <= last_snapshot:
                emit('evaluation_terminal', status=status, last_snapshot_finalized=last_snapshot)
                return
        except (subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
            # An observation failure is not evidence that inference has stopped.
            emit('observation_error', error=f'{type(error).__name__}: {error}')
        time.sleep(45)


if __name__ == '__main__':
    main()
