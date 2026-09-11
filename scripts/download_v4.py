"""Download one pinned official checkpoint and retain its revision and file inventory."""

import json
from pathlib import Path
import time

from huggingface_hub import HfApi, snapshot_download

root = Path('/workspace/prunungdsk4')
record_path = root / 'results/checkpoint-revision.json'
repo = 'deepseek-ai/DeepSeek-V4-Flash-0731'
if record_path.exists():
    record = json.loads(record_path.read_text())
    assert record['repo'] == repo
else:
    info = HfApi().model_info(repo, files_metadata=True)
    record = {'repo': repo, 'revision': info.sha,
              'files': [{'name': f.rfilename, 'size': f.size,
                         'sha256': getattr(f.lfs, 'sha256', None)} for f in info.siblings]}
    record_path.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'repo': repo, 'revision': record['revision'], 'bytes': sum(f['size'] or 0 for f in record['files'])}), flush=True)
started = time.time()
destination = root / 'models/DeepSeek-V4-Flash-0731'
snapshot_download(repo_id=repo, revision=record['revision'], local_dir=destination, max_workers=8)
for entry in record['files']:
    path = destination / entry['name']
    if not path.is_file() or entry['size'] is not None and path.stat().st_size != entry['size']:
        raise RuntimeError(f'Incomplete checkpoint file: {entry["name"]}')
(root / 'results/download-complete.json').write_text(json.dumps({
    'repo': repo, 'revision': record['revision'], 'elapsed_seconds': time.time() - started,
    'file_sizes_verified': len(record['files']), 'destination': str(destination)}, indent=2) + '\n')
