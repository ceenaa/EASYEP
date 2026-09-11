"""Offline checks for source preservation, archive validation and reconnect behavior."""

import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import vast_rebuild as local
import vast_remote as remote


class RebuildTests(unittest.TestCase):
    def fixture(self, root):
        for name in ('FreeToken', 'EASYEP', 'scripts', 'deployment', 'launcher', 'runs/easyep-smoke/input'):
            (root / name).mkdir(parents=True)
        for name in ('FreeToken/model.py', 'EASYEP/pruning.py', 'scripts/run.py', 'setup_vast_5090.sh',
                     'launcher/start_pruning_server.py', 'launcher/run_rorqual.sh',
                     'v03_extended.txt', 'metadata_full.csv', 'primevul_aligned_100_samples 2.zip', 'deployment/config.json'):
            (root / name).write_text('initial\n')

    def test_bundle_captures_edits_and_excludes_private_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            for name in ('FreeToken/.git/private', 'FreeToken/__pycache__/x.pyc', 'FreeToken/.env', 'unrelated.key'):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text('not part of the bundle')
            archive, first = local.build_bundle(root)
            self.assertNotIn('FreeToken/.env', first['files'])
            self.assertNotIn('unrelated.key', first['files'])
            self.assertIn('launcher/start_pruning_server.py', first['files'])
            self.assertIn('launcher/run_rorqual.sh', first['files'])
            self.assertFalse(any('.git/' in n or '__pycache__' in n for n in first['files']))
            (root / 'FreeToken/model.py').write_text('locally patched\n')
            archive, second = local.build_bundle(root)
            self.assertNotEqual(first['bundle_id'], second['bundle_id'])
            with tarfile.open(archive) as handle:
                self.assertEqual(handle.extractfile('FreeToken/model.py').read(), b'locally patched\n')
                self.assertIn('primevul_aligned_100_samples 2.zip', handle.getnames())
            _, repeated = local.build_bundle(root)
            self.assertEqual(second, repeated)

    def test_required_dataset_cannot_be_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / 'metadata_full.csv').unlink()
            with self.assertRaisesRegex(ValueError, 'Required input'):
                local.build_bundle(root)

    def test_archive_rejects_escape_and_links(self):
        for name, kind in (('../outside', tarfile.REGTYPE), ('/outside', tarfile.REGTYPE), ('link', tarfile.SYMTYPE)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                stream = io.BytesIO()
                with tarfile.open(fileobj=stream, mode='w') as handle:
                    info = tarfile.TarInfo(name)
                    info.type = kind
                    if kind == tarfile.SYMTYPE:
                        info.linkname = '/outside'
                    handle.addfile(info)
                stream.seek(0)
                with tarfile.open(fileobj=stream) as handle:
                    with self.assertRaises(ValueError):
                        local.checked_members(handle, Path(tmp) / 'output')

    def test_valid_archive_supports_spaces(self):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w') as handle:
            handle.addfile(tarfile.TarInfo('dataset 2.zip'))
        stream.seek(0)
        with tempfile.TemporaryDirectory() as tmp, tarfile.open(fileobj=stream) as handle:
            self.assertEqual(len(local.checked_members(handle, Path(tmp))), 1)

    def test_remote_directory_validation(self):
        self.assertEqual(local.valid_remote_path('/workspace/my-project'), '/workspace/my-project')
        for value in ('/', '/workspace', '/workspace/../other', '/workspace/a;command', '/workspace/a$(cmd)'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                local.valid_remote_path(value)

    def test_remote_manifest_detects_a_server_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'model.py'
            source.write_text('original')
            remote.save(root / 'deployment-bundle.json', {'bundle_id': 'test', 'files': {'model.py': {'sha256': remote.digest(source)}}})
            remote.verify_bundle(root)
            source.write_text('server edit')
            with self.assertRaisesRegex(ValueError, 'differs from manifest'):
                remote.verify_bundle(root)

    def test_reconnect_does_not_restart_an_active_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup, model = remote.job_names(root)
            request = {'mode': 'smoke', 'gpu': 0, 'port': 1919, 'bundle_id': 'same', 'setup_job': setup, 'model_job': model}
            remote.save(root / '.rebuild/request.json', request)
            with patch.object(remote, 'verify_bundle', return_value={'bundle_id': 'same'}), \
                 patch.object(remote.shutil, 'which', return_value='/usr/bin/supervisorctl'), \
                 patch.object(remote.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, setup + ' RUNNING', '')), \
                 patch.object(remote, 'install_job') as install, patch.object(remote, 'run') as start:
                remote.register(root, 'smoke', 0, 1919)
                install.assert_not_called()
                start.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, 'different settings'):
                    remote.register(root, 'serve', 0, 1919)

    def test_stops_only_the_owned_model_job(self):
        with patch.object(remote.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'owned RUNNING', '')), \
             patch.object(remote, 'run') as command:
            remote.stop_owned_job('owned')
            command.assert_called_once_with(['supervisorctl', 'stop', 'owned'])


if __name__ == '__main__':
    unittest.main()
