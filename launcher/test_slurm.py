import os
import subprocess
import tempfile
from pathlib import Path
import unittest

from check_slurm import memory_budget


class SlurmMemoryTests(unittest.TestCase):
    def test_allocation_caps_shared_node_memory(self):
        mem = {'MemTotal': 512 * 1024**3, 'MemAvailable': 400 * 1024**3}
        limit, available = memory_budget({'SLURM_MEM_PER_NODE': '256000'}, mem, [])
        self.assertEqual((limit, available), (250 * 1024**3, 250 * 1024**3))
        limit, _ = memory_budget({'SLURM_MEM_PER_CPU': '8000', 'SLURM_CPUS_PER_TASK': '8'}, mem, [])
        self.assertLess(limit, 240_000_000_000)

    def test_cgroup_parent_limits_and_usage_are_respected(self):
        with tempfile.TemporaryDirectory() as directory:
            group = Path(directory)
            (group / 'memory.max').write_text(str(230_000_000_000))
            (group / 'memory.current').write_text(str(50_000_000_000))
            limit, available = memory_budget({'SLURM_MEM_PER_NODE': '256000'},
                {'MemTotal': 512_000_000_000, 'MemAvailable': 400_000_000_000}, [group])
            self.assertEqual((limit, available), (230_000_000_000, 180_000_000_000))


class SlurmLaunchTests(unittest.TestCase):
    def test_login_and_compute_shells_preserve_mig_assignment(self):
        root = Path(__file__).resolve().parents[1]
        for step, mode in [('', 'serve'), ('batch', 'serve'), ('0', 'serve'), ('0', 'check')]:
            with self.subTest(step=step, mode=mode), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                fake_bin = work / 'bin'
                fake_env = work / 'fake env'
                fake_bin.mkdir()
                (fake_env / 'bin').mkdir(parents=True)
                programs = {
                    fake_bin / 'nvcc': 'echo "Cuda compilation tools, release 13.0"\n',
                    fake_bin / 'readlink': 'echo "$2"\n',
                    fake_bin / 'srun': 'echo SRUN >> "$CAPTURE"\nshift\nexec "$@"\n',
                    fake_env / 'bin/python': 'echo "PYTHON $*" >> "$CAPTURE"\necho "$CUDA_VISIBLE_DEVICES" >> "$CAPTURE"\n',
                }
                for path, body in programs.items():
                    path.write_text('#!/bin/sh\n' + body)
                    path.chmod(0o755)
                env = dict(os.environ, V4_WORK=str(work),
                    V4_ENV=str(fake_env), V4_MODULES='loaded', SLURM_JOB_ID='12345',
                    SLURM_STEP_ID=step, SLURM_STEPID='', CUDA_VISIBLE_DEVICES='MIG-owned-slice',
                    CAPTURE=str(work / 'calls'), PATH=str(fake_bin) + os.pathsep + os.environ['PATH'])
                env.pop('V4_ROOT', None)
                script = root / 'launcher/run_rorqual.sh'
                if step == 'batch':
                    spool = work / 'spool/job/slurm_script'
                    spool.parent.mkdir(parents=True)
                    spool.write_bytes(script.read_bytes())
                    script = spool
                    env['SLURM_SUBMIT_DIR'] = str(root)
                subprocess.run(['bash', str(script), mode], env=env,
                               cwd=work, check=True, capture_output=True, text=True)
                calls = (work / 'calls').read_text()
                self.assertEqual('SRUN' in calls, step in ('', 'batch'))
                self.assertIn('check_slurm.py', calls)
                self.assertIn('MIG-owned-slice', calls)
                self.assertEqual('start_pruning_server.py' in calls, mode == 'serve')
                if mode == 'serve':
                    self.assertIn('--trace-decode --inherit-gpu', calls)


if __name__ == '__main__':
    unittest.main()
