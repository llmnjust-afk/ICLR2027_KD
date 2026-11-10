"""Launcher control-flow tests with an explicit fake worker (no GPU experiments)."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rebuttal.common import ROOT


class LauncherTests(unittest.TestCase):
    def run_launcher(self, mode, failure='', extra_env=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'scripts').mkdir()
            for name in ['run_rebuttal_all.sh', 'rebuttal.sh', 'prepare_rebuttal_core.sh']:
                shutil.copyfile(ROOT / 'scripts' / name, root / 'scripts' / name)
            for name in ['rebuttal.example.env', 'REBUTTAL_RUN_GUIDE.txt']:
                (root / name).write_text('test fixture')
            (root / 'data/train').mkdir(parents=True)
            (root / 'data/val').mkdir()
            (root / 'vae').mkdir()
            (root / 'dit.pt').write_bytes(b'fake checkpoint for launcher test only')
            (root / 'rebuttal.local.env').write_text(
                f'DATA_ROOT={root}/data\nREPORT_DIR={root}/data/val\n'
                f'DIT_CKPT={root}/dit.pt\nVAE_PATH={root}/vae\n')
            fake = root / 'fake_python'
            fake.write_text('#!' + sys.executable + '\n' + '''
import os, shlex, sys, subprocess
from pathlib import Path
root = Path(__file__).parent
args = sys.argv[1:]
with (root / 'calls.txt').open('a') as f:
    f.write(shlex.join(args) + '\\n')
module = args[1]
if module == 'rebuttal.preflight':
    if '--resources-only' not in args:
        if os.environ.get('FAILURE') == 'preflight': sys.exit(12)
        print(args[args.index('--gpus') + 1])
elif module == 'rebuttal.coordination':
    env = dict(os.environ, REBUTTAL_LEGACY_GUARD='1')
    sys.exit(subprocess.call(args[args.index('--') + 1:], env=env))
elif module == 'rebuttal.resources':
    if os.environ.get('FAILURE') == 'resources': sys.exit(75)
    (root / 'data/train').mkdir(parents=True, exist_ok=True)
    (root / 'data/val').mkdir(exist_ok=True)
    (root / 'vae').mkdir(exist_ok=True)
    (root / 'dit.pt').write_bytes(b'fake checkpoint for launcher test only')
    (root / 'rebuttal.local.env').write_text(
        f'DATA_ROOT={root}/data\\nREPORT_DIR={root}/data/val\\n'
        f'DIT_CKPT={root}/dit.pt\\nVAE_PATH={root}/vae\\n')
elif module == 'rebuttal.data_setup':
    if os.environ.get('FAILURE') == 'resources': sys.exit(75)
elif module == 'rebuttal.run' and args[2] == 'smoke':
    if os.environ.get('FAILURE') == 'smoke': sys.exit(23)
''')
            fake.chmod(0o755)
            env = dict(os.environ, REBUTTAL_PYTHON=str(fake), REBUTTAL_CONFIG=str(root / 'rebuttal.local.env'),
                       WORKDIR=str(root / 'work'), FAILURE=failure, REBUTTAL_GPUS='0')
            env.pop('SLURM_JOB_ID', None)
            env.update(extra_env or {})
            result = subprocess.run(['bash', str(root / 'scripts/run_rebuttal_all.sh'), mode],
                                    env=env, capture_output=True, text=True, timeout=30)
            calls = (root / 'calls.txt').read_text() if (root / 'calls.txt').exists() else ''
            status = root / 'work/pipeline_exit.txt'
            return result, calls, status.read_text() if status.exists() else None

    def test_resource_only_never_launches_training_or_requires_gpu(self):
        result, calls, status = self.run_launcher('--prepare-only', failure='preflight')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('preflight --resources-only', calls)
        self.assertIn('resources ensure', calls)
        self.assertNotIn('rebuttal.run', calls)
        self.assertIsNone(status)

    def test_bad_gpu_environment_stops_before_downloads(self):
        result, calls, status = self.run_launcher('--foreground', failure='preflight')
        self.assertEqual(result.returncode, 12)
        self.assertNotIn('rebuttal.resources', calls)
        self.assertNotIn('rebuttal.run', calls)

    def test_smoke_failure_is_preserved_and_stops_training(self):
        result, calls, status = self.run_launcher('--foreground', failure='smoke')
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertIn('exit_code=23', status)
        self.assertIn('last_stage=smoke', status)
        self.assertNotIn('rebuttal.run prepare', calls)

    def test_network_failure_does_not_report_started(self):
        result, calls, status = self.run_launcher('--prepare-only', failure='resources')
        self.assertEqual(result.returncode, 75)
        self.assertNotIn('rebuttal.run', calls)
        self.assertNotIn('已后台启动', result.stdout)

    def test_slurm_uses_allocated_devices_and_does_not_detach(self):
        result, calls, status = self.run_launcher('start', failure='smoke', extra_env={
            'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': '4', 'REBUTTAL_GPUS': '7'})
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertIn('preflight --gpus 4', calls)
        self.assertIn('last_stage=smoke', status)
        self.assertNotIn('已后台启动', result.stdout)

    def test_slurm_cpu_only_allocation_cannot_fall_back_to_configured_gpu(self):
        result, calls, status = self.run_launcher('--foreground', extra_env={
            'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': '', 'REBUTTAL_GPUS': '0'})
        self.assertEqual(result.returncode, 1)
        self.assertIn('Slurm 作业没有可见 GPU', result.stderr)
        self.assertEqual(calls, '')

    def test_allocation_pipeline_skips_tuning_and_optional_suites(self):
        result, calls, status = self.run_launcher('--foreground', extra_env={
            'FINAL_SUITE': 'allocation', 'ARCHITECTURES': 'convnet6',
            'RUN_SENSITIVITY': '1', 'RUN_CFG_CONTROLS': '1',
            'RESNET_FEATURE_CKPT': '/unused/resnet.pt', 'CHECKPOINT_RULE': 'final'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('exit_code=0', status)
        self.assertIn('last_stage=completed', status)
        self.assertIn('--suite allocation', calls)
        self.assertIn("--k-values ''", calls)
        self.assertIn('--holdout-fraction 0.1 --seed 2027', calls)
        self.assertNotIn('--resnet-checkpoint', calls)
        self.assertNotIn('--suite tune', calls)
        self.assertNotIn('rebuttal.run select', calls)
        self.assertNotIn('--selection', calls)
        self.assertNotIn('--suite sensitivity', calls)
        self.assertNotIn('--suite cfg_controls', calls)
        self.assertIn('--checkpoint-rule final', calls)
        self.assertIn('rebuttal.run summarize', calls)
        self.assertIn('rebuttal.data_setup --verify', calls)
        self.assertNotIn('rebuttal.resources', calls)
        self.assertLess(calls.index('rebuttal.run smoke'), calls.index('rebuttal.run prepare'))
        self.assertLess(calls.index('rebuttal.run prepare'), calls.index('rebuttal.run plan'))

    def test_allocation_missing_resources_never_downloads_or_starts(self):
        result, calls, status = self.run_launcher('--foreground', failure='resources',
                                                 extra_env={'FINAL_SUITE': 'allocation'})
        self.assertEqual(result.returncode, 75)
        self.assertIn('rebuttal.data_setup --verify', calls)
        self.assertNotIn('rebuttal.resources', calls)
        self.assertNotIn('rebuttal.run', calls)
        self.assertIsNone(status)

    def test_allocation_check_is_offline_and_requires_no_gpu(self):
        result, calls, status = self.run_launcher('--check', failure='preflight',
                                                 extra_env={'FINAL_SUITE': 'allocation'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('rebuttal.data_setup --verify', calls)
        self.assertNotIn('preflight', calls)
        self.assertNotIn('rebuttal.resources', calls)

    def test_full_pipeline_retains_tuning_and_sensitivity(self):
        result, calls, status = self.run_launcher('--foreground', extra_env={
            'FINAL_SUITE': 'full', 'RUN_SENSITIVITY': '1', 'RUN_CFG_CONTROLS': '0'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--suite tune', calls)
        self.assertIn('rebuttal.run select', calls)
        self.assertIn('--selection', calls)
        self.assertIn('--suite full', calls)
        self.assertIn('--suite sensitivity', calls)
        self.assertIn('last_stage=completed', status)


if __name__ == '__main__':
    unittest.main()
