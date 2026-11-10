"""Submission tests use a fake scheduler; never allocate cluster resources."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rebuttal.common import ROOT


class SlurmLauncherTests(unittest.TestCase):
    def submit(self, mode='submit', fail_resources=False, core=False):
        with tempfile.TemporaryDirectory(prefix='cags slurm ') as tmp:
            root = Path(tmp)
            (root / 'scripts').mkdir()
            (root / 'bin').mkdir()
            shutil.copyfile(ROOT / 'scripts/submit_rebuttal.sh', root / 'scripts/submit_rebuttal.sh')
            if core:
                shutil.copyfile(ROOT / 'scripts/submit_rebuttal_core.sh', root / 'scripts/submit_rebuttal_core.sh')
                shutil.copyfile(ROOT / 'rebuttal.core.env', root / 'rebuttal.core.env')
            prepare = root / 'scripts/run_rebuttal_all.sh'
            prepare.write_text('#!/usr/bin/env bash\n'
                               'printf "%s\\n" "$*" > prepare_called.txt\n'
                               'if [[ ${FAIL_RESOURCES:-0} == 1 ]]; then exit 75; fi\n')
            (root / 'scripts/prepare_rebuttal_core.sh').write_text(
                '#!/usr/bin/env bash\n[[ "$*" == --verify ]] || exit 99\n'
                'printf "%s\\n" "$*" > prepare_called.txt\n'
                'if [[ ${FAIL_RESOURCES:-0} == 1 ]]; then exit 75; fi\n')
            stub = root / 'bin/sbatch'
            stub.write_text('#!' + sys.executable + '\n' + '''
import json, os, sys
from pathlib import Path
Path('submission.json').write_text(json.dumps({
    'args': sys.argv[1:], 'mask': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'override': os.environ.get('REBUTTAL_GPUS')}))
assert Path('prepare_called.txt').exists()
print('456;testcluster')
''')
            stub.chmod(0o755)
            env = dict(os.environ, PATH=str(root / 'bin') + ':' + os.environ.get('PATH', ''),
                       REBUTTAL_CONFIG=str(root / 'config.env'), CUDA_VISIBLE_DEVICES='-1',
                       REBUTTAL_GPUS='7', FAIL_RESOURCES=str(int(fail_resources)))
            for key in ['SLURM_JOB_ID', 'REBUTTAL_GPU_COUNT', 'REBUTTAL_SLURM_PARTITION',
                        'REBUTTAL_SLURM_GRES', 'REBUTTAL_SLURM_QOS', 'REBUTTAL_SLURM_ACCOUNT',
                        'REBUTTAL_SLURM_TIME', 'REBUTTAL_SLURM_CPUS', 'REBUTTAL_SLURM_MEM']:
                env.pop(key, None)
            if core:
                env.pop('REBUTTAL_CONFIG', None)
            entry = 'submit_rebuttal_core.sh' if core else 'submit_rebuttal.sh'
            result = subprocess.run(['bash', str(root / 'scripts' / entry), mode],
                                    env=env, capture_output=True, text=True, timeout=30)
            record = root / 'submission.json'
            job = root / 'logs/slurm/latest_job.id'
            return result, json.loads(record.read_text()) if record.exists() else None, \
                job.read_text() if job.exists() else None, (root / 'prepare_called.txt').exists()

    def test_prepares_before_submission_and_removes_login_gpu_mask(self):
        result, record, job, prepared = self.submit()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(prepared)
        self.assertEqual(job.strip(), '456')
        self.assertIn('--partition=rtx4090', record['args'])
        self.assertIn('--gres=gpu:4090:1', record['args'])
        self.assertIn('--cpus-per-task=8', record['args'])
        self.assertIn('--mem=32G', record['args'])
        self.assertIsNone(record['mask'])
        self.assertIsNone(record['override'])
        self.assertTrue(record['args'][-4].endswith('/scripts/rebuttal_slurm_job.sh'))

    def test_failed_preparation_does_not_submit_job(self):
        result, record, job, prepared = self.submit(fail_resources=True)
        self.assertEqual(result.returncode, 75)
        self.assertTrue(prepared)
        self.assertIsNone(record)
        self.assertIsNone(job)

    def test_dry_run_does_not_prepare_or_submit(self):
        result, record, job, prepared = self.submit(mode='--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(prepared)
        self.assertIsNone(record)
        self.assertIsNone(job)
        self.assertIn('REBUTTAL_CONFIG=', result.stdout)


if __name__ == '__main__':
    unittest.main()
