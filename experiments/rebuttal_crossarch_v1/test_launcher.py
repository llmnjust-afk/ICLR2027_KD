"""Isolated scheduler and provenance checks; these never submit real jobs."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('crossarch_runner', Path(__file__).with_name('runner.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CrossarchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cags-crossarch-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for key, value in [('HERE', self.root / 'extension'), ('WORK', self.root / 'work'),
                           ('CODE', self.root / 'code'), ('RUN', self.root / 'work/runs/final')]:
            value.mkdir(parents=True, exist_ok=True)
            patcher = mock.patch.object(runner, key, value)
            patcher.start(); self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start(); self.addCleanup(env.stop)
        os.environ.pop('SLURM_JOB_ID', None)

    def test_pairing_covers_each_architecture_generation_training_exactly_once(self):
        plan = {'architectures': ['resnet18', 'resnetap10'], 'generation_seeds': list(range(100, 105)),
                'training_seeds': [0, 1, 2]}
        tasks = runner.tasks_for(plan)
        self.assertEqual(len(tasks), 30)
        self.assertEqual(len({tuple(t.values()) for t in tasks}), 30)
        self.assertEqual([t['architecture'] for t in tasks[:4]], ['resnet18', 'resnetap10'] * 2)

    def test_array_limit_and_summary_dependency_and_duplicate_submission(self):
        responses = [types.SimpleNamespace(returncode=0, stdout='81001\n', stderr=''),
                     types.SimpleNamespace(returncode=0, stdout='81002\n', stderr='')]
        with mock.patch.object(runner, 'validate_setup', return_value={'plan_hash': 'frozen'}), \
             mock.patch.object(runner.shutil, 'which', return_value='/usr/bin/sbatch'), \
             mock.patch.object(runner.subprocess, 'run', side_effect=responses) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            runner.submit()
            runner.submit()
        self.assertEqual(run.call_count, 2)
        first, second = [call.args[0] for call in run.call_args_list]
        self.assertIn('--array=0-29%4', first)
        self.assertIn('--gres=gpu:4090:1', first)
        self.assertIn('--dependency=afterok:81001', second)
        self.assertFalse(any(x.startswith('--gres') for x in second))
        self.assertFalse((runner.HERE / '.submission.lock').exists())

    def test_uncertain_submission_is_not_retried(self):
        with mock.patch.object(runner, 'validate_setup', return_value={'plan_hash': 'frozen'}), \
             mock.patch.object(runner.shutil, 'which', return_value='/usr/bin/sbatch'), \
             mock.patch.object(runner.subprocess, 'run', side_effect=subprocess.TimeoutExpired('sbatch', 45)) as run:
            with self.assertRaisesRegex(RuntimeError, 'unknown'):
                runner.submit()
            with self.assertRaisesRegex(RuntimeError, 'Unresolved'):
                runner.submit()
            self.assertEqual(run.call_count, 1)

    def test_summary_submission_retry_does_not_resubmit_gpu_array(self):
        responses = [types.SimpleNamespace(returncode=0, stdout='82001\n', stderr=''),
                     types.SimpleNamespace(returncode=1, stdout='', stderr='Temporary queue rejection'),
                     types.SimpleNamespace(returncode=0, stdout='82002\n', stderr='')]
        with mock.patch.object(runner, 'validate_setup', return_value={'plan_hash': 'frozen'}), \
             mock.patch.object(runner.shutil, 'which', return_value='/usr/bin/sbatch'), \
             mock.patch.object(runner.subprocess, 'run', side_effect=responses) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'rejected'):
                runner.submit()
            runner.submit()
        self.assertEqual(run.call_count, 3)
        self.assertEqual(sum(any(x.startswith('--array=') for x in call.args[0]) for call in run.call_args_list), 1)

    def test_dry_run_never_calls_scheduler_or_writes_submission(self):
        with mock.patch.object(runner, 'validate_setup', return_value={'plan_hash': 'frozen'}), \
             mock.patch.object(runner.subprocess, 'run', side_effect=AssertionError('No scheduler call')), \
             contextlib.redirect_stdout(io.StringIO()):
            runner.submit(dry_run=True)
        self.assertFalse((runner.HERE / 'submission.json').exists())

    def test_source_dataset_hashes_and_identity_are_enforced(self):
        folder = self.root / 'source_dataset'
        folder.mkdir()
        plan = {'plan_hash': 'parent', 'conditions': {'cags': {'coefficients': [.03]}},
                'classes': ['classA'], 'protocol': {'ipc': 1}}
        photo = folder / 'classA/0000.png'
        photo.parent.mkdir(); photo.write_bytes(b'original photo bytes')
        identity = dict(plan_hash='parent', generation_seed=100, condition='cags', specification=plan['conditions']['cags'])
        manifest = dict(identity, configuration_hash=runner.digest(identity), images={'classA/0000.png': runner.sha(photo)})
        runner.write(folder / 'manifest.json', manifest)
        runner.validate_dataset(folder, plan, 100, 'cags')
        photo.write_bytes(b'changed photo bytes')
        with self.assertRaisesRegex(ValueError, 'Changed source image'):
            runner.validate_dataset(folder, plan, 100, 'cags')
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            runner.validate_dataset(folder, plan, 101, 'cags')

    def test_stale_results_or_different_gpu_are_not_silently_reused(self):
        task = dict(architecture='resnet18', generation_seed=100, training_seed=0)
        plan = {'plan_hash': 'new', 'protocol': {'epochs': 2000, 'save_checkpoints': False}}
        path = runner.result_path(task, 'cags')
        result = dict(plan_hash='new', architecture='resnet18', generation_seed=100, train_seed=0,
                      condition='cags', metrics={'top1': 25}, selected_epoch=2000, hardware={'gpu': 'RTX 4090'})
        runner.write(path, result)
        self.assertTrue(runner.existing_result_ok(path, plan, task, 'cags', 'RTX 4090'))
        with self.assertRaisesRegex(ValueError, 'another GPU'):
            runner.existing_result_ok(path, plan, task, 'cags', 'A100')
        result['plan_hash'] = 'old'; runner.write(path, result)
        with self.assertRaisesRegex(ValueError, 'stale result'):
            runner.existing_result_ok(path, plan, task, 'cags')


if __name__ == '__main__':
    unittest.main(verbosity=2)
