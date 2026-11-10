"""Provenance, pairing and scheduler tests. Never allocates a real GPU."""
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

spec = importlib.util.spec_from_file_location('a800_runner', Path(__file__).with_name('runner.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class A800Tests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='cags-a800-test-')
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for name in ['HERE', 'WORK', 'RUN', 'OLD_RUN', 'CODE']:
            path = self.root / name
            path.mkdir()
            patch = mock.patch.object(runner, name, path)
            patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.dict(os.environ, {}, clear=False)
        patch.start(); self.addCleanup(patch.stop)
        for key in ['SLURM_JOB_ID', 'CAGS_A800_QOS', 'CAGS_A800_ACCOUNT']:
            os.environ.pop(key, None)
        self.plan = {'plan_hash': 'frozen', 'hardware_migration': {
            'source_submission': {'array_job_id': '157933'}, 'a800_task_ids': [8, 9]}}

    def patch_submission(self):
        for target, value in [('validate_setup', self.plan), ('remaining_tasks', [8, 9]),
                              ('account_options', {'qos': 'a800-2', 'account': 'research'}),
                              ('query', ''), ('require_terminal', None)]:
            patch = mock.patch.object(runner, target, return_value=value)
            patch.start(); self.addCleanup(patch.stop)

    def test_only_complete_blocks_are_imported(self):
        plan = {'generation_seeds': [100, 101], 'training_seeds': [0],
                'architectures': ['resnet18'], 'conditions': {'a': {}, 'b': {}}}
        for g, names in [(100, ['a', 'b']), (101, ['a'])]:
            task = dict(generation_seed=g, training_seed=0, architecture='resnet18')
            for name in names:
                runner.write(runner.output_path(runner.OLD_RUN, task, name), {'generation_manifest_sha256': 'dataset'})
        with mock.patch.object(runner.legacy, 'existing_result_ok', side_effect=lambda p, *a: p.exists()), \
             mock.patch.object(runner, 'sha', return_value='dataset'):
            imported, pending, partial = runner.reusable_blocks(plan)
        self.assertEqual(list(imported), ['0'])
        self.assertEqual(len(imported['0']), 2)
        self.assertEqual(pending, [1])
        self.assertEqual(len(partial), 1)

    def test_import_rejects_wrong_generation_provenance(self):
        plan = {'generation_seeds': [100], 'training_seeds': [0], 'architectures': ['resnet18'], 'conditions': {'a': {}}}
        task = dict(generation_seed=100, training_seed=0, architecture='resnet18')
        runner.write(runner.output_path(runner.OLD_RUN, task, 'a'), {'generation_manifest_sha256': 'wrong'})
        with mock.patch.object(runner.legacy, 'existing_result_ok', return_value=True), \
             mock.patch.object(runner, 'sha', return_value='correct'):
            with self.assertRaisesRegex(ValueError, 'different generated dataset'):
                runner.reusable_blocks(plan)

    def test_resources_two_a800_tasks_and_cpu_only_summary(self):
        gpu, cpu = runner.scheduler_commands([8, 9], {'qos': 'a800', 'account': 'research'})
        self.assertIn('--partition=a800', gpu)
        self.assertIn('--gres=gpu:a800:1', gpu)
        self.assertIn('--array=8,9%2', gpu)
        self.assertIn('--partition=cpu', cpu)
        self.assertFalse(any(x.startswith('--gres') for x in cpu))

    def test_account_selection_uses_granted_nonurgent_qos(self):
        with mock.patch.object(runner, 'query', return_value='research||normal,a800-2,urgent|normal|\n'):
            self.assertEqual(runner.account_options(), {'qos': 'a800-2', 'account': 'research'})
        with mock.patch.object(runner, 'query', return_value='research||normal,urgent|normal|\n'):
            with self.assertRaisesRegex(RuntimeError, 'No usable'):
                runner.account_options()

    def test_terminal_check_rejects_missing_pending_and_running_states(self):
        for text in ['', '157933_8|RUNNING|\n', '157933_12|PENDING|\n']:
            with mock.patch.object(runner, 'query', return_value=text):
                with self.assertRaisesRegex(RuntimeError, 'Cannot confirm'):
                    runner.require_terminal('157933')
        with mock.patch.object(runner, 'query', return_value='157933_0|COMPLETED|\n157933_8|CANCELLED by 1001|\n'):
            runner.require_terminal('157933')

    def test_initial_submission_and_repeat_do_not_duplicate_array(self):
        self.patch_submission()
        results = [types.SimpleNamespace(returncode=0, stdout=f'{i}\n', stderr='') for i in [81001, 81002]]
        with mock.patch.object(runner.subprocess, 'run', side_effect=results) as call, contextlib.redirect_stdout(io.StringIO()):
            runner.submit()
            runner.submit()
        self.assertEqual(call.call_count, 2)
        self.assertIn('--dependency=afterok:81001', call.call_args_list[1].args[0])
        record = runner.read(runner.HERE / 'submission.json')
        self.assertEqual(record['max_parallel_gpus'], 2)

    def test_unknown_sbatch_is_never_automatically_repeated(self):
        self.patch_submission()
        with mock.patch.object(runner.subprocess, 'run', side_effect=subprocess.TimeoutExpired('sbatch', 45)) as call:
            with self.assertRaisesRegex(RuntimeError, 'unknown'):
                runner.submit()
            with self.assertRaisesRegex(RuntimeError, 'Unresolved'):
                runner.submit()
        self.assertEqual(call.call_count, 1)

    def test_invalid_success_response_is_not_retried(self):
        self.patch_submission()
        result = types.SimpleNamespace(returncode=0, stdout='unrecognized output', stderr='')
        with mock.patch.object(runner.subprocess, 'run', return_value=result) as call:
            with self.assertRaisesRegex(RuntimeError, 'Unrecognized'):
                runner.submit()
            with self.assertRaisesRegex(RuntimeError, 'Unresolved'):
                runner.submit()
        self.assertEqual(call.call_count, 1)

    def test_summary_rejection_can_retry_without_resubmitting_array(self):
        self.patch_submission()
        results = [types.SimpleNamespace(returncode=0, stdout='82001\n', stderr=''),
                   types.SimpleNamespace(returncode=1, stdout='', stderr='scheduler rejection'),
                   types.SimpleNamespace(returncode=0, stdout='82002\n', stderr='')]
        with mock.patch.object(runner.subprocess, 'run', side_effect=results) as call, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'rejected'):
                runner.submit()
            runner.submit()
        self.assertEqual(sum(any(x.startswith('--array=') for x in c.args[0]) for c in call.call_args_list), 1)

    def test_resource_preflight_failure_submits_nothing(self):
        self.patch_submission()
        with mock.patch.object(runner, 'query', side_effect=RuntimeError('Invalid qos')), \
             mock.patch.object(runner, 'invoke_submission') as submit:
            with self.assertRaisesRegex(RuntimeError, 'Invalid qos'):
                runner.submit()
            submit.assert_not_called()

    def test_retry_refuses_active_previous_array(self):
        self.patch_submission()
        runner.write(runner.HERE / 'submission.json', {'plan_hash': 'frozen', 'array_job_id': '83001',
                                                     'array_job_id_status': 'submitted'})
        with mock.patch.object(runner, 'require_terminal', side_effect=RuntimeError('still running')), \
             mock.patch.object(runner, 'invoke_submission') as submit:
            with self.assertRaisesRegex(RuntimeError, 'still running'):
                runner.submit(retry_failed=True)
            submit.assert_not_called()

    def test_dry_run_does_not_contact_scheduler_or_write_submission(self):
        self.patch_submission()
        with mock.patch.object(runner.subprocess, 'run', side_effect=AssertionError('No scheduler')), contextlib.redirect_stdout(io.StringIO()):
            runner.submit(dry_run=True)
        self.assertFalse((runner.HERE / 'submission.json').exists())

    def test_resume_rejects_another_a800_variant(self):
        task = dict(architecture='resnet18', generation_seed=101, training_seed=1)
        plan = {'plan_hash': 'new', 'protocol': {'epochs': 2000, 'save_checkpoints': False}}
        path = runner.output_path(runner.RUN, task, 'cags')
        runner.write(path, dict(plan_hash='new', architecture='resnet18', generation_seed=101, train_seed=1,
                               condition='cags', metrics={'top1': 25}, selected_epoch=2000,
                               hardware={'gpu': 'NVIDIA A800-SXM4-80GB'}))
        with self.assertRaisesRegex(ValueError, 'another GPU'):
            runner.legacy.existing_result_ok(path, plan, task, 'cags', 'NVIDIA A800 80GB PCIe')

    def test_reused_task_cannot_be_launched_as_a800_task(self):
        with mock.patch.dict(os.environ, {'SLURM_JOB_ID': '80000'}), \
             mock.patch.object(runner, 'validate_setup', return_value=self.plan):
            with self.assertRaisesRegex(ValueError, 'not assigned to A800'):
                runner.worker(0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
