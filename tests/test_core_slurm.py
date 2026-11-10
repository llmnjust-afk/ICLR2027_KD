"""Fake scheduler and real multi-process queue/lock checks; no GPUs allocated."""
import contextlib
import functools
import io
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from rebuttal import slurm
from rebuttal.coordination import locked as directory_locked
from rebuttal.common import atomic_json, digest, file_hash, read_json
from rebuttal.run import claim_job, job_state


def abandon_preflight(root):
    root = Path(root)
    with directory_locked(root / 'logs/slurm/.core_submit.lock'):
        with directory_locked(root / 'work/.all_pipeline.lock'):
            os._exit(0)  # Simulate termination without either context's cleanup.


def fake_action(root, action, *extra):
    root = Path(root)
    with (root / 'actions.log').open('a') as stream:
        stream.write(action + '\n')
    work = root / 'work'
    run = action[5:] if action.startswith('plan-') else (extra[0] if extra else 'final')
    plan = work / 'runs' / run
    if action == 'prepare':
        time.sleep(.1)  # Deliberately allow the other three workers to contend.
        if os.environ.get('FAIL_PREPARE'):
            raise ValueError('preparation failed')
        atomic_json(work / 'prepared/manifest.json', {})
        atomic_json(work / 'prepared/split.json', {'root': str(root / 'data')})
    elif action in {'plan-final', 'plan-tune'}:
        if action == 'plan-final' and os.environ.get('FINAL_SUITE') == 'full':
            if not (work / 'runs/tune/selection.json').exists():
                raise AssertionError('Final planned before tuning selection')
        for index in range(12):
            atomic_json(plan / f'queue/j{index:02}.job.json', {'stage': 'generate'})
        atomic_json(plan / 'READY.json', {})
    elif action == 'workers':
        jobs = sorted((plan / 'queue').glob('*.job.json'))
        while any(job_state(job) != 'done' for job in jobs):
            for job in jobs:
                lock = claim_job(job)
                if lock:
                    time.sleep(.02)
                    atomic_json(str(job).replace('.job.json', '.done.json'),
                                {'job_id': os.environ['SLURM_JOB_ID']})
                    with (root / 'executions.log').open('a') as stream:
                        stream.write(plan.name + '/' + job.name + '\n')
                    lock.unlink()
                    break
            else:
                time.sleep(.01)
    elif action == 'summary':
        atomic_json(plan / 'summary.json', {'complete': True})
    elif action == 'select':
        if os.environ.get('FAIL_SELECTION'):
            raise ValueError('selection failed')
        if not read_json(work / 'runs/tune/summary.json')['complete']:
            raise AssertionError('Selection before complete tuning')
        atomic_json(work / 'runs/tune/selection.json', {'frozen': True})


def child_worker(root, folder, slot, barrier, output):
    os.environ['SLURM_JOB_ID'] = str(700 + slot)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(slot)
    barrier.wait()
    try:
        with mock.patch('fcntl.flock', side_effect=AssertionError('flock is not cross-node here')), \
                mock.patch.object(slurm, 'locked', functools.partial(directory_locked, poll_interval=.01)), \
                mock.patch.object(slurm, 'command'), mock.patch.object(slurm, 'provenance', return_value='code'), \
                mock.patch.object(slurm, 'pipeline_action', side_effect=lambda *args: fake_action(root, *args)):
            slurm.worker(types.SimpleNamespace(batch=str(folder), slot=slot))
        output.put(None)
    except BaseException as error:
        output.put(str(error))


class CoreSlurmTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cags core jobs ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'data').mkdir()
        self.config = self.root / 'config.env'
        self.config.write_text('FINAL_SUITE=allocation\n')
        patch = mock.patch.dict(os.environ, {
            'FINAL_SUITE': 'allocation', 'REBUTTAL_CONFIG': str(self.config),
            'REBUTTAL_JOB_COUNT': '4', 'REBUTTAL_GPU_COUNT': '1',
            'REBUTTAL_SLURM_GRES': 'gpu:4090:1', 'REBUTTAL_SLURM_CPUS': '8',
            'REBUTTAL_SLURM_MEM': '32G', 'REBUTTAL_PYTHON': sys.executable,
            'WORKDIR': str(self.root / 'work'), 'DATA_ROOT': str(self.root / 'data'),
            'DIT_CKPT': str(self.root / 'dit.pt'),
            'CUDA_VISIBLE_DEVICES': '-1', 'REBUTTAL_GPUS': '7'}, clear=True)
        patch.start()
        self.addCleanup(patch.stop)
        for target, value in [('ROOT', self.root), ('provenance', lambda: 'code'),
                              ('scheduler_jobs', lambda: set())]:
            patch = mock.patch.object(slurm, target, value)
            patch.start()
            self.addCleanup(patch.stop)
        self.calls = []
        self.next_id = 900
        patch = mock.patch.object(slurm, 'command', side_effect=self.fake_command)
        self.command_mock = patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch('rebuttal.coordination_check.run_cluster_check')
        self.cluster_check = patch.start()
        self.addCleanup(patch.stop)

    def fake_command(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if args[0] == 'sbatch':
            self.assertNotIn('CUDA_VISIBLE_DEVICES', kwargs['env'])
            self.assertNotIn('REBUTTAL_GPUS', kwargs['env'])
            self.next_id += 1
            return str(self.next_id) + ';cluster'
        return ''

    def submit(self, dry_run=False, retry=False, after=None, recover_submit=False):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            slurm.submit(types.SimpleNamespace(dry_run=dry_run, retry_failed=retry, after_batch=after,
                                              recover_submit=recover_submit))
        return output.getvalue()

    def batch(self):
        folder = Path(read_json(self.root / 'logs/slurm/latest_core_batch.json')['batch'])
        return folder, read_json(folder / 'batch.json')

    def test_four_independent_single_gpu_jobs_verify_once_and_repeat_is_noop(self):
        self.submit()
        folder, record = self.batch()
        calls = [call for call, _ in self.calls if call[0] == 'sbatch']
        self.assertEqual(len(calls), 4)
        self.assertEqual(len({call[2] for call in calls}), 4)
        for call in calls:
            for value in ['--gres=gpu:4090:1', '--cpus-per-task=8', '--mem=32G', '--ntasks=1']:
                self.assertIn(value, call)
            self.assertIn(str(folder / 'config.env'), call)
            self.assertNotIn('--array', ' '.join(call))
        self.assertEqual([slot['job_id'] for slot in record['slots']], ['901', '902', '903', '904'])
        self.assertEqual(len([call for call, _ in self.calls if '--verify' in call]), 1)
        self.assertIn('没有重复申请', self.submit())
        self.assertEqual(len(self.calls), 6)

    def abandon_submission(self):
        process = multiprocessing.get_context('fork').Process(target=abandon_preflight, args=(self.root,))
        process.start()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            self.fail('Preflight fixture did not finish')
        self.assertEqual(process.exitcode, 0)

    def test_recover_submission_archives_both_dead_holders_then_repeat_is_noop(self):
        self.abandon_submission()
        existing = self.root / 'work/existing.json'
        atomic_json(existing, {'preserve': True})
        with self.assertRaisesRegex(BlockingIOError, '--recover-submit'):
            self.submit()
        self.assertEqual(self.calls, [])
        self.submit(recover_submit=True)
        folder, record = self.batch()
        self.assertFalse(record['retry_failed'])
        for parent, name in [('logs/slurm', '.core_submit.lock'), ('work', '.all_pipeline.lock')]:
            holders = self.root / parent / (name + '.d') / 'holders'
            self.assertEqual(len(list(holders.glob('*.recovered'))), 1)
            self.assertEqual(list(holders.glob('*.json')), [])
        self.assertEqual(read_json(existing), {'preserve': True})
        self.assertIn('没有重复申请', self.submit(recover_submit=True))
        self.assertEqual(self.batch()[0], folder)
        self.assertEqual(len([call for call, _ in self.calls if call[0] == 'sbatch']), 4)

    def test_retry_failed_can_also_recover_an_abandoned_submission(self):
        self.abandon_submission()
        self.submit(retry=True)
        self.assertTrue(self.batch()[1]['retry_failed'])

    def test_recover_submission_refuses_a_live_submitter(self):
        with directory_locked(self.root / 'logs/slurm/.core_submit.lock'):
            with self.assertRaisesRegex(BlockingIOError, '无法安全恢复'):
                self.submit(recover_submit=True)
        self.assertEqual(self.calls, [])

    def test_recover_submission_refuses_a_live_workdir_owner(self):
        with directory_locked(self.root / 'work/.all_pipeline.lock'):
            with self.assertRaisesRegex(BlockingIOError, 'Cannot prove holder terminated'):
                self.submit(recover_submit=True)
        self.assertEqual(self.calls, [])

    def test_recovery_dry_run_does_not_change_abandoned_holders(self):
        self.abandon_submission()
        original = {str(path): path.read_bytes() for path in self.root.rglob('*.json')}
        self.submit(dry_run=True, recover_submit=True)
        self.assertEqual(original, {str(path): path.read_bytes() for path in self.root.rglob('*.json')})
        self.assertEqual(self.calls, [])

    def test_terminal_hangup_releases_submission_and_work_locks(self):
        script = '''
import signal, sys, time
from pathlib import Path
from rebuttal import slurm
root = Path(sys.argv[1])
signal.signal(signal.SIGHUP, signal.SIG_DFL)
def preflight(args):
    with slurm.locked(root / 'submit.lock'):
        with slurm.locked(root / 'work.lock'):
            (root / 'ready').touch()
            time.sleep(60)
slurm.submit = preflight
sys.argv = ['slurm', 'submit']
slurm.main()
'''
        process = subprocess.Popen([sys.executable, '-c', script, str(self.root)],
                                   cwd=Path(slurm.__file__).resolve().parents[1],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 10
            while not (self.root / 'ready').exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue((self.root / 'ready').exists(), 'Child did not reach preflight')
            process.send_signal(signal.SIGHUP)
            output, error = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 128 + signal.SIGHUP, output + error)
            for name in ['submit.lock', 'work.lock']:
                with directory_locked(self.root / name, blocking=False):
                    pass
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()

    def test_partial_submission_retries_only_rejected_and_missing_slots(self):
        def reject_once(args, **kwargs):
            if args[0] == 'sbatch' and self.next_id == 902:
                raise subprocess.CalledProcessError(1, args, stderr='quota limit')
            return self.fake_command(args, **kwargs)
        self.command_mock.side_effect = reject_once
        with self.assertRaises(subprocess.CalledProcessError):
            self.submit()
        folder, record = self.batch()
        self.assertEqual([s.get('job_id') for s in record['slots']], ['901', '902', None, None])
        self.command_mock.side_effect = self.fake_command
        self.submit()
        self.assertEqual(self.batch()[0], folder)
        self.assertEqual(len([call for call, _ in self.calls if call[0] == 'sbatch']), 4)
        self.assertEqual(len([call for call, _ in self.calls if '--verify' in call]), 1)

    def test_unknown_submission_blocks_blind_retry(self):
        def timeout(args, **kwargs):
            if args[0] == 'sbatch':
                raise subprocess.TimeoutExpired(args, 60)
            return ''
        self.command_mock.side_effect = timeout
        with self.assertRaises(subprocess.TimeoutExpired):
            self.submit()
        self.command_mock.reset_mock()
        with self.assertRaisesRegex(ValueError, '结果不明'):
            self.submit()
        with self.assertRaisesRegex(ValueError, '结果不明'):
            self.submit(recover_submit=True)
        self.command_mock.assert_not_called()

    def test_missing_resources_prevent_any_submission(self):
        self.command_mock.side_effect = subprocess.CalledProcessError(75, ['verify'])
        with self.assertRaises(subprocess.CalledProcessError):
            self.submit()
        self.assertFalse((self.root / 'logs/slurm/latest_core_batch.json').exists())

    def test_controller_failure_prevents_verification_and_submission(self):
        with mock.patch.object(slurm, 'scheduler_jobs', side_effect=OSError('controller unreachable')):
            with self.assertRaises(OSError):
                self.submit()
        self.command_mock.assert_not_called()

    def test_cross_node_check_failure_prevents_gpu_submissions(self):
        self.cluster_check.side_effect = ValueError('cross-node check failed')
        with self.assertRaisesRegex(ValueError, 'cross-node'):
            self.submit()
        self.assertEqual(self.calls, [])  # Fail before repeating the expensive image verification.

    def test_status_is_read_only_and_reports_old_preparation_failures(self):
        self.submit()
        folder, _ = self.batch()
        atomic_json(folder / 'preparation.json', {'state': 'failed', 'error': 'bad weights'})
        atomic_json(folder / 'worker-1-901.json', {'exit_code': 1, 'stage': 'prepare'})
        self.command_mock.reset_mock()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            slurm.submit(types.SimpleNamespace(status=True))
        self.assertIn('bad weights', output.getvalue())
        self.assertIn('failed', output.getvalue())
        self.assertIn('完成生成 0', output.getvalue())
        self.command_mock.assert_not_called()

    def test_dry_run_has_four_commands_and_creates_nothing(self):
        output = self.submit(dry_run=True)
        self.assertEqual(output.count('sbatch --parsable'), 4)
        self.assertEqual(output.count('--gres=gpu:4090:1'), 4)
        self.assertFalse((self.root / 'logs').exists())
        self.command_mock.assert_not_called()

    def test_four_gpu_allocation_is_rejected(self):
        os.environ['REBUTTAL_SLURM_GRES'] = 'gpu:4090:4'
        with self.assertRaises(ValueError):
            self.submit()
        self.command_mock.assert_not_called()

    def test_retry_is_forbidden_while_previous_jobs_are_active(self):
        self.submit()
        with mock.patch.object(slurm, 'scheduler_jobs', return_value={'902'}):
            with self.assertRaisesRegex(ValueError, '仍有'):
                self.submit(retry=True)

    def test_explicit_retry_after_exit_preserves_existing_outputs(self):
        self.submit()
        first, _ = self.batch()
        result = self.root / 'work/runs/final/evaluation/existing.json'
        atomic_json(result, {'complete': True})
        self.submit(retry=True)
        second, record = self.batch()
        self.assertNotEqual(first, second)
        self.assertTrue(record['retry_failed'])
        self.assertEqual(read_json(result), {'complete': True})

    def test_config_changes_cannot_silently_mix_with_submitted_batch(self):
        self.submit()
        self.config.write_text('CHANGED=1\n')
        with self.assertRaisesRegex(ValueError, '配置/代码'):
            self.submit()

    def test_stale_queue_lock_is_not_automatically_deleted(self):
        lock = self.root / 'work/runs/final/queue/old.lock'
        lock.parent.mkdir(parents=True)
        lock.touch()
        with self.assertRaisesRegex(ValueError, '任务锁'):
            self.submit()
        self.assertTrue(lock.exists())

    def concurrent_workers(self, fail=False, suite='allocation', fail_selection=False):
        folder = self.root / 'batch'
        os.environ['FINAL_SUITE'] = suite
        record = {'settings': {'count': 4, 'workdir': str(self.root / 'work'), 'python': sys.executable, 'suite': suite},
                  'config_hash': file_hash(self.config), 'source': 'code', 'retry_failed': False}
        atomic_json(folder / 'batch.json', record)
        if fail:
            os.environ['FAIL_PREPARE'] = '1'
        if fail_selection:
            os.environ['FAIL_SELECTION'] = '1'
        ctx = multiprocessing.get_context('fork')
        barrier, output = ctx.Barrier(4), ctx.Queue()
        children = [ctx.Process(target=child_worker, args=(self.root, folder, i, barrier, output))
                    for i in range(1, 5)]
        for process in children:
            process.start()
        try:
            results = [output.get(timeout=15) for _ in children]
            for process in children:
                process.join(timeout=15)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in children:
                if process.is_alive():
                    process.terminate()
                    process.join()
        return folder, results, (self.root / 'actions.log').read_text().splitlines()

    def test_four_processes_prepare_once_claim_distinct_tasks_and_summarize_once(self):
        folder, results, actions = self.concurrent_workers()
        self.assertEqual(results, [None] * 4)
        self.assertEqual(actions.count('prepare'), 1)
        self.assertEqual(actions.count('plan-final'), 1)
        self.assertEqual(actions.count('workers'), 4)
        self.assertEqual(actions.count('summary'), 1)
        executed = (self.root / 'executions.log').read_text().splitlines()
        self.assertEqual(len(executed), 12)
        self.assertEqual(len(set(executed)), 12)
        owners = {read_json(p)['job_id'] for p in (self.root / 'work/runs/final/queue').glob('*.done.json')}
        self.assertEqual(len(owners), 4)
        self.assertEqual(len(list(folder.glob('worker-*.json'))), 4)

    def test_tuning_barrier_selects_once_and_then_runs_all_final_jobs(self):
        folder, results, actions = self.concurrent_workers(suite='full')
        self.assertEqual(results, [None] * 4)
        for action in ['prepare', 'plan-tune', 'select', 'plan-final']:
            self.assertEqual(actions.count(action), 1)
        self.assertEqual(actions.count('workers'), 8)
        self.assertEqual(actions.count('summary'), 2)
        self.assertLess(actions.index('summary'), actions.index('select'))
        self.assertLess(actions.index('select'), actions.index('plan-final'))
        executed = (self.root / 'executions.log').read_text().splitlines()
        self.assertEqual(len(executed), 24)
        self.assertEqual(len(set(executed)), 24)
        self.assertEqual(read_json(folder / 'final_preparation.json')['state'], 'ready')

    def test_failed_selection_stops_all_workers_before_final_plan(self):
        folder, results, actions = self.concurrent_workers(suite='full', fail_selection=True)
        self.assertTrue(all(results))
        self.assertEqual(actions.count('select'), 1)
        self.assertNotIn('plan-final', actions)
        self.assertEqual(read_json(folder / 'final_preparation.json')['state'], 'failed')

    def test_second_batch_has_all_worker_dependencies_and_separate_pointer(self):
        self.submit()
        first, _ = self.batch()
        os.environ.update(FINAL_SUITE='full', WORKDIR=str(self.root / 'tuned'))
        self.config.write_text('FINAL_SUITE=full\n')
        self.calls.clear()
        self.submit(after=str(first))
        jobs = [call for call, _ in self.calls if call[0] == 'sbatch']
        self.assertEqual(len(jobs), 4)
        for call in jobs:
            self.assertIn('--dependency=afterok:901:902:903:904', call)
            self.assertIn('--kill-on-invalid-dep=yes', call)
        self.assertEqual(self.batch()[0], first)
        pointer = self.root / 'logs/slurm/latest_tuned_batch.json'
        self.assertTrue(pointer.exists())
        self.assertNotEqual(read_json(pointer)['batch'], str(first))

    def test_full_submission_requires_dependency_and_rejects_failed_parent(self):
        os.environ['FINAL_SUITE'] = 'full'
        with self.assertRaisesRegex(ValueError, 'after-batch'):
            self.submit()
        self.command_mock.assert_not_called()
        os.environ['FINAL_SUITE'] = 'allocation'
        self.submit()
        first, _ = self.batch()
        os.environ['FINAL_SUITE'] = 'full'
        self.command_mock.reset_mock()
        atomic_json(first / 'worker-1-901.json', {'state': 'failed'})
        with self.assertRaisesRegex(ValueError, '失败'):
            self.submit()
        with self.assertRaisesRegex(ValueError, '失败'):
            self.submit(after=str(first))
        self.command_mock.assert_not_called()

    def complete_parent(self):
        self.submit()
        folder, record = self.batch()
        upstream = Path(record['settings']['workdir']) / 'runs/final'
        plan = {'suite': 'allocation'}
        plan['plan_hash'] = digest(plan)
        atomic_json(upstream / 'plan.json', plan)
        atomic_json(upstream / 'summary.json', {'complete': True, 'missing': [], 'plan_hash': plan['plan_hash']})
        atomic_json(upstream / '.slurm_summary.json', {'sha256': file_hash(upstream / 'summary.json')})
        for slot in record['slots']:
            atomic_json(folder / f'worker-{slot["slot"]}-{slot["job_id"]}.json',
                        {'state': 'completed', 'exit_code': 0, 'job_id': slot['job_id']})
        os.environ.update(FINAL_SUITE='full', WORKDIR=str(self.root / 'tuned'))
        self.config.write_text('FINAL_SUITE=full\n')
        self.calls.clear()
        return folder, upstream

    def test_completed_parent_is_detected_without_resubmitting_or_expired_dependencies(self):
        folder, upstream = self.complete_parent()
        self.submit()
        calls = [call for call, _ in self.calls if call[0] == 'sbatch']
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(not any(arg.startswith('--dependency') for arg in call) for call in calls))
        self.assertEqual(self.batch()[0], folder)
        tuned = read_json(self.root / 'logs/slurm/latest_tuned_batch.json')['batch']
        dependency = read_json(Path(tuned) / 'batch.json')['settings']['dependency']
        self.assertTrue(dependency['completed'])
        self.assertEqual(dependency['summary_sha256'], file_hash(upstream / 'summary.json'))

    def test_completion_record_cannot_override_mismatched_parent_summary(self):
        folder, upstream = self.complete_parent()
        atomic_json(upstream / 'summary.json', {'complete': True, 'plan_hash': 'another-plan'})
        with self.assertRaisesRegex(ValueError, '不匹配'):
            self.submit()
        self.assertEqual(self.calls, [])

    def test_retry_recovers_target_workdir_not_only_previous_batch(self):
        self.submit()
        os.environ['WORKDIR'] = str(self.root / 'new-work')
        with mock.patch.object(slurm, 'recover') as recovery:
            self.submit(retry=True)
        self.assertIn(mock.call(self.root / 'new-work/.all_pipeline.lock'), recovery.call_args_list)

    def test_shared_preparation_failure_stops_all_workers(self):
        folder, results, actions = self.concurrent_workers(fail=True)
        self.assertTrue(all(results))
        self.assertEqual(actions.count('prepare'), 1)
        self.assertNotIn('workers', actions)
        self.assertNotIn('summary', actions)
        self.assertEqual(read_json(folder / 'preparation.json')['state'], 'failed')

    def test_incomplete_queue_cannot_report_success(self):
        plan = self.root / 'work/runs/final'
        atomic_json(plan / 'queue/missing.job.json', {})
        with self.assertRaisesRegex(ValueError, '未全部成功'):
            slurm.finish_summary(plan)

    def test_worker_rejects_an_accidental_multi_gpu_allocation(self):
        folder = self.root / 'batch'
        atomic_json(folder / 'batch.json', {'settings': {'count': 4}})
        os.environ.update(SLURM_JOB_ID='701', CUDA_VISIBLE_DEVICES='0,1,2,3')
        with self.assertRaisesRegex(ValueError, '恰好有一张'):
            slurm.worker(types.SimpleNamespace(batch=str(folder), slot=1))
        self.command_mock.assert_not_called()

    def test_batch_shell_forwards_snapshot_slot_and_slurm_device(self):
        from rebuttal.common import ROOT
        script = self.root / 'job.sh'
        shutil.copyfile(ROOT / 'scripts/rebuttal_slurm_job.sh', script)
        binary = self.root / 'bin'
        binary.mkdir()
        srun = binary / 'srun'
        srun.write_text('#!/bin/bash\nshift 2\nexec "$@"\n')
        srun.chmod(0o755)
        fake_python = binary / 'python'
        fake_python.write_text('#!' + sys.executable + '\n'
            'import json, os, sys\nfrom pathlib import Path\n'
            'Path("invocation.json").write_text(json.dumps({"args":sys.argv[1:],'
            '"config":os.environ["REBUTTAL_CONFIG"],"mask":os.environ["CUDA_VISIBLE_DEVICES"],'
            '"override":os.environ.get("REBUTTAL_GPUS")}))\n')
        fake_python.chmod(0o755)
        env = dict(os.environ, PATH=str(binary) + ':' + os.defpath, SLURM_JOB_ID='712',
                   CUDA_VISIBLE_DEVICES='2', REBUTTAL_GPUS='99')
        result = subprocess.run(['bash', str(script), str(self.root), str(self.config), str(fake_python),
                                 str(self.root / 'batch'), '3'], env=env, text=True,
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        call = read_json(self.root / 'invocation.json')
        self.assertEqual(call['args'], ['-m', 'rebuttal.slurm', 'worker', '--batch',
                                        str(self.root / 'batch'), '--slot', '3'])
        self.assertEqual(call['config'], str(self.config))
        self.assertEqual(call['mask'], '2')
        self.assertIsNone(call['override'])


if __name__ == '__main__':
    unittest.main()
