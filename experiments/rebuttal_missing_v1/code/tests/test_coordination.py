import contextlib
import errno
import io
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest import mock

from rebuttal import coordination as locks
from rebuttal.common import atomic_json, read_json
from rebuttal.coordination_check import implementation_hash, run_cluster_check, valid_result


class CoordinationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'shared.lock'

    def test_transient_communication_error_never_hides_an_active_owner(self):
        real_read = locks.read_json
        failures = []
        def read(path):
            if path.parent.name == 'holders' and len(failures) < 2:
                failures.append(str(path))
                raise OSError(errno.ECOMM, 'Communication error on send')
            return real_read(path)
        with locks.locked(self.path):
            with mock.patch.object(locks, 'read_json', side_effect=read):
                with self.assertRaises(BlockingIOError):
                    with locks.locked(self.path, blocking=False):
                        self.fail('Unreadable ownership was mistaken for an available lock')
        self.assertEqual(len(failures), 2)
        with locks.locked(self.path, blocking=False):
            pass

    def test_persistent_storage_error_stops_after_bounded_read_retries(self):
        operation = mock.Mock(side_effect=OSError(errno.ECOMM, 'still unavailable'))
        with mock.patch.object(locks.time, 'monotonic', side_effect=[0, 0, 11]), \
                mock.patch.object(locks.time, 'sleep'):
            with self.assertRaises(OSError) as failure:
                locks._metadata_read(operation, self.path)
        self.assertEqual(failure.exception.errno, errno.ECOMM)
        self.assertEqual(operation.call_count, 2)

    def test_permission_errors_are_not_retried(self):
        operation = mock.Mock(side_effect=PermissionError(errno.EACCES, 'denied'))
        with mock.patch.object(locks.time, 'sleep') as sleep:
            with self.assertRaises(PermissionError):
                locks._metadata_read(operation, self.path)
        operation.assert_called_once()
        sleep.assert_not_called()

    def test_remote_io_reads_are_retried(self):
        operation = mock.Mock(side_effect=[OSError(errno.EREMOTEIO, 'remote I/O'), 'complete'])
        with mock.patch.object(locks.time, 'sleep'):
            self.assertEqual(locks._metadata_read(operation, self.path), 'complete')
        self.assertEqual(operation.call_count, 2)

    def test_waiting_for_holder_does_not_mutate_registration_gate(self):
        holder = locks.lock_root(self.path) / 'holders/active.json'
        atomic_json(holder, locks.owner_record())
        with mock.patch.object(locks.os, 'link') as link:
            def waiting():
                link.assert_not_called()
                raise RuntimeError('stop waiting')
            with self.assertRaisesRegex(RuntimeError, 'stop waiting'):
                with locks.locked(self.path, on_wait=waiting):
                    self.fail('Entered while another holder exists')

    def test_holder_hint_is_rechecked_under_registration_gate(self):
        holder = locks.lock_root(self.path) / 'holders/other.json'
        real_link = os.link
        def link(source, gate):
            # A new holder appears after the optimistic hint was read.
            atomic_json(holder, locks.owner_record())
            real_link(source, gate)
        with mock.patch.object(locks.os, 'link', side_effect=link):
            with self.assertRaises(BlockingIOError):
                with locks.locked(self.path, blocking=False):
                    self.fail('Trusted a stale hint instead of serialized ownership')
        self.assertTrue(holder.exists())

    def test_readers_can_join_but_exclusive_owner_is_rejected_without_flock(self):
        with mock.patch('fcntl.flock', side_effect=AssertionError('not supported')):
            with locks.locked(self.path, shared=True):
                with locks.locked(self.path, shared=True, blocking=False):
                    with self.assertRaises(BlockingIOError):
                        with locks.locked(self.path, blocking=False):
                            self.fail('exclusive owner overlapped with readers')
            with locks.locked(self.path, blocking=False):
                with self.assertRaises(BlockingIOError):
                    with locks.locked(self.path, shared=True, blocking=False):
                        self.fail('reader overlapped with exclusive owner')

    def test_failure_releases_owned_registration(self):
        with self.assertRaises(ValueError):
            with locks.locked(self.path):
                raise ValueError('failed work')
        with locks.locked(self.path, blocking=False):
            pass

    def test_long_queue_then_normally_disappearing_gate_does_not_fail(self):
        root = locks.lock_root(self.path)
        gate = root / 'gate'
        active = root / 'holders/active.json'
        atomic_json(active, locks.owner_record())
        clock = {'now': 0., 'race': False, 'inject': False}
        real_link, real_read = os.link, locks.read_json
        def sleep(_):
            if clock['now'] == 0:
                clock['now'] = 121
                active.unlink(missing_ok=True)
                clock['race'] = True
        def link(source, path, *args, **kwargs):
            if path == gate and clock['race']:
                clock.update(race=False, inject=True)
                real_link(source, path)
                raise FileExistsError()
            return real_link(source, path, *args, **kwargs)
        def read(path):
            if path == gate and clock['inject']:
                clock['inject'] = False
                gate.unlink()  # The other registrant finished between symlink and read.
                raise FileNotFoundError()
            return real_read(path)
        with mock.patch.object(locks.os, 'link', link), mock.patch.object(locks, 'read_json', read), \
                mock.patch.object(locks.time, 'monotonic', lambda: clock['now']), \
                mock.patch.object(locks.time, 'sleep', sleep):
            with locks.locked(self.path):
                self.assertGreater(clock['now'], 120)

    def test_only_same_continuously_ownerless_gate_times_out(self):
        gate = self.root / 'gate'
        gate.mkdir()
        state = locks.WaitState()
        clock = {'now': 1000.}
        with mock.patch.object(locks.time, 'monotonic', lambda: clock['now']):
            identity = locks._gate_identity(gate)
            state.missing_registration(gate, identity)
            clock['now'] = 1119
            state.missing_registration(gate, identity)
            clock['now'] = 1120
            with self.assertRaisesRegex(TimeoutError, 'persisted'):
                state.missing_registration(gate, identity)
        self.assertTrue(gate.exists())

    def test_gate_recreation_or_owner_recovery_resets_fault_timer(self):
        gate = self.root / 'gate'
        gate.mkdir()
        state = locks.WaitState()
        clock = {'now': 0.}
        with mock.patch.object(locks.time, 'monotonic', lambda: clock['now']):
            old = locks._gate_identity(gate)
            state.missing_registration(gate, old)
            gate.rmdir()
            clock['now'] = 1000
            state.missing_registration(gate, old)
            self.assertIsNone(state.registration)
            gate.mkdir()
            current = locks._gate_identity(gate)
            state.missing_registration(gate, current)
            state.registration_ok()  # A successful owner.json read.
            clock['now'] = 3000
            state.missing_registration(gate, current)
            clock['now'] = 3119
            state.missing_registration(gate, current)

    def test_changed_gate_during_read_is_not_an_abandoned_registration(self):
        state = locks.WaitState()
        with mock.patch.object(locks, '_gate_identity', return_value=(1, 2, 4)):
            state.missing_registration(self.root, (1, 2, 3))
        self.assertIsNone(state.registration)

    def test_lock_and_recovery_never_remove_directories(self):
        with mock.patch.object(Path, 'rmdir', side_effect=OSError(errno.EBUSY, 'BeeGFS busy')):
            with locks.locked(self.path):
                pass
            self.orphan()
            with mock.patch.object(locks, 'owner_terminated', return_value=True):
                locks.recover(self.path)
            with locks.locked(self.path, blocking=False):
                pass

    def test_gate_is_published_only_after_its_owner_record_is_complete(self):
        gate = self.root / 'gate'
        owner = locks.owner_record()
        real_link = os.link
        def link(source, path, *args, **kwargs):
            self.assertEqual(read_json(source), owner)
            return real_link(source, path, *args, **kwargs)
        with mock.patch.object(locks.os, 'link', link):
            with locks._registration(gate, owner):
                self.assertFalse(gate.is_symlink())
                self.assertEqual(gate.stat().st_nlink, 2)
                self.assertEqual(locks._read_gate(gate), owner)
        self.assertFalse(gate.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_failed_gate_release_preserves_owner_metadata_for_recovery(self):
        gate = self.root / 'gate'
        owner = locks.owner_record()
        unlink = Path.unlink
        def fail_gate(path, *args, **kwargs):
            if path == gate:
                raise OSError(errno.EIO, 'failed gate release')
            return unlink(path, *args, **kwargs)
        with mock.patch.object(Path, 'unlink', fail_gate):
            with self.assertRaisesRegex(OSError, 'failed gate release'):
                with locks._registration(gate, owner):
                    pass
        self.assertEqual(read_json(gate), owner)

    def test_successful_unlink_with_remote_error_is_not_replayed(self):
        gate = self.root / 'gate'
        real_unlink, calls = Path.unlink, []
        def unlink(path, *args, **kwargs):
            if path == gate:
                calls.append(path)
                real_unlink(path, *args, **kwargs)
                raise OSError(errno.EREMOTEIO, 'reply lost after deletion')
            return real_unlink(path, *args, **kwargs)
        with mock.patch.object(Path, 'unlink', side_effect=unlink, autospec=True):
            with locks._registration(gate, locks.owner_record()):
                pass
        self.assertEqual(calls, [gate])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_ambiguous_unlink_never_removes_the_next_registrant(self):
        gate = self.root / 'gate'
        next_owner = locks.owner_record()
        real_unlink, calls = Path.unlink, []
        def unlink(path, *args, **kwargs):
            if path == gate:
                calls.append(path)
                real_unlink(path, *args, **kwargs)
                atomic_json(gate, next_owner)
                raise OSError(errno.EREMOTEIO, 'new owner already registered')
            return real_unlink(path, *args, **kwargs)
        with mock.patch.object(Path, 'unlink', side_effect=unlink, autospec=True):
            with locks._registration(gate, locks.owner_record()):
                pass
        self.assertEqual(calls, [gate])
        self.assertEqual(read_json(gate), next_owner)

    def test_unconfirmed_unlink_preserves_owner_and_never_replays_mutation(self):
        gate, owner = self.root / 'gate', locks.owner_record()
        real_unlink, calls = Path.unlink, []
        clock = [0.]
        def unlink(path, *args, **kwargs):
            if path == gate:
                calls.append(path)
                raise OSError(errno.EREMOTEIO, 'deletion did not complete')
            return real_unlink(path, *args, **kwargs)
        def sleep(_):
            clock[0] += 11
        with mock.patch.object(Path, 'unlink', side_effect=unlink, autospec=True), \
                mock.patch.object(locks.time, 'monotonic', side_effect=lambda: clock[0]), \
                mock.patch.object(locks.time, 'sleep', side_effect=sleep):
            with self.assertRaisesRegex(OSError, 'did not complete'):
                with locks._registration(gate, owner):
                    pass
        self.assertEqual(calls, [gate])
        self.assertEqual(read_json(gate), owner)
        self.assertTrue((self.root / ('gate-owner-' + owner['token'] + '.json')).exists())

    def test_successful_link_with_remote_error_requires_matching_owner(self):
        gate, owner = self.root / 'gate', locks.owner_record()
        real_link = os.link
        def link(source, target):
            real_link(source, target)
            raise OSError(errno.EREMOTEIO, 'reply lost after link')
        with mock.patch.object(locks.os, 'link', side_effect=link) as call:
            with locks._registration(gate, owner):
                self.assertEqual(read_json(gate), owner)
        call.assert_called_once()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_ambiguous_link_cannot_enter_when_another_owner_exists(self):
        gate, other = self.root / 'gate', locks.owner_record()
        atomic_json(gate, other)
        with mock.patch.object(locks.os, 'link', side_effect=OSError(errno.EREMOTEIO, 'unknown result')):
            with self.assertRaisesRegex(OSError, 'unknown result'):
                with locks._registration(gate, locks.owner_record()):
                    self.fail('Entered without proving ownership')
        self.assertEqual(read_json(gate), other)

    def test_private_uuid_cleanup_can_retry_without_touching_the_shared_gate(self):
        gate, owner = self.root / 'gate', locks.owner_record()
        candidate = self.root / ('gate-owner-' + owner['token'] + '.json')
        real_unlink, attempts = Path.unlink, []
        def unlink(path, *args, **kwargs):
            if path == candidate:
                attempts.append(path)
                if len(attempts) == 1:
                    real_unlink(path, *args, **kwargs)
                    raise OSError(errno.EREMOTEIO, 'private deletion reply lost')
            return real_unlink(path, *args, **kwargs)
        with mock.patch.object(Path, 'unlink', side_effect=unlink, autospec=True), \
                mock.patch.object(locks.time, 'sleep'):
            with locks._registration(gate, owner):
                pass
        self.assertEqual(len(attempts), 2)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_live_or_unknown_gate_cannot_be_recovered(self):
        root = locks.lock_root(self.path)
        root.mkdir()
        gate = root / 'gate'
        with locks._registration(gate, locks.owner_record()):
            with mock.patch.object(locks, 'owner_terminated', return_value=False):
                with self.assertRaises(BlockingIOError):
                    locks.recover(self.path)
            self.assertTrue(gate.is_file())

    def test_unserialized_recovery_never_replaces_a_registration_gate(self):
        gate = locks.lock_root(self.path) / 'gate'
        atomic_json(gate, locks.owner_record())
        original = gate.read_bytes()
        with mock.patch.object(locks, 'owner_terminated', return_value=True):
            with self.assertRaisesRegex(BlockingIOError, 'Registration gate already exists'):
                locks.recover(self.path, recover_gate=False)
        self.assertEqual(gate.read_bytes(), original)

    def test_other_pid_namespace_cannot_prove_an_owner_dead(self):
        owner = dict(locks.owner_record(), pid_namespace='another-namespace')
        with mock.patch.object(locks.os, 'kill', side_effect=ProcessLookupError) as kill:
            self.assertFalse(locks.owner_terminated(owner))
        kill.assert_not_called()

    def test_legacy_pid_cannot_be_checked_from_an_isolated_tool(self):
        owner = locks.owner_record()
        owner.pop('pid_namespace')
        with mock.patch.object(Path, 'read_text', return_value='codex-linux-san\n'), \
                mock.patch.object(locks.os, 'kill', side_effect=ProcessLookupError) as kill:
            self.assertFalse(locks.owner_terminated(owner))
        kill.assert_not_called()

    def test_legacy_pid_checked_from_host_requires_absence_not_just_age(self):
        owner = dict(locks.owner_record(), heartbeat=0)
        owner.pop('pid_namespace')
        with mock.patch.object(Path, 'read_text', return_value='systemd\n'):
            for error, expected in [(None, False), (PermissionError, False), (ProcessLookupError, True)]:
                with self.subTest(error=error), mock.patch.object(locks.os, 'kill', side_effect=error):
                    self.assertEqual(locks.owner_terminated(owner), expected)

    def test_gate_type_uses_lstat_instead_of_following_symlink(self):
        gate = self.root / 'gate'
        owner = locks.owner_record()
        atomic_json(self.root / 'owner.json', owner)
        gate.symlink_to('owner.json')
        with mock.patch.object(Path, 'is_dir', return_value=True):
            self.assertEqual(locks._read_gate(gate), owner)

    def test_legacy_link_resolving_to_directory_is_preserved(self):
        gate = locks.lock_root(self.path) / 'gate'
        gate.parent.mkdir()
        gate.symlink_to('.')
        with self.assertRaises(FileNotFoundError):
            locks._read_gate(gate)
        with self.assertRaisesRegex(BlockingIOError, 'preserve gate'):
            locks.recover(self.path)
        self.assertTrue(gate.is_symlink())

    def test_legacy_gate_requires_confirmed_recovery_without_rmdir(self):
        gate = locks.lock_root(self.path) / 'gate'
        owner = locks.owner_record()
        atomic_json(gate / 'owner.json', owner)
        with self.assertRaisesRegex(BlockingIOError, 'Legacy'):
            with locks.locked(self.path, blocking=False):
                self.fail('legacy owner was bypassed')
        with mock.patch.object(locks, 'owner_terminated', return_value=True), \
                mock.patch.object(Path, 'rmdir', side_effect=OSError(errno.EBUSY, 'busy')):
            locks.recover(self.path)
            with locks.locked(self.path):
                pass
        self.assertEqual(len(list(gate.parent.glob('abandoned-gate-*'))), 1)

    def test_broken_gate_link_is_preserved_without_owner_proof(self):
        gate = locks.lock_root(self.path) / 'gate'
        gate.parent.mkdir()
        gate.symlink_to('missing-owner.json')
        state = locks.WaitState()
        with mock.patch.object(locks.time, 'monotonic', return_value=0):
            state.missing_registration(gate, locks._gate_identity(gate))
        with mock.patch.object(locks.time, 'monotonic', return_value=121):
            with self.assertRaises(TimeoutError):
                state.missing_registration(gate, locks._gate_identity(gate))
        with self.assertRaises(BlockingIOError):
            locks.recover(self.path)
        self.assertTrue(gate.is_symlink())

    def test_stale_owner_timer_starts_at_fault_and_resets_after_heartbeat(self):
        state = locks.WaitState()
        owner = dict(locks.owner_record(), heartbeat=200)
        clock = {'wall': 200., 'mono': 0.}
        with mock.patch.object(locks.time, 'time', lambda: clock['wall']), \
                mock.patch.object(locks.time, 'monotonic', lambda: clock['mono']), \
                mock.patch.object(locks, 'owner_terminated', return_value=False):
            state.check_owner(owner)
            clock.update(wall=340, mono=5000)
            state.check_owner(owner)  # Long previous queue is not a stale-owner timeout.
            clock['mono'] = 5599
            state.check_owner(owner)
            owner['heartbeat'] = clock['wall']
            state.check_owner(owner)
            clock.update(wall=500, mono=10000)
            state.check_owner(owner)
            clock['mono'] = 10599
            state.check_owner(owner)
            clock['mono'] = 10600
            with self.assertRaisesRegex(TimeoutError, '600 seconds'):
                state.check_owner(owner)

    def orphan(self):
        root = locks.lock_root(self.path)
        owner = locks.owner_record()
        owner.update(job_id='123', heartbeat=time.time()-200)
        atomic_json(root / 'holders/dead.json', owner)
        return root

    def test_stale_time_alone_does_not_allow_recovery(self):
        root = self.orphan()
        with mock.patch.object(locks, 'owner_terminated', return_value=False):
            with self.assertRaises(BlockingIOError):
                locks.recover(self.path)
        self.assertTrue((root / 'holders/dead.json').exists())

    def test_recovery_requires_confirmed_terminal_owner(self):
        root = self.orphan()
        with mock.patch.object(locks, 'owner_terminated', return_value=True):
            locks.recover(self.path)
        self.assertFalse((root / 'holders/dead.json').exists())
        self.assertTrue((root / 'holders/dead.recovered').exists())
        with locks.locked(self.path, blocking=False):
            pass

    def test_dead_prepare_owner_ends_wait_without_stealing_lock(self):
        root = self.orphan()
        with mock.patch.object(locks, 'owner_terminated', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'terminated'):
                with locks.locked(self.path, poll_interval=.01):
                    self.fail('abandoned lock was stolen')
        self.assertTrue((root / 'holders/dead.json').exists())

    def test_scheduler_absence_or_running_is_not_terminal_proof(self):
        for output, expected in [('123|RUNNING|\n', False), ('', False),
                                  ('123|FAILED|\n', True), ('123|CANCELLED by 42|\n', True)]:
            with self.subTest(output=output), mock.patch.object(locks.subprocess, 'run',
                    return_value=mock.Mock(stdout=output)):
                self.assertEqual(locks.owner_terminated({'job_id': '123'}), expected)

    def test_two_node_check_requests_no_gpu_and_reuses_only_matching_code(self):
        cfg = {'workdir': str(self.root), 'partition': 'rtx4090', 'python': '/python', 'options': {}}
        calls = []
        def command(args, **kwargs):
            calls.append(args)
            folder = Path(args[-1])
            atomic_json(folder / 'PASSED.json', {'complete': True, 'hosts': ['node-a', 'node-b'],
                        'implementation': implementation_hash(), 'counter': 48, 'requested_gpus': 0,
                        'participants': 4, 'held_s': 125, 'minimum_waited_s': 125,
                        'preparations': 1, 'summaries': 1})
            return 'passed'
        with contextlib.redirect_stdout(io.StringIO()):
            run_cluster_check(cfg, command)
            run_cluster_check(cfg, command)
        self.assertEqual(len(calls), 1)
        self.assertIn('--nodes=2', calls[0])
        self.assertIn('--ntasks=4', calls[0])
        self.assertIn('--ntasks-per-node=2', calls[0])
        self.assertIn('--time=00:05:00', calls[0])
        self.assertFalse(any(arg.startswith(('--gres', '--gpus')) for arg in calls[0]))

    def test_old_short_check_cannot_authorize_new_gpu_batch(self):
        short = {'complete': True, 'hosts': ['a', 'b'], 'counter': 24,
                 'implementation': implementation_hash(), 'requested_gpus': 0}
        self.assertFalse(valid_result(short))

    def test_transient_check_failure_retries_in_a_new_directory_and_requires_full_proof(self):
        cfg = {'workdir': str(self.root), 'partition': 'rtx4090', 'python': '/python', 'options': {}}
        folders = []
        def command(args, **kwargs):
            folder = Path(args[-1])
            folders.append(folder)
            if len(folders) == 1:
                raise subprocess.CalledProcessError(143, args, stderr='OSError: [Errno 121] Remote I/O error\n')
            atomic_json(folder / 'PASSED.json', {'complete': True, 'hosts': ['node-a', 'node-b'],
                        'implementation': implementation_hash(), 'counter': 48, 'requested_gpus': 0,
                        'participants': 4, 'held_s': 125, 'minimum_waited_s': 125,
                        'preparations': 1, 'summaries': 1})
            return 'passed'
        with contextlib.redirect_stdout(io.StringIO()), mock.patch('rebuttal.coordination_check.time.sleep'):
            passed = run_cluster_check(cfg, command)
        self.assertEqual(len(folders), 2)
        self.assertNotEqual(folders[0], folders[1])
        self.assertTrue((folders[0] / 'FAILED.json').exists())
        self.assertIn('Errno 121', (folders[0] / 'srun.log').read_text())
        self.assertEqual(passed, folders[1] / 'PASSED.json')
        pointer = self.root / 'coordination_checks/latest_passed.json'
        self.assertEqual(read_json(pointer)['directory'], str(folders[1]))

    def test_persistent_remote_io_stops_after_two_checks(self):
        cfg = {'workdir': str(self.root), 'partition': 'rtx4090', 'python': '/python', 'options': {}}
        failure = subprocess.CalledProcessError(143, ['srun'], stderr='OSError: [Errno 121] Remote I/O error\n')
        command = mock.Mock(side_effect=failure)
        with contextlib.redirect_stdout(io.StringIO()), mock.patch('rebuttal.coordination_check.time.sleep'):
            with self.assertRaises(subprocess.CalledProcessError):
                run_cluster_check(cfg, command)
        self.assertEqual(command.call_count, 2)
        self.assertFalse((self.root / 'coordination_checks/latest_passed.json').exists())

    def test_check_invariant_failure_is_never_retried_or_hidden_by_storage_error(self):
        cfg = {'workdir': str(self.root), 'partition': 'rtx4090', 'python': '/python', 'options': {}}
        failure = subprocess.CalledProcessError(1, ['srun'], stderr=(
            'OSError: [Errno 121] Remote I/O error\nAssertionError: Two nodes entered the critical section at once.\n'))
        command = mock.Mock(side_effect=failure)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
            run_cluster_check(cfg, command)
        command.assert_called_once()

    def test_missing_diagnostic_artifact_blocks_experiment_submission(self):
        cfg = {'workdir': str(self.root), 'partition': 'rtx4090', 'python': '/python', 'options': {}}
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'PASSED'):
            run_cluster_check(cfg, lambda *a, **k: 'exit zero without proof')


if __name__ == '__main__':
    unittest.main()
