import contextlib
import errno
import io
import os
from pathlib import Path
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

    def test_live_or_unknown_gate_cannot_be_recovered(self):
        root = locks.lock_root(self.path)
        root.mkdir()
        gate = root / 'gate'
        with locks._registration(gate, locks.owner_record()):
            with mock.patch.object(locks, 'owner_terminated', return_value=False):
                with self.assertRaises(BlockingIOError):
                    locks.recover(self.path)
            self.assertTrue(gate.is_file())

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

    def test_missing_diagnostic_artifact_blocks_experiment_submission(self):
        cfg = {'workdir': str(self.root), 'partition': 'rtx4090', 'python': '/python', 'options': {}}
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'PASSED'):
            run_cluster_check(cfg, lambda *a, **k: 'exit zero without proof')


if __name__ == '__main__':
    unittest.main()
