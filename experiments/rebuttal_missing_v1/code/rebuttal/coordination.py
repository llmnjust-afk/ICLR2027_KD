"""Shared-filesystem reader/writer locks using atomic hard links, never flock.

The short-lived gate is a hard link to an already complete owner record. No directory is
removed during normal locking: concurrent rmdir can fail with EBUSY on BeeGFS.
No new symlinks are used: an empty symlink target was observed across nodes.
Holder files carry leases
for diagnostics; expired leases are NOT permission to steal a live process's lock.
Recovery is explicit and requires proof that every recorded owner has terminated.
"""
import argparse
import contextlib
import errno
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import threading
import time
import uuid

from .common import atomic_json, now, read_json


TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
TRANSIENT_METADATA_ERRORS = {errno.ECOMM, errno.ESTALE, errno.EREMOTEIO}


def pid_namespace():
    try:
        return os.readlink('/proc/self/ns/pid')
    except OSError:
        return None


def owner_record(shared=False, pid=None):
    return {'token': uuid.uuid4().hex, 'host': socket.gethostname(),
            'pid': pid or os.getpid(), 'job_id': os.environ.get('SLURM_JOB_ID'),
            'pid_namespace': pid_namespace(),
            'shared': shared, 'created': now(), 'heartbeat': time.time()}


def owner_terminated(owner):
    """False/unknown never permits reclamation. Absent squeue alone is not proof."""
    job = owner.get('job_id')
    if job:
        try:
            result = subprocess.run(['sacct', '-n', '-X', '-j', str(job), '-o', 'JobIDRaw,State', '-P'],
                                    capture_output=True, text=True, timeout=8, check=True)
        except (OSError, subprocess.SubprocessError):
            return False
        states = [line.split('|')[1].split()[0].rstrip('+') for line in result.stdout.splitlines()
                  if line.split('|')[0] == str(job) and len(line.split('|')) >= 2 and line.split('|')[1]]
        return bool(states) and all(state in TERMINAL for state in states)
    if owner.get('host') == socket.gethostname() and owner.get('pid'):
        recorded_scope = owner.get('pid_namespace')
        if recorded_scope:
            if recorded_scope != pid_namespace():
                return False
        else:
            # Legacy records must be checked from the host login terminal. A
            # sandbox can share the hostname but cannot see host process IDs.
            try:
                if Path('/proc/1/comm').read_text().strip() not in {'systemd', 'init'}:
                    return False
            except OSError:
                return False
        try:
            os.kill(int(owner['pid']), 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
    return False


def lock_root(path):
    return Path(str(path) + '.d')


def _metadata_read(operation, path):
    """Retry transient BeeGFS reads without treating unknown ownership as free.

    Mutations are never replayed: their outcome may be ambiguous after ECOMM.
    Persistent storage errors still stop submission/work rather than steal a lock.
    """
    deadline = time.monotonic() + 10
    while True:
        try:
            return operation(path)
        except OSError as error:
            if error.errno not in TRANSIENT_METADATA_ERRORS or time.monotonic() >= deadline:
                raise
            time.sleep(.2)


def _gate_identity(gate):
    info = _metadata_read(Path.lstat, gate)
    return info.st_dev, info.st_ino, info.st_ctime_ns


def _read_gate(gate):
    # lstat inspects the gate itself; is_dir follows broken/empty legacy links.
    if stat.S_ISDIR(_metadata_read(Path.lstat, gate).st_mode):
        raise BlockingIOError(f'Legacy directory gate requires explicit recovery after its owner exits: {gate}')
    try:
        return _metadata_read(read_json, gate)
    except IsADirectoryError as error:
        # An empty legacy symlink can resolve to its parent on BeeGFS. Preserve
        # it and apply the existing incomplete-registration timeout, never steal it.
        raise FileNotFoundError(f'Unreadable legacy gate metadata: {gate}') from error


def _unlink_private(path):
    """Only for this process's UUID path, never a reusable shared gate name."""
    deadline = time.monotonic() + 10
    while True:
        try:
            path.unlink(missing_ok=True)
            return
        except OSError as error:
            if error.errno not in TRANSIENT_METADATA_ERRORS or time.monotonic() >= deadline:
                raise
            time.sleep(.2)


def _release_registration(gate, owner):
    """Issue unlink once; reconcile an ambiguous reply using read-only checks.

    A remote I/O error may arrive after the server removed our link. Another
    process can already own this name, so replaying unlink is never safe.
    """
    try:
        gate.unlink()
    except OSError as error:
        if error.errno not in TRANSIENT_METADATA_ERRORS:
            raise
        deadline = time.monotonic() + 10
        while True:
            try:
                current = _read_gate(gate)
            except FileNotFoundError:
                return  # The original deletion completed.
            if current.get('token') and current['token'] != owner['token']:
                return  # Our link is gone; leave the new registrant untouched.
            if time.monotonic() >= deadline:
                raise error  # Preserve the owner record when release is unconfirmed.
            time.sleep(.2)


@contextlib.contextmanager
def _registration(gate, owner):
    """Publish complete metadata before competing for the one shared gate name."""
    candidate = gate.with_name('gate-owner-' + owner['token'] + '.json')
    owner['heartbeat'] = time.time()
    atomic_json(candidate, owner)
    try:
        os.link(candidate, gate)
    except OSError as error:
        if error.errno in TRANSIENT_METADATA_ERRORS:
            # A link may also have succeeded despite a lost reply. Enter only
            # after observing our complete owner token, and never replay link.
            try:
                current = _read_gate(gate)
            except FileNotFoundError:
                raise error
            if current.get('token') != owner['token']:
                raise
        else:
            _unlink_private(candidate)
            raise
    # Keep the metadata if gate removal fails; recovery must still identify us.
    try:
        yield
    finally:
        _release_registration(gate, owner)
        _unlink_private(candidate)


def _conflicting_holders(holders, shared):
    conflicts = []
    for path in _metadata_read(lambda folder: list(folder.glob('*.json')), holders):
        try:
            current = _metadata_read(read_json, path)
        except FileNotFoundError:  # A holder may finish during inspection.
            continue
        if not shared or not current['shared']:
            conflicts.append(current)
    return conflicts


class WaitState:
    """Timers describe one observed fault, never the total time waiting for work."""
    def __init__(self):
        self.registration = None
        self.stale_owners = {}

    def registration_ok(self):
        self.registration = None

    def missing_registration(self, gate, before):
        try:
            after = _gate_identity(gate)
        except FileNotFoundError:
            self.registration_ok()
            return
        # The owner may have finished or replaced the gate while we read it.
        if before != after:
            self.registration_ok()
            return
        current = time.monotonic()
        if self.registration is None or self.registration[0] != after:
            self.registration = (after, current)
        elif current - self.registration[1] >= 120:
            raise TimeoutError(f'Incomplete lock registration persisted for 120 seconds; preserve and inspect: {gate}')

    def check_owner(self, owner):
        identity = owner.get('token') or (owner.get('host'), owner.get('pid'), owner.get('job_id'))
        heartbeat = owner.get('heartbeat', time.time())
        if time.time() - heartbeat <= 120:
            self.stale_owners.pop(identity, None)
            return
        current = time.monotonic()
        previous = self.stale_owners.get(identity)
        if previous is None or previous[0] != heartbeat:
            previous = self.stale_owners[identity] = (heartbeat, current)
        if owner_terminated(owner):
            raise RuntimeError(f'Lock owner has terminated: {owner}. Retry after recovering this batch.')
        if current - previous[1] >= 600:
            raise TimeoutError(f'Stale lock heartbeat; owner status unconfirmed for 600 seconds: {owner}')


def _check_wait(owners, state, on_wait):
    if on_wait:
        on_wait()
    for owner in owners:
        state.check_owner(owner)


@contextlib.contextmanager
def locked(path, shared=False, blocking=True, on_wait=None, poll_interval=2):
    root = lock_root(path)
    holders = root / 'holders'
    holders.mkdir(parents=True, exist_ok=True)
    gate = root / 'gate'
    owner = owner_record(shared)
    holder = holders / (owner['token'] + '.json')
    since = time.monotonic()
    wait_state = WaitState()
    while True:
        # Wait on existing holders without repeatedly creating/deleting the
        # shared gate. This is only a hint: acquisition rechecks under the gate.
        conflicts = _conflicting_holders(holders, shared)
        if conflicts:
            if not blocking:
                raise BlockingIOError(f'Work directory already in use: {root}; owners={conflicts}')
            _check_wait(conflicts, wait_state, on_wait)
            time.sleep(poll_interval)
            continue
        registration = _registration(gate, owner)
        try:
            registration.__enter__()
        except FileExistsError:
            if not blocking and time.monotonic() - since > 10:
                raise BlockingIOError(f'Lock registration in progress: {root}')
            try:
                before = _gate_identity(gate)
            except FileNotFoundError:
                wait_state.registration_ok()
                if on_wait:
                    on_wait()
                continue
            try:
                owners = [_read_gate(gate)]
            except FileNotFoundError:
                owners = []
                wait_state.missing_registration(gate, before)
            else:
                wait_state.registration_ok()
            _check_wait(owners, wait_state, on_wait)
            time.sleep(poll_interval)
            continue
        conflicts = []
        wait_state.registration_ok()
        try:
            conflicts = _conflicting_holders(holders, shared)
            if not conflicts:
                owner['heartbeat'] = time.time()
                atomic_json(holder, owner)
        finally:
            registration.__exit__(None, None, None)
        if not conflicts:
            break
        if not blocking:
            raise BlockingIOError(f'Work directory already in use: {root}; owners={conflicts}')
        _check_wait(conflicts, wait_state, on_wait)
        time.sleep(poll_interval)
    stopped = threading.Event()
    def heartbeat():
        while not stopped.wait(15):
            owner['heartbeat'] = time.time()
            atomic_json(holder, owner)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield owner
    finally:
        stopped.set()
        thread.join()
        _unlink_private(holder)


def recover(path, *, recover_gate=True):
    """Archive only confirmed-dead owners, never reclaim an expired lease alone.

    Gate replacement requires external serialization. Global submission-lock
    recovery has no outer lock and must use recover_gate=False: acquiring the
    existing registration gate then serializes holder inspection and archival.
    """
    root = lock_root(path)
    if not root.exists():
        return
    gate = root / 'gate'
    if gate.exists() or gate.is_symlink():
        if not recover_gate:
            raise BlockingIOError(f'Registration gate already exists; retry after it is released: {gate}')
        metadata = gate / 'owner.json' if stat.S_ISDIR(gate.lstat().st_mode) else gate
        try:
            owner = read_json(metadata)
        except (OSError, ValueError) as error:
            raise BlockingIOError(f'Cannot read registration owner; preserve gate: {gate}') from error
        if not owner_terminated(owner):
            raise BlockingIOError(f'Cannot prove registration owner terminated: {gate}')
        archive = root / ('abandoned-gate-' + uuid.uuid4().hex)
        gate.rename(archive)  # Live owners never remove somebody else's gate.
    with _registration(gate, owner_record()):
        candidates = list((root / 'holders').glob('*.json'))
        for holder in candidates:
            owner = read_json(holder)
            if not owner_terminated(owner):
                raise BlockingIOError(f'Cannot prove holder terminated: {holder}; owner={owner}')
        for holder in candidates:
            holder.rename(holder.with_suffix('.recovered'))


def legacy_guard(args):
    """Hold the same exclusive workdir lock around the old foreground pipeline."""
    with locked(Path(args.workdir) / '.all_pipeline.lock', blocking=False):
        env = dict(os.environ, REBUTTAL_LEGACY_GUARD='1')
        process = subprocess.Popen(args.command, env=env, start_new_session=True)
        try:
            raise SystemExit(process.wait())
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == '--':
        args.command.pop(0)
    if not args.command:
        parser.error('A foreground command is required.')
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    legacy_guard(args)


if __name__ == '__main__':
    main()
