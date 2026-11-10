"""Four CPU processes on two nodes must survive a 125-second shared preparation."""
import argparse
import os
from pathlib import Path
import socket
import time
import uuid

from .common import ROOT, atomic_json, digest, file_hash, now, read_json
from .coordination import locked

PARTICIPANTS = 4
HOLD_SECONDS = 125


def implementation_hash():
    return digest({name: file_hash(ROOT / 'rebuttal' / name)
                   for name in ['coordination.py', 'coordination_check.py', 'common.py']})


def wait_for_files(folder, pattern, expected, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        files = sorted(folder.glob(pattern))
        if len(files) == expected:
            return files
        time.sleep(.2)
    raise TimeoutError(f'Coordination diagnostic participants missing: {folder}/{pattern}')


def participate(folder, rank, expected_nodes=2):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    info = {'rank': rank, 'host': socket.gethostname(), 'job_id': os.environ.get('SLURM_JOB_ID'),
            'implementation': implementation_hash()}
    atomic_json(folder / f'arrival-{rank}.json', info)
    wait_for_files(folder, 'arrival-*.json', PARTICIPANTS)
    if rank == 0:
        with locked(folder / 'prepare.lock', poll_interval=.1):
            atomic_json(folder / 'holding.json', info)
            wait_for_files(folder, 'waiting-*.json', PARTICIPANTS - 1)
            started = time.monotonic()
            time.sleep(HOLD_SECONDS)
            info['held_s'] = time.monotonic() - started
            atomic_json(folder / 'prepared.json', {'count': 1, 'ready': True})
    else:
        wait_for_files(folder, 'holding.json', 1)
        started = time.monotonic()
        atomic_json(folder / f'waiting-{rank}.json', info)
        with locked(folder / 'prepare.lock', poll_interval=.1):
            info['waited_s'] = time.monotonic() - started
            if read_json(folder / 'prepared.json') != {'count': 1, 'ready': True}:
                raise AssertionError('Preparation was duplicated or observed before completion.')
            if info['waited_s'] < 120:
                raise AssertionError('Diagnostic did not exercise waiting longer than 120 seconds.')
    for _ in range(12):
        with locked(folder / 'counter.lock', poll_interval=.02):
            sentinel = folder / 'critical-section.json'
            if sentinel.exists():
                raise AssertionError('Two nodes entered the critical section at once.')
            atomic_json(sentinel, info)
            counter = folder / 'counter.json'
            value = read_json(counter)['value'] if counter.exists() else 0
            time.sleep(.02)
            atomic_json(counter, {'value': value + 1})
            sentinel.unlink()
    atomic_json(folder / f'done-{rank}.json', info)
    records = [read_json(path) for path in wait_for_files(folder, 'done-*.json', PARTICIPANTS)]
    with locked(folder / 'summary.lock', poll_interval=.1):
        if (folder / 'PASSED.json').exists():
            return
        hosts = sorted({item['host'] for item in records})
        if len(hosts) != expected_nodes or any(item['implementation'] != implementation_hash() for item in records):
            raise AssertionError(f'Diagnostic requires {expected_nodes} distinct nodes running identical code.')
        if read_json(folder / 'counter.json')['value'] != PARTICIPANTS * 12:
            raise AssertionError('Concurrent update lost or duplicated.')
        waited = min(item['waited_s'] for item in records if item['rank'] != 0)
        held = next(item['held_s'] for item in records if item['rank'] == 0)
        if waited < 120 or held < HOLD_SECONDS:
            raise AssertionError('Long-wait diagnostic did not reach the required duration.')
        atomic_json(folder / 'PASSED.json', {'complete': True, 'hosts': hosts, 'created': now(),
                    'implementation': implementation_hash(), 'counter': PARTICIPANTS * 12, 'requested_gpus': 0,
                    'participants': PARTICIPANTS, 'held_s': held, 'minimum_waited_s': waited,
                    'preparations': 1, 'summaries': 1})
    print(f'Coordination check rank={rank}, host={info["host"]} finished.', flush=True)


def run_cluster_check(cfg, command):
    base = Path(cfg['workdir']) / 'coordination_checks'
    pointer = base / 'latest_passed.json'
    if pointer.exists():
        previous = Path(read_json(pointer)['directory']) / 'PASSED.json'
        if previous.exists():
            record = read_json(previous)
            if valid_result(record):
                print(f'复用已通过的两节点协调检查：{previous}', flush=True)
                return previous
    folder = base / uuid.uuid4().hex[:12]
    folder.mkdir(parents=True)
    args = ['srun', f'--partition={cfg["partition"]}', '--nodes=2', '--ntasks=4',
            '--ntasks-per-node=2', '--cpus-per-task=1', '--mem=1G', '--time=00:05:00',
            '--immediate=60', '--kill-on-bad-exit=1', '--job-name=cags-coordination-check',
            f'--chdir={ROOT}']
    args += [f'--{key}={value}' for key, value in cfg['options'].items() if key in {'account', 'qos'}]
    args += [cfg['python'], '-m', 'rebuttal.coordination_check', '--directory', str(folder)]
    print('正在执行两节点/四进程 CPU 协调检查（持锁 125 秒，不申请 GPU，最多等待资源 60 秒）……', flush=True)
    try:
        output = command(args, capture=True, timeout=360)
        (folder / 'srun.log').write_text(output + '\n')
        passed = folder / 'PASSED.json'
        if not passed.exists():
            raise ValueError('Two-node coordination check did not produce PASSED.json.')
        result = read_json(passed)
        if not valid_result(result):
            raise ValueError('Two-node coordination check returned an invalid result.')
        atomic_json(pointer, {'directory': str(folder)})
        print(f'两节点协调检查通过：{result["hosts"]}', flush=True)
        return passed
    except BaseException as error:
        atomic_json(folder / 'FAILED.json', {'error': str(error), 'stderr': getattr(error, 'stderr', None),
                                            'created': now()})
        raise


def valid_result(record):
    return (record.get('complete') and len(set(record.get('hosts', []))) == 2
            and record.get('implementation') == implementation_hash()
            and record.get('counter') == PARTICIPANTS * 12 and record.get('requested_gpus') == 0
            and record.get('participants') == PARTICIPANTS and record.get('held_s', 0) >= HOLD_SECONDS
            and record.get('minimum_waited_s', 0) >= 120
            and record.get('preparations') == 1 and record.get('summaries') == 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    args = parser.parse_args()
    if os.environ.get('SLURM_NTASKS') != str(PARTICIPANTS):
        raise ValueError('Launch with srun --nodes=2 --ntasks=4 --ntasks-per-node=2 (no GPUs).')
    participate(args.directory, int(os.environ['SLURM_PROCID']))


if __name__ == '__main__':
    main()
