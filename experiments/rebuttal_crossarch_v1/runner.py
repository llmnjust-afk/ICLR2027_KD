"""Isolated, evaluation-only cross-architecture extension of the finished core run.

Existing experiments, source files, generation manifests, and images are read-only
inputs. One Slurm array task runs all six conditions for one architecture /
generation seed / training seed, preserving paired comparisons.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import tarfile
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT / 'results/rebuttal_core_v2/runs/final'
WORK = ROOT / 'results/rebuttal_crossarch_v1'
RUN = WORK / 'runs/final'
CODE = HERE / 'code'
PYTHON = ROOT / 'ICLR/bin/python'
SOURCE_ARCHIVE = ROOT / 'backups/rebuttal_20261107T113403_861938Z/source.tar.gz'
ARCHITECTURES = ['resnet18', 'resnetap10']
CONCURRENCY = 4
SCRIPT_NAMES = ['runner.py', 'submit.sh', 'job.sh', 'summary_job.sh']


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n')
    temp.replace(path)


def checked_plan(path):
    plan = read(path)
    if digest({k: v for k, v in plan.items() if k != 'plan_hash'}) != plan['plan_hash']:
        raise ValueError(f'Plan checksum mismatch: {path}')
    return plan


def tasks_for(plan):
    return [dict(architecture=arch, generation_seed=g, training_seed=t)
            for g in plan['generation_seeds'] for t in plan['training_seeds']
            for arch in plan['architectures']]


def result_path(task, condition):
    return RUN / 'evaluation' / f"g{task['generation_seed']}" / f"t{task['training_seed']}" / task['architecture'] / f'{condition}.json'


def validate_dataset(folder, plan, seed, condition, full=True):
    manifest = read(folder / 'manifest.json')
    identity = {'plan_hash': plan['plan_hash'], 'generation_seed': seed,
                'condition': condition, 'specification': plan['conditions'][condition]}
    if manifest['configuration_hash'] != digest(identity):
        raise ValueError(f'Generation identity mismatch: {folder}')
    for key in ['plan_hash', 'generation_seed', 'condition', 'specification']:
        if manifest[key] != identity[key]:
            raise ValueError(f'Generation metadata mismatch ({key}): {folder}')
    expected = {f'{cls}/{i:04d}.png' for cls in plan['classes'] for i in range(plan['protocol']['ipc'])}
    if set(manifest['images']) != expected:
        raise ValueError(f'Unexpected generated-image inventory: {folder}')
    if full:
        actual = {str(p.relative_to(folder)) for p in folder.rglob('*.png')}
        if actual != expected:
            raise ValueError(f'Extra or missing generated images: {folder}')
        def check_image(item):
            rel, checksum = item
            if sha(folder / rel) != checksum:
                raise ValueError(f'Changed source image: {folder / rel}')
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(check_image, manifest['images'].items()))
    return manifest


def prepare():
    if (HERE / 'frozen_manifest.json').exists():
        validate_setup()
        print('Existing isolated plan verified; no source or result was overwritten.')
        return
    if (HERE / 'submission.json').exists():
        raise ValueError('Submission already exists; refusing to rebuild.')
    parent = checked_plan(PARENT / 'plan.json')
    summary = read(PARENT / 'summary.json')
    if not summary['complete'] or summary['missing'] or summary['plan_hash'] != parent['plan_hash']:
        raise ValueError('The source core experiment must be complete.')
    if parent['suite'] != 'allocation' or len(parent['conditions']) != 6:
        raise ValueError('Expected the six-condition allocation experiment.')
    if parent['protocol']['checkpoint_rule'] != 'final' or parent['protocol']['evaluation_split'] != 'report':
        raise ValueError('This extension requires the fixed-final reporting protocol.')
    manifests = {}
    for seed in parent['generation_seeds']:
        for name in parent['conditions']:
            folder = PARENT / 'datasets' / f'g{seed}' / name
            validate_dataset(folder, parent, seed, name)
            manifests[f'g{seed}/{name}'] = sha(folder / 'manifest.json')
            print(f'Verified source dataset g{seed}/{name} ({len(manifests)}/30).', flush=True)
    copied = {}
    with tarfile.open(SOURCE_ARCHIVE, 'r:gz') as archive:
        for rel, checksum in parent['source'].items():
            if Path(rel).is_absolute() or '..' in Path(rel).parts:
                raise ValueError(f'Unsafe source path: {rel}')
            member = archive.getmember(rel)
            if not member.isfile():
                raise ValueError(f'Expected a regular source file: {rel}')
            with archive.extractfile(member) as stream:
                content = stream.read()
            if hashlib.sha256(content).hexdigest() != checksum:
                raise ValueError(f'Backup source does not match the parent experiment: {rel}')
            destination = CODE / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and sha(destination) != checksum:
                raise ValueError(f'Conflicting isolated source copy: {destination}')
            if not destination.exists():
                destination.write_bytes(content)
            copied[rel] = checksum
    RUN.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PARENT / 'evaluation_images.json', RUN / 'evaluation_images.json')
    plan = copy.deepcopy(parent)
    plan.pop('plan_hash')
    plan['architectures'] = ARCHITECTURES
    plan['evaluation_only'] = True
    plan['generation_reuse'] = {'source_run': str(PARENT), 'source_plan_hash': parent['plan_hash'],
                              'source_summary_sha256': sha(PARENT / 'summary.json'),
                              'dataset_manifest_sha256': manifests,
                              'note': 'Original generation metadata and files are preserved; no images are regenerated.'}
    plan['execution_extension'] = {name: sha(HERE / name) for name in SCRIPT_NAMES}
    plan['plan_hash'] = digest(plan)
    write(RUN / 'plan.json', plan)
    write(HERE / 'tasks.json', tasks_for(plan))
    write(HERE / 'frozen_manifest.json', {
        'created': now(), 'plan_hash': plan['plan_hash'], 'source_files': copied,
        'source_archive': str(SOURCE_ARCHIVE), 'source_archive_sha256': sha(SOURCE_ARCHIVE),
        'scripts': plan['execution_extension'], 'tasks_sha256': sha(HERE / 'tasks.json'),
        'evaluation_images_sha256': sha(RUN / 'evaluation_images.json'),
        'python': str(PYTHON), 'max_parallel_gpus': CONCURRENCY,
        'source_datasets_verified': len(manifests), 'source_images_verified': len(manifests) * len(plan['classes']) * plan['protocol']['ipc'],
        'expected_fits': len(tasks_for(plan)) * len(plan['conditions']),
        'protocol_changes': ['classifier architecture only'],
        'gpu_smoke': 'Performed on the assigned RTX 4090 before each paired training block; isolated from reported results.'})
    write(RUN / 'READY.json', {'plan_hash': plan['plan_hash'], 'created': now()})
    print(f'Prepared {len(tasks_for(plan))} paired blocks / 180 classifier fits. No GPU jobs submitted.')


def validate_setup():
    frozen = read(HERE / 'frozen_manifest.json')
    plan = checked_plan(RUN / 'plan.json')
    if plan['plan_hash'] != frozen['plan_hash']:
        raise ValueError('Frozen plan was changed.')
    for name, checksum in frozen['scripts'].items():
        if sha(HERE / name) != checksum:
            raise ValueError(f'Launcher changed after preparation: {name}')
    for rel, checksum in frozen['source_files'].items():
        if sha(CODE / rel) != checksum:
            raise ValueError(f'Isolated evaluation source changed: {rel}')
    if sha(HERE / 'tasks.json') != frozen['tasks_sha256'] or read(HERE / 'tasks.json') != tasks_for(plan):
        raise ValueError('Task list was changed.')
    if sha(RUN / 'evaluation_images.json') != frozen['evaluation_images_sha256']:
        raise ValueError('Reporting image inventory changed.')
    parent = checked_plan(PARENT / 'plan.json')
    reuse = plan['generation_reuse']
    if parent['plan_hash'] != reuse['source_plan_hash'] or sha(PARENT / 'summary.json') != reuse['source_summary_sha256']:
        raise ValueError('Parent experiment identity changed.')
    for rel, checksum in reuse['dataset_manifest_sha256'].items():
        if sha(PARENT / 'datasets' / rel / 'manifest.json') != checksum:
            raise ValueError(f'Parent generation manifest changed: {rel}')
    if sha(Path(plan['protocol']['prepared']) / 'manifest.json') != plan['prepared_manifest_hash']:
        raise ValueError('Prepared manifest changed.')
    return plan


def scheduler_commands():
    logdir = WORK / 'slurm'
    common = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--partition=rtx4090', '--export=ALL']
    array = common + ['--job-name=cags-xarch-v1', '--array=0-29%4', '--gres=gpu:4090:1',
                      '--cpus-per-task=8', '--mem=32G', '--chdir=' + str(CODE),
                      '--output=' + str(logdir / 'eval-%A_%a.log'),
                      '--error=' + str(logdir / 'eval-%A_%a.log'), str(HERE / 'job.sh'), str(HERE), str(PYTHON)]
    report = common + ['--job-name=cags-xarch-report', '--cpus-per-task=2', '--mem=8G',
                       '--chdir=' + str(CODE), '--output=' + str(logdir / 'summary-%j.log'),
                       '--error=' + str(logdir / 'summary-%j.log')]
    return array, report


def invoke_submission(command, record, field):
    record[field + '_status'] = 'submitting'
    write(HERE / 'submission.json', record)
    env = dict(os.environ)
    for key in ['CUDA_VISIBLE_DEVICES', 'REBUTTAL_GPUS']:
        env.pop(key, None)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=45, env=env)
    except (subprocess.TimeoutExpired, OSError) as error:
        record[field + '_status'] = 'unknown'
        record['last_error'] = str(error)
        write(HERE / 'submission.json', record)
        raise RuntimeError('Submission result is unknown; inspect squeue before any retry.') from error
    if result.returncode:
        record[field + '_status'] = 'rejected'
        record['last_error'] = result.stderr.strip()
        write(HERE / 'submission.json', record)
        raise RuntimeError('sbatch rejected submission: ' + result.stderr.strip())
    jobid = result.stdout.strip().split(';')[0]
    if not jobid.isdigit():
        record[field + '_status'] = 'unknown'
        record['last_error'] = result.stdout.strip()
        write(HERE / 'submission.json', record)
        raise RuntimeError('Unrecognized sbatch response; inspect squeue before any retry.')
    record[field] = jobid
    record[field + '_status'] = 'submitted'
    write(HERE / 'submission.json', record)
    return jobid


def submit(dry_run=False):
    plan = validate_setup()
    array, report = scheduler_commands()
    if dry_run:
        print(shlex.join(array))
        print(shlex.join(report + ['--dependency=afterok:ARRAY_JOB_ID', '--kill-on-invalid-dep=yes',
                                  str(HERE / 'summary_job.sh'), str(HERE), str(PYTHON)]))
        print('Preview only: 30 paired tasks, at most 4 extra RTX 4090 GPUs; no jobs submitted.')
        return
    if os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Run submit.sh from a login terminal, not inside an existing GPU job.')
    if not shutil.which('sbatch'):
        raise RuntimeError('sbatch is unavailable; run this command on the cluster login node.')
    (WORK / 'slurm').mkdir(parents=True, exist_ok=True)
    lock = HERE / '.submission.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump({'host': socket.gethostname(), 'pid': os.getpid(), 'created': now()}, stream)
    try:
        record = read(HERE / 'submission.json') if (HERE / 'submission.json').exists() else {
            'created': now(), 'plan_hash': plan['plan_hash'], 'max_parallel_gpus': CONCURRENCY}
        if record['plan_hash'] != plan['plan_hash']:
            raise ValueError('Submission belongs to another plan.')
        for field in ['array_job_id', 'summary_job_id']:
            if record.get(field + '_status') in {'submitting', 'unknown'}:
                raise RuntimeError(f'Unresolved earlier {field}; inspect Slurm before retrying.')
        arrayid = record.get('array_job_id')
        if not arrayid:
            arrayid = invoke_submission(array, record, 'array_job_id')
        summaryid = record.get('summary_job_id')
        if not summaryid:
            command = report + [f'--dependency=afterok:{arrayid}', '--kill-on-invalid-dep=yes',
                                str(HERE / 'summary_job.sh'), str(HERE), str(PYTHON)]
            summaryid = invoke_submission(command, record, 'summary_job_id')
        print(f'GPU array job: {arrayid}; maximum 4 RTX 4090 GPUs concurrently.')
        print(f'Automatic CPU summary job: {summaryid}; starts after all array tasks succeed.')
        print(f'Logs: {WORK / "slurm"}')
        print('Repeated submission reuses these IDs; it does not launch another array.')
    finally:
        lock.unlink()


def activate_snapshot():
    sys.path.insert(0, str(CODE))
    os.chdir(CODE)
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/cags-crossarch-matplotlib')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
    from rebuttal.common import source_manifest
    if source_manifest() != read(HERE / 'frozen_manifest.json')['source_files']:
        raise ValueError('Unexpected or changed files in the isolated source snapshot.')


def existing_result_ok(path, plan, task, condition, gpu=None):
    if not path.exists():
        return False
    result = read(path)
    expected = {'plan_hash': plan['plan_hash'], 'architecture': task['architecture'],
                'generation_seed': task['generation_seed'], 'train_seed': task['training_seed'], 'condition': condition}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f'Mixed or stale result at {path}: {key}')
    if 'metrics' not in result or result.get('selected_epoch') != plan['protocol']['epochs']:
        raise ValueError(f'Incomplete or nonfinal result: {path}')
    if gpu and result['hardware']['gpu'] != gpu:
        raise ValueError(f'Paired training block cannot resume on another GPU model: {path}')
    checkpoint = path.with_suffix('.pt')
    if plan['protocol']['save_checkpoints'] and (not checkpoint.exists() or sha(checkpoint) != result['checkpoint_sha256']):
        raise ValueError(f'Classifier checkpoint is missing or changed: {checkpoint}')
    return True


def worker(task_id):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Workers must run inside a Slurm allocation.')
    plan = validate_setup()
    tasks = tasks_for(plan)
    if not 0 <= task_id < len(tasks):
        raise ValueError('Task ID out of range.')
    task = tasks[task_id]
    activate_snapshot()
    import torch
    from rebuttal.common import hardware
    from rebuttal.evaluate import image_manifest, loader, train_one
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Each array task requires exactly one visible CUDA GPU.')
    machine = hardware()
    if machine['gpu'] != 'NVIDIA GeForce RTX 4090':
        raise RuntimeError(f'This batch is frozen to RTX 4090; allocated {machine["gpu"]}.')
    if machine['versions'] != plan['environment']:
        raise RuntimeError('Package versions differ from the completed core experiment.')
    reporting = read(RUN / 'evaluation_images.json')
    cfg = plan['protocol']
    if reporting['root'] != cfg['evaluation_root'] or image_manifest(reporting['root'], reporting['paths']) != cfg['evaluation_image_manifest']:
        raise RuntimeError('Reporting images changed after planning.')
    parent = checked_plan(PARENT / 'plan.json')
    pending = [name for name in plan['conditions'] if not existing_result_ok(result_path(task, name), plan, task, name, machine['gpu'])]
    if not pending:
        print(f'Task {task_id} is already complete; no training repeated.', flush=True)
        return
    folder = WORK / 'task_state'
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / f'{task_id:02d}.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump({'job_id': os.environ['SLURM_JOB_ID'], 'pid': os.getpid(), 'host': socket.gethostname()}, stream)
    state = dict(task, task_id=task_id, job_id=os.environ['SLURM_JOB_ID'], started=now(), hardware=machine, state='running')
    write(folder / f'{task_id:02d}.json', state)
    try:
        batches = loader(reporting['root'], reporting['paths'], cfg['classifier_size'], cfg['batch_size'], cfg['num_workers'])
        smoke_root = WORK / 'smoke_tests' / f'task_{task_id:02d}'
        smoke_root.mkdir(parents=True, exist_ok=True)
        dataset = PARENT / 'datasets' / f"g{task['generation_seed']}" / pending[0]
        validate_dataset(dataset, parent, task['generation_seed'], pending[0])
        # Full 100-class / 1000-image / batch-128 memory and architecture check.
        # Diagnostic output is excluded from the evaluation directory and summaries.
        print(f'Task {task_id}: {task}; running full-size one-epoch GPU diagnostic.', flush=True)
        torch.cuda.reset_peak_memory_stats()
        smoke_cfg = dict(cfg, epochs=1, eval_interval=1, save_checkpoints=False)
        # measure() requires every class, so use one report image from each class.
        selected = {}
        for pair in reporting['paths']:
            selected.setdefault(pair[1], pair)
        smoke_batches = loader(reporting['root'], list(selected.values()), cfg['classifier_size'], cfg['batch_size'], 0)
        train_one(dataset, plan['classes'], smoke_cfg, task['training_seed'], task['architecture'],
                  smoke_batches, None, smoke_root / 'diagnostic.json', {'smoke_test_only': True, 'plan_hash': plan['plan_hash']})
        write(smoke_root / 'SMOKE_PASSED.json', {'created': now(), 'architecture': task['architecture'],
              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
              'batch_size': cfg['batch_size'], 'ipc': cfg['ipc'], 'nclass': len(plan['classes']),
              'hardware': machine, 'smoke_test_only': True})
        for name in pending:
            state.update(condition=name, updated=now())
            write(folder / f'{task_id:02d}.json', state)
            dataset = PARENT / 'datasets' / f"g{task['generation_seed']}" / name
            validate_dataset(dataset, parent, task['generation_seed'], name)
            provenance = {'plan_hash': plan['plan_hash'], 'condition': name,
                'generation_seed': task['generation_seed'], 'source_generation_plan_hash': parent['plan_hash'],
                'source_generation_directory': str(dataset), 'generation_manifest_sha256': sha(dataset / 'manifest.json'),
                'prepared_manifest_sha256': plan['prepared_manifest_hash']}
            print(f'Task {task_id}: starting {name}, {task["architecture"]}, generation={task["generation_seed"]}, training={task["training_seed"]}.', flush=True)
            train_one(dataset, plan['classes'], cfg, task['training_seed'], task['architecture'],
                      batches, None, result_path(task, name), provenance)
        state.update(state='complete', finished=now())
        write(folder / f'{task_id:02d}.json', state)
    except BaseException:
        state.update(state='failed', finished=now(), traceback=traceback.format_exc())
        write(folder / f'{task_id:02d}.json', state)
        raise
    finally:
        lock.unlink()


def status():
    plan = checked_plan(RUN / 'plan.json')
    complete = []
    for task in tasks_for(plan):
        for name in plan['conditions']:
            path = result_path(task, name)
            if path.exists():
                record = read(path)
                if record.get('plan_hash') == plan['plan_hash'] and 'metrics' in record:
                    complete.append(record)
    states = [read(p) for p in (WORK / 'task_state').glob('*.json')]
    counts = {state: sum(x['state'] == state for x in states) for state in ['running', 'complete', 'failed']}
    print(f'Classifier fits: {len(complete)}/180; task state files: {counts}')
    for arch in ARCHITECTURES:
        done = [r for r in complete if r['architecture'] == arch]
        print(f'{arch}: {len(done)}/90 fits', end='')
        if done:
            print(f'; median fit {statistics.median(r["elapsed_s"] for r in done)/60:.1f} min')
        else:
            print('; GPU runtime not measured yet')
    for row in states:
        if row['state'] == 'running':
            task = {key: row[key] for key in ['architecture', 'generation_seed', 'training_seed']}
            name = row.get('condition')
            history = result_path(task, name).with_suffix('.history.jsonl') if name else None
            last = history.read_text().splitlines()[-1] if history and history.exists() and history.stat().st_size else 'GPU diagnostic / starting'
            print(f'Task {row["task_id"]}: {name or "diagnostic"}; {last}')
    if (HERE / 'submission.json').exists():
        record = read(HERE / 'submission.json')
        print('Slurm array:', record.get('array_job_id'), '; summary job:', record.get('summary_job_id'))
    else:
        print('Not submitted yet.')
    print(f'Results: {RUN}')
    print('State files describe recorded progress; scheduler state is available via squeue.')


def summarize():
    plan = validate_setup()
    for task in tasks_for(plan):
        for name in plan['conditions']:
            if not existing_result_ok(result_path(task, name), plan, task, name):
                raise ValueError(f'Missing result: {task}, {name}; refusing incomplete inference.')
    activate_snapshot()
    from rebuttal.statistics import summarize as original_summary
    output = original_summary(RUN)
    output['scope'] += '; evaluation-only extension reusing the same completed core synthetic datasets across architectures'
    from rebuttal.statistics import summary_text
    write(RUN / 'summary.json', output)
    (RUN / 'summary.txt').write_text(summary_text(output, plan))
    write(RUN / 'runtime.json', {'complete': True, 'expected_fits': 180, 'measured_fits': 180,
          'reused_datasets': 30, 'new_generation_tasks': 0, 'estimated_remaining_gpu_hours': 0})
    print('Complete. Summary:', RUN / 'summary.txt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'submit', 'status', 'worker', 'summarize'])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--task-id', type=int)
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'submit': submit(args.dry_run)
    elif args.action == 'status': status()
    elif args.action == 'worker': worker(args.task_id)
    elif args.action == 'summarize': summarize()


if __name__ == '__main__':
    main()
