"""Finish cross-architecture paired blocks with at most two A800 GPUs.

Only complete six-condition RTX 4090 blocks are reused. Partial blocks restart
on A800, in a separate directory, with the original frozen evaluator/protocol.
"""
from __future__ import annotations

import argparse
import copy
import getpass
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import signal
import statistics
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'experiments/rebuttal_crossarch_v1'
OLD_RUN = ROOT / 'results/rebuttal_crossarch_v1/runs/final'
WORK = ROOT / 'results/rebuttal_crossarch_a800_v1'
RUN = WORK / 'runs/final'
CODE = OLD / 'code'
PYTHON = ROOT / 'ICLR/bin/python'
spec = importlib.util.spec_from_file_location('cags_crossarch_original', OLD / 'runner.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
from guard.coordination import locked, recover, TERMINAL

read, write, sha, digest = legacy.read, legacy.write, legacy.sha, legacy.digest
tasks_for, now = legacy.tasks_for, legacy.now
SCRIPT_NAMES = ['runner.py', 'submit.sh', 'job.sh', 'summary_job.sh',
                'guard/__init__.py', 'guard/common.py', 'guard/coordination.py']


def output_path(run, task, condition):
    return run / 'evaluation' / f"g{task['generation_seed']}" / f"t{task['training_seed']}" / task['architecture'] / f'{condition}.json'


def reusable_blocks(plan):
    """No partial RTX block may contribute to an A800 paired comparison."""
    reused, pending, partial = {}, [], []
    for task_id, task in enumerate(tasks_for(plan)):
        records = []
        for condition in plan['conditions']:
            path = output_path(OLD_RUN, task, condition)
            if legacy.existing_result_ok(path, plan, task, condition, 'NVIDIA GeForce RTX 4090'):
                result = read(path)
                dataset = legacy.PARENT / 'datasets' / f"g{task['generation_seed']}" / condition
                if result.get('generation_manifest_sha256') != sha(dataset / 'manifest.json'):
                    raise ValueError(f'Result used a different generated dataset: {path}')
                records.append({'source': str(path), 'sha256': sha(path),
                                'destination': str(path.relative_to(OLD_RUN))})
        if len(records) == len(plan['conditions']):
            reused[str(task_id)] = records
        else:
            pending.append(task_id)
            partial.extend(records)
    return reused, pending, partial


def prepare():
    if (HERE / 'frozen_manifest.json').exists():
        return validate_setup()
    if (HERE / 'submission.json').exists():
        raise ValueError('Submission exists without a frozen plan; refusing to rebuild.')
    old_plan = legacy.validate_setup()
    reused, pending, partial = reusable_blocks(old_plan)
    plan = copy.deepcopy(old_plan)
    plan.pop('plan_hash')
    plan['execution_extension'] = {name: sha(HERE / name) for name in SCRIPT_NAMES}
    plan['hardware_migration'] = {
        'source_run': str(OLD_RUN), 'source_plan_hash': old_plan['plan_hash'],
        'source_launcher_sha256': sha(OLD / 'runner.py'),
        'source_submission': read(OLD / 'submission.json'),
        'reused_blocks': reused, 'a800_task_ids': pending,
        'partial_results_not_reused': partial,
        'selection_rule': 'Reuse every complete six-condition block; restart every incomplete block on A800. Selection does not depend on accuracies.',
        'hardware_rule': 'All conditions within an architecture/generation/training-seed block use the same exact GPU model. Completed RTX blocks remain explicitly identified.',
        'max_concurrent_gpus': 2,
        'protocol_changes': ['GPU hardware for unfinished paired blocks only; generation is reused.'],
    }
    plan['plan_hash'] = digest(plan)
    if (RUN / 'plan.json').exists() and read(RUN / 'plan.json') != plan:
        raise ValueError('Incomplete preparation differs from source records; preserve it and inspect.')
    write(RUN / 'plan.json', plan)
    reporting = OLD_RUN / 'evaluation_images.json'
    write(RUN / 'evaluation_images.json', read(reporting))
    imports = {}
    for records in reused.values():
        for entry in records:
            original = read(entry['source'])
            result = dict(original, plan_hash=plan['plan_hash'],
                          source_evaluation_plan_hash=old_plan['plan_hash'],
                          reused_evaluation={'file': entry['source'], 'sha256': entry['sha256'],
                                             'note': 'Unchanged measured result from a complete RTX 4090 paired block.'})
            path = RUN / entry['destination']
            if path.exists() and read(path) != result:
                raise ValueError(f'Conflicting imported result: {path}')
            write(path, result)
            checkpoint = path.with_suffix('.pt')
            original_checkpoint = Path(entry['source']).with_suffix('.pt')
            if plan['protocol']['save_checkpoints']:
                if checkpoint.is_symlink():
                    if checkpoint.resolve() != original_checkpoint.resolve():
                        raise ValueError(f'Conflicting checkpoint link: {checkpoint}')
                elif checkpoint.exists():
                    raise ValueError(f'Refusing to replace checkpoint: {checkpoint}')
                else:
                    checkpoint.symlink_to(original_checkpoint)
            imports[entry['destination']] = sha(path)
    write(HERE / 'frozen_manifest.json', {
        'created': now(), 'plan_hash': plan['plan_hash'],
        'scripts': plan['execution_extension'], 'imports': imports,
        'reporting_manifest_sha256': sha(RUN / 'evaluation_images.json'),
        'source_files': old_plan['source'], 'expected_fits': 180,
        'reused_fits': len(imports), 'new_fits': len(pending) * len(plan['conditions']),
        'restarted_completed_fits_in_partial_blocks': len(partial),
    })
    write(RUN / 'READY.json', {'created': now(), 'plan_hash': plan['plan_hash']})
    return validate_setup()


def validate_setup():
    frozen = read(HERE / 'frozen_manifest.json')
    plan = legacy.checked_plan(RUN / 'plan.json')
    old_plan = legacy.validate_setup()
    migration = plan['hardware_migration']
    if plan['plan_hash'] != frozen['plan_hash'] or old_plan['plan_hash'] != migration['source_plan_hash']:
        raise ValueError('Frozen plan identity changed.')
    if sha(OLD / 'runner.py') != migration['source_launcher_sha256']:
        raise ValueError('Source launcher changed.')
    for name, checksum in frozen['scripts'].items():
        if sha(HERE / name) != checksum:
            raise ValueError(f'Frozen A800 launcher changed: {name}')
    for name in ['protocol', 'classes', 'conditions', 'architectures', 'generation_seeds', 'training_seeds', 'source', 'environment']:
        if plan[name] != old_plan[name]:
            raise ValueError(f'Unexpected protocol change: {name}')
    if sha(RUN / 'evaluation_images.json') != frozen['reporting_manifest_sha256']:
        raise ValueError('Reporting image inventory changed.')
    for records in migration['reused_blocks'].values():
        for entry in records:
            if sha(entry['source']) != entry['sha256'] or sha(RUN / entry['destination']) != frozen['imports'][entry['destination']]:
                raise ValueError(f'Reused result changed: {entry["source"]}')
    return plan


def query(command, timeout=15):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{shlex.join(command)} failed: {result.stderr.strip()}')
    return result.stdout


def require_terminal(job_id):
    """An empty queue is insufficient proof: accounting must confirm termination."""
    text = query(['sacct', '-n', '-X', '-j', str(job_id), '-o', 'JobIDRaw,State', '-P'])
    states = [line.split('|')[1].split()[0].rstrip('+') for line in text.splitlines()
              if len(line.split('|')) >= 2 and line.split('|')[1].strip()]
    if not states or not all(state in TERMINAL for state in states):
        raise RuntimeError(f'Cannot confirm job/array {job_id} has ended: {states}. No overlapping batch was submitted.')


def account_options(dry_run=False):
    explicit_qos = os.environ.get('CAGS_A800_QOS')
    explicit_account = os.environ.get('CAGS_A800_ACCOUNT')
    if explicit_qos or dry_run:
        return {'qos': explicit_qos or 'a800', 'account': explicit_account}
    text = query(['sacctmgr', '-nP', 'show', 'assoc', f'user={getpass.getuser()}',
                  'format=Account,Partition,QOS,DefaultQOS'])
    for line in text.splitlines():
        parts = line.split('|')
        if len(parts) < 4:
            continue
        account, partition, qos_list, default = [p.strip() for p in parts[:4]]
        if partition not in {'', 'a800'} or (explicit_account and explicit_account != account):
            continue
        allowed = set(qos_list.split(','))
        for candidate in [default, 'a800', 'a800-2', 'a800-3']:
            if candidate in {'a800', 'a800-2', 'a800-3'} and candidate in allowed:
                return {'qos': candidate, 'account': explicit_account or account}
    raise RuntimeError('No usable A800 QOS was found for this account. Set CAGS_A800_QOS and, if needed, CAGS_A800_ACCOUNT to your allocated values; no job was submitted.')


def scheduler_commands(task_ids, options):
    if not task_ids:
        raise ValueError('No GPU tasks remain.')
    common = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--export=ALL', '--chdir=' + str(CODE)]
    account = ['--account=' + options['account']] if options.get('account') else []
    gpu = common + account + ['--partition=a800', '--qos=' + options['qos'], '--gres=gpu:a800:1',
        '--cpus-per-task=8', '--mem=32G', '--job-name=cags-xarch-a800',
        '--array=' + ','.join(map(str, task_ids)) + '%2',
        '--output=' + str(WORK / 'slurm/eval-%A_%a.log'), '--error=' + str(WORK / 'slurm/eval-%A_%a.log'),
        str(HERE / 'job.sh'), str(HERE), str(PYTHON)]
    cpu = common + account + ['--partition=cpu', '--cpus-per-task=2', '--mem=8G',
        '--job-name=cags-xarch-a800-report', '--output=' + str(WORK / 'slurm/summary-%j.log'),
        '--error=' + str(WORK / 'slurm/summary-%j.log')]
    return gpu, cpu


def invoke_submission(command, record, field):
    record[field + '_status'] = 'submitting'
    record[field + '_command'] = command
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
        raise RuntimeError('Submission outcome unknown; inspect Slurm before retrying.') from error
    if result.returncode:
        record[field + '_status'] = 'rejected'
        record['last_error'] = result.stderr.strip()
        write(HERE / 'submission.json', record)
        raise RuntimeError(f'Slurm rejected submission: {result.stderr.strip()}')
    value = result.stdout.strip().split(';')[0]
    if not re.fullmatch(r'\d+', value):
        record[field + '_status'] = 'unknown'
        record['last_error'] = result.stdout
        write(HERE / 'submission.json', record)
        raise RuntimeError('Unrecognized sbatch response; will not resubmit automatically.')
    record[field], record[field + '_status'] = value, 'submitted'
    write(HERE / 'submission.json', record)
    return value


def remaining_tasks(plan):
    tasks = tasks_for(plan)
    return [i for i in plan['hardware_migration']['a800_task_ids']
            if not all(legacy.existing_result_ok(output_path(RUN, tasks[i], c), plan, tasks[i], c)
                       for c in plan['conditions'])]


def submit(dry_run=False, retry_failed=False):
    plan = validate_setup()
    old_job = plan['hardware_migration']['source_submission']['array_job_id']
    if dry_run:
        tasks = remaining_tasks(plan)
        if not tasks:
            print('All paired blocks complete; only summary may remain.')
            return
        gpu, cpu = scheduler_commands(tasks, account_options(dry_run=True))
        print(shlex.join(gpu))
        print(shlex.join(cpu + ['--dependency=afterok:ARRAY_ID', '--kill-on-invalid-dep=yes', str(HERE / 'summary_job.sh'), str(HERE), str(PYTHON)]))
        print(f'Preview: {len(tasks)} paired blocks, at most 2 A800 GPUs. Real submission discovers your A800 QOS/account and verifies old array {old_job} is terminal. No job submitted.')
        return
    if os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run the submission command from a login terminal, outside an existing allocation.')
    recover(HERE / '.submission.lock')
    with locked(HERE / '.submission.lock', blocking=False):
        record = read(HERE / 'submission.json') if (HERE / 'submission.json').exists() else None
        if record and record['plan_hash'] != plan['plan_hash']:
            raise ValueError('Submission belongs to another plan.')
        if record and any(record.get(f + '_status') in {'unknown', 'submitting'} for f in ['array_job_id', 'summary_job_id']):
            raise RuntimeError('Unresolved previous submission; inspect Slurm before retrying.')
        if record and retry_failed and record.get('array_job_id'):
            require_terminal(record['array_job_id'])
            write(HERE / f'submission-{record["array_job_id"]}.json', record)
            for i in remaining_tasks(plan):
                recover(WORK / 'task_state' / f'{i:02d}.lock')
            record = None
        if record and record.get('array_job_id') and record.get('summary_job_id'):
            print(f'Already submitted: GPU array {record["array_job_id"]}, CPU summary {record["summary_job_id"]}. No duplicate jobs created. Use --status; use --retry-failed only after the batch has ended.')
            return
        require_terminal(old_job)
        tasks = remaining_tasks(plan)
        if not tasks:
            summarize()
            return
        options = record.get('account_options') if record else account_options()
        gpu, cpu = scheduler_commands(tasks, options)
        (WORK / 'slurm').mkdir(parents=True, exist_ok=True)
        record = record or {'created': now(), 'plan_hash': plan['plan_hash'], 'max_parallel_gpus': 2,
                            'task_ids': tasks, 'account_options': options}
        if not record.get('array_job_id'):
            # Validate both resource requests before committing a GPU allocation.
            query([gpu[0], '--test-only'] + gpu[1:])
            query([cpu[0], '--test-only'] + cpu[1:] + [str(HERE / 'summary_job.sh'), str(HERE), str(PYTHON)])
            invoke_submission(gpu, record, 'array_job_id')
        if not record.get('summary_job_id'):
            invoke_submission(cpu + [f'--dependency=afterok:{record["array_job_id"]}', '--kill-on-invalid-dep=yes',
                                     str(HERE / 'summary_job.sh'), str(HERE), str(PYTHON)], record, 'summary_job_id')
        print(f'Submitted GPU array {record["array_job_id"]}: at most 2 A800 GPUs; CPU summary {record["summary_job_id"]}.')
        print(f'Logs: {WORK / "slurm"}\nResults: {RUN}')


def worker(task_id):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Worker requires a Slurm allocation.')
    plan = validate_setup()
    if task_id not in plan['hardware_migration']['a800_task_ids']:
        raise ValueError('This task is not assigned to A800; completed RTX blocks must not be rerun.')
    task = tasks_for(plan)[task_id]
    legacy.activate_snapshot()
    import torch
    from rebuttal.common import hardware
    from rebuttal.evaluate import loader, image_manifest, train_one
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible CUDA GPU per task.')
    machine = hardware()
    if not re.search(r'\bA800\b', machine['gpu'], re.IGNORECASE):
        raise RuntimeError(f'Expected A800; received {machine["gpu"]}.')
    if machine['versions'] != plan['environment']:
        raise RuntimeError('Python package versions differ from the frozen evaluation protocol.')
    cfg = plan['protocol']
    reporting = read(RUN / 'evaluation_images.json')
    if reporting['root'] != cfg['evaluation_root'] or image_manifest(reporting['root'], reporting['paths']) != cfg['evaluation_image_manifest']:
        raise RuntimeError('Reporting images changed.')
    parent = legacy.checked_plan(legacy.PARENT / 'plan.json')
    state_path = WORK / 'task_state' / f'{task_id:02d}.json'
    with locked(WORK / 'task_state' / f'{task_id:02d}.lock', blocking=False):
        binding = WORK / 'task_state' / f'{task_id:02d}.gpu.json'
        if binding.exists() and read(binding)['gpu'] != machine['gpu']:
            raise RuntimeError('Resume this paired block on the same A800 model; mixing GPU variants is forbidden.')
        if not binding.exists():
            write(binding, machine)
        pending = [c for c in plan['conditions'] if not legacy.existing_result_ok(output_path(RUN, task, c), plan, task, c, machine['gpu'])]
        if not pending:
            print(f'Task {task_id} already complete; no training repeated.')
            return
        state = dict(task, task_id=task_id, job_id=os.environ['SLURM_JOB_ID'], state='running', started=now(), hardware=machine)
        write(state_path, state)
        def stop(signum, frame):
            raise KeyboardInterrupt('Slurm terminated the worker.')
        signal.signal(signal.SIGTERM, stop)
        try:
            batches = loader(reporting['root'], reporting['paths'], cfg['classifier_size'], cfg['batch_size'], cfg['num_workers'])
            dataset = legacy.PARENT / 'datasets' / f"g{task['generation_seed']}" / pending[0]
            legacy.validate_dataset(dataset, parent, task['generation_seed'], pending[0])
            smoke_root = WORK / 'smoke_tests' / f'task_{task_id:02d}'
            smoke_root.mkdir(parents=True, exist_ok=True)
            selected = {}
            for pair in reporting['paths']:
                selected.setdefault(pair[1], pair)
            smoke_loader = loader(reporting['root'], list(selected.values()), cfg['classifier_size'], cfg['batch_size'], 0)
            torch.cuda.reset_peak_memory_stats()
            print(f'Task {task_id}: 20-epoch memory/speed diagnostic; excluded from reported experiments.', flush=True)
            train_one(dataset, plan['classes'], dict(cfg, epochs=20, eval_interval=5, save_checkpoints=False),
                      task['training_seed'], task['architecture'], smoke_loader, None, smoke_root / 'diagnostic.json',
                      {'plan_hash': plan['plan_hash'], 'smoke_test_only': True})
            history = [json.loads(s) for s in (smoke_root / 'diagnostic.history.jsonl').read_text().splitlines()]
            seconds_per_epoch = (history[-1]['elapsed_s'] - history[-3]['elapsed_s']) / (history[-1]['epoch'] - history[-3]['epoch'])
            write(smoke_root / 'SMOKE_PASSED.json', {'created': now(), 'architecture': task['architecture'],
                'hardware': machine, 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                'batch_size': cfg['batch_size'], 'precision': cfg['precision'], 'nclass': len(plan['classes']),
                'measured_seconds_per_epoch': seconds_per_epoch,
                'estimated_2000_epoch_minutes': seconds_per_epoch * cfg['epochs'] / 60,
                'estimate_scope': 'Short diagnostic after warmup; excludes final evaluation/checkpoint and may differ from sustained throughput.',
                'smoke_test_only': True})
            print(f'A800 diagnostic estimate: {seconds_per_epoch * cfg["epochs"] / 60:.1f} minutes per full {task["architecture"]} fit; measure sustained results before claiming a speedup.', flush=True)
            for condition in pending:
                state.update(condition=condition, updated=now())
                write(state_path, state)
                dataset = legacy.PARENT / 'datasets' / f"g{task['generation_seed']}" / condition
                legacy.validate_dataset(dataset, parent, task['generation_seed'], condition)
                provenance = {'plan_hash': plan['plan_hash'], 'condition': condition,
                    'generation_seed': task['generation_seed'], 'source_generation_plan_hash': parent['plan_hash'],
                    'source_generation_directory': str(dataset), 'generation_manifest_sha256': sha(dataset / 'manifest.json'),
                    'prepared_manifest_sha256': plan['prepared_manifest_hash']}
                print(f'START task={task_id} {task} condition={condition} GPU={machine["gpu"]}', flush=True)
                train_one(dataset, plan['classes'], cfg, task['training_seed'], task['architecture'],
                          batches, None, output_path(RUN, task, condition), provenance)
            state.update(state='complete', finished=now())
            write(state_path, state)
        except BaseException:
            state.update(state='failed', finished=now(), traceback=traceback.format_exc())
            write(state_path, state)
            raise


def status():
    plan = validate_setup()
    imported = len(read(HERE / 'frozen_manifest.json')['imports'])
    rows = [read(p) for p in RUN.glob('evaluation/*/*/*/*.json')]
    print(f'Total: {len(rows)}/180 fits; reused RTX 4090 fits: {imported}; completed new A800 fits: {len(rows)-imported}.')
    for arch in plan['architectures']:
        fresh = [r for r in rows if r['architecture'] == arch and 'reused_evaluation' not in r]
        benchmarks = [read(p) for p in (WORK / 'smoke_tests').glob('*/SMOKE_PASSED.json') if read(p)['architecture'] == arch]
        if fresh:
            print(f'{arch}: measured A800 median fit {statistics.median(r["elapsed_s"] for r in fresh)/60:.1f} min ({len(fresh)} fits).')
        elif benchmarks:
            print(f'{arch}: diagnostic estimate {statistics.median(r["estimated_2000_epoch_minutes"] for r in benchmarks):.1f} min; full-fit runtime not measured yet.')
    if (HERE / 'submission.json').exists():
        record = read(HERE / 'submission.json')
        print('Recorded GPU array:', record.get('array_job_id'), '; summary:', record.get('summary_job_id'))
    else:
        print('Prepared, not submitted.')
    for p in sorted((WORK / 'task_state').glob('[0-9][0-9].json')):
        s = read(p)
        if s['state'] != 'complete':
            print(f'Task {s["task_id"]}: recorded state={s["state"]}; condition={s.get("condition", "diagnostic")}; updated={s.get("updated", s["started"])}')
    print(f'Results: {RUN}\nLogs: {WORK / "slurm"}\nFile states do not replace live scheduler status.')


def summarize():
    plan = validate_setup()
    hardware_blocks = []
    for task_id, task in enumerate(tasks_for(plan)):
        rows = []
        for condition in plan['conditions']:
            path = output_path(RUN, task, condition)
            if not legacy.existing_result_ok(path, plan, task, condition):
                raise ValueError(f'Missing result: {task}, {condition}')
            rows.append(read(path))
        gpu_models = {r['hardware']['gpu'] for r in rows}
        if len(gpu_models) != 1:
            raise ValueError(f'Mixed GPU models within paired block {task_id}')
        model = next(iter(gpu_models))
        if task_id in plan['hardware_migration']['a800_task_ids'] and not re.search(r'\bA800\b', model, re.IGNORECASE):
            raise ValueError(f'Expected A800 results in task {task_id}')
        hardware_blocks.append(dict(task, gpu=model, reused=str(task_id) in plan['hardware_migration']['reused_blocks']))
    legacy.activate_snapshot()
    from rebuttal.statistics import summarize as original_summary, summary_text
    output = original_summary(RUN)
    output['hardware_blocks'] = hardware_blocks
    output['scope'] += '; hardware varies between paired blocks (complete RTX 4090 blocks reused; remaining blocks trained on A800), never between conditions within a block. No pure single-GPU-model replication is claimed.'
    write(RUN / 'summary.json', output)
    (RUN / 'summary.txt').write_text(summary_text(output, plan))
    write(RUN / 'runtime.json', {'complete': True, 'expected_fits': 180, 'measured_fits': 180,
          'reused_datasets': len(plan['generation_reuse']['dataset_manifest_sha256']),
          'new_generation_tasks': 0, 'estimated_remaining_gpu_hours': 0,
          'reused_rtx4090_fits': len(read(HERE / 'frozen_manifest.json')['imports']),
          'hardware_blocks': hardware_blocks})
    print('Complete:', RUN / 'summary.txt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'submit', 'worker', 'status', 'summarize'])
    parser.add_argument('--task-id', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'submit':
        submit(args.dry_run, args.retry_failed)
    elif args.action == 'worker':
        worker(args.task_id)
    elif args.action == 'status':
        status()
    else:
        summarize()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'A800 experiment: {type(error).__name__}: {error}', file=sys.stderr)
        sys.exit(1)
