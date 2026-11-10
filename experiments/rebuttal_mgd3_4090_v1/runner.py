"""MGD3 official stop_t=25 under the completed core experiment's protocol.

Only five new datasets and fifteen ConvNet6 fits. Forty-five paired baseline
results are imported with original provenance; no baseline training is queued.
"""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
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
CODE = HERE / 'code'
WORK = ROOT / 'results/rebuttal_mgd3_4090_v1'
RUN = WORK / 'runs/final'
PARENT = ROOT / 'results/rebuttal_core_v2/runs/final'
PYTHON = ROOT / 'ICLR/bin/python'
BASELINES = ['cags', 'fixed_mean', 'no_mode']
METHOD = 'mgd3_shared_stop25'
GPU = 'NVIDIA GeForce RTX 4090'
sys.path.insert(0, str(CODE))
from rebuttal.common import (atomic_json as write, read_json as read, read_plan,
                            file_hash as sha, digest, now, source_manifest)
from guard.coordination import locked, recover, TERMINAL


def experiment_files():
    result = [HERE / name for name in ['runner.py', 'test_launcher.py', 'job.sh']]
    for folder in ['code', 'guard', 'reference']:
        result += [p for p in (HERE / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    result.append(ROOT / 'scripts/submit_rebuttal_mgd3_4090.sh')
    return sorted(result)


def result_path(run, g, t, name):
    return run / 'evaluation' / f'g{g}' / f't{t}' / 'convnet6' / (name + '.json')


def dataset_identity(plan, g, name):
    return digest({'plan_hash': plan['plan_hash'], 'generation_seed': g,
                   'condition': name, 'specification': plan['conditions'][name]})


def check_result(path, plan, g, t, name, manifest_hash=None):
    r = read(path)
    expected = {'plan_hash': plan['plan_hash'], 'generation_seed': g, 'train_seed': t,
                'condition': name, 'architecture': 'convnet6', 'checkpoint_rule': 'final',
                'selected_epoch': plan['protocol']['epochs'], 'prepared_manifest_sha256': plan['prepared_manifest_hash']}
    for key, value in expected.items():
        if r.get(key) != value:
            raise ValueError(f'Invalid result identity {key}: {path}')
    if r['hardware']['gpu'] != GPU or r['hardware']['versions'] != plan['environment']:
        raise ValueError('Paired result hardware/environment mismatch: ' + str(path))
    if manifest_hash and r['generation_manifest_sha256'] != manifest_hash:
        raise ValueError('Result used a different dataset: ' + str(path))
    import math
    if not math.isfinite(r['metrics']['top1']) or not 0 <= r['metrics']['top1'] <= 100:
        raise ValueError('Invalid accuracy: ' + str(path))
    return r


def verify_parent_dataset(parent, g, name, full=False):
    folder = PARENT / 'datasets' / f'g{g}' / name
    m = read(folder / 'manifest.json')
    if m['configuration_hash'] != dataset_identity(parent, g, name):
        raise ValueError('Parent generation identity differs: ' + str(folder))
    if m['hardware']['gpu'] != GPU or m['hardware']['versions'] != parent['environment']:
        raise ValueError('Parent generation hardware/environment differs.')
    expected = {f'{c}/{i:04d}.png' for c in parent['classes'] for i in range(parent['protocol']['ipc'])}
    if set(m['images']) != expected:
        raise ValueError('Parent image inventory differs.')
    if full:
        if {str(p.relative_to(folder)) for p in folder.rglob('*.png')} != expected:
            raise ValueError('Parent image files differ.')
        def verify_image(item):
            rel, checksum = item
            if sha(folder / rel) != checksum:
                raise ValueError('Parent image changed: ' + str(folder / rel))
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(verify_image, m['images'].items()))
    return m


def prepare():
    if (HERE / 'frozen_manifest.json').exists():
        validate()
        print('Existing frozen plan verified.')
        return
    if (HERE / 'submission.json').exists():
        raise RuntimeError('Submission exists without a frozen plan; refusing to rebuild.')
    parent = read_plan(PARENT / 'plan.json')
    summary = read(PARENT / 'summary.json')
    if not summary['complete'] or summary['missing'] or summary['plan_hash'] != parent['plan_hash']:
        raise ValueError('The paired source experiment must be complete.')
    if source_manifest() != parent['source']:
        raise ValueError('Snapshot differs from the completed core experiment.')
    if parent['generation_seeds'] != [100, 101, 102, 103, 104] or parent['training_seeds'] != [0, 1, 2]:
        raise ValueError('Unexpected parent seed grid.')
    if parent['architectures'] != ['convnet6'] or parent['protocol']['checkpoint_rule'] != 'final':
        raise ValueError('Unexpected evaluation protocol.')
    from rebuttal.common import hardware
    from rebuttal.evaluate import image_manifest
    if hardware()['versions'] != parent['environment']:
        raise ValueError('Python packages differ from the completed experiment.')
    reporting = read(PARENT / 'evaluation_images.json')
    if image_manifest(reporting['root'], reporting['paths']) != parent['protocol']['evaluation_image_manifest']:
        raise ValueError('Reporting image inventory changed.')
    if sha(parent['protocol']['checkpoint']) != parent['protocol']['checkpoint_sha256']:
        raise ValueError('DiT checkpoint changed.')
    manifests, originals = {}, {}
    for g in parent['generation_seeds']:
        noises = None
        for name in BASELINES:
            m = verify_parent_dataset(parent, g, name, full=True)
            if noises is not None and noises != m['noise_seeds']:
                raise ValueError('Original baselines were not paired.')
            noises = m['noise_seeds']
            rel = f'g{g}/{name}'
            manifests[rel] = sha(PARENT / 'datasets' / rel / 'manifest.json')
            for t in parent['training_seeds']:
                path = result_path(PARENT, g, t, name)
                check_result(path, parent, g, t, name, manifests[rel])
                originals[str(path.relative_to(PARENT))] = sha(path)
            print(f'Verified reusable {name}, generation {g}.', flush=True)
    plan = copy.deepcopy(parent)
    plan.pop('plan_hash')
    plan['suite'] = 'mgd3_official_window_matched_core'
    plan['conditions'] = {name: copy.deepcopy(parent['conditions'][name]) for name in BASELINES}
    plan['conditions'][METHOD] = {'coefficients': [.1] * len(plan['classes']), 'stop_t': 25,
        'kind': 'verified_mgd3_kernel_shared_protocol',
        'note': 'Official sampling update and example stop_t=25: 24 active steps of 50. Shared center-crop VAE-mean features, K=IPC targets, paired RNG, batch=4 and common evaluator. Not a reproduction of published table scores.'}
    plan['baseline_reuse'] = {'source_run': str(PARENT), 'source_plan_hash': parent['plan_hash'],
        'source_dataset_manifests': manifests, 'source_results': originals,
        'rule': 'Reuse ALL completed seeds for three prespecified references; no outcome-based filtering.',
        'new_generation_condition': METHOD, 'new_datasets': 5, 'new_fits': 15, 'reused_fits': 45}
    plan['extension_files'] = {str(p.relative_to(ROOT)): sha(p) for p in experiment_files()}
    plan['plan_hash'] = digest(plan)
    if (RUN / 'plan.json').exists() and read_plan(RUN / 'plan.json')['plan_hash'] != plan['plan_hash']:
        raise ValueError('A different immutable plan already exists.')
    write(RUN / 'plan.json', plan)
    write(RUN / 'evaluation_images.json', reporting)
    imported = {}
    for rel, checksum in originals.items():
        record = read(PARENT / rel)
        record['plan_hash'] = plan['plan_hash']
        record['reused_result'] = {'source': str(PARENT / rel), 'source_sha256': checksum,
                                   'source_plan_hash': parent['plan_hash']}
        destination = RUN / rel
        if destination.exists() and read(destination) != record:
            raise ValueError('Refusing to overwrite a different imported result: ' + str(destination))
        write(destination, record)
        imported[rel] = sha(destination)
    write(RUN / 'READY.json', {'created': now(), 'plan_hash': plan['plan_hash']})
    write(HERE / 'frozen_manifest.json', {'created': now(), 'plan_hash': plan['plan_hash'],
          'imports': imported, 'reporting_sha256': sha(RUN / 'evaluation_images.json')})
    validate()
    print('Prepared: 5 new MGD3 datasets + 15 fits; 45 baseline fits reused, 0 baseline fits scheduled.')


def validate():
    frozen = read(HERE / 'frozen_manifest.json')
    plan = read_plan(RUN / 'plan.json')
    if plan['plan_hash'] != frozen['plan_hash'] or read(RUN / 'READY.json')['plan_hash'] != plan['plan_hash']:
        raise ValueError('Frozen plan identity differs.')
    if source_manifest() != plan['source']:
        raise ValueError('Frozen source snapshot changed.')
    for rel, checksum in plan['extension_files'].items():
        if sha(ROOT / rel) != checksum:
            raise ValueError('Frozen launcher/reference changed: ' + rel)
    parent = read_plan(PARENT / 'plan.json')
    if parent['plan_hash'] != plan['baseline_reuse']['source_plan_hash']:
        raise ValueError('Parent identity changed.')
    for key in ['protocol', 'source', 'classes', 'generation_seeds', 'training_seeds', 'environment']:
        if plan[key] != parent[key]:
            raise ValueError('Pairing protocol changed: ' + key)
    prepared = Path(plan['protocol']['prepared'])
    if sha(prepared / 'manifest.json') != plan['prepared_manifest_hash']:
        raise ValueError('Preparation manifest changed.')
    for name in ['mode_targets.npz', 'split.json']:
        if sha(prepared / name) != plan['prepared_manifest']['files'][name]:
            raise ValueError('Shared preparation input changed: ' + name)
    if sha(RUN / 'evaluation_images.json') != frozen['reporting_sha256']:
        raise ValueError('Reporting list changed.')
    for rel, checksum in plan['baseline_reuse']['source_dataset_manifests'].items():
        if sha(PARENT / 'datasets' / rel / 'manifest.json') != checksum:
            raise ValueError('Reused dataset metadata changed: ' + rel)
    for rel, checksum in plan['baseline_reuse']['source_results'].items():
        if sha(PARENT / rel) != checksum or sha(RUN / rel) != frozen['imports'][rel]:
            raise ValueError('Reused baseline result changed: ' + rel)
    return plan


def require_gpu(plan):
    import torch
    from rebuttal.common import hardware
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one Slurm-allocated GPU.')
    machine = hardware()
    if machine['gpu'] != GPU or machine['versions'] != plan['environment']:
        raise ValueError('New runs must use the same RTX4090 model and package versions as the paired baselines.')
    return machine


def smoke(plan):
    passed = WORK / 'smoke_tests/PASSED.json'
    if passed.exists():
        if read(passed)['plan_hash'] != plan['plan_hash']:
            raise ValueError('Diagnostics belong to another experiment.')
        return
    import torch
    from rebuttal.generate import GenerationEngine
    from rebuttal.evaluate import train_one, loader
    cfg = plan['protocol']
    if sha(cfg['checkpoint']) != cfg['checkpoint_sha256']:
        raise ValueError('DiT checkpoint changed.')
    toy = copy.deepcopy(plan)
    toy['classes'] = plan['classes'][:2]
    toy['prepared_manifest']['imagenet_labels'] = plan['prepared_manifest']['imagenet_labels'][:2]
    toy['protocol']['ipc'] = 4
    toy['plan_hash'] = digest({'plan_hash': plan['plan_hash'], 'smoke_only': True})
    torch.cuda.reset_peak_memory_stats()
    engine = GenerationEngine(toy)
    try:
        engine.generate(toy, WORK / 'smoke_tests/generation', 100, METHOD,
                        {'coefficients': [.1, .1], 'stop_t': 25, 'smoke_test_only': True})
    finally:
        engine.close()
    generation_peak = torch.cuda.max_memory_allocated() / 2**30
    dataset = PARENT / 'datasets/g100/cags'
    paths = [(c + '/0000.png', i) for i, c in enumerate(plan['classes'])]
    torch.cuda.reset_peak_memory_stats()
    train_one(dataset, plan['classes'], dict(cfg, epochs=2, eval_interval=1, save_checkpoints=False),
        0, 'convnet6', loader(dataset, paths, cfg['classifier_size'], 128, 0), None,
        WORK / 'smoke_tests/classifier.json', {'plan_hash': plan['plan_hash'], 'smoke_test_only': True,
        'note': 'Two-epoch 100-class memory diagnostic; evaluation uses synthetic training images. Excluded from scientific results.'})
    write(passed, {'created': now(), 'plan_hash': plan['plan_hash'], 'smoke_test_only': True,
        'generation_peak_gib': generation_peak, 'training_peak_gib': torch.cuda.max_memory_allocated()/2**30})


def require_smoke(plan):
    if read(WORK / 'smoke_tests/PASSED.json')['plan_hash'] != plan['plan_hash']:
        raise ValueError('Matching GPU self-test has not passed.')


def verify_new_dataset(plan, seed):
    from rebuttal.generate import verify_dataset
    folder = RUN / 'datasets' / f'g{seed}' / METHOD
    m = verify_dataset(folder, dataset_identity(plan, seed, METHOD))
    if m['hardware']['gpu'] != GPU or m['hardware']['versions'] != plan['environment']:
        raise ValueError('New dataset hardware/environment differs from baselines.')
    reference = read(PARENT / 'datasets' / f'g{seed}/cags/manifest.json')
    if m['noise_seeds'] != reference['noise_seeds'] or set(m['images']) != set(reference['images']):
        raise ValueError('MGD3 and the reused baselines are not paired.')
    if m['active_guidance_steps'] != list(range(26, 50)):
        raise ValueError('Expected exact official example stopping boundary t > 25.')
    return m


def generate(plan, task_id):
    require_smoke(plan)
    seed = plan['generation_seeds'][task_id]
    folder = RUN / 'datasets' / f'g{seed}' / METHOD
    if (folder / 'manifest.json').exists():
        verify_new_dataset(plan, seed)
        return
    if sha(plan['protocol']['checkpoint']) != plan['protocol']['checkpoint_sha256']:
        raise ValueError('DiT checkpoint changed.')
    from rebuttal.generate import GenerationEngine
    engine = GenerationEngine(plan)
    try:
        engine.generate(plan, RUN, seed, METHOD, plan['conditions'][METHOD])
    finally:
        engine.close()
    verify_new_dataset(plan, seed)


def evaluate(plan, task_id):
    require_smoke(plan)
    seed, training = [(g, t) for g in plan['generation_seeds'] for t in plan['training_seeds']][task_id]
    verify_new_dataset(plan, seed)
    manifest_hash = sha(RUN / 'datasets' / f'g{seed}' / METHOD / 'manifest.json')
    output = result_path(RUN, seed, training, METHOD)
    if output.exists():
        check_result(output, plan, seed, training, METHOD, manifest_hash)
        return
    from rebuttal.run import execute_evaluation
    # All baseline result identities were checked by validate(). The original
    # evaluator skips those files; it trains only the new MGD3 condition.
    execute_evaluation(plan, RUN, {'generation_seed': seed, 'training_seed': training, 'architecture': 'convnet6'})
    check_result(output, plan, seed, training, METHOD, manifest_hash)


def summarize(plan):
    for g in plan['generation_seeds']:
        verify_new_dataset(plan, g)
        checksum = sha(RUN / 'datasets' / f'g{g}' / METHOD / 'manifest.json')
        for t in plan['training_seeds']:
            check_result(result_path(RUN, g, t, METHOD), plan, g, t, METHOD, checksum)
    from rebuttal.statistics import summarize as make_summary
    summary = make_summary(RUN)
    summary['scope'] += '; MGD3 official sampling update and stopping boundary under the common core protocol; baseline results reused with recorded provenance.'
    summary['baseline_reuse'] = plan['baseline_reuse']
    write(RUN / 'summary.json', summary)
    comparison = summary['by_architecture']['convnet6']['comparisons']['cags_minus_' + METHOD]
    write(RUN / 'mgd3_comparison.json', {'plan_hash': plan['plan_hash'],
        'reviewer_questions': ['7hx5 W1.3 / Q3', 'PrL8 Q1 (related protocol clarification)'],
        'comparison': 'CAGS minus MGD3 official stop_t=25 under the shared protocol',
        'architecture': 'convnet6', 'statistics': comparison,
        'scope': 'Common ImageNet100 IPC10 protocol, five paired generation replicates, averaging three classifier seeds. This is not a reproduction of the published paper accuracy tables.'})
    from rebuttal.statistics import summary_text
    (RUN / 'summary.txt').write_text(summary_text(summary, plan), encoding='utf-8')
    write(RUN / 'runtime.json', {'plan_hash': plan['plan_hash'], 'new_generation_datasets': 5,
        'new_fits': 15, 'reused_fits': 45, 'remaining_gpu_work': 0, 'complete': True})
    print('Complete: ' + str(RUN / 'summary.txt'))


def run_stage(action, task_id):
    plan = validate()
    if action != 'summarize':
        machine = require_gpu(plan)
    else:
        machine = {'gpu': 'CPU'}
    with locked(WORK / 'locks' / f'{action}_{task_id}.lock', blocking=False):
        state = {'stage': action, 'task_id': task_id, 'plan_hash': plan['plan_hash'],
                 'job_id': os.environ.get('SLURM_JOB_ID'), 'hardware': machine,
                 'state': 'running', 'started': now()}
        output = WORK / 'task_state' / f'{action}_{task_id}.json'
        write(output, state)
        try:
            if action == 'smoke':
                smoke(plan)
            elif action == 'generate':
                generate(plan, task_id)
            elif action == 'evaluate':
                evaluate(plan, task_id)
            else:
                summarize(plan)
        except BaseException:
            write(output, dict(state, state='failed', finished=now(), traceback=traceback.format_exc()))
            raise
        write(output, dict(state, state='complete', finished=now()))


def query(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=40)
    if result.returncode:
        raise RuntimeError(shlex.join(command) + ': ' + result.stderr.strip())
    return result.stdout


def scheduler_requests(parallel, options):
    common = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--export=ALL',
              '--account=' + options['account'], '--chdir=' + str(CODE)]
    gpu = ['--partition=rtx4090', '--gres=gpu:4090:1', '--cpus-per-task=8', '--mem=32G']
    if options.get('qos'):
        gpu += ['--qos=' + options['qos']]
    result = []
    stages = [('smoke', gpu, None, None), ('generate', gpu, f'0-4%{parallel}', 'smoke'),
              ('evaluate', gpu, f'0-14%{parallel}', 'generate'),
              ('summarize', ['--partition=cpu', '--cpus-per-task=2', '--mem=8G'], None, 'evaluate')]
    for name, resources, array, dependency in stages:
        command = common + resources + ['--job-name=cags-mgd3-' + name,
            '--output=' + str(WORK / 'slurm' / (name + '-%A_%a.log')),
            '--error=' + str(WORK / 'slurm' / (name + '-%A_%a.log'))]
        if array:
            command += ['--array=' + array]
        result.append((name, command, dependency))
    return result


def tail(stage):
    return [str(HERE / 'job.sh'), str(HERE), str(PYTHON), stage]


def invoke(command, record, stage):
    record['stages'][stage] = {'status': 'submitting', 'command': command}
    write(HERE / 'submission.json', record)
    env = dict(os.environ)
    for key in ['CUDA_VISIBLE_DEVICES', 'REBUTTAL_GPUS']:
        env.pop(key, None)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=45, env=env)
    except (OSError, subprocess.TimeoutExpired) as error:
        record['stages'][stage].update(status='unknown', error=str(error))
        write(HERE / 'submission.json', record)
        raise RuntimeError('Submission outcome unknown. Inspect Slurm before retrying.') from error
    if result.returncode:
        record['stages'][stage].update(status='rejected', error=result.stderr.strip())
        write(HERE / 'submission.json', record)
        raise RuntimeError('Slurm rejected ' + stage + ': ' + result.stderr.strip())
    job = result.stdout.strip().split(';')[0]
    if not re.fullmatch(r'\d+', job):
        record['stages'][stage].update(status='unknown', error=result.stdout)
        write(HERE / 'submission.json', record)
        raise RuntimeError('Unexpected sbatch response; no blind duplicate submission allowed.')
    record['stages'][stage].update(status='submitted', job_id=job)
    write(HERE / 'submission.json', record)
    print(f'{stage}: submitted {job}', flush=True)


def require_terminal(record):
    for value in record['stages'].values():
        if 'job_id' not in value:
            continue
        output = query(['sacct', '-n', '-X', '-j', value['job_id'], '-o', 'JobIDRaw,State', '-P'])
        states = [l.split('|')[1].split()[0].rstrip('+') for l in output.splitlines()
                  if len(l.split('|')) > 1 and l.split('|')[1].strip()]
        if not states or any(s not in TERMINAL for s in states):
            raise RuntimeError('Previous job is active or its termination is unconfirmed: ' + value['job_id'])


def status(plan):
    path = HERE / 'submission.json'
    print(json.dumps(read(path) if path.exists() else {'submitted': False}, indent=2))
    gen = sum((RUN / 'datasets' / f'g{g}' / METHOD / 'manifest.json').exists() for g in plan['generation_seeds'])
    fits = sum(result_path(RUN, g, t, METHOD).exists() for g in plan['generation_seeds'] for t in plan['training_seeds'])
    print(f'New MGD3 datasets: {gen}/5; new classifier fits: {fits}/15; reused baseline fits: 45.')
    for f in sorted((WORK / 'task_state').glob('*.json')):
        s = read(f)
        print(f.name, s['state'], 'Slurm job', s['job_id'])
    print('Task states are recorded progress; use Slurm for live queue state.')


def submit(args):
    plan = validate()
    if args.status:
        return status(plan)
    old = read(HERE / 'submission.json') if (HERE / 'submission.json').exists() else None
    parallel = args.max_parallel or (old['max_parallel'] if old else 2)
    options = old['options'] if old else {'account': os.environ.get('CAGS_4090_ACCOUNT', 'general'),
                                         'qos': os.environ.get('CAGS_4090_QOS')}
    specs = scheduler_requests(parallel, options)
    if args.dry_run:
        for name, command, dependency in specs:
            suffix = [f'--dependency=afterok:{dependency.upper()}_JOB_ID', '--kill-on-invalid-dep=yes'] if dependency else []
            print(shlex.join(command + suffix + tail(name)))
        print(f'Preview only: max {parallel} RTX4090 GPUs; 5 new datasets, 15 new fits, 45 reused baseline fits. No job submitted.')
        return
    if os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit from the login terminal outside an existing allocation.')
    recover(HERE / '.submission.lock')
    with locked(HERE / '.submission.lock', blocking=False):
        record = read(HERE / 'submission.json') if (HERE / 'submission.json').exists() else None
        if record:
            if record['plan_hash'] != plan['plan_hash'] or record['max_parallel'] != parallel or record['options'] != options:
                raise ValueError('Existing submission has different settings; no jobs created.')
            if any(v['status'] in {'submitting', 'unknown'} for v in record['stages'].values()):
                raise RuntimeError('Resolve the recorded unknown submission before retrying.')
            if args.retry_failed:
                require_terminal(record)
                write(HERE / ('submission-history-' + record['created'].replace(':', '-') + '.json'), record)
                for path in (WORK / 'locks').glob('*.lock.d'):
                    recover(Path(str(path)[:-2]))
                record = None
            elif len(record['stages']) == 4 and all(v['status'] == 'submitted' for v in record['stages'].values()):
                print('Already submitted; no duplicate jobs created. Use --status for recorded IDs.')
                return
        (WORK / 'slurm').mkdir(parents=True, exist_ok=True)
        # Read-only scheduler validation of every resource request before the first submission.
        for name, command, _ in specs:
            query([command[0], '--test-only'] + command[1:] + tail(name))
        record = record or {'created': now(), 'plan_hash': plan['plan_hash'], 'max_parallel': parallel,
                            'options': options, 'stages': {}}
        for name, command, dependency in specs:
            if record['stages'].get(name, {}).get('status') == 'submitted':
                continue
            if dependency:
                command += ['--dependency=afterok:' + record['stages'][dependency]['job_id'], '--kill-on-invalid-dep=yes']
            invoke(command + tail(name), record, name)
        print(f'Queued MGD3 only: at most {parallel} RTX4090 GPUs, 5 datasets and 15 fits. Baselines reused.')
        print('Logs: ' + str(WORK / 'slurm'))
        print('Final report: ' + str(RUN / 'summary.txt'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'submit', 'smoke', 'generate', 'evaluate', 'summarize'])
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--max-parallel', type=int, choices=[1, 2])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'submit':
        submit(args)
    else:
        def terminated(signum, frame):
            raise KeyboardInterrupt('Worker terminated by Slurm.')
        signal.signal(signal.SIGTERM, terminated)
        run_stage(args.action, args.task_id)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
