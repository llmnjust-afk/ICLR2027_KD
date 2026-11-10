"""Isolated missing-experiment campaign; submitting never edits an active run.

Stages: ResNet features -> CPU estimator profiles -> frozen plan -> generation
-> paired classifier blocks -> CPU summaries. All jobs are queued in one call.
"""
from __future__ import annotations
import argparse
import copy
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CODE = HERE / 'code'
WORK = ROOT / 'results/rebuttal_missing_v1'
RUN = WORK / 'runs/final'
PARENT = ROOT / 'results/rebuttal_tuned_v1/runs/final'
PYTHON = ROOT / 'ICLR/bin/python'
WEIGHTS = Path('/home/lulm/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth').resolve()
sys.path.insert(0, str(CODE))
from rebuttal.common import atomic_json as write, read_json as read, read_plan, file_hash as sha, digest, now
from guard.coordination import locked, recover, TERMINAL
from profiles import PROFILES

GEN_SEEDS = [300, 301, 302, 303, 304]
TRAIN_SEEDS = [0, 1, 2]
BASELINES = ['no_mode', 'fixed_mean', 'fixed_best_convnet6', 'cags', 'cags_selected_convnet6']
EXTERNAL = ['mgd3_shared_stop25', 'beta_cfg_dit', 'cags_beta_cfg_dit']
CONDITIONS = BASELINES + EXTERNAL + [name + suffix for name in PROFILES for suffix in ['', '_mean_matched']]


def files():
    paths = [HERE / n for n in ['runner.py', 'profiles.py', 'adaptive_cfg.py', 'analysis.py', 'job.sh', 'test_campaign.py']]
    for folder in ['code', 'guard', 'reference']:
        paths.extend(p for p in (HERE / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.append(ROOT / 'scripts/submit_rebuttal_missing.sh')
    return sorted(paths)


def freeze():
    if (HERE / 'frozen_manifest.json').exists():
        return validate()
    if (HERE / 'submission.json').exists():
        raise RuntimeError('Cannot rebuild a submitted experiment.')
    parent = read_plan(PARENT / 'plan.json')
    for rel, checksum in parent['source'].items():
        if sha(CODE / rel) != checksum:
            raise ValueError('Copied baseline differs from the frozen parent: ' + rel)
    if set(GEN_SEEDS) & set(parent['generation_seeds'] + parent['selection']['generation_seeds']):
        raise ValueError('New independent generation seeds overlap an existing experiment.')
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    # Verify the torchvision filename checksum before using the existing local weights.
    if not sha(WEIGHTS).startswith('f37072fd'):
        raise ValueError('ResNet18 checkpoint does not match its official filename checksum.')
    record = {'created': now(), 'parent_plan_hash': parent['plan_hash'],
              'files': {str(p.relative_to(ROOT)): sha(p) for p in files()},
              'resnet_checkpoint': str(WEIGHTS), 'resnet_sha256': sha(WEIGHTS),
              'generation_seeds': GEN_SEEDS, 'training_seeds': TRAIN_SEEDS,
              'condition_names': CONDITIONS, 'new_datasets': len(CONDITIONS) * len(GEN_SEEDS),
              'new_classifier_fits': len(CONDITIONS) * len(GEN_SEEDS) * len(TRAIN_SEEDS),
              'comparability': 'All new conditions including shared baselines use A800 and fresh paired generation seeds; no existing run is overwritten or resumed.'}
    record['campaign_hash'] = digest(record)
    write(HERE / 'frozen_manifest.json', record)
    print(json.dumps(record | {'files': f"{len(record['files'])} frozen files"}, indent=2))
    return record


def validate():
    frozen = read(HERE / 'frozen_manifest.json')
    if digest({k: v for k, v in frozen.items() if k != 'campaign_hash'}) != frozen['campaign_hash']:
        raise ValueError('Campaign manifest hash mismatch.')
    for rel, checksum in frozen['files'].items():
        if sha(ROOT / rel) != checksum:
            raise ValueError('Frozen extension changed: ' + rel)
    parent = read_plan(PARENT / 'plan.json')
    if parent['plan_hash'] != frozen['parent_plan_hash']:
        raise ValueError('Parent plan changed.')
    prepared = Path(parent['protocol']['prepared'])
    if sha(prepared / 'manifest.json') != parent['prepared_manifest_hash']:
        raise ValueError('Preparation manifest changed.')
    for rel in ['split.json', 'mode_targets.npz', 'profiles/vae_auto.json']:
        if sha(prepared / rel) != parent['prepared_manifest']['files'][rel]:
            raise ValueError('Shared input changed: ' + rel)
    return frozen


def gpu(stage):
    import torch
    from rebuttal.common import hardware
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('This stage requires one Slurm-allocated CUDA GPU.')
    machine = hardware()
    if not re.search(r'\bA800\b', machine['gpu'], re.I):
        raise RuntimeError('Expected an A800; received ' + machine['gpu'])
    if machine['versions'] != read_plan(PARENT / 'plan.json')['environment']:
        raise ValueError('Package versions differ from the frozen evaluation environment.')
    binding = WORK / 'hardware' / (stage + '.json')
    # A single binding for the entire campaign prevents mixing A800 variants.
    with locked(WORK / '.hardware.lock', blocking=True):
        shared = WORK / 'hardware/gpu_model.json'
        if shared.exists() and read(shared)['gpu'] != machine['gpu']:
            raise ValueError('All campaign conditions must use the same exact GPU model.')
        if not shared.exists():
            write(shared, machine)
        write(binding, machine)
    return machine


def features():
    frozen = validate()
    gpu('features')
    import numpy as np
    import torch
    from torchvision.models import resnet18
    from torch.utils.data import DataLoader
    from rebuttal.prepare import SourceImages
    from rebuttal.common import seed_everything
    if sha(WEIGHTS) != frozen['resnet_sha256']:
        raise ValueError('ResNet checkpoint changed.')
    parent = read_plan(PARENT / 'plan.json')
    # Exercise generation, the new adapter, and the classifier on the allocated
    # GPU before any expensive preparation. Diagnostics never enter results.
    smoke_path = WORK / 'smoke_tests/PASSED.json'
    if not smoke_path.exists():
        from rebuttal.smoke import smoke
        from types import SimpleNamespace
        cfg = parent['protocol']
        smoke(SimpleNamespace(workdir=str(WORK), checkpoint=cfg['checkpoint'], vae=cfg['vae'],
              local_files_only=True, deterministic=cfg['deterministic'], generation_batch_size=4,
              classifier_batch_size=128))
        from rebuttal.generate import GenerationEngine
        from adaptive_cfg import make_forward
        toy = copy.deepcopy(parent)
        toy['classes'] = parent['classes'][:2]
        toy['prepared_manifest']['imagenet_labels'] = parent['prepared_manifest']['imagenet_labels'][:2]
        toy['protocol']['ipc'] = 4
        toy['plan_hash'] = digest({'campaign_hash': frozen['campaign_hash'], 'smoke_test_only': True})
        engine = GenerationEngine(toy)
        try:
            engine.model.forward_with_cfg = make_forward(engine.model, engine.diffusion.timestep_map,
                {'scale': 7.5, 'a': 2., 'b': 3., 'gamma': 1.})
            engine.generate(toy, WORK / 'smoke_tests/beta_cfg', 0, 'beta_cfg_with_modes',
                            {'coefficients': [.03, .03], 'smoke_test_only': True})
        finally:
            engine.close()
        write(smoke_path, {'campaign_hash': frozen['campaign_hash'], 'smoke_test_only': True, 'created': now()})
    elif read(smoke_path)['campaign_hash'] != frozen['campaign_hash']:
        raise ValueError('GPU diagnostics belong to another campaign.')
    prepared = Path(parent['protocol']['prepared'])
    split = read(prepared / 'split.json')
    source_hashes = read(prepared / 'source_image_hashes.json')
    if sha(prepared / 'source_image_hashes.json') != parent['prepared_manifest']['files']['source_image_hashes.json']:
        raise ValueError('Training-image provenance changed.')
    with locked(WORK / '.features.lock', blocking=False):
        seed_everything(2027, parent['protocol']['deterministic'])
        net = resnet18(weights=None)
        net.load_state_dict(torch.load(WEIGHTS, map_location='cpu'), strict=True)
        net.fc = torch.nn.Identity()
        net.cuda().eval()
        folder = WORK / 'features_resnet18'
        folder.mkdir(parents=True, exist_ok=True)
        for cls in parent['classes']:
            output = folder / (cls + '.npy')
            done = folder / (cls + '.json')
            if done.exists() and read(done)['campaign_hash'] == frozen['campaign_hash'] and sha(output) == read(done)['sha256']:
                continue
            data = SourceImages(split['root'], split['train'][cls], 224, resnet=True)
            batches = DataLoader(data, batch_size=32, num_workers=4, shuffle=False)
            chunks = []
            with torch.no_grad():
                for x, paths, hashes in batches:
                    if any(source_hashes[p] != h for p, h in zip(paths, hashes)):
                        raise ValueError('Training image changed during feature extraction.')
                    chunks.append(net(x.cuda()).cpu().numpy())
            x = np.concatenate(chunks).astype(np.float32)
            if not np.isfinite(x).all():
                raise FloatingPointError('Nonfinite ResNet features.')
            temp = output.with_suffix('.partial.npy')
            np.save(temp, x)
            temp.replace(output)
            write(done, {'campaign_hash': frozen['campaign_hash'], 'sha256': sha(output), 'shape': list(x.shape)})
            print('ResNet18 features:', cls, x.shape, flush=True)
        write(WORK / 'FEATURES_READY.json', {'campaign_hash': frozen['campaign_hash'], 'created': now()})


def profile(task_id):
    frozen = validate()
    from profiles import compute
    name = PROFILES[task_id]
    parent = read_plan(PARENT / 'plan.json')
    prepared = Path(parent['protocol']['prepared'])
    with locked(WORK / ('profile_' + name + '.lock'), blocking=False):
        output = WORK / 'profiles' / (name + '.json')
        if output.exists():
            if read(output)['campaign_hash'] != frozen['campaign_hash']:
                raise ValueError('Stale estimator profile.')
            return
        for cls in parent['classes']:
            if name == 'resnet18_auto':
                meta = read(WORK / 'features_resnet18' / (cls + '.json'))
                if meta['campaign_hash'] != frozen['campaign_hash'] or sha(WORK / 'features_resnet18' / (cls + '.npy')) != meta['sha256']:
                    raise ValueError('ResNet feature changed.')
            else:
                rel = 'features/' + cls + '.npy'
                if sha(prepared / rel) != parent['prepared_manifest']['files'][rel]:
                    raise ValueError('VAE feature changed: ' + cls)
        result = compute(name, prepared, WORK)
        result.update(campaign_hash=frozen['campaign_hash'], name=name, targets='unchanged shared mode_targets.npz')
        write(output, result)


def make_plan():
    frozen = validate()
    from rebuttal.common import coefficient_vectors, source_manifest, hardware
    from rebuttal.run import mean_matched
    from rebuttal.evaluate import image_manifest
    parent = read_plan(PARENT / 'plan.json')
    plan = copy.deepcopy(parent)
    plan.pop('plan_hash')
    plan.update(suite='missing_prespecified_v1', generation_seeds=GEN_SEEDS, training_seeds=TRAIN_SEEDS,
                source=source_manifest(), conditions={}, selected_cags_references={'convnet6': 'cags_selected_convnet6'})
    for name in BASELINES:
        plan['conditions'][name] = copy.deepcopy(parent['conditions'][name])
    plan['conditions']['mgd3_shared_stop25'] = {
        'coefficients': [.1] * len(plan['classes']), 'stop_t': 25, 'kind': 'verified_mgd3_kernel_shared_protocol',
        'note': 'Official update/default lambda with exact example stop_t=25 (24 active steps); shared train-only center-crop features, targets, paired RNG, batching and evaluator. Not a reproduction of published accuracy.'}
    beta = {'scale': 7.5, 'a': 2., 'b': 3., 'gamma': 1.}
    for name, reference in [('beta_cfg_dit', 'no_mode'), ('cags_beta_cfg_dit', 'cags')]:
        plan['conditions'][name] = {'coefficients': plan['conditions'][reference]['coefficients'],
            'kind': 'published_beta_cfg_adaptation', 'beta_cfg': beta,
            'note': 'Official README preset, no final-set tuning; SD/DDIM rule adapted to DiT/DDPM, original three guided channels retained. Not a claim of optimal beta-CFG or published SD-score reproduction.'}
    target = sum(plan['conditions']['cags']['coefficients']) / len(plan['classes'])
    hashes = {}
    for name in PROFILES:
        path = WORK / 'profiles' / (name + '.json')
        stats = read(path)
        if stats['campaign_hash'] != frozen['campaign_hash'] or stats['classes'] != plan['classes']:
            raise ValueError('Estimator profile identity differs.')
        hashes[name] = sha(path)
        values = coefficient_vectors(stats)['cags']
        plan['conditions'][name] = {'coefficients': values, 'kind': 'estimator_sensitivity', 'profile': name}
        plan['conditions'][name + '_mean_matched'] = {'coefficients': mean_matched(values, target),
            'kind': 'estimator_sensitivity', 'profile': name, 'matching': 'logit intercept to baseline CAGS mean'}
    assert list(plan['conditions']) == CONDITIONS
    if hardware()['versions'] != parent['environment']:
        raise ValueError('Environment changed.')
    if sha(plan['protocol']['checkpoint']) != plan['protocol']['checkpoint_sha256']:
        raise ValueError('DiT checkpoint changed.')
    reporting = read(PARENT / 'evaluation_images.json')
    if image_manifest(reporting['root'], reporting['paths']) != plan['protocol']['evaluation_image_manifest']:
        raise ValueError('Evaluation images changed.')
    plan['missing_campaign'] = {'campaign_hash': frozen['campaign_hash'], 'profiles_sha256': hashes,
        'feature_checkpoint_sha256': frozen['resnet_sha256'], 'parent_plan_hash': parent['plan_hash'],
        'new_baselines_reason': 'Fresh A800 paired generation/evaluation; do not compare unpaired hardware results.',
        'beta_cfg_selection': 'One published default preset fixed before outcomes; no equal-tuning-budget claim.'}
    plan['plan_hash'] = digest(plan)
    with locked(WORK / '.plan.lock', blocking=False):
        if (RUN / 'plan.json').exists() and read_plan(RUN / 'plan.json')['plan_hash'] != plan['plan_hash']:
            raise ValueError('Refusing to replace a different frozen plan.')
        write(RUN / 'plan.json', plan)
        write(RUN / 'evaluation_images.json', reporting)
        write(RUN / 'READY.json', {'plan_hash': plan['plan_hash'], 'created': now()})
    print(f"Plan ready: {len(CONDITIONS)} conditions, {len(CONDITIONS)*5} datasets, {len(CONDITIONS)*15} fits.")


def checked_plan():
    frozen = validate()
    plan = read_plan(RUN / 'plan.json')
    if plan['missing_campaign']['campaign_hash'] != frozen['campaign_hash'] or read(RUN / 'READY.json')['plan_hash'] != plan['plan_hash']:
        raise ValueError('Plan not ready or belongs to another campaign.')
    return plan


def generate(task_id):
    plan = checked_plan()
    gpu('generation_' + str(task_id))
    from rebuttal.generate import GenerationEngine
    from adaptive_cfg import make_forward
    seed, name = [(g, c) for g in GEN_SEEDS for c in CONDITIONS][task_id]
    with locked(WORK / f'gen_{task_id}.lock', blocking=False):
        from rebuttal.generate import verify_dataset
        output = RUN / 'datasets' / f'g{seed}' / name
        expected = digest({'plan_hash': plan['plan_hash'], 'generation_seed': seed, 'condition': name,
                           'specification': plan['conditions'][name]})
        if (output / 'manifest.json').exists():
            verify_dataset(output, expected)
            return
        engine = GenerationEngine(plan)
        try:
            if 'beta_cfg' in plan['conditions'][name]:
                engine.model.forward_with_cfg = make_forward(engine.model, engine.diffusion.timestep_map,
                                                             plan['conditions'][name]['beta_cfg'])
            engine.generate(plan, RUN, seed, name, plan['conditions'][name])
        finally:
            engine.close()


def evaluate(task_id):
    plan = checked_plan()
    gpu('evaluation_' + str(task_id))
    from rebuttal.run import execute_evaluation
    seed, training = [(g, t) for g in GEN_SEEDS for t in TRAIN_SEEDS][task_id]
    with locked(WORK / f'eval_{task_id}.lock', blocking=False):
        execute_evaluation(plan, RUN, {'generation_seed': seed, 'training_seed': training, 'architecture': 'convnet6'})


def query(command):
    result = subprocess.run(command, text=True, capture_output=True, timeout=40)
    if result.returncode:
        raise RuntimeError(shlex.join(command) + ': ' + result.stderr.strip())
    return result.stdout


def account_options(dry_run=False):
    if dry_run:
        return {'account': os.environ.get('CAGS_A800_ACCOUNT', 'general'), 'qos': os.environ.get('CAGS_A800_QOS', 'a800')}
    data = query(['sacctmgr', '-nP', 'show', 'assoc', 'user=' + getpass.getuser(), 'format=Account,Partition,QOS,DefaultQOS'])
    for line in data.splitlines():
        fields = line.strip().split('|')
        if len(fields) < 4:
            continue
        account, partition, allowed, default = fields[:4]
        if partition not in {'', 'a800'} or os.environ.get('CAGS_A800_ACCOUNT', account) != account:
            continue
        for qos in [os.environ.get('CAGS_A800_QOS'), default, 'a800', 'a800-2', 'a800-3']:
            if qos in {'a800', 'a800-2', 'a800-3'} and qos in allowed.split(','):
                return {'account': account, 'qos': qos}
    raise RuntimeError('No authorized A800 account/QOS found; no jobs submitted.')


def requests(options, parallel):
    common = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--export=ALL', '--chdir=' + str(CODE), '--account=' + options['account']]
    gpu_args = ['--partition=a800', '--qos=' + options['qos'], '--gres=gpu:a800:1', '--cpus-per-task=8', '--mem=32G']
    cpu_args = ['--partition=cpu', '--cpus-per-task=8', '--mem=32G']
    stages = [('analyze-existing', ['--partition=cpu', '--cpus-per-task=2', '--mem=8G'], None, None),
              ('features', gpu_args, None, None),
              ('profile', cpu_args, '0-4%2', 'features'), ('plan', cpu_args, None, 'profile'),
              ('generate', gpu_args, f'0-{len(CONDITIONS)*5-1}%{parallel}', 'plan'),
              ('evaluate', gpu_args, f'0-14%{parallel}', 'generate'),
              ('summarize', ['--partition=cpu', '--cpus-per-task=2', '--mem=8G'], None, 'evaluate')]
    result = []
    for name, resources, array, dependency in stages:
        cmd = common + resources + ['--job-name=cags-missing-' + name,
            '--output=' + str(WORK / 'slurm' / (name + '-%A_%a.log')),
            '--error=' + str(WORK / 'slurm' / (name + '-%A_%a.log'))]
        if array:
            cmd += ['--array=' + array]
        result.append((name, cmd, dependency))
    return result


def command_tail(stage):
    return [str(HERE / 'job.sh'), str(HERE), str(PYTHON), stage]


def invoke(command, record, name):
    record['stages'][name] = {'status': 'submitting', 'command': command}
    write(HERE / 'submission.json', record)
    env = dict(os.environ)
    for key in ['CUDA_VISIBLE_DEVICES', 'REBUTTAL_GPUS']:
        env.pop(key, None)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=45, env=env)
    except (subprocess.TimeoutExpired, OSError) as error:
        record['stages'][name].update(status='unknown', error=str(error))
        write(HERE / 'submission.json', record)
        raise RuntimeError('Submission outcome unknown; inspect Slurm before retrying.') from error
    if result.returncode:
        record['stages'][name].update(status='rejected', error=result.stderr)
        write(HERE / 'submission.json', record)
        raise RuntimeError('Slurm rejected ' + name + ': ' + result.stderr.strip())
    job_id = result.stdout.strip().split(';')[0]
    if not re.fullmatch(r'\d+', job_id):
        record['stages'][name].update(status='unknown', error=result.stdout)
        write(HERE / 'submission.json', record)
        raise RuntimeError('Unrecognized sbatch response; do not resubmit blindly.')
    record['stages'][name].update(status='submitted', job_id=job_id)
    write(HERE / 'submission.json', record)
    print(f'{name}: submitted {job_id}', flush=True)


def submit(args):
    frozen = validate()
    if args.status:
        print(json.dumps(read(HERE / 'submission.json') if (HERE / 'submission.json').exists() else {'submitted': False}, indent=2))
        if (RUN / 'plan.json').exists():
            datasets = sum((RUN / 'datasets' / f'g{g}' / c / 'manifest.json').exists() for g in GEN_SEEDS for c in CONDITIONS)
            fits = sum((RUN / 'evaluation' / f'g{g}' / f't{t}' / 'convnet6' / (c + '.json')).exists() for g in GEN_SEEDS for t in TRAIN_SEEDS for c in CONDITIONS)
            print(f'Completed datasets {datasets}/{len(CONDITIONS)*5}; fits {fits}/{len(CONDITIONS)*15}.')
        return
    if args.dry_run:
        for name, cmd, dep in requests(account_options(True), args.max_parallel):
            print(shlex.join(cmd + ([f'--dependency=afterok:{dep.upper()}_JOB_ID', '--kill-on-invalid-dep=yes'] if dep else []) + command_tail(name)))
        print(f'Preview only: {len(CONDITIONS)} conditions, {len(CONDITIONS)*5} datasets, {len(CONDITIONS)*15} fits; max {args.max_parallel} A800 GPUs. No jobs submitted.')
        return
    if os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit from the login terminal, outside an existing GPU allocation.')
    recover(HERE / '.submission.lock')
    with locked(HERE / '.submission.lock', blocking=False):
        path = HERE / 'submission.json'
        record = read(path) if path.exists() else None
        if record:
            if record['campaign_hash'] != frozen['campaign_hash'] or record['max_parallel'] != args.max_parallel:
                raise ValueError('Submission configuration differs from the existing campaign.')
            if any(v['status'] in {'unknown', 'submitting'} for v in record['stages'].values()):
                raise RuntimeError('An earlier submission is unresolved. Inspect recorded job commands and Slurm; no duplicate submission attempted.')
            if args.retry_failed:
                for value in record['stages'].values():
                    if 'job_id' not in value:
                        continue
                    accounting = query(['sacct', '-n', '-X', '-j', value['job_id'], '-o', 'JobIDRaw,State', '-P'])
                    states = [l.split('|')[1].split()[0].rstrip('+') for l in accounting.splitlines()
                              if len(l.split('|')) > 1 and l.split('|')[1].strip()]
                    if not states or any(s not in TERMINAL for s in states):
                        raise RuntimeError('Retry requires every previous job to be terminal: ' + value['job_id'])
                write(HERE / ('submission-history-' + record['created'].replace(':', '-') + '.json'), record)
                for path in WORK.glob('*.lock.d'):
                    recover(Path(str(path)[:-2]))
                record = None
            elif len(record['stages']) == 7 and all(v['status'] == 'submitted' for v in record['stages'].values()):
                print('Already queued; no duplicate jobs created. Use --status to see job IDs.')
                return
        options = record['account_options'] if record else account_options()
        specs = requests(options, args.max_parallel)
        (WORK / 'slurm').mkdir(parents=True, exist_ok=True)
        # Validate ALL independent resource requests before submitting the first job.
        for name, command, _ in specs:
            query([command[0], '--test-only'] + command[1:] + command_tail(name))
        record = record or {'created': now(), 'campaign_hash': frozen['campaign_hash'], 'account_options': options,
                            'max_parallel': args.max_parallel, 'stages': {}}
        for name, command, dep in specs:
            if record['stages'].get(name, {}).get('status') == 'submitted':
                continue
            if dep:
                command += ['--dependency=afterok:' + record['stages'][dep]['job_id'], '--kill-on-invalid-dep=yes']
            invoke(command + command_tail(name), record, name)
        print(f'All stages queued; at most {args.max_parallel} additional A800 GPUs, one per task. CPU stages use no GPU.')
        print('Logs: ' + str(WORK / 'slurm'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'submit', 'features', 'profile', 'plan', 'generate', 'evaluate', 'summarize', 'analyze-existing'])
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--max-parallel', type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze()
    elif args.action == 'submit':
        submit(args)
    elif args.action == 'features':
        features()
    elif args.action == 'profile':
        profile(args.task_id)
    elif args.action == 'plan':
        make_plan()
    elif args.action == 'generate':
        generate(args.task_id)
    elif args.action == 'evaluate':
        evaluate(args.task_id)
    elif args.action == 'analyze-existing':
        validate()
        analyze_existing()
    else:
        checked_plan()
        from rebuttal.statistics import summarize
        summarize(RUN)
        from rebuttal.reporting import runtime_estimate
        workers = read(HERE / 'submission.json')['max_parallel']
        write(RUN / 'runtime.json', runtime_estimate(RUN, checked_plan(), workers=workers))
        from analysis import analyze
        analyze(RUN, WORK / 'analysis/rebuttal_missing_v1_final')
        analyze_existing()


def analyze_existing():
    from analysis import analyze
    for folder in [PARENT, PARENT.parent / 'tune', ROOT / 'results/rebuttal_core_v2/runs/final']:
        analyze(folder, WORK / 'analysis' / (folder.parents[1].name + '_' + folder.name))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
