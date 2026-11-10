"""Four single-GPU jobs: allocation controls, or held-out tuning then full evaluation."""
import argparse
import contextlib
import getpass
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import uuid
import socket
import time

from .common import ROOT, atomic_json, digest, file_hash, now, read_json, source_manifest
from .coordination import locked, recover


def command(args, capture=False, timeout=None, env=None):
    """Forward termination to the entire child process group, including loaders."""
    with subprocess.Popen(args, cwd=ROOT, env=env, text=True, start_new_session=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, args, stdout, stderr)
        return (stdout or '').strip()


def scheduler_jobs():
    # Check connectivity before spending minutes verifying the dataset.
    output = command(['squeue', '--noheader', '--user', getpass.getuser(), '--format=%i'],
                     capture=True, timeout=20)
    return set(output.split())


def provenance():
    files = source_manifest()
    for name in ['scripts/rebuttal.sh', 'scripts/rebuttal_slurm_job.sh',
                 'scripts/submit_rebuttal_core.sh', 'scripts/submit_rebuttal_tuned.sh',
                 'scripts/submit_rebuttal_campaign.sh', 'scripts/prepare_rebuttal_core.sh']:
        files[name] = file_hash(ROOT / name)
    return digest(files)


def settings():
    count = int(os.environ.get('REBUTTAL_JOB_COUNT', '4'))
    gpu_count = int(os.environ.get('REBUTTAL_GPU_COUNT', '1'))
    gres = os.environ.get('REBUTTAL_SLURM_GRES', 'gpu:4090:1')
    if not 1 <= count <= 4 or gpu_count != 1 or gres != 'gpu:4090:1':
        raise ValueError('核心入口要求每个作业 gpu:4090:1；用 REBUTTAL_JOB_COUNT 设置独立作业数。')
    suite = os.environ.get('FINAL_SUITE')
    if suite not in {'allocation', 'full'}:
        raise ValueError('此入口仅用于 FINAL_SUITE=allocation 或 full。')
    return {'count': count, 'gres': gres, 'suite': suite,
            'partition': os.environ.get('REBUTTAL_SLURM_PARTITION', 'rtx4090'),
            'cpus': os.environ.get('REBUTTAL_SLURM_CPUS', '8'),
            'mem': os.environ.get('REBUTTAL_SLURM_MEM', '32G'),
            'python': os.environ.get('REBUTTAL_PYTHON', sys.executable),
            'workdir': str(Path(os.environ['WORKDIR']).resolve()),
            'options': {key: os.environ['REBUTTAL_SLURM_' + key.upper()]
                        for key in ['account', 'qos', 'time']
                        if os.environ.get('REBUTTAL_SLURM_' + key.upper())}}


def batch_namespace(cfg=None):
    suite = (cfg or {}).get('suite', os.environ.get('FINAL_SUITE', 'allocation'))
    return 'tuned' if suite == 'full' else 'core'


def initial_run(record):
    return 'tune' if record['settings'].get('suite', 'allocation') == 'full' else 'final'


def guard_paths(work):
    return [work / name for name in ['.all_pipeline.lock', '.core_prepare.lock', '.core_final.lock',
            'runs/tune/.slurm_summary.lock', 'runs/final/.slurm_summary.lock']]


def dependency_settings(folder):
    folder = Path(folder).resolve()
    record = read_json(folder / 'batch.json')
    slots = record['slots']
    if (record['settings'].get('suite', 'allocation') != 'allocation'
            or len(slots) != record['settings']['count']
            or any(s.get('state') != 'submitted' or not re.fullmatch(r'[0-9]+', s.get('job_id', '')) for s in slots)):
        raise ValueError('第二批必须依赖已完整提交的第一批，不能依赖不明或部分提交记录。')
    for slot in slots:
        state = folder / f'worker-{slot["slot"]}-{slot["job_id"]}.json'
        if state.exists() and read_json(state).get('state') == 'failed':
            raise ValueError('第一批已有失败作业；先恢复第一批，再提交后续依赖。')
    return {'batch': str(folder), 'job_ids': [s['job_id'] for s in slots],
            'workdir': record['settings']['workdir']}


def sbatch_args(cfg, folder, slot):
    args = ['sbatch', '--parsable', f'--job-name=cags-{folder.name}-w{slot}',
            '--nodes=1', '--ntasks=1', f'--partition={cfg["partition"]}',
            f'--gres={cfg["gres"]}', f'--cpus-per-task={cfg["cpus"]}', f'--mem={cfg["mem"]}',
            f'--chdir={ROOT}', '--export=ALL',
            f'--output={ROOT}/logs/slurm/rebuttal-%j.log',
            f'--error={ROOT}/logs/slurm/rebuttal-%j.log']
    args += [f'--{key}={value}' for key, value in cfg['options'].items()]
    if cfg.get('dependency'):
        args += ['--dependency=afterok:' + ':'.join(cfg['dependency']['job_ids']), '--kill-on-invalid-dep=yes']
    return args + [str(ROOT / 'scripts/rebuttal_slurm_job.sh'), str(ROOT),
                   str(folder / 'config.env'), cfg['python'], str(folder), str(slot)]


def report_batch(folder, record):
    jobs = [item['job_id'] for item in record['slots'] if item.get('job_id')]
    atomic_json(ROOT / f'logs/slurm/latest_{batch_namespace(record["settings"])}_jobs.json',
                {'batch': str(folder), 'job_ids': jobs, 'slots': record['slots']})
    print(f'提交记录：{folder}/batch.json', flush=True)
    for item in record['slots']:
        if item.get('job_id'):
            print(f'工作进程 {item["slot"]}：作业 {item["job_id"]}，1 张 RTX 4090；'
                  f'日志 logs/slurm/rebuttal-{item["job_id"]}.log', flush=True)
    if jobs:
        print('查看队列：squeue -j ' + ','.join(jobs), flush=True)


def submit(args):
    if getattr(args, 'status', False):
        show_status()
        return
    if os.environ.get('SLURM_JOB_ID'):
        raise ValueError('请在登录节点提交，不能在已有 Slurm 作业内申请另一批。')
    cfg = settings()
    after = getattr(args, 'after_batch', None)
    if after:
        if cfg['suite'] != 'full':
            raise ValueError('--after-batch 仅用于第二批 full。')
        cfg['dependency'] = dependency_settings(after)
    elif cfg['suite'] == 'full' and not args.dry_run:
        raise ValueError('第二批必须指定 --after-batch，确保第一批完成后才占用 GPU；可使用 submit_rebuttal_campaign.sh。')
    config = Path(os.environ['REBUTTAL_CONFIG']).resolve()
    logdir = ROOT / 'logs/slurm'
    verify = ['bash', str(ROOT / 'scripts/prepare_rebuttal_core.sh'), '--verify']
    if args.dry_run:
        print('先完成两节点 CPU 协调检查，再校验本地资源一次（不下载）：' + shlex.join(verify))
        print('随后检查本地 DiT 权重键名/尺寸。全部通过才提交 GPU 作业。')
        print(f'{cfg["count"]} 个独立作业，每个 1 张 RTX 4090；以下命令未执行：')
        for slot in range(1, cfg['count'] + 1):
            print(shlex.join(sbatch_args(cfg, logdir / f'{batch_namespace(cfg)}_batches/PREVIEW', slot)))
        return
    active = scheduler_jobs()
    with locked(logdir / '.core_submit.lock', blocking=False):
        pointer = logdir / f'latest_{batch_namespace(cfg)}_batch.json'
        previous = read_json(pointer) if pointer.exists() else None
        folder = Path(previous['batch']) if previous else None
        record = read_json(folder / 'batch.json') if folder else None
        if record:
            ids = {s['job_id'] for s in record['slots'] if s.get('job_id')}
            if args.retry_failed:
                if active & ids:
                    raise ValueError('上一批仍有排队/运行作业；停止它们后才能 --retry-failed。')
                if any(s['state'] == 'uncertain' for s in record['slots']):
                    raise ValueError('存在结果不明的 sbatch 请求，必须先按记录中的作业名核实，不能重复提交。')
                for lock in guard_paths(Path(record['settings']['workdir'])):
                    recover(lock)
                folder, record = None, None
            else:
                if record['config_hash'] != file_hash(config) or record['settings'] != cfg or record['source'] != provenance():
                    raise ValueError('配置/代码与已提交批次不同。先检查旧作业；结束后用 --retry-failed 创建新批次。')
                if any(s['state'] == 'uncertain' for s in record['slots']):
                    report_batch(folder, record)
                    raise ValueError('上次提交结果不明，已阻止重复提交；请按 batch.json 的作业名核实。')
                if all(s.get('job_id') for s in record['slots']):
                    report_batch(folder, record)
                    print('已有完整提交记录，没有重复申请 GPU。失败任务在所有作业结束后用 --retry-failed 重试。')
                    return
        if record is None:
            work = Path(cfg['workdir'])
            if args.retry_failed:
                for lock in guard_paths(work):
                    recover(lock)
            # Do not alter a queue owned by an old pipeline or orphaned worker.
            with locked(work / '.all_pipeline.lock', blocking=False):
                if list((work / 'runs').glob('*/queue/*.lock')):
                    raise ValueError('实验队列仍有任务锁；确认对应工作进程已停止后处理，不能自动删除。')
                # Exercise cross-node locking before the expensive image verification.
                from .coordination_check import run_cluster_check
                run_cluster_check(cfg, command)
                command(verify)
                command([cfg['python'], '-m', 'rebuttal.checkpoints', '--checkpoint', os.environ['DIT_CKPT']])
            folder = logdir / f'{batch_namespace(cfg)}_batches' / uuid.uuid4().hex[:12]
            folder.mkdir(parents=True)
            (folder / 'config.env').write_bytes(config.read_bytes())
            record = {'created': now(), 'settings': cfg, 'config_hash': file_hash(config),
                      'source': provenance(), 'retry_failed': args.retry_failed,
                      'slots': [{'slot': i, 'state': 'pending'} for i in range(1, cfg['count'] + 1)]}
            atomic_json(folder / 'batch.json', record)
            atomic_json(pointer, {'batch': str(folder)})
        submit_env = dict(os.environ)
        for key in ['CUDA_VISIBLE_DEVICES', 'REBUTTAL_GPUS']:
            submit_env.pop(key, None)
        try:
            for item in record['slots']:
                if item.get('job_id'):
                    continue
                call = sbatch_args(cfg, folder, item['slot'])
                item.update(state='uncertain', command=call)
                atomic_json(folder / 'batch.json', record)
                try:
                    output = command(call, capture=True, timeout=60, env=submit_env)
                except subprocess.CalledProcessError as error:
                    item.update(state='rejected', error=error.stderr or str(error))
                    atomic_json(folder / 'batch.json', record)
                    raise
                job_id = output.split(';')[0]
                if not re.fullmatch(r'[0-9]+', job_id):
                    raise ValueError(f'sbatch 返回值不明确：{output!r}；先核实记录中的作业名。')
                item.update(state='submitted', job_id=job_id)
                atomic_json(folder / 'batch.json', record)
        finally:
            report_batch(folder, record)


def pipeline_action(action, *extra):
    command(['bash', str(ROOT / 'scripts/rebuttal.sh'), action, *extra])


def initialize(folder, record):
    work = Path(record['settings']['workdir'])
    run = initial_run(record)
    plan = work / 'runs' / run
    state_file = folder / 'preparation.json'
    print('等待跨节点共享准备锁；首个获得锁的单卡作业负责准备。', flush=True)
    last_report = [0.0]
    def waiting():
        if state_file.exists() and read_json(state_file).get('state') == 'failed':
            raise ValueError(f'共享准备失败，停止本批：{state_file}')
        if time.monotonic() - last_report[0] > 30:
            print('共享准备仍在运行，等待完成；未启动重复准备。', flush=True)
            last_report[0] = time.monotonic()
    with locked(work / '.core_prepare.lock', on_wait=waiting):
        if state_file.exists():
            state = read_json(state_file)
            if state['state'] != 'ready':
                raise ValueError(f'共享准备失败或中断，停止本批：{state_file}')
            if not (plan / 'READY.json').exists():
                raise ValueError('共享准备记录已就绪，但冻结计划不存在。')
            print('复用本批已完成的共享准备和实验计划。', flush=True)
            return
        preparation = {'state': 'running', 'job_id': os.environ['SLURM_JOB_ID'],
                       'host': socket.gethostname(), 'created': now(), 'stage': 'smoke'}
        atomic_json(state_file, preparation)
        try:
            pipeline_action('smoke')
            preparation['stage'] = 'features'
            atomic_json(state_file, preparation)
            prepared = work / 'prepared'
            if os.environ.get('PREPARED_SOURCE'):
                reuse_prepared(work, os.environ['PREPARED_SOURCE'])
            if (prepared / 'manifest.json').exists():
                split = read_json(prepared / 'split.json')
                if not os.path.samefile(split['root'], os.environ['DATA_ROOT']):
                    raise ValueError('已有特征缓存属于另一数据目录，请使用另一 WORKDIR。')
            else:
                pipeline_action('prepare')
            preparation['stage'] = 'plan'
            atomic_json(state_file, preparation)
            pipeline_action('plan-' + run)
            if not (plan / 'READY.json').exists():
                raise ValueError('计划生成未完成，不能启动工作进程。')
            if list((plan / 'queue').glob('*.lock')):
                raise ValueError('实验任务仍被锁定，不能重置或重复执行。')
            if record['retry_failed']:
                pipeline_action('retry-failed', run)
            atomic_json(state_file, dict(preparation, state='ready', stage='ready', finished=now()))
        except BaseException as error:
            atomic_json(state_file, dict(preparation, state='failed', error=str(error), finished=now()))
            raise


def reuse_prepared(work, source):
    """Reuse only a complete, compatible immutable cache; never edit its manifest."""
    from .run import validate_prepared
    source = Path(source).resolve()
    manifest = validate_prepared(source)
    config = manifest['config']
    expected = {'nclass': int(os.environ.get('NCLASS', 100)), 'ipc': int(os.environ.get('IPC', 10)),
                'holdout_fraction': float(os.environ.get('HOLDOUT_FRACTION', .1)),
                'seed': int(os.environ.get('PREPARE_SEED', 2027))}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('准备缓存与本批类别数、IPC、留出划分或种子不同。')
    split = read_json(source / 'split.json')
    if not os.path.samefile(split['root'], os.environ['DATA_ROOT']):
        raise ValueError('准备缓存属于另一数据目录。')
    classes = Path(os.environ.get('CLASS_FILE', ROOT / 'misc/class100.txt')).read_text().split()
    if classes[:expected['nclass']] != manifest['classes']:
        raise ValueError('准备缓存类别顺序不同。')
    target = work / 'prepared'
    if target.exists() or target.is_symlink():
        if target.resolve() != source:
            raise ValueError('拒绝替换已有准备缓存。')
    else:
        target.symlink_to(source, target_is_directory=True)
    receipt = work / 'cache_reuse.json'
    if not receipt.exists():
        atomic_json(receipt, {'source': str(source), 'prepared_manifest_sha256': file_hash(source / 'manifest.json'),
                             'verified_files': len(manifest['files']), 'created': now()})


def initialize_final(folder, record):
    """A barrier: all tuning fits finish before one worker selects and freezes final."""
    work = Path(record['settings']['workdir'])
    state_file = folder / 'final_preparation.json'
    with locked(work / '.core_final.lock'):
        if state_file.exists():
            if read_json(state_file).get('state') != 'ready' or not (work / 'runs/final/READY.json').exists():
                raise ValueError('第二批正式计划准备失败或中断，保留记录后重试。')
            return
        atomic_json(state_file, {'state': 'running', 'stage': 'selection', 'created': now()})
        try:
            pipeline_action('select')
            pipeline_action('plan-final')
            if not (work / 'runs/final/READY.json').exists():
                raise ValueError('正式计划尚未冻结。')
            if record['retry_failed']:
                pipeline_action('retry-failed', 'final')
            atomic_json(state_file, {'state': 'ready', 'stage': 'ready', 'finished': now()})
        except BaseException as error:
            atomic_json(state_file, {'state': 'failed', 'error': str(error), 'finished': now()})
            raise


def finish_summary(plan):
    from .run import job_state
    with locked(plan / '.slurm_summary.lock'):
        jobs = list((plan / 'queue').glob('*.job.json'))
        if not jobs or any(job_state(job) != 'done' for job in jobs):
            raise ValueError('队列未全部成功，不能标记实验完成。')
        marker = plan / '.slurm_summary.json'
        summary = plan / 'summary.json'
        if marker.exists() and summary.exists() and read_json(marker)['sha256'] == file_hash(summary):
            return
        pipeline_action('summary', plan.name)
        if not read_json(summary).get('complete'):
            raise ValueError('汇总缺少实验结果，不能标记完成。')
        atomic_json(marker, {'finished': now(), 'sha256': file_hash(summary)})


def worker(args):
    folder = Path(args.batch).resolve()
    record = read_json(folder / 'batch.json')
    job_id = os.environ.get('SLURM_JOB_ID')
    if not job_id or not 1 <= args.slot <= record['settings']['count']:
        raise ValueError('工作进程必须通过本批 Slurm 作业启动。')
    stage, code, error_text = 'preflight', 1, None
    status_file = folder / f'worker-{args.slot}-{job_id}.json'
    def progress():
        atomic_json(status_file, {'job_id': job_id, 'slot': args.slot, 'host': socket.gethostname(),
                                 'state': 'running', 'exit_code': None, 'stage': stage, 'updated': now()})
    try:
        progress()
        mask = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not re.fullmatch(r'[A-Za-z0-9-]+', mask) or mask == '-1':
            raise ValueError(f'每个作业必须恰好有一张可见 GPU，当前为 {mask!r}。')
        os.environ.update(REBUTTAL_GPUS=mask, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                          TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
        if file_hash(os.environ['REBUTTAL_CONFIG']) != record['config_hash'] or provenance() != record['source']:
            raise ValueError('提交后配置或代码发生变化，停止以避免混用实验版本。')
        command([record['settings']['python'], '-m', 'rebuttal.preflight', '--gpus', mask])
        if record['settings'].get('dependency'):
            upstream = Path(record['settings']['dependency']['workdir']) / 'runs/final'
            from .common import read_plan
            parent_plan = read_plan(upstream / 'plan.json')
            parent_summary = read_json(upstream / 'summary.json')
            if not parent_summary.get('complete') or parent_summary.get('plan_hash') != parent_plan['plan_hash']:
                raise ValueError('第一批尚无完整匹配的汇总，拒绝开始第二批。')
        work = Path(record['settings']['workdir'])
        # Shared for these workers, exclusive for the original all-in-one launcher.
        with locked(work / '.all_pipeline.lock', shared=True, blocking=False):
            stage = 'prepare'
            progress()
            initialize(folder, record)
            if initial_run(record) == 'tune':
                stage = 'tune'
                progress()
                pipeline_action('workers', 'tune')
                finish_summary(work / 'runs/tune')
                stage = 'selection'
                progress()
                initialize_final(folder, record)
            stage = 'workers'
            progress()
            print(f'开始领取生成/训练任务；详细日志：{work}/runs/final/worker_logs', flush=True)
            pipeline_action('workers', 'final')
            stage = 'summary'
            progress()
            finish_summary(work / 'runs/final')
        stage, code = 'completed', 0
    except BaseException as error:
        error_text = str(error)
        raise
    finally:
        atomic_json(status_file, {'job_id': job_id, 'slot': args.slot, 'host': socket.gethostname(),
                    'exit_code': code, 'state': 'completed' if code == 0 else 'failed',
                    'stage': stage, 'error': error_text, 'finished': now()})


def show_status():
    """Read-only progress, including failures before an experiment plan exists."""
    pointer = ROOT / f'logs/slurm/latest_{batch_namespace()}_batch.json'
    if not pointer.exists():
        print('尚无本批实验提交记录。')
        return
    folder = Path(read_json(pointer)['batch'])
    record = read_json(folder / 'batch.json')
    print(f'批次：{folder.name}（以下为共享文件记录，不代表实时调度器查询）')
    for slot in record['slots']:
        job = slot.get('job_id')
        status = folder / f'worker-{slot["slot"]}-{job}.json'
        data = read_json(status) if status.exists() else {}
        state = data.get('state', ('failed' if data['exit_code'] else 'completed')
                         if data.get('exit_code') is not None else slot['state'])
        print(f'作业 {job or "未提交"}：{state}，阶段 {data.get("stage", "等待启动")}，'
              f'节点 {data.get("host", "未知")}；日志 logs/slurm/rebuttal-{job}.log')
        if data.get('error'):
            print('  错误：' + data['error'])
    prep_file = folder / 'preparation.json'
    if prep_file.exists():
        prep = read_json(prep_file)
        print(f'共享准备：{prep["state"]}，阶段 {prep.get("stage", "未知")}')
        if prep.get('error'):
            print('  原因：' + prep['error'])
    work = Path(record['settings']['workdir'])
    if initial_run(record) == 'tune':
        show_plan_status(work / 'runs/tune', '留出集调参')
        state = folder / 'final_preparation.json'
        if state.exists():
            print('参数选择/正式计划：' + str(read_json(state)))
    show_plan_status(work / 'runs/final', '正式评估')


def show_plan_status(plan_dir, label):
    print(label + '：')
    if not (plan_dir / 'plan.json').exists():
        print('正式实验计划尚未生成；完成生成 0 份、分类器训练 0 次。')
        return
    from .common import read_plan
    plan = read_plan(plan_dir / 'plan.json')
    generated = 0
    for seed in plan['generation_seeds']:
        for name in plan['conditions']:
            marker = plan_dir / 'queue' / f'gen_g{seed}_{name}.done.json'
            if marker.exists() and read_json(marker).get('plan_hash') == plan['plan_hash']:
                generated += 1
    completed, expected = 0, 0
    for seed in plan['generation_seeds']:
        for training in plan['training_seeds']:
            for arch in plan['architectures']:
                for name, condition in plan['conditions'].items():
                    if condition.get('only_arch', arch) != arch:
                        continue
                    expected += 1
                    result = plan_dir / 'evaluation' / f'g{seed}' / f't{training}' / arch / f'{name}.json'
                    if result.exists():
                        value = read_json(result)
                        if value.get('plan_hash') == plan['plan_hash'] and 'metrics' in value:
                            completed += 1
    print(f'生成：{generated}/{len(plan["generation_seeds"]) * len(plan["conditions"])}；训练：{completed}/{expected}')
    summary = plan_dir / 'summary.json'
    summary_data = read_json(summary) if summary.exists() else {}
    complete = (summary_data.get('complete', False) and summary_data.get('plan_hash') == plan['plan_hash']
                and completed == expected and generated == len(plan['generation_seeds']) * len(plan['conditions']))
    print(f'正式汇总完整：{complete}；结果目录：{plan_dir}')
    from .reporting import runtime_estimate
    runtime = runtime_estimate(plan_dir, plan, int(os.environ.get('REBUTTAL_JOB_COUNT', 4)))
    if runtime['median_generation_seconds'] is not None or runtime['median_training_seconds'] is not None:
        print('实测耗时估计（仅本计划，未计排队/失败）：' + str(runtime))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    submit_parser = sub.add_parser('submit')
    submit_parser.add_argument('--dry-run', action='store_true')
    submit_parser.add_argument('--retry-failed', action='store_true')
    submit_parser.add_argument('--status', action='store_true')
    submit_parser.add_argument('--after-batch', help='Full suite waits for every job in this allocation batch to succeed')
    worker_parser = sub.add_parser('worker')
    worker_parser.add_argument('--batch', required=True)
    worker_parser.add_argument('--slot', type=int, required=True)
    args = parser.parse_args()
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    try:
        (submit if args.action == 'submit' else worker)(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'单卡作业启动失败：{error}', file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
