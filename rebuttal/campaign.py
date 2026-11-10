"""Submit allocation first, then tuning/full with an all-workers afterok dependency."""
import argparse
import os
import subprocess
import sys

from .common import ROOT, read_json
from .slurm import command


def run(args):
    configs = [ROOT / 'rebuttal.core.env', ROOT / 'rebuttal.tuned.env']
    scripts = ['submit_rebuttal_core.sh', 'submit_rebuttal_tuned.sh']
    mode = '--status' if args.status else '--dry-run' if args.dry_run else None
    for index, (config, script) in enumerate(zip(configs, scripts)):
        call = ['bash', str(ROOT / 'scripts' / script)]
        if mode:
            call.append(mode)
        else:
            if args.retry_failed:
                call.append('--retry-failed')
            if index:
                pointer = read_json(ROOT / 'logs/slurm/latest_core_batch.json')
                call += ['--after-batch', pointer['batch']]
        print(('第一批：六组分配对照' if index == 0 else '第二批：独立调参及统一消融'), flush=True)
        command(call, env=dict(os.environ, REBUTTAL_CONFIG=str(config)))
    if not args.status:
        print('第二批使用 afterok 依赖第一批的全部作业；最多四个单卡实验作业同时运行。', flush=True)
        if args.dry_run:
            print('以上为预览，未提交作业或写入配置。', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--status', action='store_true')
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f'补实验提交停止：{error}', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
