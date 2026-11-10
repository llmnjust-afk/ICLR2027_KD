"""Fail before downloads/background launch when the execution environment is unusable."""
import argparse
import os
import re
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', default='auto')
    parser.add_argument('--resources-only', action='store_true')
    args = parser.parse_args()
    try:
        import requests
        import huggingface_hub
        from PIL import Image
        if args.resources_only:
            print('资源准备环境检查通过。', file=sys.stderr)
            return
        selected = args.gpus
        if selected != 'auto':
            ids = selected.split(',')
            if not ids or len(ids) != len(set(ids)) or any(
                    not re.fullmatch(r'[A-Za-z0-9-]+', x) or x == '-1' for x in ids):
                raise ValueError(f'无效或重复 GPU 编号：{selected}')
            os.environ['CUDA_VISIBLE_DEVICES'] = selected
        import torch
        from diffusers import AutoencoderKL
        from .generate import GenerationEngine
        from .evaluate import train_one
        import sklearn
        if not torch.cuda.is_available():
            from .common import ROOT
            raise RuntimeError('PyTorch 无法使用 CUDA。若当前位于登录节点，请提交 GPU 作业：'
                               f'bash {ROOT}/scripts/submit_rebuttal.sh；'
                               '仅下载数据/模型可用 --prepare-only。')
        count = torch.cuda.device_count()
        if selected == 'auto':
            selected = ','.join(str(i) for i in range(count))
        elif count != len(selected.split(',')):
            raise RuntimeError(f'请求 GPU={selected}，但 PyTorch 只看见 {count} 张卡。')
        for index in range(count):
            with torch.cuda.device(index):
                torch.ones(1, device=f'cuda:{index}').sum().item()
                print(f'GPU {selected.split(",")[index]}: {torch.cuda.get_device_name(index)} 可用', file=sys.stderr)
        print(selected)
    except Exception as error:
        print(f'启动前检查失败：{type(error).__name__}: {error}', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
