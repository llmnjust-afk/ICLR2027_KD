import sys, os, time
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
sys.argv = ['quick_eval_v3_tagsfix.py',
    '--train-dir', './results/sweep_in100/high_noise_tags_linear_cagsv2_d25/dataset_0',
    '--val-dir', '/root/data/imagenet100/val',
    '--class-file', './misc/class100.txt',
    '--nclass', '100', '--epochs', '1300', '--seeds', '0', '1', '2']

# Patch print to also write to a progress file
import builtins
_orig_print = builtins.print
_prog = open('/root/ICLR2027_KD/logs/tags_progress.txt', 'w', buffering=1)
def patched_print(*args, **kwargs):
    _orig_print(*args, **kwargs)
    _prog.write(' '.join(str(a) for a in args) + '\n')
    _prog.flush()
builtins.print = patched_print

sys.path.insert(0, '/root/ICLR2027_KD')
os.chdir('/root/ICLR2027_KD')
exec(open('quick_eval_v3_tagsfix.py').read())
_prog.close()
