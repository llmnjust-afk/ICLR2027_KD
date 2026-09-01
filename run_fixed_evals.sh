#!/bin/bash
# Run ImageNette + ImageWoof evals with correct class files
set -uo pipefail
cd /root/ICLR2027_KD

GPU=0
SEEDS="0 1 2"
DEPTH=6

mkdir -p logs/phase2

echo "========================================"
echo "  Fixed Evals: ImageNette + ImageWoof"
echo "  Started: $(date)"
echo "========================================"

# ImageNette unguided
echo ""
echo "=== ImageNette unguided ==="
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python3 quick_eval_v2.py \
    --train-dir results/sweep_nette/high_noise_unguided_d25/dataset_0 \
    --val-dir /root/data/imagenette2/val \
    --class-file ./misc/class_nette.txt \
    --nclass 10 --arch convnet --depth $DEPTH \
    --epochs 1000 --seeds $SEEDS \
    2>&1 | tee logs/phase2/nette_unguided_eval_fixed.log

# ImageNette lambda005
echo ""
echo "=== ImageNette lambda005 ==="
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python3 quick_eval_v2.py \
    --train-dir results/sweep_nette/high_noise_lambda005_d25/dataset_0 \
    --val-dir /root/data/imagenette2/val \
    --class-file ./misc/class_nette.txt \
    --nclass 10 --arch convnet --depth $DEPTH \
    --epochs 1000 --seeds $SEEDS \
    2>&1 | tee logs/phase2/nette_lambda005_eval_fixed.log

# ImageWoof unguided
echo ""
echo "=== ImageWoof unguided ==="
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python3 quick_eval_v2.py \
    --train-dir results/sweep_woof/high_noise_unguided_d25/dataset_0 \
    --val-dir /root/data/imagewoof2/val \
    --class-file ./misc/class_woof.txt \
    --nclass 10 --arch convnet --depth $DEPTH \
    --epochs 1000 --seeds $SEEDS \
    2>&1 | tee logs/phase2/woof_unguided_eval_fixed.log

# ImageWoof lambda005
echo ""
echo "=== ImageWoof lambda005 ==="
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python3 quick_eval_v2.py \
    --train-dir results/sweep_woof/high_noise_lambda005_d25/dataset_0 \
    --val-dir /root/data/imagewoof2/val \
    --class-file ./misc/class_woof.txt \
    --nclass 10 --arch convnet --depth $DEPTH \
    --epochs 1000 --seeds $SEEDS \
    2>&1 | tee logs/phase2/woof_lambda005_eval_fixed.log

echo ""
echo "========================================"
echo "  ALL FIXED EVALS DONE — $(date)"
echo "========================================"
