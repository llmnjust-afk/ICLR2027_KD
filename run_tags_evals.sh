#!/bin/bash
# Run TAGS evals (10-class ImageNet-100) after fixed evals complete
set -uo pipefail
cd /root/ICLR2027_KD

GPU=0
SEEDS="0 1 2"
DEPTH=6
VAL_DIR="/root/data/imagenet100_10class/val"
CLASS_FILE="./misc/class100.txt"

mkdir -p logs/phase2

# Wait for fixed evals to finish
echo "Waiting for fixed evals to complete..."
while ps aux | grep -q "[r]un_fixed_evals.sh"; do
    sleep 30
done
echo "Fixed evals done. Starting TAGS evals."

echo "========================================"
echo "  TAGS Evals: 10-class ImageNet-100"
echo "  Started: $(date)"
echo "========================================"

# TAGS cosine
echo ""
echo "=== TAGS cosine (lambda=0.1) ==="
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python3 quick_eval_v2.py \
    --train-dir results/sweep_in10/high_noise_tags_cosine_l0.1_d25/dataset_0 \
    --val-dir "$VAL_DIR" --class-file "$CLASS_FILE" \
    --nclass 10 --arch convnet --depth $DEPTH \
    --epochs 1000 --seeds $SEEDS \
    2>&1 | tee logs/phase2/tags_cosine_eval.log

# TAGS linear
echo ""
echo "=== TAGS linear (lambda=0.1) ==="
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python3 quick_eval_v2.py \
    --train-dir results/sweep_in10/high_noise_tags_linear_l0.1_d25/dataset_0 \
    --val-dir "$VAL_DIR" --class-file "$CLASS_FILE" \
    --nclass 10 --arch convnet --depth $DEPTH \
    --epochs 1000 --seeds $SEEDS \
    2>&1 | tee logs/phase2/tags_linear_eval.log

# TAGS exponential
echo ""
echo "=== TAGS exponential (lambda=0.1) ==="
CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python3 quick_eval_v2.py \
    --train-dir results/sweep_in10/high_noise_tags_exp_l0.1_d25/dataset_0 \
    --val-dir "$VAL_DIR" --class-file "$CLASS_FILE" \
    --nclass 10 --arch convnet --depth $DEPTH \
    --epochs 1000 --seeds $SEEDS \
    2>&1 | tee logs/phase2/tags_exp_eval.log

echo ""
echo "========================================"
echo "  ALL TAGS EVALS DONE — $(date)"
echo "========================================"
