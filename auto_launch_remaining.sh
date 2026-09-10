#!/bin/bash
# Auto-launch remaining evaluations
cd /root/ICLR2027_KD

echo "=== Auto-launch remaining evals started at $(date) ==="

# ===== Wait for IN-1K unguided generation to complete, then start eval =====
echo "Waiting for IN-1K unguided generation to complete..."
while [ "$(ls ./results/in1k/unguided/dataset_0/ 2>/dev/null | wc -l)" -lt 1000 ]; do
    sleep 60
    COUNT=$(ls ./results/in1k/unguided/dataset_0/ 2>/dev/null | wc -l)
    echo "  IN-1K unguided gen: $COUNT/1000 at $(date)"
done
echo "IN-1K unguided generation complete at $(date)"

CUDA_VISIBLE_DEVICES=1 env PYTHONUNBUFFERED=1 python3 -u /root/ICLR2027_KD/quick_eval_v3.py \
    --train-dir ./results/in1k/unguided/dataset_0 \
    --val-dir /root/data/imagenet1k/val \
    --class-file ./misc/class_indices.txt \
    --nclass 1000 --epochs 1000 --seeds 0 \
    --lr 0.1 --weight-decay 1e-4 --batch-size 128 \
    --arch convnet --depth 6 --norm-type instance \
    > /root/ICLR2027_KD/logs/eval_in1k_unguided.log 2>&1 &
echo "IN-1K unguided eval started on GPU 1 at $(date) (PID: $!)"

# ===== Wait for Food-101 CAGS eval to complete (GPU 3 frees), then start unguided eval =====
echo "Waiting for Food-101 CAGS eval to complete..."
while true; do
    UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 3)
    FOOD_PROC=$(ps aux | grep 'quick_eval.*food101' | grep -v grep | wc -l)
    if [ "$FOOD_PROC" -eq 0 ]; then
        echo "Food-101 CAGS eval complete at $(date)"
        break
    fi
    sleep 60
done

CUDA_VISIBLE_DEVICES=3 env PYTHONUNBUFFERED=1 python3 -u /root/ICLR2027_KD/quick_eval_v3.py \
    --train-dir ./results/sd_food101_unguided/dataset_0 \
    --val-dir /root/data/food101/val \
    --class-file ./misc/class_food101.txt \
    --nclass 101 --epochs 2000 --seeds 0 1 2 \
    --lr 0.1 --weight-decay 1e-4 --batch-size 128 \
    --arch convnet --depth 6 --norm-type instance \
    > /root/ICLR2027_KD/logs/eval_sd_food101_unguided.log 2>&1 &
echo "Food-101 unguided eval started on GPU 3 at $(date) (PID: $!)"

echo "=== All remaining evals launched. Script complete at $(date) ==="
