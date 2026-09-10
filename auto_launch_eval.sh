#!/bin/bash
# Auto-launch evaluation for IN-1K and Food-101 after generation completes
# Monitors for generated dataset directories and launches evaluation

cd /root/ICLR2027_KD
mkdir -p logs

echo "=== Evaluation auto-launch started at $(date) ==="

# ===== Wait for IN-1K CAGS generation to complete =====
echo "Waiting for IN-1K CAGS generation to complete..."
while [ ! -d "./results/in1k/optimal/dataset_0" ] || [ "$(ls ./results/in1k/optimal/dataset_0/ 2>/dev/null | wc -l)" -lt 1000 ]; do
    sleep 120
    COUNT=$(ls ./results/in1k/optimal/dataset_0/ 2>/dev/null | wc -l)
    echo "  IN-1K CAGS: $COUNT/1000 classes generated at $(date)"
done
echo "IN-1K CAGS generation complete at $(date)"

# Start IN-1K CAGS evaluation (1000 epochs for time efficiency)
CUDA_VISIBLE_DEVICES=0 env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
    --train-dir ./results/in1k/optimal/dataset_0 \
    --val-dir /root/data/imagenet1k/val \
    --class-file ./misc/class_indices.txt \
    --nclass 1000 --epochs 1000 --seeds 0 \
    --lr 0.1 --weight-decay 1e-4 --batch-size 128 \
    --arch convnet --depth 6 --norm-type instance \
    > logs/eval_in1k_cags.log 2>&1 &
echo "IN-1K CAGS evaluation started on GPU 0 at $(date)"

# ===== Wait for IN-1K unguided generation to complete =====
echo "Waiting for IN-1K unguided generation to complete..."
while [ ! -d "./results/in1k/unguided/dataset_0" ] || [ "$(ls ./results/in1k/unguided/dataset_0/ 2>/dev/null | wc -l)" -lt 1000 ]; do
    sleep 120
    COUNT=$(ls ./results/in1k/unguided/dataset_0/ 2>/dev/null | wc -l)
    echo "  IN-1K unguided: $COUNT/1000 classes generated at $(date)"
done
echo "IN-1K unguided generation complete at $(date)"

CUDA_VISIBLE_DEVICES=1 env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
    --train-dir ./results/in1k/unguided/dataset_0 \
    --val-dir /root/data/imagenet1k/val \
    --class-file ./misc/class_indices.txt \
    --nclass 1000 --epochs 1000 --seeds 0 \
    --lr 0.1 --weight-decay 1e-4 --batch-size 128 \
    --arch convnet --depth 6 --norm-type instance \
    > logs/eval_in1k_unguided.log 2>&1 &
echo "IN-1K unguided evaluation started on GPU 1 at $(date)"

# ===== Wait for Food-101 CAGS generation to complete =====
echo "Waiting for Food-101 CAGS generation to complete..."
while [ ! -d "./results/sd_food101_cags/dataset_0" ] || [ "$(ls ./results/sd_food101_cags/dataset_0/ 2>/dev/null | wc -l)" -lt 101 ]; do
    sleep 60
    COUNT=$(ls ./results/sd_food101_cags/dataset_0/ 2>/dev/null | wc -l)
    echo "  Food-101 CAGS: $COUNT/101 classes generated at $(date)"
done
echo "Food-101 CAGS generation complete at $(date)"

# Wait for a GPU to be free
echo "Waiting for GPU to be free for Food-101 evaluation..."
while true; do
    for gpu in 2 3 0 1; do
        UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu)
        if [ "$UTIL" -lt 5 ]; then
            FREE_GPU=$gpu
            break 2
        fi
    done
    sleep 60
done

CUDA_VISIBLE_DEVICES=$FREE_GPU env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
    --train-dir ./results/sd_food101_cags/dataset_0 \
    --val-dir /root/data/food101/val \
    --class-file ./misc/class_food101.txt \
    --nclass 101 --epochs 2000 --seeds 0 1 2 \
    --lr 0.1 --weight-decay 1e-4 --batch-size 128 \
    --arch convnet --depth 6 --norm-type instance \
    > logs/eval_sd_food101_cags.log 2>&1 &
echo "Food-101 CAGS evaluation started on GPU $FREE_GPU at $(date)"

# ===== Wait for Food-101 unguided generation to complete =====
echo "Waiting for Food-101 unguided generation to complete..."
while [ ! -d "./results/sd_food101_unguided/dataset_0" ] || [ "$(ls ./results/sd_food101_unguided/dataset_0/ 2>/dev/null | wc -l)" -lt 101 ]; do
    sleep 60
    COUNT=$(ls ./results/sd_food101_unguided/dataset_0/ 2>/dev/null | wc -l)
    echo "  Food-101 unguided: $COUNT/101 classes generated at $(date)"
done
echo "Food-101 unguided generation complete at $(date)"

# Wait for a GPU to be free
echo "Waiting for GPU to be free for Food-101 unguided evaluation..."
while true; do
    for gpu in 2 3 0 1; do
        UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu)
        if [ "$UTIL" -lt 5 ]; then
            FREE_GPU=$gpu
            break 2
        fi
    done
    sleep 60
done

CUDA_VISIBLE_DEVICES=$FREE_GPU env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
    --train-dir ./results/sd_food101_unguided/dataset_0 \
    --val-dir /root/data/food101/val \
    --class-file ./misc/class_food101.txt \
    --nclass 101 --epochs 2000 --seeds 0 1 2 \
    --lr 0.1 --weight-decay 1e-4 --batch-size 128 \
    --arch convnet --depth 6 --norm-type instance \
    > logs/eval_sd_food101_unguided.log 2>&1 &
echo "Food-101 unguided evaluation started on GPU $FREE_GPU at $(date)"

echo "=== All evaluation jobs launched. Script complete at $(date) ==="
