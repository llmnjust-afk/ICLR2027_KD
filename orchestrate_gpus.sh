#!/bin/bash
# Orchestrates GPU usage:
# 1. Kills GPU 3 Food-101 unguided after seed 0 finishes (avoid redundant seed 1)
# 2. Launches Food-101 unguided seed 2 on GPU 0 after IN-1K CAGS finishes
# 3. Kills GPU 1 IN-1K unguided after it finishes (optional, for cleanup)

LOG_DIR=/root/ICLR2027_KD/logs
GPU3_PID=4107560  # Food-101 unguided seeds 0,1,2 on GPU 3

# Phase 1: Wait for GPU 3 seed 0 to finish, then kill it
echo "$(date '+%H:%M:%S') Waiting for GPU 3 Food-101 seed 0 to complete..."
while true; do
    if grep -q "Seed 1:" "$LOG_DIR/eval_sd_food101_unguided.log" 2>/dev/null; then
        echo "$(date '+%H:%M:%S') GPU 3 started seed 1 - killing to avoid redundancy"
        kill $GPU3_PID 2>/dev/null
        break
    fi
    # Also check if process died
    if ! ps -p $GPU3_PID > /dev/null 2>&1; then
        echo "$(date '+%H:%M:%S') GPU 3 process already exited"
        break
    fi
    sleep 30
done

# Phase 2: Wait for GPU 0 IN-1K CAGS to finish, then launch seed 2
echo "$(date '+%H:%M:%S') Waiting for GPU 0 IN-1K CAGS to complete..."
while true; do
    if grep -q "Mean Best Top-1" "$LOG_DIR/eval_in1k_cags.log" 2>/dev/null; then
        echo "$(date '+%H:%M:%S') IN-1K CAGS completed, launching Food-101 unguided seed 2 on GPU 0"
        cd /root/ICLR2027_KD && CUDA_VISIBLE_DEVICES=0 nohup python3 -u quick_eval_v3.py \
            --train-dir ./results/sd_food101_unguided/dataset_2 \
            --val-dir /root/data/food101/val \
            --class-file ./misc/class_food101.txt \
            --nclass 101 --epochs 2000 --seeds 2 \
            --lr 0.1 --weight-decay 1e-4 --batch-size 128 \
            --arch convnet --depth 6 --norm-type instance \
            > "$LOG_DIR/eval_sd_food101_unguided_seed2.log" 2>&1 &
        echo "$(date '+%H:%M:%S') Launched seed 2 on GPU 0, PID=$!"
        break
    fi
    sleep 30
done

echo "$(date '+%H:%M:%S') Orchestration complete. All GPUs now busy."
