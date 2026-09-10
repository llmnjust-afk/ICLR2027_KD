#!/bin/bash
# Auto-launch IN-1K and Food-101 experiments when GPUs become free
# Monitors GPU status and launches generation jobs

cd /root/ICLR2027_KD
mkdir -p logs

OPTIMAL="--alpha 0.0 --beta 0.5 --gamma 0.5 --delta 0.0"
COMMON_GEN="--cags-min-scale 0.0 --cags-max-scale 0.06 --sigmoid-slope 3.0 --sigmoid-center 0.6"

echo "=== Auto-launch script started at $(date) ==="

# ===== Wait for GPU 0 to be free, then start IN-1K cache pre-computation =====
echo "Waiting for GPU 0 to be free for IN-1K cache pre-computation..."
while true; do
    GPU0_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0)
    if [ "$GPU0_UTIL" -lt 5 ]; then
        echo "GPU 0 is free (util=$GPU0_UTIL). Starting IN-1K cache pre-computation at $(date)"
        break
    fi
    sleep 60
done

# Pre-compute cluster cache for IN-1K (VAE only, no DiT needed)
CUDA_VISIBLE_DEVICES=0 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py \
    --spec imagenet1k --nclass 1000 \
    --imagenet-dir /root/data/imagenet1k/ \
    --save-base ./results/in1k --tag cache \
    --ipc 10 --cache-only \
    $OPTIMAL $COMMON_GEN \
    > logs/gen_in1k_cache.log 2>&1

echo "IN-1K cache pre-computation done at $(date)"

# Now launch IN-1K CAGS generation on GPU 0
CUDA_VISIBLE_DEVICES=0 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py \
    --spec imagenet1k --nclass 1000 \
    --imagenet-dir /root/data/imagenet1k/ \
    --save-base ./results/in1k --tag optimal \
    --ipc 10 --window high_noise \
    --recompute-only \
    $OPTIMAL $COMMON_GEN \
    > logs/gen_in1k_optimal.log 2>&1 &

echo "IN-1K CAGS generation started on GPU 0 at $(date)"

# ===== Wait for GPU 3 to be free, then start Food-101 CAGS (SD) =====
echo "Waiting for GPU 3 to be free for Food-101 SD generation..."
while true; do
    GPU3_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 3)
    if [ "$GPU3_UTIL" -lt 5 ]; then
        echo "GPU 3 is free (util=$GPU3_UTIL). Starting Food-101 CAGS generation at $(date)"
        break
    fi
    sleep 60
done

# Start Food-101 CAGS generation with Stable Diffusion
CUDA_VISIBLE_DEVICES=3 env PYTHONUNBUFFERED=1 python3 -u sample_ags_sd.py \
    --dataset food101 \
    --num-samples 10 --num-datasets 3 \
    --save-dir ./results/sd_food101_cags \
    --cags-alpha 0.0 --cags-beta 0.5 --cags-gamma 0.5 --cags-delta 0.0 \
    --guidance-scale-min 0.0 --guidance-scale-max 0.06 \
    --guidance-window high_noise \
    > logs/gen_sd_food101_cags.log 2>&1 &

echo "Food-101 CAGS generation started on GPU 3 at $(date)"

# ===== Wait for GPU 1 to be free, then start IN-1K unguided =====
echo "Waiting for GPU 1 to be free for IN-1K unguided generation..."
while true; do
    GPU1_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1)
    if [ "$GPU1_UTIL" -lt 5 ]; then
        echo "GPU 1 is free (util=$GPU1_UTIL). Starting IN-1K unguided generation at $(date)"
        break
    fi
    sleep 60
done

CUDA_VISIBLE_DEVICES=1 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py \
    --spec imagenet1k --nclass 1000 \
    --imagenet-dir /root/data/imagenet1k/ \
    --save-base ./results/in1k --tag unguided \
    --ipc 10 --window high_noise \
    --recompute-only \
    --no-cags --fixed-scale 0.0 \
    > logs/gen_in1k_unguided.log 2>&1 &

echo "IN-1K unguided generation started on GPU 1 at $(date)"

# ===== Wait for GPU 2 to be free, then start Food-101 unguided (SD) =====
echo "Waiting for GPU 2 to be free for Food-101 unguided SD generation..."
while true; do
    GPU2_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 2)
    if [ "$GPU2_UTIL" -lt 5 ]; then
        echo "GPU 2 is free (util=$GPU2_UTIL). Starting Food-101 unguided SD generation at $(date)"
        break
    fi
    sleep 60
done

CUDA_VISIBLE_DEVICES=2 env PYTHONUNBUFFERED=1 python3 -u sample_ags_sd.py \
    --dataset food101 \
    --num-samples 10 --num-datasets 3 \
    --save-dir ./results/sd_food101_unguided \
    --no-cags --no-iast --no-tags \
    --guidance-scale-min 0.0 --guidance-scale-max 0.06 \
    --guidance-window high_noise \
    > logs/gen_sd_food101_unguided.log 2>&1 &

echo "Food-101 unguided SD generation started on GPU 2 at $(date)"
echo "=== All generation jobs launched. Auto-launch script complete at $(date) ==="
