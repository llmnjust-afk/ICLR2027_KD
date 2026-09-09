#!/bin/bash
# P2-8: Weight Learning — Grid search over complexity weights
# Uses cached VAE features (cluster_cache.pkl), only recomputes complexity scores
# Fast proxy: 1 seed, 500 epochs for evaluation
#
# Round 1: High β, δ=0 (4 configs on 4 GPUs)
# Round 2: Very high β, δ=0 (4 configs on 4 GPUs)
# Round 3: Small δ>0 (4 configs on 4 GPUs)

cd /root/ICLR2027_KD
mkdir -p logs

IMAGENET_DIR=/root/data/imagenet100/
SAVE_BASE=./results/sweep_in100
EVAL_FLAGS="--val-dir /root/data/imagenet100/val --class-file ./misc/class100.txt --nclass 100 --epochs 500 --seeds 0 --lr 0.1 --weight-decay 1e-4 --batch-size 128 --arch convnet --norm-type instance"
GEN_FLAGS="--spec imagenet100 --nclass 100 --imagenet-dir $IMAGENET_DIR --save-base $SAVE_BASE --ipc 10 --window high_noise --cags-min-scale 0.0 --cags-max-scale 0.06 --sigmoid-slope 3.0 --sigmoid-center 0.6 --recompute-only"

echo "============================================"
echo "P2-8: Weight Learning Grid Search"
echo "Start: $(date)"
echo "============================================"

# --- Round 1: High β, δ=0 ---
echo "=== Round 1: High entropy, no separability ==="
echo "Start: $(date)"

CUDA_VISIBLE_DEVICES=0 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py $GEN_FLAGS \
    --tag wl_r1_c1 --alpha 0.2 --beta 0.5 --gamma 0.3 --delta 0.0 \
    > logs/gen_wl_r1_c1.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=1 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py $GEN_FLAGS \
    --tag wl_r1_c2 --alpha 0.1 --beta 0.6 --gamma 0.3 --delta 0.0 \
    > logs/gen_wl_r1_c2.log 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=2 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py $GEN_FLAGS \
    --tag wl_r1_c3 --alpha 0.2 --beta 0.6 --gamma 0.2 --delta 0.0 \
    > logs/gen_wl_r1_c3.log 2>&1 &
PID3=$!

CUDA_VISIBLE_DEVICES=3 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py $GEN_FLAGS \
    --tag wl_r1_c4 --alpha 0.3 --beta 0.5 --gamma 0.2 --delta 0.0 \
    > logs/gen_wl_r1_c4.log 2>&1 &
PID4=$!

wait $PID1 $PID2 $PID3 $PID4
echo "Round 1 generation done: $(date)"

# Evaluate Round 1
for i in 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$((i-1)) env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
        --train-dir $SAVE_BASE/wl_r1_c${i}/dataset_0 $EVAL_FLAGS \
        > logs/eval_wl_r1_c${i}.log 2>&1 &
done
wait
echo "Round 1 evaluation done: $(date)"
for i in 1 2 3 4; do
    echo -n "r1_c${i}: "; grep "Mean Best" logs/eval_wl_r1_c${i}.log 2>/dev/null || echo "N/A"
done

# --- Round 2: Very high β, δ=0 ---
echo "=== Round 2: Very high entropy, no separability ==="
echo "Start: $(date)"

CUDA_VISIBLE_DEVICES=0 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py $GEN_FLAGS \
    --tag wl_r2_c1 --alpha 0.1 --beta 0.7 --gamma 0.2 --delta 0.0 \
    > logs/gen_wl_r2_c1.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=1 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py $GEN_FLAGS \
    --tag wl_r2_c2 --alpha 0.0 --beta 0.5 --gamma 0.5 --delta 0.0 \
    > logs/gen_wl_r2_c2.log 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=2 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py $GEN_FLAGS \
    --tag wl_r2_c3 --alpha 0.2 --beta 0.4 --gamma 0.4 --delta 0.0 \
    > logs/gen_wl_r2_c3.log 2>&1 &
PID3=$!

CUDA_VISIBLE_DEVICES=3 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py $GEN_FLAGS \
    --tag wl_r2_c4 --alpha 0.1 --beta 0.5 --gamma 0.4 --delta 0.0 \
    > logs/gen_wl_r2_c4.log 2>&1 &
PID4=$!

wait $PID1 $PID2 $PID3 $PID4
echo "Round 2 generation done: $(date)"

# Evaluate Round 2
for i in 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$((i-1)) env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
        --train-dir $SAVE_BASE/wl_r2_c${i}/dataset_0 $EVAL_FLAGS \
        > logs/eval_wl_r2_c${i}.log 2>&1 &
done
wait
echo "Round 2 evaluation done: $(date)"
for i in 1 2 3 4; do
    echo -n "r2_c${i}: "; grep "Mean Best" logs/eval_wl_r2_c${i}.log 2>/dev/null || echo "N/A"
done

# --- Round 3: Small δ>0 ---
echo "=== Round 3: Small separability weight ==="
echo "Start: $(date)"

CUDA_VISIBLE_DEVICES=0 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py $GEN_FLAGS \
    --tag wl_r3_c1 --alpha 0.2 --beta 0.5 --gamma 0.2 --delta 0.1 \
    > logs/gen_wl_r3_c1.log 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=1 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py $GEN_FLAGS \
    --tag wl_r3_c2 --alpha 0.1 --beta 0.6 --gamma 0.2 --delta 0.1 \
    > logs/gen_wl_r3_c2.log 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=2 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py $GEN_FLAGS \
    --tag wl_r3_c3 --alpha 0.15 --beta 0.55 --gamma 0.25 --delta 0.05 \
    > logs/gen_wl_r3_c3.log 2>&1 &
PID3=$!

CUDA_VISIBLE_DEVICES=3 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py $GEN_FLAGS \
    --tag wl_r3_c4 --alpha 0.25 --beta 0.45 --gamma 0.25 --delta 0.05 \
    > logs/gen_wl_r3_c4.log 2>&1 &
PID4=$!

wait $PID1 $PID2 $PID3 $PID4
echo "Round 3 generation done: $(date)"

# Evaluate Round 3
for i in 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$((i-1)) env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
        --train-dir $SAVE_BASE/wl_r3_c${i}/dataset_0 $EVAL_FLAGS \
        > logs/eval_wl_r3_c${i}.log 2>&1 &
done
wait
echo "Round 3 evaluation done: $(date)"
for i in 1 2 3 4; do
    echo -n "r3_c${i}: "; grep "Mean Best" logs/eval_wl_r3_c${i}.log 2>/dev/null || echo "N/A"
done

# --- Summary ---
echo ""
echo "============================================"
echo "WEIGHT LEARNING SUMMARY"
echo "============================================"
echo ""
echo "Config              α     β     γ     δ     Top-1%"
echo "-----------------------------------------------------------"
echo "Existing factor ablation (3 seeds, 2000 epochs):"
echo "  entropy_only      0.0   1.0   0.0   0.0   25.31 ± 0.62"
echo "  equal_weights     0.25  0.25  0.25  0.25  24.69 ± 0.29"
echo "  no_separability   0.25  0.25  0.5   0.0   24.61 ± 0.44"
echo "  intra_var_only    0.0   0.0   1.0   0.0   24.46 ± 0.16"
echo "  sep_dominated     0.1   0.1   0.2   0.6   24.25 ± 0.44"
echo "  full_cags         0.3   0.3   0.2   0.2   24.24 ± 0.26"
echo "  mode_count_only   1.0   0.0   0.0   0.0   23.92 ± 0.12"
echo "  sep_only          0.0   0.0   0.0   1.0   23.75 ± 0.09"
echo ""
echo "New grid search (1 seed, 500 epochs — fast proxy):"

declare -A CONFIGS
CONFIGS["r1_c1"]="0.2  0.5  0.3  0.0"
CONFIGS["r1_c2"]="0.1  0.6  0.3  0.0"
CONFIGS["r1_c3"]="0.2  0.6  0.2  0.0"
CONFIGS["r1_c4"]="0.3  0.5  0.2  0.0"
CONFIGS["r2_c1"]="0.1  0.7  0.2  0.0"
CONFIGS["r2_c2"]="0.0  0.5  0.5  0.0"
CONFIGS["r2_c3"]="0.2  0.4  0.4  0.0"
CONFIGS["r2_c4"]="0.1  0.5  0.4  0.0"
CONFIGS["r3_c1"]="0.2  0.5  0.2  0.1"
CONFIGS["r3_c2"]="0.1  0.6  0.2  0.1"
CONFIGS["r3_c3"]="0.15 0.55 0.25 0.05"
CONFIGS["r3_c4"]="0.25 0.45 0.25 0.05"

for key in r1_c1 r1_c2 r1_c3 r1_c4 r2_c1 r2_c2 r2_c3 r2_c4 r3_c1 r3_c2 r3_c3 r3_c4; do
    weights="${CONFIGS[$key]}"
    result=$(grep "Mean Best Top-1" logs/eval_wl_${key}.log 2>/dev/null | awk '{print $NF}')
    echo "  ${key}            ${weights}  ${result:-N/A}"
done

echo ""
echo "Done: $(date)"
