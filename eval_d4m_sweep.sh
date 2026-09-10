#!/bin/bash
# Evaluate D4M sensitivity sweep + real images baseline
# Uses 1 seed for D4M sensitivity curve (trend), 3 seeds for real images baseline
# Usage: CUDA_VISIBLE_DEVICES=X bash eval_d4m_sweep.sh
cd /root/ICLR2027_KD

# D4M datasets (t_start = 5, 10, 15, 20, 30, 35, 45) - 1 seed for sensitivity curve
for T in 5 10 15 20 30 35 45; do
    DIR="results/sweep_in100/d4m_t${T}/dataset_0"
    COUNT=$(find "$DIR" -name "*.png" 2>/dev/null | wc -l)
    if [ "$COUNT" -lt 1000 ]; then
        echo "SKIP d4m_t${T}: only ${COUNT} images (not ready)"
        continue
    fi
    LOG="logs/eval_d4m_t${T}.log"
    if [ -f "$LOG" ] && grep -q "Mean Best" "$LOG" 2>/dev/null; then
        echo "SKIP d4m_t${T}: already evaluated"
        continue
    fi
    echo "=== Evaluating D4M t_start=${T} (1 seed) ==="
    python3 quick_eval_v3.py \
        --train-dir "$DIR" \
        --val-dir /root/data/imagenet100/val \
        --class-file misc/class100.txt \
        --nclass 100 \
        --epochs 2000 \
        --seeds 0 \
        --depth 6 --arch convnet \
        > "$LOG" 2>&1
    echo "  Result: $(grep 'Mean Best' "$LOG" 2>/dev/null | head -1)"
done

# Real images baseline - 3 seeds
DIR="results/sweep_in100/real_images/dataset_0"
LOG="logs/eval_real_images.log"
if [ ! -f "$LOG" ] || ! grep -q "Mean Best" "$LOG" 2>/dev/null; then
    echo "=== Evaluating Real Images baseline (3 seeds) ==="
    python3 quick_eval_v3.py \
        --train-dir "$DIR" \
        --val-dir /root/data/imagenet100/val \
        --class-file misc/class100.txt \
        --nclass 100 \
        --epochs 2000 \
        --seeds 0 1 2 \
        --depth 6 --arch convnet \
        > "$LOG" 2>&1
    echo "  Result: $(grep 'Mean Best' "$LOG" 2>/dev/null | head -1)"
fi

echo "ALL EVALUATIONS COMPLETE: $(date)"
