#!/bin/bash
# Run CAGS v2 experiments with regenerated v2 cache
# GPU 0: 10-class CAGS v2 (3 configs)
# GPU 1: 100-class cache regen + CAGS v2 (2 configs)

cd /root/ICLR2027_KD

BASE_10="./results/sweep_in10"
BASE_100="./results/sweep_in100"
DATA_10="/root/data/imagenet100_10class"
DATA_100="/root/data/imagenet100"

run_10class() {
    local TAG=$1
    local MIN_S=$2
    local MAX_S=$3

    local CONFIG="high_noise_${TAG}_d25"
    local SAVE_DIR="${BASE_10}/${CONFIG}"

    echo "  [10-class] ${TAG}: range=[${MIN_S}, ${MAX_S}] — $(date)"

    # Check if eval already done
    if grep -q "Mean Top-1" "logs/10class_${TAG}_eval.log" 2>/dev/null; then
        echo "    Already done, skipping"
        return 0
    fi

    # Generate images (1-epoch sanity eval inside duration_sweep)
    CUDA_VISIBLE_DEVICES=0 python3 duration_sweep.py \
        --window high_noise \
        --imagenet-dir "${DATA_10}" \
        --save-base "${BASE_10}" \
        --ipc 10 \
        --num-datasets 1 \
        --epochs 1 \
        --durations 25 \
        --tag "${TAG}" \
        --spec imagenet100 \
        --nclass 10 \
        --depth 6 \
        --cags-min-scale ${MIN_S} --cags-max-scale ${MAX_S} \
        > "logs/10class_${TAG}_gen.log" 2>&1

    # Real evaluation (3 seeds, 1000 epochs, ConvNet-6)
    CUDA_VISIBLE_DEVICES=0 python3 quick_eval_v2.py \
        --train-dir "${SAVE_DIR}/dataset_0" \
        --val-dir "${DATA_10}/val" \
        --nclass 10 \
        --arch convnet \
        --depth 6 \
        --epochs 1000 \
        --seeds 0 1 2 \
        > "logs/10class_${TAG}_eval.log" 2>&1

    echo "    ${TAG} done — $(date)"
    grep "Mean Top-1" "logs/10class_${TAG}_eval.log" 2>/dev/null
}

run_100class() {
    local TAG=$1
    local MIN_S=$2
    local MAX_S=$3

    local CONFIG="high_noise_${TAG}_d25"
    local SAVE_DIR="${BASE_100}/${CONFIG}"

    echo "  [100-class] ${TAG}: range=[${MIN_S}, ${MAX_S}] — $(date)"

    if grep -q "Mean Top-1" "logs/100class_${TAG}_eval.log" 2>/dev/null; then
        echo "    Already done, skipping"
        return 0
    fi

    # Generate images
    CUDA_VISIBLE_DEVICES=1 python3 duration_sweep.py \
        --window high_noise \
        --imagenet-dir "${DATA_100}" \
        --save-base "${BASE_100}" \
        --ipc 10 \
        --num-datasets 1 \
        --epochs 1 \
        --durations 25 \
        --tag "${TAG}" \
        --spec imagenet100 \
        --nclass 100 \
        --depth 6 \
        --cags-min-scale ${MIN_S} --cags-max-scale ${MAX_S} \
        > "logs/100class_${TAG}_gen.log" 2>&1

    # Evaluate
    CUDA_VISIBLE_DEVICES=1 python3 quick_eval_v2.py \
        --train-dir "${SAVE_DIR}/dataset_0" \
        --val-dir "${DATA_100}/val" \
        --nclass 100 \
        --arch convnet \
        --depth 6 \
        --epochs 1000 \
        --seeds 0 1 2 \
        > "logs/100class_${TAG}_eval.log" 2>&1

    echo "    ${TAG} done — $(date)"
    grep "Mean Top-1" "logs/100class_${TAG}_eval.log" 2>/dev/null
}

# ================================================================
# GPU 0: 10-class CAGS v2 experiments
# ================================================================
{
    echo "========================================"
    echo "GPU 0: 10-class CAGS v2 — $(date)"
    echo "========================================"

    # Config 1: Range [0.0, 0.06] — complex classes near optimal 0.05
    run_10class "cagsv2_0.0_0.06" 0.0 0.06

    # Config 2: Range [0.02, 0.06] — moderate range
    run_10class "cagsv2_0.02_0.06" 0.02 0.06

    # Config 3: Range [0.0, 0.08] — wider range
    run_10class "cagsv2_0.0_0.08" 0.0 0.08

    echo "GPU 0: All 10-class CAGS v2 complete! — $(date)"
} &

# ================================================================
# GPU 1: 100-class cache regen + CAGS v2 experiments
# ================================================================
{
    echo "========================================"
    echo "GPU 1: Regenerating 100-class cache — $(date)"
    echo "========================================"

    # Regenerate 100-class cache with v2 params
    CUDA_VISIBLE_DEVICES=1 python3 regen_cache.py \
        --spec imagenet100 --nclass 100 \
        --save-base "$BASE_100" \
        --sigmoid-slope 5.0 --sigmoid-center 0.5 \
        --complexity-k 10 \
        --alpha 0.15 --beta 0.15 --gamma 0.35 --delta 0.35 \
        > /tmp/regen_100class.log 2>&1

    echo "GPU 1: 100-class cache regenerated — $(date)"

    # Print stats
    python3 -c "
import pickle, numpy as np
with open('$BASE_100/cluster_cache.pkl', 'rb') as f:
    a = pickle.load(f)
vals = list(a.complexity_scores.values())
print(f'100-class complexity: [{min(vals):.4f}, {max(vals):.4f}], std={np.std(vals):.4f}')
modes = list(a.mode_counts.values())
print(f'Mode counts: unique K={sorted(set(modes))}')
vars = list(a.intra_variances.values())
print(f'Intra-var: [{min(vars):.4f}, {max(vars):.4f}], std={np.std(vars):.4f}')
seps = list(a.separabilities.values())
print(f'Separability: [{min(seps):.4f}, {max(seps):.4f}], std={np.std(seps):.4f}')
" 2>&1

    # Run CAGS v2 experiments
    run_100class "cagsv2_0.0_0.06" 0.0 0.06
    run_100class "cagsv2_0.02_0.06" 0.02 0.06

    echo "GPU 1: All 100-class CAGS v2 complete! — $(date)"
} &

# Wait for both GPUs
wait
echo "========================================"
echo "All CAGS v2 experiments complete! — $(date)"
echo "========================================"

# Print summary
echo ""
echo "=== 10-CLASS RESULTS ==="
for f in logs/10class_cagsv2_*_eval.log; do
    tag=$(basename $f | sed 's/10class_//;s/_eval.log//')
    echo -n "  $tag: "
    grep "Mean Top-1" "$f" 2>/dev/null || echo "no result"
done

echo ""
echo "=== 100-CLASS RESULTS ==="
for f in logs/100class_cagsv2_*_eval.log; do
    tag=$(basename $f | sed 's/100class_//;s/_eval.log//')
    echo -n "  $tag: "
    grep "Mean Top-1" "$f" 2>/dev/null || echo "no result"
done
