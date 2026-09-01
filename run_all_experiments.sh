#!/bin/bash
# AGS-DD Comprehensive Experiment Runner (v3)
# Uses 10-class symlink dir for generation (avoids empty dir errors),
# full val dir for evaluation.
#
# Usage: bash run_all_experiments.sh

set -uo pipefail
cd /root/ICLR2027_KD

SEEDS="0 1 2"
EPOCHS=1000
DEPTH=6
IPC=10
WINDOW="high_noise"
STOP_T=25
BASE_10="./results/sweep_in10"
BASE_100="./results/sweep_in100"
VAL_DIR="/root/data/imagenet100/val"
DATA_10="/root/data/imagenet100_10class"
DATA_100="/root/data/imagenet100"

mkdir -p logs

count_pngs() { find "$1" -name "*.png" 2>/dev/null | wc -l; }

run_experiment() {
    local GPU=$1 NCLASS=$2 TAG=$3 BASE=$4 EXTRA_ARGS=$5
    local DATA_DIR
    if [ $NCLASS -eq 10 ]; then
        DATA_DIR=$DATA_10
    else
        DATA_DIR=$DATA_100
    fi
    local DIR="$BASE/high_noise_${TAG}_d25/dataset_0"
    local NEEDED=$((NCLASS * IPC))
    local PREFIX=$([ $NCLASS -eq 10 ] && echo "10class" || echo "100class")

    echo ""
    echo "================================================"
    echo "  ${PREFIX}: ${TAG} (GPU ${GPU}) — $(date)"
    echo "================================================"

    if [ ! -d "$DIR" ] || [ "$(count_pngs "$DIR")" -lt "$NEEDED" ]; then
        echo "  Generating ${NEEDED} images..."
        CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python3 duration_sweep.py \
            --window $WINDOW --durations $STOP_T --schedule constant \
            --spec imagenet100 --nclass $NCLASS --ipc $IPC \
            --imagenet-dir $DATA_DIR --save-base $BASE \
            --epochs 1 --depth $DEPTH \
            --fixed-stop-t $STOP_T \
            --seeds 0 --num-datasets 1 --tag $TAG $EXTRA_ARGS \
            2>&1 | tee logs/${PREFIX}_${TAG}_gen.log
    else
        echo "  Images already generated ($(count_pngs "$DIR")/${NEEDED}), skipping"
    fi

    echo "  Evaluating (${EPOCHS} epochs, seeds ${SEEDS})..."
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python3 quick_eval.py \
        --train-dir "$DIR" \
        --val-dir $VAL_DIR \
        --class-file ./misc/class100.txt \
        --nclass $NCLASS --epochs $EPOCHS --depth $DEPTH --seeds $SEEDS \
        2>&1 | tee logs/${PREFIX}_${TAG}_eval.log

    echo "  Done: ${TAG} — $(date)"
}

# ================================================================
# GPU 0: 10-class experiments
# ================================================================
run_gpu0() {
    echo "========================================"
    echo "GPU 0: 10-class experiments — started $(date)"
    echo "========================================"

    # Phase 1: Fixed lambda search
    for LAMBDA in 0.0 0.05 0.08 0.1 0.12 0.15; do
        run_experiment 0 10 "fixed_l${LAMBDA}" "$BASE_10" "--no-cags --fixed-scale ${LAMBDA}"
    done

    # Phase 2: CAGS tuning
    run_experiment 0 10 "cags_0.05_0.12" "$BASE_10" "--cags-min-scale 0.05 --cags-max-scale 0.12"
    run_experiment 0 10 "cags_0.05_0.15" "$BASE_10" "--cags-min-scale 0.05 --cags-max-scale 0.15"
    run_experiment 0 10 "cags_0.08_0.12" "$BASE_10" "--cags-min-scale 0.08 --cags-max-scale 0.12"

    # Phase 4: TAGS ablation (using fixed lambda=0.1)
    run_experiment 0 10 "tags_cosine_l0.1" "$BASE_10" "--no-cags --fixed-scale 0.1 --schedule cosine"
    run_experiment 0 10 "tags_linear_l0.1" "$BASE_10" "--no-cags --fixed-scale 0.1 --schedule linear"
    run_experiment 0 10 "tags_exp_l0.1" "$BASE_10" "--no-cags --fixed-scale 0.1 --schedule exponential"

    echo "GPU 0: All 10-class experiments complete! — $(date)"
}

# ================================================================
# GPU 1: 100-class experiments (waits for data)
# ================================================================
run_gpu1() {
    echo "========================================"
    echo "GPU 1: Waiting for 100-class data download..."
    echo "========================================"

    while true; do
        CLASSES_WITH_DATA=$(find $DATA_100/train/ -name "*.JPEG" 2>/dev/null | sed "s|$DATA_100/train/||;s|/.*||" | sort -u | wc -l)
        TOTAL_IMAGES=$(find $DATA_100/train/ -name "*.JPEG" 2>/dev/null | wc -l)
        echo "  Progress: ${CLASSES_WITH_DATA}/100 classes, ${TOTAL_IMAGES} images — $(date)"

        if [ "$CLASSES_WITH_DATA" -ge 100 ]; then
            echo "  All 100 classes have data! Starting 100-class experiments."
            break
        fi
        sleep 60
    done

    echo "========================================"
    echo "GPU 1: 100-class experiments — started $(date)"
    echo "========================================"

    run_experiment 1 100 "unguided" "$BASE_100" "--no-cags --fixed-scale 0.0"
    run_experiment 1 100 "fixed_l0.1" "$BASE_100" "--no-cags --fixed-scale 0.1"
    run_experiment 1 100 "cags_0.05_0.15" "$BASE_100" "--cags-min-scale 0.05 --cags-max-scale 0.15"
    run_experiment 1 100 "cags_0.05_0.12" "$BASE_100" "--cags-min-scale 0.05 --cags-max-scale 0.12"

    echo "GPU 1: All 100-class experiments complete! — $(date)"
}

# ================================================================
# Main
# ================================================================
echo "========================================"
echo "AGS-DD Comprehensive Experiments"
echo "Started: $(date)"
echo "========================================"

# Start GPU 1 first (waits for data, no GPU contention)
run_gpu1 &
GPU1_PID=$!
sleep 5

# Start GPU 0 (10-class, data is ready)
run_gpu0 &
GPU0_PID=$!

echo "GPU 0 PID: $GPU0_PID (10-class)"
echo "GPU 1 PID: $GPU1_PID (100-class, waiting for data)"

# Wait for both
wait $GPU0_PID
GPU0_EXIT=$?
echo "GPU 0 finished (exit $GPU0_EXIT) — $(date)"

wait $GPU1_PID
GPU1_EXIT=$?
echo "GPU 1 finished (exit $GPU1_EXIT) — $(date)"

echo ""
echo "========================================"
echo "All experiments complete! — $(date)"
echo "========================================"

# Collect results
echo "Collecting results..."
python3 collect_results.py 2>&1 | tee logs/collect_results.log

# Push to GitHub
echo "Pushing results to GitHub..."
bash auto_push.sh 2>&1 | tee logs/auto_push.log

echo "All done! — $(date)"
