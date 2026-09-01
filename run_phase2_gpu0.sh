#!/bin/bash
# Phase 2: ImageNette + ImageWoof on GPU 0
# Baseline (unguided) + best fixed lambda (0.05)
set -uo pipefail
cd /root/ICLR2027_KD

SEEDS="0 1 2"
DEPTH=6
WINDOW="high_noise"
STOP_T=25
NETTE_DIR="/root/data/imagenette2"
WOOF_DIR="/root/data/imagewoof2"
NETTE_VAL="/root/data/imagenette2/val"
WOOF_VAL="/root/data/imagewoof2/val"
GPU=0

mkdir -p logs/phase2 results/sweep_nette results/sweep_woof

count_pngs() { find "$1" -name "*.png" 2>/dev/null | wc -l; }

gen_and_eval() {
    local SPEC=$1 NCLASS=$2 IPC=$3 TAG=$4 BASE=$5 DATA_DIR=$6 VAL_DIR=$7 LAMBDA=$8

    local DIR="$BASE/high_noise_${TAG}_d${STOP_T}/dataset_0"
    local NEEDED=$((NCLASS * IPC))
    local PREFIX="${SPEC}_${NCLASS}cls_ipc${IPC}_${TAG}"

    echo ""
    echo "================================================"
    echo "  ${PREFIX} lambda=${LAMBDA} (GPU ${GPU})"
    echo "  $(date)"
    echo "================================================"

    # Generate if needed
    if [ ! -d "$DIR" ] || [ "$(count_pngs "$DIR")" -lt "$NEEDED" ]; then
        echo "  Generating ${NEEDED} images..."
        if [ "$LAMBDA" = "unguided" ]; then
            CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python3 duration_sweep.py \
                --window $WINDOW --durations $STOP_T --schedule constant \
                --spec $SPEC --nclass $NCLASS --ipc $IPC \
                --imagenet-dir $DATA_DIR --save-base $BASE \
                --epochs 1 --depth $DEPTH \
                --fixed-stop-t $STOP_T \
                --seeds 0 --num-datasets 1 --tag $TAG \
                2>&1 | tee logs/phase2/${PREFIX}_gen.log
        else
            CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python3 duration_sweep.py \
                --window $WINDOW --durations $STOP_T --schedule constant \
                --spec $SPEC --nclass $NCLASS --ipc $IPC \
                --imagenet-dir $DATA_DIR --save-base $BASE \
                --epochs 1 --depth $DEPTH \
                --fixed-stop-t $STOP_T \
                --seeds 0 --num-datasets 1 --tag $TAG \
                --fixed-lambda $LAMBDA \
                2>&1 | tee logs/phase2/${PREFIX}_gen.log
        fi
    else
        echo "  Images exist ($(count_pngs "$DIR")/${NEEDED}), skipping gen"
    fi

    # Evaluate
    echo "  Evaluating (3 seeds, 1000 epochs, ConvNet-6)..."
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python3 quick_eval_v2.py \
        --train-dir "$DIR" --val-dir "$VAL_DIR" \
        --nclass $NCLASS --arch convnet --depth $DEPTH \
        --epochs 1000 --seeds $SEEDS \
        2>&1 | tee logs/phase2/${PREFIX}_eval.log

    echo "  ${PREFIX} done — $(date)"
}

echo "========================================"
echo "  Phase 2: ImageNette + ImageWoof on GPU 0"
echo "  Started: $(date)"
echo "========================================"

# ImageNette (9 classes, IPC=10)
gen_and_eval imagenette 9 10 unguided results/sweep_nette $NETTE_DIR $NETTE_VAL unguided
gen_and_eval imagenette 9 10 lambda005 results/sweep_nette $NETTE_DIR $NETTE_VAL 0.05

# ImageWoof (9 classes, IPC=10)
gen_and_eval imagewoof 9 10 unguided results/sweep_woof $WOOF_DIR $WOOF_VAL unguided
gen_and_eval imagewoof 9 10 lambda005 results/sweep_woof $WOOF_DIR $WOOF_VAL 0.05

echo ""
echo "========================================"
echo "  Phase 2 (GPU 0) ALL DONE — $(date)"
echo "========================================"
