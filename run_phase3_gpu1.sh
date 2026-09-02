#!/bin/bash
# Phase 3 GPU 1: Slow experiments (100-class TAGS + ResNet + IPC + IAST)
# All on CUDA_VISIBLE_DEVICES=1

set -e
cd /root/ICLR2027_KD
LOG="logs/phase3_gpu1_master.log"
echo "========================================" | tee "$LOG"
echo "Phase 3 GPU 1 — $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

DATA_100="/root/data/imagenet100"
BASE_100="./results/sweep_in100"
EPOCHS=1000
SEEDS="0 1 2"
mkdir -p logs/phase3

# Helper: generate + eval for 100-class
gen_eval_100() {
    local TAG=$1; shift
    local CONFIG="high_noise_${TAG}_d25"
    local SAVE="${BASE_100}/${CONFIG}"
    echo "  [100-class $TAG] gen — $(date)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=1 python3 duration_sweep.py \
        --window high_noise --imagenet-dir "$DATA_100" --save-base "$BASE_100" \
        --ipc $IPC --num-datasets 1 --epochs 1 --durations 25 \
        --tag "$TAG" --spec imagenet100 --nclass 100 --depth 6 \
        "$@" > "logs/100class_${TAG}_gen.log" 2>&1
    echo "  [100-class $TAG] eval — $(date)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=1 python3 quick_eval_v2.py \
        --train-dir "${SAVE}/dataset_0" --val-dir "${DATA_100}/val" \
        --nclass 100 --arch convnet --depth 6 --epochs $EPOCHS --seeds $SEEDS \
        > "logs/100class_${TAG}_eval.log" 2>&1
    grep "Mean Top-1" "logs/100class_${TAG}_eval.log" | tee -a "$LOG"
}

# Helper: re-eval existing 100-class images with different arch
reeval_100() {
    local TAG=$1; local ARCH=$2; local DEPTH=$3; local NORM=$4
    local CONFIG="high_noise_${TAG}_d25"
    local SAVE="${BASE_100}/${CONFIG}"
    local EVAL_TAG="${TAG}_${ARCH}${DEPTH}"
    echo "  [reeval 100-class $EVAL_TAG] — $(date)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=1 python3 quick_eval_v2.py \
        --train-dir "${SAVE}/dataset_0" --val-dir "${DATA_100}/val" \
        --nclass 100 --arch $ARCH --depth $DEPTH --norm-type $NORM \
        --epochs $EPOCHS --seeds $SEEDS \
        > "logs/100class_${EVAL_TAG}_eval.log" 2>&1
    grep "Mean Top-1" "logs/100class_${EVAL_TAG}_eval.log" | tee -a "$LOG"
}

# ========== BATCH 1: TAGS + CAGS on 100-class (full AGS system) ==========
echo "=== BATCH 1: TAGS + CAGS v2 on 100-class ===" | tee -a "$LOG"
IPC=10
gen_eval_100 "tags_linear_cagsv2_0.0_0.06" \
    --schedule linear --cags-min-scale 0.0 --cags-max-scale 0.06
gen_eval_100 "tags_exp_cagsv2_0.0_0.06" \
    --schedule exponential --cags-min-scale 0.0 --cags-max-scale 0.06

# ========== BATCH 2: ResNet-18 re-evals on 100-class ==========
echo "=== BATCH 2: ResNet-18 on 100-class ===" | tee -a "$LOG"
reeval_100 "unguided" resnet 18 batch
reeval_100 "fixed_l0.1" resnet 18 batch
reeval_100 "cagsv2_0.0_0.06" resnet 18 batch

# ========== BATCH 3: IPC=1 on 100-class ==========
echo "=== BATCH 3: IPC=1 on 100-class ===" | tee -a "$LOG"
IPC=1
gen_eval_100 "ipc1_unguided" --no-cags --fixed-scale 0.0
gen_eval_100 "ipc1_cagsv2_0.0_0.06" --cags-min-scale 0.0 --cags-max-scale 0.06 --complexity-k 1
IPC=10

# ========== BATCH 4: IAST + CAGS on 100-class ==========
echo "=== BATCH 4: IAST + CAGS v2 on 100-class ===" | tee -a "$LOG"
IPC=10
gen_eval_100 "iast_cagsv2_0.0_0.06" \
    --cags-min-scale 0.0 --cags-max-scale 0.06 --use-iast

# ========== BATCH 5: ResNet-50 re-evals on 100-class ==========
echo "=== BATCH 5: ResNet-50 on 100-class ===" | tee -a "$LOG"
reeval_100 "unguided" resnet 50 batch
reeval_100 "cagsv2_0.0_0.06" resnet 50 batch

# ========== BATCH 6: IPC=50 on 100-class ==========
echo "=== BATCH 6: IPC=50 on 100-class ===" | tee -a "$LOG"
IPC=50
gen_eval_100 "ipc50_unguided" --no-cags --fixed-scale 0.0
gen_eval_100 "ipc50_cagsv2_0.0_0.06" --cags-min-scale 0.0 --cags-max-scale 0.06 --complexity-k 50
IPC=10

echo "========================================" | tee -a "$LOG"
echo "Phase 3 GPU 1 ALL DONE — $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
