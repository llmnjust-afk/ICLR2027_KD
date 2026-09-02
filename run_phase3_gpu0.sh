#!/bin/bash
# Phase 3 GPU 0: Fast experiments (10-class + ImageNette/Woof + re-evals)
# All on CUDA_VISIBLE_DEVICES=0

set -e
cd /root/ICLR2027_KD
LOG="logs/phase3_gpu0_master.log"
echo "========================================" | tee "$LOG"
echo "Phase 3 GPU 0 — $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

DATA_10="/root/data/imagenet100_10class"
DATA_NETTE="/root/data/imagenette2"
DATA_WOOF="/root/data/imagewoof2"
BASE_10="./results/sweep_in10"
BASE_NETTE="./results/sweep_nette"
BASE_WOOF="./results/sweep_woof"
EPOCHS=1000
SEEDS="0 1 2"

# Helper: generate + eval for 10-class
gen_eval_10() {
    local TAG=$1; shift
    local CONFIG="high_noise_${TAG}_d25"
    local SAVE="${BASE_10}/${CONFIG}"
    echo "  [$TAG] gen — $(date)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 python3 duration_sweep.py \
        --window high_noise --imagenet-dir "$DATA_10" --save-base "$BASE_10" \
        --ipc $IPC --num-datasets 1 --epochs 1 --durations 25 \
        --tag "$TAG" --spec imagenet100 --nclass 10 --depth 6 \
        "$@" > "logs/10class_${TAG}_gen.log" 2>&1
    echo "  [$TAG] eval — $(date)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 python3 quick_eval_v2.py \
        --train-dir "${SAVE}/dataset_0" --val-dir "${DATA_10}/val" \
        --nclass 10 --arch convnet --depth 6 --epochs $EPOCHS --seeds $SEEDS \
        > "logs/10class_${TAG}_eval.log" 2>&1
    grep "Mean Top-1" "logs/10class_${TAG}_eval.log" | tee -a "$LOG"
}

# Helper: re-eval existing 10-class images with different arch
reeval_10() {
    local TAG=$1; local ARCH=$2; local DEPTH=$3; local NORM=$4
    local CONFIG="high_noise_${TAG}_d25"
    local SAVE="${BASE_10}/${CONFIG}"
    local EVAL_TAG="${TAG}_${ARCH}${DEPTH}"
    echo "  [reeval $EVAL_TAG] — $(date)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 python3 quick_eval_v2.py \
        --train-dir "${SAVE}/dataset_0" --val-dir "${DATA_10}/val" \
        --nclass 10 --arch $ARCH --depth $DEPTH --norm-type $NORM \
        --epochs $EPOCHS --seeds $SEEDS \
        > "logs/10class_${EVAL_TAG}_eval.log" 2>&1
    grep "Mean Top-1" "logs/10class_${EVAL_TAG}_eval.log" | tee -a "$LOG"
}

# Helper: gen + eval for ImageNette
gen_eval_nette() {
    local TAG=$1; shift
    local CONFIG="high_noise_${TAG}_d25"
    local SAVE="${BASE_NETTE}/${CONFIG}"
    echo "  [nette $TAG] gen — $(date)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 python3 duration_sweep.py \
        --window high_noise --imagenet-dir "$DATA_NETTE" --save-base "$BASE_NETTE" \
        --ipc 10 --num-datasets 1 --epochs 1 --durations 25 \
        --tag "$TAG" --spec nette --nclass 10 --depth 6 \
        --regen-cache --complexity-k 10 \
        "$@" > "logs/phase3/nette_${TAG}_gen.log" 2>&1
    echo "  [nette $TAG] eval — $(date)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 python3 quick_eval_v2.py \
        --train-dir "${SAVE}/dataset_0" --val-dir "${DATA_NETTE}/val" \
        --class-file ./misc/class_nette.txt --nclass 10 \
        --arch convnet --depth 6 --epochs $EPOCHS --seeds $SEEDS \
        > "logs/phase3/nette_${TAG}_eval.log" 2>&1
    grep "Mean Top-1" "logs/phase3/nette_${TAG}_eval.log" | tee -a "$LOG"
}

# Helper: gen + eval for ImageWoof
gen_eval_woof() {
    local TAG=$1; shift
    local CONFIG="high_noise_${TAG}_d25"
    local SAVE="${BASE_WOOF}/${CONFIG}"
    echo "  [woof $TAG] gen — $(date)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 python3 duration_sweep.py \
        --window high_noise --imagenet-dir "$DATA_WOOF" --save-base "$BASE_WOOF" \
        --ipc 10 --num-datasets 1 --epochs 1 --durations 25 \
        --tag "$TAG" --spec woof --nclass 10 --depth 6 \
        --regen-cache --complexity-k 10 \
        "$@" > "logs/phase3/woof_${TAG}_gen.log" 2>&1
    echo "  [woof $TAG] eval — $(date)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 python3 quick_eval_v2.py \
        --train-dir "${SAVE}/dataset_0" --val-dir "${DATA_WOOF}/val" \
        --class-file ./misc/class_woof.txt --nclass 10 \
        --arch convnet --depth 6 --epochs $EPOCHS --seeds $SEEDS \
        > "logs/phase3/woof_${TAG}_eval.log" 2>&1
    grep "Mean Top-1" "logs/phase3/woof_${TAG}_eval.log" | tee -a "$LOG"
}

mkdir -p logs/phase3

# ========== BATCH 1: CAGS on ImageNette/ImageWoof ==========
echo "=== BATCH 1: CAGS on ImageNette/ImageWoof ===" | tee -a "$LOG"
gen_eval_nette "cagsv2_0.0_0.06" --cags-min-scale 0.0 --cags-max-scale 0.06
gen_eval_woof "cagsv2_0.0_0.06" --cags-min-scale 0.0 --cags-max-scale 0.06

# ========== BATCH 2: ResNet-18 re-evals on 10-class ==========
echo "=== BATCH 2: ResNet-18 on 10-class ===" | tee -a "$LOG"
reeval_10 "fixed_l0.0" resnet 18 batch
reeval_10 "fixed_l0.05" resnet 18 batch
reeval_10 "cagsv2_0.0_0.08" resnet 18 batch

# ========== BATCH 3: ResNet-50 re-evals on 10-class ==========
echo "=== BATCH 3: ResNet-50 on 10-class ===" | tee -a "$LOG"
reeval_10 "fixed_l0.0" resnet 50 batch
reeval_10 "fixed_l0.05" resnet 50 batch
reeval_10 "cagsv2_0.0_0.08" resnet 50 batch

# ========== BATCH 4: IPC=1 on 10-class ==========
echo "=== BATCH 4: IPC=1 on 10-class ===" | tee -a "$LOG"
IPC=1
gen_eval_10 "ipc1_unguided" --no-cags --fixed-scale 0.0
gen_eval_10 "ipc1_fixed_l0.05" --no-cags --fixed-scale 0.05
gen_eval_10 "ipc1_cagsv2_0.0_0.06" --cags-min-scale 0.0 --cags-max-scale 0.06 --complexity-k 1
IPC=10  # reset

# ========== BATCH 5: TAGS low-lambda on 10-class ==========
echo "=== BATCH 5: TAGS lambda=0.05 on 10-class ===" | tee -a "$LOG"
gen_eval_10 "tags_linear_l0.05" --no-cags --fixed-scale 0.05 --schedule linear
gen_eval_10 "tags_exp_l0.05" --no-cags --fixed-scale 0.05 --schedule exponential
gen_eval_10 "tags_cosine_l0.05" --no-cags --fixed-scale 0.05 --schedule cosine

# ========== BATCH 6: IAST on 10-class ==========
echo "=== BATCH 6: IAST + CAGS on 10-class ===" | tee -a "$LOG"
gen_eval_10 "iast_cagsv2_0.0_0.08" --cags-min-scale 0.0 --cags-max-scale 0.08 --use-iast

# ========== BATCH 7: IPC=50 on 10-class ==========
echo "=== BATCH 7: IPC=50 on 10-class ===" | tee -a "$LOG"
IPC=50
gen_eval_10 "ipc50_unguided" --no-cags --fixed-scale 0.0
gen_eval_10 "ipc50_fixed_l0.05" --no-cags --fixed-scale 0.05
gen_eval_10 "ipc50_cagsv2_0.0_0.06" --cags-min-scale 0.0 --cags-max-scale 0.06 --complexity-k 50
IPC=10  # reset

echo "========================================" | tee -a "$LOG"
echo "Phase 3 GPU 0 ALL DONE — $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
