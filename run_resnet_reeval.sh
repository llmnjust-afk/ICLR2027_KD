#!/bin/bash
# ResNet re-evaluation with INSTANCE NORM (correct for small datasets)
# Run after Phase 3 main scripts complete

set -e
cd /root/ICLR2027_KD
LOG="logs/phase3_resnet_master.log"
echo "========================================" | tee "$LOG"
echo "ResNet Instance-Norm Re-evals — $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

DATA_10="/root/data/imagenet100_10class"
DATA_100="/root/data/imagenet100"
BASE_10="./results/sweep_in10"
BASE_100="./results/sweep_in100"
EPOCHS=1000
SEEDS="0 1 2"

reeval() {
    local BASE=$1; local TAG=$2; local ARCH=$3; local DEPTH=$4; local DATA=$5; local NCLASS=$6; local CLASS_FILE=$7
    local CONFIG="high_noise_${TAG}_d25"
    local SAVE="${BASE}/${CONFIG}"
    local EVAL_TAG="${TAG}_${ARCH}${DEPTH}_inst"
    echo "  [reeval $EVAL_TAG] — $(date)" | tee -a "$LOG"
    local EXTRA=""
    if [ -n "$CLASS_FILE" ]; then EXTRA="--class-file $CLASS_FILE"; fi
    CUDA_VISIBLE_DEVICES=0 python3 quick_eval_v2.py \
        --train-dir "${SAVE}/dataset_0" --val-dir "${DATA}/val" \
        --nclass $NCLASS --arch $ARCH --depth $DEPTH --norm-type instance \
        --epochs $EPOCHS --seeds $SEEDS $EXTRA \
        > "logs/phase3/${EVAL_TAG}_eval.log" 2>&1
    grep "Mean Top-1" "logs/phase3/${EVAL_TAG}_eval.log" | tee -a "$LOG"
}

mkdir -p logs/phase3

echo "=== ResNet-18 (instance norm) on 10-class ===" | tee -a "$LOG"
reeval "$BASE_10" "fixed_l0.0" resnet 18 "$DATA_10" 10 ""
reeval "$BASE_10" "fixed_l0.05" resnet 18 "$DATA_10" 10 ""
reeval "$BASE_10" "cagsv2_0.0_0.08" resnet 18 "$DATA_10" 10 ""

echo "=== ResNet-50 (instance norm) on 10-class ===" | tee -a "$LOG"
reeval "$BASE_10" "fixed_l0.0" resnet 50 "$DATA_10" 10 ""
reeval "$BASE_10" "fixed_l0.05" resnet 50 "$DATA_10" 10 ""
reeval "$BASE_10" "cagsv2_0.0_0.08" resnet 50 "$DATA_10" 10 ""

echo "=== ResNet-18 (instance norm) on 100-class ===" | tee -a "$LOG"
reeval "$BASE_100" "unguided" resnet 18 "$DATA_100" 100 ""
reeval "$BASE_100" "fixed_l0.1" resnet 18 "$DATA_100" 100 ""
reeval "$BASE_100" "cagsv2_0.0_0.06" resnet 18 "$DATA_100" 100 ""

echo "=== ResNet-50 (instance norm) on 100-class ===" | tee -a "$LOG"
reeval "$BASE_100" "unguided" resnet 50 "$DATA_100" 100 ""
reeval "$BASE_100" "cagsv2_0.0_0.06" resnet 50 "$DATA_100" 100 ""

echo "========================================" | tee -a "$LOG"
echo "ResNet Re-evals ALL DONE — $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
