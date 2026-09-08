#!/bin/bash
set -e
cd /root/ICLR2027_KD

EVAL="python3 -u quick_eval_v3.py"
VAL100="/root/data/imagenet100/val"
CLASS100="./misc/class100.txt"
LOG="/root/ICLR2027_KD/logs/rerun_v3_gpu0.log"

echo "========================================" | tee "$LOG"
echo "V3 Rerun GPU0 - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

echo "=== 100-class IPC=50 unguided (1000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_ipc50_unguided_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 1000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 100-class IPC=10 CAGS v2 (1300 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 1300 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 100-class ResNet-18 unguided (1300 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_unguided_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 1300 --seeds 0 1 2 \
  --arch resnet --depth 18 | tee -a "$LOG"

echo "=== 100-class ResNet-18 CAGS v2 (1300 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 1300 --seeds 0 1 2 \
  --arch resnet --depth 18 | tee -a "$LOG"

echo "=== 10-class IPC=50 unguided (1500 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_ipc50_unguided_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 1500 --seeds 0 1 2 | tee -a "$LOG"

echo "========================================" | tee -a "$LOG"
echo "GPU0 V3 RERUN DONE - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
