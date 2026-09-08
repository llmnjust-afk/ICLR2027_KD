#!/bin/bash
set -e
cd /root/ICLR2027_KD

EVAL="python3 -u quick_eval_v3.py"
VAL100="/root/data/imagenet100/val"
CLASS100="./misc/class100.txt"
LOG="/root/ICLR2027_KD/logs/eval_gpu3_resnetap.log"

echo "========================================" | tee "$LOG"
echo "GPU3 ResNetAP-10 + IAST - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

echo "=== 100-class ResNetAP-10 unguided (1300 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_unguided_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 1300 --seeds 0 1 2 \
  --arch resnet_ap --depth 10 | tee -a "$LOG"

echo "=== 100-class ResNetAP-10 CAGS v2 (1300 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 1300 --seeds 0 1 2 \
  --arch resnet_ap --depth 10 | tee -a "$LOG"

echo "=== 100-class IPC=1 IAST+CAGS v2 (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_iast_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "========================================" | tee -a "$LOG"
echo "GPU3 RESNETAP+IAST DONE - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
