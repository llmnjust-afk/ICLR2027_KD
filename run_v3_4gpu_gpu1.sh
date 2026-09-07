#!/bin/bash
set -e
cd /root/ICLR2027_KD

EVAL="python3 -u quick_eval_v3.py"
VAL100="/root/data/imagenet100/val"
CLASS100="./misc/class100.txt"
LOG="/root/ICLR2027_KD/logs/rerun_v3_gpu1.log"

echo "========================================" | tee "$LOG"
echo "V3 Rerun GPU1 - 100-class IPC=1 + IPC=50 - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# === 100-class IPC=50 ===
echo "=== 100-class IPC=50 Random Herding (1000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_random_herding_ipc50_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 1000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 100-class IPC=50 unguided (1000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_ipc50_unguided_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 1000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 100-class IPC=50 CAGS v2 (1000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_ipc50_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 1000 --seeds 0 1 2 | tee -a "$LOG"

# === 100-class IPC=1 ===
echo "=== 100-class IPC=1 Random Herding (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_random_herding_ipc1_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 100-class IPC=1 unguided (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_ipc1_unguided_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 100-class IPC=1 CAGS v2 (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_ipc1_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 100-class IPC=1 IAST+CAGS v2 (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_iast_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "========================================" | tee -a "$LOG"
echo "GPU1 V3 RERUN DONE - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
