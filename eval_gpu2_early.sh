#!/bin/bash
set -e
cd /root/ICLR2027_KD

EVAL="python3 -u quick_eval_v3.py"
VAL100="/root/data/imagenet100/val"
CLASS100="./misc/class100.txt"
VAL_NETTE="/root/data/imagenette2/val"
CLASS_NETTE="./misc/class_nette.txt"
LOG="/root/ICLR2027_KD/logs/eval_gpu2_early.log"

echo "========================================" | tee "$LOG"
echo "GPU2 Early Eval (IN10+Nette ready) - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

echo "=== 10-class IPC=10 unguided (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_fixed_l0.0_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 10-class IPC=10 fixed lambda=0.05 (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_fixed_l0.05_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 10-class IPC=10 CAGS v2 (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_cagsv2_0.0_0.08_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== Nette unguided (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_nette/high_noise_unguided_d25/dataset_0 \
  --val-dir $VAL_NETTE --class-file $CLASS_NETTE --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== Nette CAGS v2 (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_nette/high_noise_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL_NETTE --class-file $CLASS_NETTE --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 10-class IPC=1 unguided (3000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_ipc1_unguided_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 3000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 10-class IPC=1 CAGS v2 (3000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_ipc1_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 3000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 10-class IPC=1 IAST+CAGS v2 (3000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_iast_cagsv2_0.0_0.08_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 3000 --seeds 0 1 2 | tee -a "$LOG"

echo "========================================" | tee -a "$LOG"
echo "GPU2 EARLY EVAL DONE - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
