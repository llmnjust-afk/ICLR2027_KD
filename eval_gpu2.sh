#!/bin/bash
set -e
cd /root/ICLR2027_KD

EVAL="python3 -u quick_eval_v3.py"
VAL100="/root/data/imagenet100/val"
VAL_NETTE="/root/data/imagenette2/val"
CLASS100="./misc/class100.txt"
LOG="/root/ICLR2027_KD/logs/rerun_v3_gpu2.log"

echo "========================================" | tee "$LOG"
echo "V3 Rerun GPU2 (balanced) - IPC50-CAGS + 10-class + Nette - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

echo "=== 100-class IPC=50 CAGS v2 (1000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_ipc50_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 1000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 10-class IPC=1 Random Herding (3000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_random_herding_ipc1_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 3000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 10-class IPC=1 unguided (3000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_ipc1_unguided_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 3000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 10-class IPC=1 CAGS v2 (3000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_ipc1_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 3000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 10-class IPC=1 IAST+CAGS v2 (3000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_iast_cagsv2_0.0_0.08_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 3000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 10-class IPC=10 Random Herding (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_random_herding_ipc10_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 10-class IPC=10 unguided/fixed λ=0.0 (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_fixed_l0.0_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 10-class IPC=10 fixed λ=0.05 (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_fixed_l0.05_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 10-class IPC=10 CAGS v2 (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_cagsv2_0.0_0.08_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 10-class IPC=50 Random Herding (1500 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_random_herding_ipc50_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 1500 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 10-class IPC=50 unguided (1500 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_ipc50_unguided_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 1500 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 10-class IPC=50 CAGS v2 (1500 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_ipc50_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 1500 --seeds 0 1 2 | tee -a "$LOG"

echo "=== ImageNette Random Herding (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_nette/high_noise_random_herding_ipc10_d25/dataset_0 \
  --val-dir $VAL_NETTE --class-file ./misc/class_nette.txt --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== ImageNette unguided (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_nette/high_noise_unguided_d25/dataset_0 \
  --val-dir $VAL_NETTE --class-file ./misc/class_nette.txt --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== ImageNette CAGS v2 (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_nette/high_noise_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL_NETTE --class-file ./misc/class_nette.txt --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "========================================" | tee -a "$LOG"
echo "GPU2 V3 RERUN DONE - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
