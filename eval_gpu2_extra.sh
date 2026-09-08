#!/bin/bash
set -e
cd /root/ICLR2027_KD

EVAL="python3 -u quick_eval_v3.py"
VAL100="/root/data/imagenet100/val"
CLASS100="./misc/class100.txt"
VAL_NETTE="/root/data/imagenette2/val"
CLASS_NETTE="./misc/class_nette.txt"
VAL_WOOF="/root/data/imagewoof2/val"
CLASS_WOOF="./misc/class_woof.txt"
LOG="/root/ICLR2027_KD/logs/eval_gpu2_extra.log"

echo "========================================" | tee "$LOG"
echo "GPU2 Extra Eval - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

echo "=== 10-class IPC=50 unguided (1500 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_ipc50_unguided_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 1500 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 10-class IPC=50 CAGS v2 (1500 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_ipc50_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 1500 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 10-class IPC=1 Random Herding (3000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_random_herding_ipc1_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 3000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 10-class IPC=10 Random Herding (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_random_herding_ipc10_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 10-class IPC=50 Random Herding (1500 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_random_herding_ipc50_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 1500 --seeds 0 1 2 | tee -a "$LOG"

echo "=== Nette Random Herding (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_nette/high_noise_random_herding_ipc10_d25/dataset_0 \
  --val-dir $VAL_NETTE --class-file $CLASS_NETTE --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== Woof CAGS v2 (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_woof/high_noise_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL_WOOF --class-file $CLASS_WOOF --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "========================================" | tee -a "$LOG"
echo "GPU2 EXTRA EVAL DONE - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
