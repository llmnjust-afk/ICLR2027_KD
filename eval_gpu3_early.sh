#!/bin/bash
set -e
cd /root/ICLR2027_KD

EVAL="python3 -u quick_eval_v3.py"
VAL100="/root/data/imagenet100/val"
CLASS100="./misc/class100.txt"
VAL_WOOF="/root/data/imagewoof2/val"
CLASS_WOOF="./misc/class_woof.txt"
LOG="/root/ICLR2027_KD/logs/eval_gpu3_early.log"

echo "========================================" | tee "$LOG"
echo "GPU3 Early Eval (ready configs) - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

echo "=== 100-class IPC=1 Random Herding (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_random_herding_ipc1_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 100-class IPC=10 Random Herding (1300 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_random_herding_ipc10_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 1300 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 100-class IPC=50 Random Herding (1000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_random_herding_ipc50_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 1000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== Woof Random Herding (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_woof/high_noise_random_herding_ipc10_d25/dataset_0 \
  --val-dir $VAL_WOOF --class-file $CLASS_WOOF --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== Woof unguided (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_woof/high_noise_unguided_d25/dataset_0 \
  --val-dir $VAL_WOOF --class-file $CLASS_WOOF --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== Woof CAGS v2 (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_woof/high_noise_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL_WOOF --class-file $CLASS_WOOF --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

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
  --val-dir /root/data/imagenette2/val --class-file ./misc/class_nette.txt --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "========================================" | tee -a "$LOG"
echo "GPU3 EARLY EVAL DONE - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
