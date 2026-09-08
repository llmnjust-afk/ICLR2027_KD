#!/bin/bash
set -e
cd /root/ICLR2027_KD

EVAL="python3 -u quick_eval_v3.py"
VAL100="/root/data/imagenet100/val"
CLASS100="./misc/class100.txt"
LOG="/root/ICLR2027_KD/logs/rerun_v3_gpu2.log"

echo "========================================" | tee "$LOG"
echo "V3 Rerun GPU2 (IN100 IPC=1 only - post-gen) - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# Wait for GPU2 early eval to finish
echo "Waiting for GPU2 early eval to finish..." | tee -a "$LOG"
while pgrep -f "eval_gpu2_early" > /dev/null 2>&1; do
    sleep 30
done
echo "GPU2 early eval done. Starting GPU2 main eval - $(date)" | tee -a "$LOG"

echo "=== 100-class IPC=1 unguided (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_ipc1_unguided_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 100-class IPC=1 CAGS v2 (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_ipc1_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "========================================" | tee -a "$LOG"
echo "GPU2 V3 RERUN DONE - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
