#!/bin/bash
set -e
cd /root/ICLR2027_KD

EVAL="python3 -u quick_eval_v3.py"
VAL100="/root/data/imagenet100/val"
CLASS100="./misc/class100.txt"
LOG="/root/ICLR2027_KD/logs/eval_iast_redo.log"

echo "========================================" | tee "$LOG"
echo "IAST Redo (missed by GPU3 main) - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# Wait for GPU3 reassign to finish
echo "Waiting for GPU3 reassign to finish..." | tee -a "$LOG"
while pgrep -f "eval_gpu3_reassign" > /dev/null 2>&1; do
    sleep 30
done
echo "GPU3 reassign done. Starting IAST redo - $(date)" | tee -a "$LOG"

echo "=== 100-class IPC=1 IAST+CAGS v2 (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_iast_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "========================================" | tee -a "$LOG"
echo "IAST REDO DONE - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
