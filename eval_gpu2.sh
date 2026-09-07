#!/bin/bash
set -e
cd /root/ICLR2027_KD

EVAL="python3 -u quick_eval_v3.py"
VAL100="/root/data/imagenet100/val"
CLASS100="./misc/class100.txt"
LOG="/root/ICLR2027_KD/logs/rerun_v3_gpu2.log"

echo "========================================" | tee "$LOG"
echo "V3 Rerun GPU2 (100-class IPC=50 CAGS - waiting for 10-class eval to finish)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# Wait for 10-class + Nette eval to finish (running separately on GPU2)
echo "Waiting for IN10+Nette eval to finish..." | tee -a "$LOG"
while pgrep -f "eval_in10_nette\|sweep_in10.*quick_eval\|sweep_nette.*quick_eval" > /dev/null 2>&1; do
    sleep 30
done
echo "IN10+Nette eval done. Starting GPU2 eval - $(date)" | tee -a "$LOG"

echo "=== 100-class IPC=50 CAGS v2 (1000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_ipc50_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 1000 --seeds 0 1 2 | tee -a "$LOG"

echo "========================================" | tee -a "$LOG"
echo "GPU2 V3 RERUN DONE - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
