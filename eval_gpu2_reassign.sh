#!/bin/bash
set -e
cd /root/ICLR2027_KD

EVAL="python3 -u quick_eval_v3.py"
VAL100="/root/data/imagenet100/val"
CLASS100="./misc/class100.txt"
LOG="/root/ICLR2027_KD/logs/eval_gpu2_reassign.log"

echo "========================================" | tee "$LOG"
echo "GPU2 Reassign (GPU1 task: IN10 IPC=50 CAGS) - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# Wait for GPU2 to be free by checking nvidia-smi memory
echo "Waiting for GPU2 memory to be free..." | tee -a "$LOG"
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 2>/dev/null | tr -d ' ')
    if [ "$MEM" -lt 500 ] 2>/dev/null; then
        break
    fi
    sleep 30
done
echo "GPU2 free (mem=${MEM}MiB). Starting reassigned task - $(date)" | tee -a "$LOG"

echo "=== 10-class IPC=50 CAGS v2 (1500 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in10/high_noise_ipc50_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 10 --epochs 1500 --seeds 0 1 2 | tee -a "$LOG"

echo "========================================" | tee -a "$LOG"
echo "GPU2 REASSIGN DONE - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
