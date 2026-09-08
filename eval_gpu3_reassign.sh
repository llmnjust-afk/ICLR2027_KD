#!/bin/bash
set -e
cd /root/ICLR2027_KD

EVAL="python3 -u quick_eval_v3.py"
VAL100="/root/data/imagenet100/val"
CLASS100="./misc/class100.txt"
LOG="/root/ICLR2027_KD/logs/eval_gpu3_reassign.log"

echo "========================================" | tee "$LOG"
echo "GPU3 Reassign (GPU1 tasks) - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# Wait for GPU3 to be free by checking nvidia-smi memory
echo "Waiting for GPU3 memory to be free..." | tee -a "$LOG"
while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 2>/dev/null | tr -d ' ')
    if [ "$MEM" -lt 500 ] 2>/dev/null; then
        break
    fi
    sleep 30
done
echo "GPU3 free (mem=${MEM}MiB). Starting reassigned tasks - $(date)" | tee -a "$LOG"

echo "=== 100-class IPC=10 TAGS linear + CAGS (1300 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_tags_linear_cagsv2_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 1300 --seeds 0 1 2 | tee -a "$LOG"

echo "=== 100-class ResNetAP-10 unguided (1300 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_unguided_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 1300 --seeds 0 1 2 \
  --arch resnet_ap --depth 10 | tee -a "$LOG"

echo "=== 100-class ResNetAP-10 CAGS v2 (1300 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 1300 --seeds 0 1 2 \
  --arch resnet_ap --depth 10 | tee -a "$LOG"

echo "========================================" | tee -a "$LOG"
echo "GPU3 REASSIGN DONE - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
