#!/bin/bash
set -e
cd /root/ICLR2027_KD

EVAL="python3 -u quick_eval_v3.py"
VAL100="/root/data/imagenet100/val"
VAL_NETTE="/root/data/imagenette2/val"
VAL_WOOF="/root/data/imagewoof2/val"
CLASS100="./misc/class100.txt"
LOG="/root/ICLR2027_KD/logs/rerun_v3_gpu3.log"

echo "========================================" | tee "$LOG"
echo "V3 Rerun GPU3 - Cross-arch + Nette/Woof - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# === Cross-architecture (100-class IPC=10) ===
echo "=== 100-class ResNet-18 unguided (1300 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_unguided_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 1300 --seeds 0 1 2 \
  --arch resnet --depth 18 | tee -a "$LOG"

echo "=== 100-class ResNet-18 CAGS v2 (1300 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 1300 --seeds 0 1 2 \
  --arch resnet --depth 18 | tee -a "$LOG"

echo "=== 100-class ResNetAP-10 unguided (1300 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_unguided_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 1300 --seeds 0 1 2 \
  --arch resnet_ap --depth 10 | tee -a "$LOG"

echo "=== 100-class ResNetAP-10 CAGS v2 (1300 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_in100/high_noise_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL100 --class-file $CLASS100 --nclass 100 --epochs 1300 --seeds 0 1 2 \
  --arch resnet_ap --depth 10 | tee -a "$LOG"

# === ImageNette (IPC=10) ===
echo "=== ImageNette Random Herding (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_nette/high_noise_random_herding_ipc10_d25/dataset_0 \
  --val-dir $VAL_NETTE --class-file ./misc/class_nette.txt --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== ImageNette unguided (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_nette/high_noise_unguided_d25/dataset_0 \
  --val-dir $VAL_NETTE --class-file ./misc/class_nette.txt --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== ImageNette CAGS v2 (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_nette/high_noise_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL_NETTE --class-file ./misc/class_nette.txt --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

# === ImageWoof (IPC=10) ===
echo "=== ImageWoof Random Herding (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_woof/high_noise_random_herding_ipc10_d25/dataset_0 \
  --val-dir $VAL_WOOF --class-file ./misc/class_woof.txt --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== ImageWoof unguided (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_woof/high_noise_unguided_d25/dataset_0 \
  --val-dir $VAL_WOOF --class-file ./misc/class_woof.txt --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "=== ImageWoof CAGS v2 (2000 ep) ===" | tee -a "$LOG"
$EVAL --train-dir ./results/sweep_woof/high_noise_cagsv2_0.0_0.06_d25/dataset_0 \
  --val-dir $VAL_WOOF --class-file ./misc/class_woof.txt --nclass 10 --epochs 2000 --seeds 0 1 2 | tee -a "$LOG"

echo "========================================" | tee -a "$LOG"
echo "GPU3 V3 RERUN DONE - $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
