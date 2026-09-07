#!/bin/bash
# Master launch script: cluster computation + generation + evaluation
# Run after ImageNet-100 download is complete
set -e
cd /root/ICLR2027_KD

LOGDIR=/root/ICLR2027_KD/logs
mkdir -p "$LOGDIR"

echo "============================================" | tee "$LOGDIR/master.log"
echo "AGS-DD Full Pipeline - $(date)" | tee -a "$LOGDIR/master.log"
echo "============================================" | tee -a "$LOGDIR/master.log"

# === Phase 0: Random Herding (no GPU needed) ===
echo "" | tee -a "$LOGDIR/master.log"
echo "=== Phase 0: Random Herding datasets ===" | tee -a "$LOGDIR/master.log"
RH="python3 run_random_herding.py"

# ImageNet-100 Random Herding
$RH --src-dir /root/data/imagenet100/train --dst-dir ./results/sweep_in100/high_noise_random_herding_ipc10_d25 --class-file ./misc/class100.txt --nclass 100 --ipc 10 2>&1 | tee -a "$LOGDIR/master.log"
$RH --src-dir /root/data/imagenet100/train --dst-dir ./results/sweep_in100/high_noise_random_herding_ipc1_d25 --class-file ./misc/class100.txt --nclass 100 --ipc 1 2>&1 | tee -a "$LOGDIR/master.log"
$RH --src-dir /root/data/imagenet100/train --dst-dir ./results/sweep_in100/high_noise_random_herding_ipc50_d25 --class-file ./misc/class100.txt --nclass 100 --ipc 50 2>&1 | tee -a "$LOGDIR/master.log"

# ImageNet-10 Random Herding (first 10 classes of IN100)
$RH --src-dir /root/data/imagenet100/train --dst-dir ./results/sweep_in10/high_noise_random_herding_ipc1_d25 --class-file ./misc/class100.txt --nclass 10 --ipc 1 2>&1 | tee -a "$LOGDIR/master.log"
$RH --src-dir /root/data/imagenet100/train --dst-dir ./results/sweep_in10/high_noise_random_herding_ipc10_d25 --class-file ./misc/class100.txt --nclass 10 --ipc 10 2>&1 | tee -a "$LOGDIR/master.log"
$RH --src-dir /root/data/imagenet100/train --dst-dir ./results/sweep_in10/high_noise_random_herding_ipc50_d25 --class-file ./misc/class100.txt --nclass 10 --ipc 50 2>&1 | tee -a "$LOGDIR/master.log"

# ImageNette Random Herding
$RH --src-dir /root/data/imagenette2/train --dst-dir ./results/sweep_nette/high_noise_random_herding_ipc10_d25 --class-file ./misc/class_nette.txt --nclass 10 --ipc 10 2>&1 | tee -a "$LOGDIR/master.log"

# ImageWoof Random Herding
$RH --src-dir /root/data/imagewoof2/train --dst-dir ./results/sweep_woof/high_noise_random_herding_ipc10_d25 --class-file ./misc/class_woof.txt --nclass 10 --ipc 10 2>&1 | tee -a "$LOGDIR/master.log"

echo "Random Herding DONE - $(date)" | tee -a "$LOGDIR/master.log"

# === Phase 1: Compute all CAGS clusters (sequential, GPU0) ===
echo "" | tee -a "$LOGDIR/master.log"
echo "=== Phase 1: Computing CAGS clusters ===" | tee -a "$LOGDIR/master.log"
CUDA_VISIBLE_DEVICES=0 python3 gen_all.py --compute-clusters-only 2>&1 | tee -a "$LOGDIR/master.log"
echo "Clusters DONE - $(date)" | tee -a "$LOGDIR/master.log"

# === Phase 2: Generate datasets (4 GPUs parallel) ===
echo "" | tee -a "$LOGDIR/master.log"
echo "=== Phase 2: Generating datasets on 4 GPUs ===" | tee -a "$LOGDIR/master.log"

CUDA_VISIBLE_DEVICES=0 nohup python3 gen_all.py --gpu-group 0 > "$LOGDIR/gen_gpu0.log" 2>&1 &
PID0=$!
echo "GPU0 generation PID: $PID0"

CUDA_VISIBLE_DEVICES=1 nohup python3 gen_all.py --gpu-group 1 > "$LOGDIR/gen_gpu1.log" 2>&1 &
PID1=$!
echo "GPU1 generation PID: $PID1"

CUDA_VISIBLE_DEVICES=2 nohup python3 gen_all.py --gpu-group 2 > "$LOGDIR/gen_gpu2.log" 2>&1 &
PID2=$!
echo "GPU2 generation PID: $PID2"

CUDA_VISIBLE_DEVICES=3 nohup python3 gen_all.py --gpu-group 3 > "$LOGDIR/gen_gpu3.log" 2>&1 &
PID3=$!
echo "GPU3 generation PID: $PID3"

echo "Waiting for all generation to complete..." | tee -a "$LOGDIR/master.log"
wait $PID0 $PID1 $PID2 $PID3
echo "Generation DONE - $(date)" | tee -a "$LOGDIR/master.log"

# === Phase 3: Evaluation (4 GPUs parallel) ===
echo "" | tee -a "$LOGDIR/master.log"
echo "=== Phase 3: Evaluation with quick_eval_v3.py ===" | tee -a "$LOGDIR/master.log"

CUDA_VISIBLE_DEVICES=0 nohup bash run_v3_4gpu_gpu0.sh > "$LOGDIR/eval_gpu0.log" 2>&1 &
echo "GPU0 eval PID: $!"

CUDA_VISIBLE_DEVICES=1 nohup bash run_v3_4gpu_gpu1.sh > "$LOGDIR/eval_gpu1.log" 2>&1 &
echo "GPU1 eval PID: $!"

CUDA_VISIBLE_DEVICES=2 nohup bash run_v3_4gpu_gpu2.sh > "$LOGDIR/eval_gpu2.log" 2>&1 &
echo "GPU2 eval PID: $!"

CUDA_VISIBLE_DEVICES=3 nohup bash run_v3_4gpu_gpu3.sh > "$LOGDIR/eval_gpu3.log" 2>&1 &
echo "GPU3 eval PID: $!"

echo "Evaluation launched. Monitor with:" | tee -a "$LOGDIR/master.log"
echo "  tail -f $LOGDIR/eval_gpu0.log" | tee -a "$LOGDIR/master.log"
echo "  tail -f $LOGDIR/eval_gpu1.log" | tee -a "$LOGDIR/master.log"
echo "  tail -f $LOGDIR/eval_gpu2.log" | tee -a "$LOGDIR/master.log"
echo "  tail -f $LOGDIR/eval_gpu3.log" | tee -a "$LOGDIR/master.log"
echo "============================================" | tee -a "$LOGDIR/master.log"
