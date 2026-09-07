#!/bin/bash
# Launch generation (Phase 2) on 4 GPUs, then evaluation (Phase 3)
# Run AFTER cluster computation is complete
set -e
cd /root/ICLR2027_KD

LOGDIR=/root/ICLR2027_KD/logs
mkdir -p "$LOGDIR"

echo "============================================" | tee -a "$LOGDIR/pipeline.log"
echo "Phase 2: Dataset Generation - $(date)" | tee -a "$LOGDIR/pipeline.log"
echo "============================================" | tee -a "$LOGDIR/pipeline.log"

# Launch generation on 4 GPUs
CUDA_VISIBLE_DEVICES=0 nohup python3 gen_all.py --gpu-group 0 > "$LOGDIR/gen_gpu0.log" 2>&1 &
PID0=$!
echo "GPU0 gen PID: $PID0" | tee -a "$LOGDIR/pipeline.log"

CUDA_VISIBLE_DEVICES=1 nohup python3 gen_all.py --gpu-group 1 > "$LOGDIR/gen_gpu1.log" 2>&1 &
PID1=$!
echo "GPU1 gen PID: $PID1" | tee -a "$LOGDIR/pipeline.log"

CUDA_VISIBLE_DEVICES=2 nohup python3 gen_all.py --gpu-group 2 > "$LOGDIR/gen_gpu2.log" 2>&1 &
PID2=$!
echo "GPU2 gen PID: $PID2" | tee -a "$LOGDIR/pipeline.log"

CUDA_VISIBLE_DEVICES=3 nohup python3 gen_all.py --gpu-group 3 > "$LOGDIR/gen_gpu3.log" 2>&1 &
PID3=$!
echo "GPU3 gen PID: $PID3" | tee -a "$LOGDIR/pipeline.log"

echo "Waiting for generation to complete..." | tee -a "$LOGDIR/pipeline.log"
wait $PID0 && echo "GPU0 gen DONE" | tee -a "$LOGDIR/pipeline.log" || echo "GPU0 gen FAILED" | tee -a "$LOGDIR/pipeline.log"
wait $PID1 && echo "GPU1 gen DONE" | tee -a "$LOGDIR/pipeline.log" || echo "GPU1 gen FAILED" | tee -a "$LOGDIR/pipeline.log"
wait $PID2 && echo "GPU2 gen DONE" | tee -a "$LOGDIR/pipeline.log" || echo "GPU2 gen FAILED" | tee -a "$LOGDIR/pipeline.log"
wait $PID3 && echo "GPU3 gen DONE" | tee -a "$LOGDIR/pipeline.log" || echo "GPU3 gen FAILED" | tee -a "$LOGDIR/pipeline.log"

echo "" | tee -a "$LOGDIR/pipeline.log"
echo "============================================" | tee -a "$LOGDIR/pipeline.log"
echo "Phase 3: Evaluation - $(date)" | tee -a "$LOGDIR/pipeline.log"
echo "============================================" | tee -a "$LOGDIR/pipeline.log"

# Launch balanced evaluation on 4 GPUs
CUDA_VISIBLE_DEVICES=0 nohup bash eval_gpu0.sh > "$LOGDIR/eval_gpu0.log" 2>&1 &
echo "GPU0 eval PID: $!" | tee -a "$LOGDIR/pipeline.log"

CUDA_VISIBLE_DEVICES=1 nohup bash eval_gpu1.sh > "$LOGDIR/eval_gpu1.log" 2>&1 &
echo "GPU1 eval PID: $!" | tee -a "$LOGDIR/pipeline.log"

CUDA_VISIBLE_DEVICES=2 nohup bash eval_gpu2.sh > "$LOGDIR/eval_gpu2.log" 2>&1 &
echo "GPU2 eval PID: $!" | tee -a "$LOGDIR/pipeline.log"

CUDA_VISIBLE_DEVICES=3 nohup bash eval_gpu3.sh > "$LOGDIR/eval_gpu3.log" 2>&1 &
echo "GPU3 eval PID: $!" | tee -a "$LOGDIR/pipeline.log"

echo "Evaluation launched on 4 GPUs." | tee -a "$LOGDIR/pipeline.log"
echo "Monitor: tail -f $LOGDIR/eval_gpu*.log" | tee -a "$LOGDIR/pipeline.log"
echo "============================================" | tee -a "$LOGDIR/pipeline.log"
