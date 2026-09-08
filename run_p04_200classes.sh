#!/bin/bash
# P0-4: Scale to 200 classes
# Phase 1: Cache generation (GPU 0)
# Phase 2: Dataset generation: CAGS (GPU 0,1), Entropy (GPU 2), Unguided (GPU 3) 
# Phase 3: Evaluation: 3 configs x 3 seeds on 4 GPUs

cd /root/ICLR2027_KD
mkdir -p logs results

COMMON_GEN="--cags-min-scale 0.0 --cags-max-scale 0.06 --sigmoid-slope 3.0 --sigmoid-center 0.6"
EVAL_ARGS="--epochs 2000 --seeds 0 1 2 --lr 0.1 --weight-decay 1e-4 --batch-size 128 --arch convnet --norm-type instance"

echo "============================================"
echo "P0-4: Scale to 200 classes"
echo "Started: $(date)"
echo "============================================"

# Phase 1: Cache generation
echo "Phase 1: Cache generation for imagenet200..."
CUDA_VISIBLE_DEVICES=0 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py \
    --spec imagenet200 --nclass 200 \
    --imagenet-dir /root/data/imagenet200/ \
    --save-base ./results/sweep_in200 \
    --tag cache_warmup --cache-only \
    > logs/cache_in200.log 2>&1
echo "Cache ready! $(date)"

# Phase 2: Dataset generation (3 configs on 3 GPUs)
echo "Phase 2: Dataset generation..."

# GPU 0: CAGS
CUDA_VISIBLE_DEVICES=0 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py \
    --spec imagenet200 --nclass 200 \
    --imagenet-dir /root/data/imagenet200/ \
    --save-base ./results/sweep_in200 \
    --tag cags_main --alpha 0.3 --beta 0.3 --gamma 0.2 --delta 0.2 \
    $COMMON_GEN \
    > logs/gen_in200_cags.log 2>&1 &
PID_CAGS=$!

# GPU 1: Entropy
CUDA_VISIBLE_DEVICES=1 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py \
    --spec imagenet200 --nclass 200 \
    --imagenet-dir /root/data/imagenet200/ \
    --save-base ./results/sweep_in200 \
    --tag entropy_only --alpha 0.0 --beta 1.0 --gamma 0.0 --delta 0.0 \
    $COMMON_GEN \
    > logs/gen_in200_entropy.log 2>&1 &
PID_ENT=$!

# GPU 2: Unguided
CUDA_VISIBLE_DEVICES=2 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py \
    --spec imagenet200 --nclass 200 \
    --imagenet-dir /root/data/imagenet200/ \
    --save-base ./results/sweep_in200 \
    --tag unguided --no-cags --fixed-scale 0.0 \
    > logs/gen_in200_unguided.log 2>&1 &
PID_UNG=$!

echo "Waiting for generation..."
wait $PID_CAGS
echo "CAGS generation done! $(date)"
wait $PID_ENT
echo "Entropy generation done! $(date)"
wait $PID_UNG
echo "Unguided generation done! $(date)"

# Phase 3: Evaluation (3 configs on 3 GPUs)
echo "Phase 3: Evaluation..."

CUDA_VISIBLE_DEVICES=0 env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
    --train-dir ./results/sweep_in200/cags_main/dataset_0 \
    --val-dir /root/data/imagenet200/val --class-file ./misc/class200.txt \
    --nclass 200 $EVAL_ARGS \
    > logs/eval_in200_cags.log 2>&1 &
PID_EVAL_CAGS=$!

CUDA_VISIBLE_DEVICES=1 env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
    --train-dir ./results/sweep_in200/entropy_only/dataset_0 \
    --val-dir /root/data/imagenet200/val --class-file ./misc/class200.txt \
    --nclass 200 $EVAL_ARGS \
    > logs/eval_in200_entropy.log 2>&1 &
PID_EVAL_ENT=$!

CUDA_VISIBLE_DEVICES=2 env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
    --train-dir ./results/sweep_in200/unguided/dataset_0 \
    --val-dir /root/data/imagenet200/val --class-file ./misc/class200.txt \
    --nclass 200 $EVAL_ARGS \
    > logs/eval_in200_unguided.log 2>&1 &
PID_EVAL_UNG=$!

wait $PID_EVAL_CAGS
wait $PID_EVAL_ENT
wait $PID_EVAL_UNG
echo "Evaluation done! $(date)"

echo ""
echo "============================================"
echo "=== P0-4 RESULTS SUMMARY ==="
echo "============================================"
for cfg in cags entropy unguided; do
    if [ -f logs/eval_in200_${cfg}.log ]; then
        echo -n "  ${cfg}: "
        grep "Mean Best Top-1" logs/eval_in200_${cfg}.log | tail -1
    fi
done
echo ""
echo "P0-4 complete! $(date)"
