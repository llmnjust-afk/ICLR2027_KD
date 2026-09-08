#!/bin/bash
# Objects pipeline: CAGS (already running), then entropy, then unguided on GPU 3
# Then start objects evaluation once animals evals finish

cd /root/ICLR2027_KD
COMMON_GEN="--cags-min-scale 0.0 --cags-max-scale 0.06 --sigmoid-slope 3.0 --sigmoid-center 0.6"
EVAL_ARGS="--epochs 2000 --seeds 0 1 2 --lr 0.1 --weight-decay 1e-4 --batch-size 128 --arch convnet --norm-type instance"

echo "============================================"
echo "Objects pipeline started: $(date)"
echo "============================================"

# Wait for objects CAGS generation (already running on GPU 3)
echo "Waiting for objects CAGS generation..."
while ! grep -q "Done!" logs/gen_objects_cags.log 2>/dev/null; do
    sleep 30
done
echo "Objects CAGS done! $(date)"

# Start objects entropy on GPU 3
echo "Starting objects entropy generation..."
CUDA_VISIBLE_DEVICES=3 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py \
    --spec objects --nclass 56 \
    --imagenet-dir /root/data/imagenet100/ \
    --save-base ./results/sweep_objects \
    --tag entropy_only --alpha 0.0 --beta 1.0 --gamma 0.0 --delta 0.0 \
    $COMMON_GEN \
    > logs/gen_objects_entropy.log 2>&1
echo "Objects entropy done! $(date)"

# Start objects unguided on GPU 3
echo "Starting objects unguided generation..."
CUDA_VISIBLE_DEVICES=3 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py \
    --spec objects --nclass 56 \
    --imagenet-dir /root/data/imagenet100/ \
    --save-base ./results/sweep_objects \
    --tag unguided --no-cags --fixed-scale 0.0 \
    > logs/gen_objects_unguided.log 2>&1
echo "Objects unguided done! $(date)"

# Wait for animals evals to finish (frees up GPUs 0,1,2)
echo "Waiting for animals evaluations to finish..."
while ! grep -q "Mean Best" logs/eval_animals_cags.log 2>/dev/null || \
      ! grep -q "Mean Best" logs/eval_animals_entropy.log 2>/dev/null || \
      ! grep -q "Mean Best" logs/eval_animals_unguided.log 2>/dev/null; do
    sleep 30
done
echo "Animals evals done! $(date)"

# Start objects evaluations
echo "Starting objects evaluations..."

# GPU 0: Objects CAGS eval
CUDA_VISIBLE_DEVICES=0 env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
    --train-dir ./results/sweep_objects/cags_main/dataset_0 \
    --val-dir /root/data/imagenet100/val --class-file ./misc/class_objects_subset.txt \
    --nclass 56 $EVAL_ARGS \
    > logs/eval_objects_cags.log 2>&1 &
PID_OBJ_CAGS=$!

# GPU 1: Objects entropy eval
CUDA_VISIBLE_DEVICES=1 env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
    --train-dir ./results/sweep_objects/entropy_only/dataset_0 \
    --val-dir /root/data/imagenet100/val --class-file ./misc/class_objects_subset.txt \
    --nclass 56 $EVAL_ARGS \
    > logs/eval_objects_entropy.log 2>&1 &
PID_OBJ_ENT=$!

# GPU 2: Objects unguided eval
CUDA_VISIBLE_DEVICES=2 env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
    --train-dir ./results/sweep_objects/unguided/dataset_0 \
    --val-dir /root/data/imagenet100/val --class-file ./misc/class_objects_subset.txt \
    --nclass 56 $EVAL_ARGS \
    > logs/eval_objects_unguided.log 2>&1 &
PID_OBJ_UNG=$!

wait $PID_OBJ_CAGS
wait $PID_OBJ_ENT
wait $PID_OBJ_UNG
echo "Objects evals done! $(date)"

echo ""
echo "============================================"
echo "=== P0-3 RESULTS SUMMARY ==="
echo "============================================"
echo "--- Animals (44 classes) ---"
for cfg in cags entropy unguided; do
    if [ -f logs/eval_animals_${cfg}.log ]; then
        echo -n "  ${cfg}: "
        grep "Mean Best Top-1" logs/eval_animals_${cfg}.log | tail -1
    fi
done
echo "--- Objects (56 classes) ---"
for cfg in cags entropy unguided; do
    if [ -f logs/eval_objects_${cfg}.log ]; then
        echo -n "  ${cfg}: "
        grep "Mean Best Top-1" logs/eval_objects_${cfg}.log | tail -1
    fi
done
echo ""
echo "P0-3 complete! $(date)"
