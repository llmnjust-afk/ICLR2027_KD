#!/bin/bash
# P0-3: Semantic subset experiments (animals/objects)
# Runs after P0-2 experiments complete
# Uses 4 GPUs: 0,1 for animals, 2,3 for objects

cd /root/ICLR2027_KD
mkdir -p logs results

COMMON_GEN="--cags-min-scale 0.0 --cags-max-scale 0.06 --sigmoid-slope 3.0 --sigmoid-center 0.6"
EVAL_ARGS="--epochs 2000 --seeds 0 1 2 --lr 0.1 --weight-decay 1e-4 --batch-size 128 --arch convnet --norm-type instance"

echo "============================================"
echo "P0-3: Semantic subset experiments"
echo "Started: $(date)"
echo "============================================"

# Phase 1: Cache generation (parallel on GPU 0,2)
echo "Phase 1: Cache generation for animals and objects..."

CUDA_VISIBLE_DEVICES=0 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py \
    --spec animals --nclass 44 \
    --imagenet-dir /root/data/imagenet100/ \
    --save-base ./results/sweep_animals \
    --tag cache_warmup --cache-only \
    > logs/cache_animals.log 2>&1 &
PID_CACHE_ANIMALS=$!

CUDA_VISIBLE_DEVICES=2 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py \
    --spec objects --nclass 56 \
    --imagenet-dir /root/data/imagenet100/ \
    --save-base ./results/sweep_objects \
    --tag cache_warmup --cache-only \
    > logs/cache_objects.log 2>&1 &
PID_CACHE_OBJECTS=$!

wait $PID_CACHE_ANIMALS
echo "Animals cache ready!"
wait $PID_CACHE_OBJECTS
echo "Objects cache ready!"

# Phase 2: Dataset generation (6 configs parallel on 4 GPUs)
echo "Phase 2: Dataset generation..."

# GPU 0: Animals CAGS + Objects CAGS (sequential)
CUDA_VISIBLE_DEVICES=0 env PYTHONUNBUFFERED=1 bash -c '
cd /root/ICLR2027_KD
echo "=== ANIMALS CAGS ==="
python3 -u gen_single_config.py --spec animals --nclass 44 \
    --imagenet-dir /root/data/imagenet100/ --save-base ./results/sweep_animals \
    --tag cags_main --alpha 0.3 --beta 0.3 --gamma 0.2 --delta 0.2 '"$COMMON_GEN"'
echo "=== ANIMALS ENTROPY ==="
python3 -u gen_single_config.py --spec animals --nclass 44 \
    --imagenet-dir /root/data/imagenet100/ --save-base ./results/sweep_animals \
    --tag entropy_only --alpha 0.0 --beta 1.0 --gamma 0.0 --delta 0.0 '"$COMMON_GEN"'
echo "=== ANIMALS DONE ==="
' > logs/gen_animals_all.log 2>&1 &
PID_GEN_ANIMALS=$!

# GPU 1: Animals unguided
CUDA_VISIBLE_DEVICES=1 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py \
    --spec animals --nclass 44 \
    --imagenet-dir /root/data/imagenet100/ \
    --save-base ./results/sweep_animals \
    --tag unguided --no-cags --fixed-scale 0.0 \
    > logs/gen_animals_unguided.log 2>&1 &
PID_GEN_ANIMALS_UNG=$!

# GPU 2: Objects CAGS + entropy
CUDA_VISIBLE_DEVICES=2 env PYTHONUNBUFFERED=1 bash -c '
cd /root/ICLR2027_KD
echo "=== OBJECTS CAGS ==="
python3 -u gen_single_config.py --spec objects --nclass 56 \
    --imagenet-dir /root/data/imagenet100/ --save-base ./results/sweep_objects \
    --tag cags_main --alpha 0.3 --beta 0.3 --gamma 0.2 --delta 0.2 '"$COMMON_GEN"'
echo "=== OBJECTS ENTROPY ==="
python3 -u gen_single_config.py --spec objects --nclass 56 \
    --imagenet-dir /root/data/imagenet100/ --save-base ./results/sweep_objects \
    --tag entropy_only --alpha 0.0 --beta 1.0 --gamma 0.0 --delta 0.0 '"$COMMON_GEN"'
echo "=== OBJECTS DONE ==="
' > logs/gen_objects_all.log 2>&1 &
PID_GEN_OBJECTS=$!

# GPU 3: Objects unguided
CUDA_VISIBLE_DEVICES=3 env PYTHONUNBUFFERED=1 python3 -u gen_single_config.py \
    --spec objects --nclass 56 \
    --imagenet-dir /root/data/imagenet100/ \
    --save-base ./results/sweep_objects \
    --tag unguided --no-cags --fixed-scale 0.0 \
    > logs/gen_objects_unguided.log 2>&1 &
PID_GEN_OBJECTS_UNG=$!

echo "Waiting for generation..."
wait $PID_GEN_ANIMALS
wait $PID_GEN_ANIMALS_UNG
echo "Animals generation done!"
wait $PID_GEN_OBJECTS
wait $PID_GEN_OBJECTS_UNG
echo "Objects generation done!"

# Phase 3: Evaluation (6 configs, 2 per GPU)
echo "Phase 3: Evaluation..."

CUDA_VISIBLE_DEVICES=0 env PYTHONUNBUFFERED=1 bash -c '
cd /root/ICLR2027_KD
echo "=== EVAL ANIMALS CAGS ==="
python3 -u quick_eval_v3.py --train-dir ./results/sweep_animals/cags_main/dataset_0 \
    --val-dir /root/data/imagenet100/val --class-file ./misc/class_animals_subset.txt \
    --nclass 44 '"$EVAL_ARGS"' 2>&1 | tee logs/eval_animals_cags.log
echo "=== EVAL ANIMALS ENTROPY ==="
python3 -u quick_eval_v3.py --train-dir ./results/sweep_animals/entropy_only/dataset_0 \
    --val-dir /root/data/imagenet100/val --class-file ./misc/class_animals_subset.txt \
    --nclass 44 '"$EVAL_ARGS"' 2>&1 | tee logs/eval_animals_entropy.log
echo "=== ANIMALS EVAL DONE ==="
' > logs/eval_animals_all.log 2>&1 &
PID_EVAL_ANIMALS=$!

CUDA_VISIBLE_DEVICES=1 env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
    --train-dir ./results/sweep_animals/unguided/dataset_0 \
    --val-dir /root/data/imagenet100/val --class-file ./misc/class_animals_subset.txt \
    --nclass 44 $EVAL_ARGS \
    > logs/eval_animals_unguided.log 2>&1 &
PID_EVAL_ANIMALS_UNG=$!

CUDA_VISIBLE_DEVICES=2 env PYTHONUNBUFFERED=1 bash -c '
cd /root/ICLR2027_KD
echo "=== EVAL OBJECTS CAGS ==="
python3 -u quick_eval_v3.py --train-dir ./results/sweep_objects/cags_main/dataset_0 \
    --val-dir /root/data/imagenet100/val --class-file ./misc/class_objects_subset.txt \
    --nclass 56 '"$EVAL_ARGS"' 2>&1 | tee logs/eval_objects_cags.log
echo "=== EVAL OBJECTS ENTROPY ==="
python3 -u quick_eval_v3.py --train-dir ./results/sweep_objects/entropy_only/dataset_0 \
    --val-dir /root/data/imagenet100/val --class-file ./misc/class_objects_subset.txt \
    --nclass 56 '"$EVAL_ARGS"' 2>&1 | tee logs/eval_objects_entropy.log
echo "=== OBJECTS EVAL DONE ==="
' > logs/eval_objects_all.log 2>&1 &
PID_EVAL_OBJECTS=$!

CUDA_VISIBLE_DEVICES=3 env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
    --train-dir ./results/sweep_objects/unguided/dataset_0 \
    --val-dir /root/data/imagenet100/val --class-file ./misc/class_objects_subset.txt \
    --nclass 56 $EVAL_ARGS \
    > logs/eval_objects_unguided.log 2>&1 &
PID_EVAL_OBJECTS_UNG=$!

wait $PID_EVAL_ANIMALS
wait $PID_EVAL_ANIMALS_UNG
echo "Animals evaluation done!"
wait $PID_EVAL_OBJECTS
wait $PID_EVAL_OBJECTS_UNG
echo "Objects evaluation done!"

echo ""
echo "============================================"
echo "=== P0-3 RESULTS SUMMARY ==="
echo "============================================"
echo "--- Animals (44 classes) ---"
for cfg in cags_main entropy_only unguided; do
    if [ -f logs/eval_animals_${cfg}.log ]; then
        echo -n "  ${cfg}: "
        grep "Mean Best Top-1" logs/eval_animals_${cfg}.log | tail -1
    fi
done
echo "--- Objects (56 classes) ---"
for cfg in cags_main entropy_only unguided; do
    if [ -f logs/eval_objects_${cfg}.log ]; then
        echo -n "  ${cfg}: "
        grep "Mean Best Top-1" logs/eval_objects_${cfg}.log | tail -1
    fi
done
echo ""
echo "P0-3 complete! — $(date)"
