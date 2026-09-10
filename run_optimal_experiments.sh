#!/bin/bash
# Run optimal config (0, 0.5, 0.5, 0) on all datasets + cross-architecture
set -e

cd /root/ICLR2027_KD
mkdir -p logs

OPTIMAL="--alpha 0.0 --beta 0.5 --gamma 0.5 --delta 0.0"
COMMON="--cags-min-scale 0.0 --cags-max-scale 0.06 --sigmoid-slope 3.0 --sigmoid-center 0.6 --recompute-only"
EVAL_COMMON="--val-dir /root/data/imagenet100/val --class-file ./misc/class100.txt --nclass 100 --lr 0.1 --weight-decay 1e-4 --batch-size 128 --arch convnet --norm-type instance"

echo "============================================"
echo "Optimal Config Experiments: (0, 0.5, 0.5, 0)"
echo "Start: $(date)"
echo "============================================"

# ===== Phase 1: Generation + IN-100 cross-arch eval =====
echo "=== Phase 1: Generation + cross-arch eval ==="
echo "Start: $(date)"

# GPU 0: Generate ImageNette with optimal config
CUDA_VISIBLE_DEVICES=0 python3 -u gen_single_config.py \
  --spec nette --nclass 10 --imagenet-dir /root/data/imagenette2/ \
  --save-base ./results/sweep_nette --tag optimal \
  --ipc 10 --window high_noise $COMMON $OPTIMAL \
  > logs/gen_optimal_nette.log 2>&1 &
PID_GEN_NETTE=$!

# GPU 1: Generate ImageWoof with optimal config
CUDA_VISIBLE_DEVICES=1 python3 -u gen_single_config.py \
  --spec woof --nclass 10 --imagenet-dir /root/data/imagewoof2/ \
  --save-base ./results/sweep_woof --tag optimal \
  --ipc 10 --window high_noise $COMMON $OPTIMAL \
  > logs/gen_optimal_woof.log 2>&1 &
PID_GEN_WOOF=$!

# GPU 2: Generate IN-200 with optimal config (reuse cache)
CUDA_VISIBLE_DEVICES=2 python3 -u gen_single_config.py \
  --spec imagenet200 --nclass 200 --imagenet-dir /root/data/imagenet200/ \
  --save-base ./results/sweep_in200 --tag optimal \
  --ipc 10 --window high_noise $COMMON $OPTIMAL \
  > logs/gen_optimal_in200.log 2>&1 &
PID_GEN_IN200=$!

# GPU 3: Eval IN-100 optimal (wl_r2_c2) on ResNet-18
CUDA_VISIBLE_DEVICES=3 python3 -u quick_eval_v3.py \
  --train-dir ./results/sweep_in100/wl_r2_c2/dataset_0 \
  --val-dir /root/data/imagenet100/val --class-file ./misc/class100.txt \
  --nclass 100 --epochs 2000 --seeds 0 1 2 --lr 0.1 --weight-decay 1e-4 \
  --batch-size 128 --arch resnet18 --norm-type instance \
  > logs/eval_optimal_in100_resnet18.log 2>&1 &
PID_EVAL_R18=$!

echo "Waiting for generation to complete..."
wait $PID_GEN_NETTE && echo "ImageNette gen done: $(date)"
wait $PID_GEN_WOOF && echo "ImageWoof gen done: $(date)"
wait $PID_GEN_IN200 && echo "IN-200 gen done: $(date)"

echo "=== Phase 1 generation done: $(date) ==="

# ===== Phase 2: Evaluation of new datasets =====
echo "=== Phase 2: Dataset evaluation ==="
echo "Start: $(date)"

# GPU 0: Eval ImageNette optimal (3 seeds, 2000 epochs)
CUDA_VISIBLE_DEVICES=0 python3 -u quick_eval_v3.py \
  --train-dir ./results/sweep_nette/optimal/dataset_0 \
  --val-dir /root/data/imagenette2/val --class-file ./misc/class_nette.txt \
  --nclass 10 --epochs 2000 --seeds 0 1 2 --lr 0.1 --weight-decay 1e-4 \
  --batch-size 128 --arch convnet --norm-type instance \
  > logs/eval_optimal_nette.log 2>&1 &
PID_EVAL_NETTE=$!

# GPU 1: Eval ImageWoof optimal (3 seeds, 2000 epochs)
CUDA_VISIBLE_DEVICES=1 python3 -u quick_eval_v3.py \
  --train-dir ./results/sweep_woof/optimal/dataset_0 \
  --val-dir /root/data/imagewoof2/val --class-file ./misc/class_woof.txt \
  --nclass 10 --epochs 2000 --seeds 0 1 2 --lr 0.1 --weight-decay 1e-4 \
  --batch-size 128 --arch convnet --norm-type instance \
  > logs/eval_optimal_woof.log 2>&1 &
PID_EVAL_WOOF=$!

# GPU 2: Eval IN-200 optimal (3 seeds, 2000 epochs)
CUDA_VISIBLE_DEVICES=2 python3 -u quick_eval_v3.py \
  --train-dir ./results/sweep_in200/optimal/dataset_0 \
  --val-dir /root/data/imagenet200/val --class-file ./misc/class200.txt \
  --nclass 200 --epochs 2000 --seeds 0 1 2 --lr 0.1 --weight-decay 1e-4 \
  --batch-size 128 --arch convnet --norm-type instance \
  > logs/eval_optimal_in200.log 2>&1 &
PID_EVAL_IN200=$!

# Wait for ResNet-18 eval from Phase 1
wait $PID_EVAL_R18 && echo "IN-100 ResNet-18 done: $(date)"

# GPU 3: Eval IN-100 optimal (wl_r2_c2) on ResNetAP-10
CUDA_VISIBLE_DEVICES=3 python3 -u quick_eval_v3.py \
  --train-dir ./results/sweep_in100/wl_r2_c2/dataset_0 \
  --val-dir /root/data/imagenet100/val --class-file ./misc/class100.txt \
  --nclass 100 --epochs 2000 --seeds 0 1 2 --lr 0.1 --weight-decay 1e-4 \
  --batch-size 128 --arch resnetap10 --norm-type instance \
  > logs/eval_optimal_in100_resnetap10.log 2>&1 &
PID_EVAL_RAP=$!

echo "Waiting for all Phase 2 evaluations..."
wait $PID_EVAL_NETTE && echo "ImageNette eval done: $(date)"
wait $PID_EVAL_WOOF && echo "ImageWoof eval done: $(date)"
wait $PID_EVAL_IN200 && echo "IN-200 eval done: $(date)"
wait $PID_EVAL_RAP && echo "IN-100 ResNetAP-10 done: $(date)"

echo ""
echo "============================================"
echo "ALL EXPERIMENTS COMPLETE: $(date)"
echo "============================================"
echo ""
echo "=== RESULTS SUMMARY ==="
echo ""
echo "IN-100 (ConvNet-6, 3 seeds, 2000ep):"
grep "Mean Best" logs/eval_full_r2_c2.log 2>/dev/null
echo ""
echo "IN-100 (ResNet-18, 3 seeds, 2000ep):"
grep "Mean Best" logs/eval_optimal_in100_resnet18.log 2>/dev/null
echo ""
echo "IN-100 (ResNetAP-10, 3 seeds, 2000ep):"
grep "Mean Best" logs/eval_optimal_in100_resnetap10.log 2>/dev/null
echo ""
echo "ImageNette (ConvNet-6, 3 seeds, 2000ep):"
grep "Mean Best" logs/eval_optimal_nette.log 2>/dev/null
echo ""
echo "ImageWoof (ConvNet-6, 3 seeds, 2000ep):"
grep "Mean Best" logs/eval_optimal_woof.log 2>/dev/null
echo ""
echo "IN-200 (ConvNet-6, 3 seeds, 2000ep):"
grep "Mean Best" logs/eval_optimal_in200.log 2>/dev/null
