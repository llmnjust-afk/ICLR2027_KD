#!/bin/bash
# Evaluate best CAGS+D4M hybrid with ResNet-18 and ResNetAP-10
# Usage: bash eval_hybrid_resnet.sh <best_tag>
# Example: bash eval_hybrid_resnet.sh cags_d4m_4f_t40

cd /root/ICLR2027_KD

TAG=${1:-cags_d4m_4f_t40}
echo "Evaluating hybrid dataset: $TAG"

# GPU 0: ResNet-18
CUDA_VISIBLE_DEVICES=0 nohup env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
    --train-dir ./results/sweep_in100/${TAG}/dataset_0 \
    --val-dir /root/data/imagenet100/val \
    --class-file ./misc/class100.txt --nclass 100 \
    --epochs 2000 --seeds 0 1 2 \
    --lr 0.1 --weight-decay 1e-4 --batch-size 128 \
    --arch resnet --depth 18 --norm-type instance \
    > logs/eval_hybrid_${TAG}_resnet18.log 2>&1 &
PID_R18=$!

# GPU 1: ResNetAP-10
CUDA_VISIBLE_DEVICES=1 nohup env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
    --train-dir ./results/sweep_in100/${TAG}/dataset_0 \
    --val-dir /root/data/imagenet100/val \
    --class-file ./misc/class100.txt --nclass 100 \
    --epochs 2000 --seeds 0 1 2 \
    --lr 0.1 --weight-decay 1e-4 --batch-size 128 \
    --arch resnet_ap --depth 10 --norm-type instance \
    > logs/eval_hybrid_${TAG}_resnetap10.log 2>&1 &
PID_RAP=$!

echo "ResNet-18 (GPU 0): PID $PID_R18"
echo "ResNetAP-10 (GPU 1): PID $PID_RAP"

echo "Waiting for evaluations..."
wait $PID_R18 $PID_RAP

echo "=== Results ==="
echo "ResNet-18:" && tail -3 logs/eval_hybrid_${TAG}_resnet18.log
echo "ResNetAP-10:" && tail -3 logs/eval_hybrid_${TAG}_resnetap10.log
