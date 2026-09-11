#!/bin/bash
# Evaluate CAGS+D4M hybrid datasets with ConvNet-6 first (find best t_start)
# Then evaluate best with ResNet-18 and ResNetAP-10

cd /root/ICLR2027_KD

# Wait for generation to finish (check if all 300 classes are done)
echo "Waiting for generation to finish..."
while true; do
    T40=$(grep -c '^  n[0-9]' logs/gen_cags_d4m_4f_t40.log 2>/dev/null || echo 0)
    T35=$(grep -c '^  n[0-9]' logs/gen_cags_d4m_4f_t35.log 2>/dev/null || echo 0)
    T45=$(grep -c '^  n[0-9]' logs/gen_cags_d4m_4f_t45.log 2>/dev/null || echo 0)
    echo "  t40: $T40/300, t35: $T35/300, t45: $T45/300"
    if [ "$T40" -ge 300 ] && [ "$T35" -ge 300 ] && [ "$T45" -ge 300 ]; then
        echo "All generation complete!"
        break
    fi
    sleep 60
done

echo "Starting ConvNet-6 evaluation on all 3 t_start values..."

# GPU 0: ConvNet-6 eval of t40
CUDA_VISIBLE_DEVICES=0 nohup env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
    --train-dir ./results/sweep_in100/cags_d4m_4f_t40/dataset_0 \
    --val-dir /root/data/imagenet100/val \
    --class-file ./misc/class100.txt --nclass 100 \
    --epochs 2000 --seeds 0 1 2 \
    --lr 0.1 --weight-decay 1e-4 --batch-size 128 \
    --arch convnet --norm-type instance \
    > logs/eval_hybrid_t40_convnet.log 2>&1 &
PID_T40_CONV=$!

# GPU 1: ConvNet-6 eval of t35
CUDA_VISIBLE_DEVICES=1 nohup env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
    --train-dir ./results/sweep_in100/cags_d4m_4f_t35/dataset_0 \
    --val-dir /root/data/imagenet100/val \
    --class-file ./misc/class100.txt --nclass 100 \
    --epochs 2000 --seeds 0 1 2 \
    --lr 0.1 --weight-decay 1e-4 --batch-size 128 \
    --arch convnet --norm-type instance \
    > logs/eval_hybrid_t35_convnet.log 2>&1 &
PID_T35_CONV=$!

# GPU 2: ConvNet-6 eval of t45
CUDA_VISIBLE_DEVICES=2 nohup env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
    --train-dir ./results/sweep_in100/cags_d4m_4f_t45/dataset_0 \
    --val-dir /root/data/imagenet100/val \
    --class-file ./misc/class100.txt --nclass 100 \
    --epochs 2000 --seeds 0 1 2 \
    --lr 0.1 --weight-decay 1e-4 --batch-size 128 \
    --arch convnet --norm-type instance \
    > logs/eval_hybrid_t45_convnet.log 2>&1 &
PID_T45_CONV=$!

echo "ConvNet-6 evaluation started:"
echo "  t40 (GPU 0): PID $PID_T40_CONV"
echo "  t35 (GPU 1): PID $PID_T35_CONV"
echo "  t45 (GPU 2): PID $PID_T45_CONV"

# Wait for ConvNet-6 evals to finish
echo "Waiting for ConvNet-6 evaluations to finish..."
wait $PID_T40_CONV $PID_T35_CONV $PID_T45_CONV

echo "ConvNet-6 evaluations complete!"
echo "=== Results ==="
echo "t40:" && tail -3 logs/eval_hybrid_t40_convnet.log
echo "t35:" && tail -3 logs/eval_hybrid_t35_convnet.log
echo "t45:" && tail -3 logs/eval_hybrid_t45_convnet.log
