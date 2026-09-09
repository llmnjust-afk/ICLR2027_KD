#!/bin/bash
# Run D4M baseline on all datasets
cd /root/ICLR2027_KD

# IN-100 with different t_start values
for T in 25 40; do
    echo "=== D4M IN-100 t_start=$T ==="
    CUDA_VISIBLE_DEVICES=3 env PYTHONUNBUFFERED=1 python3 -u gen_d4m.py \
        --spec imagenet100 --nclass 100 \
        --imagenet-dir /root/data/imagenet100/ \
        --save-base ./results/sweep_in100 \
        --tag d4m_t${T} --ipc 10 --t-start $T \
        > logs/gen_d4m_in100_t${T}.log 2>&1
    
    echo "Evaluating D4M IN-100 t_start=$T..."
    CUDA_VISIBLE_DEVICES=3 env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
        --train-dir ./results/sweep_in100/d4m_t${T}/dataset_0 \
        --val-dir /root/data/imagenet100/val \
        --class-file ./misc/class100.txt --nclass 100 \
        --epochs 2000 --seeds 0 1 2 \
        --lr 0.1 --weight-decay 1e-4 --batch-size 128 \
        --arch convnet --norm-type instance \
        > logs/eval_d4m_in100_t${T}.log 2>&1
    
    grep "Mean Best" logs/eval_d4m_in100_t${T}.log
done

# ImageNette
echo "=== D4M ImageNette ==="
CUDA_VISIBLE_DEVICES=3 env PYTHONUNBUFFERED=1 python3 -u gen_d4m.py \
    --spec nette --nclass 10 \
    --imagenet-dir /root/data/imagenette2 \
    --save-base ./results/sweep_nette \
    --tag d4m --ipc 10 --t-start 25 \
    > logs/gen_d4m_nette.log 2>&1

CUDA_VISIBLE_DEVICES=3 env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
    --train-dir ./results/sweep_nette/d4m/dataset_0 \
    --val-dir /root/data/imagenette2/val \
    --class-file ./misc/class_nette.txt --nclass 10 \
    --epochs 2000 --seeds 0 1 2 \
    --lr 0.1 --weight-decay 1e-4 --batch-size 128 \
    --arch convnet --norm-type instance \
    > logs/eval_d4m_nette.log 2>&1

grep "Mean Best" logs/eval_d4m_nette.log

# ImageWoof
echo "=== D4M ImageWoof ==="
CUDA_VISIBLE_DEVICES=3 env PYTHONUNBUFFERED=1 python3 -u gen_d4m.py \
    --spec woof --nclass 10 \
    --imagenet-dir /root/data/imagewoof2 \
    --save-base ./results/sweep_woof \
    --tag d4m --ipc 10 --t-start 25 \
    > logs/gen_d4m_woof.log 2>&1

CUDA_VISIBLE_DEVICES=3 env PYTHONUNBUFFERED=1 python3 -u quick_eval_v3.py \
    --train-dir ./results/sweep_woof/d4m/dataset_0 \
    --val-dir /root/data/imagewoof2/val \
    --class-file ./misc/class_woof.txt --nclass 10 \
    --epochs 2000 --seeds 0 1 2 \
    --lr 0.1 --weight-decay 1e-4 --batch-size 128 \
    --arch convnet --norm-type instance \
    > logs/eval_d4m_woof.log 2>&1

grep "Mean Best" logs/eval_d4m_woof.log

echo "=== All D4M experiments complete ==="
