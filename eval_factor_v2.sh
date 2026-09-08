#!/bin/bash
# Evaluate all v2 factor ablation configs (3 seeds each)
# Run on 4 GPUs in parallel, 2 configs per GPU

cd /root/ICLR2027_KD

SEEDS="0 1 2"
EPOCHS=2000
DATASET="imagenet100"
NCCLASS=100
IPC=10

# Config list (must match gen_factor_ablation_v2.py FACTOR_CONFIGS)
CONFIGS=(
    "v2_full_cags"
    "v2_intra_var_only"
    "v2_separability_only"
    "v2_mode_count_only"
    "v2_entropy_only"
    "v2_equal_weights"
    "v2_no_separability"
    "v2_separability_dominated"
)

# Assign 2 configs per GPU
# GPU0: configs 0,4; GPU1: configs 1,5; GPU2: configs 2,6; GPU3: configs 3,7

run_eval() {
    local gpu=$1
    local config=$2
    local data_dir="results/sweep_in100/factor_ablation_${config}"
    local log_file="logs/eval_factor_v2_${config}.log"
    
    echo "GPU${gpu}: Evaluating ${config} on ${DATASET} ${NCCLASS}-class IPC=${IPC}" > ${log_file}
    
    for seed in ${SEEDS}; do
        echo "Seed ${seed}:" >> ${log_file}
        CUDA_VISIBLE_DEVICES=${gpu} python3 quick_eval_v3.py \
            --data_dir "${data_dir}/dataset_0" \
            --dataset ${DATASET} \
            --nclass ${NCCLASS} \
            --ipc ${IPC} \
            --epochs ${EPOCHS} \
            --seed ${seed} \
            --lr 0.1 \
            --wd 1e-4 \
            --batch_size 128 \
            --net_type convnet \
            --depth 6 \
            --width 1.0 \
            --norm_type instance \
            >> ${log_file} 2>&1
    done
    echo "Done: ${config}" >> ${log_file}
}

# Run 2 configs per GPU, 4 GPUs in parallel
for i in 0 1 2 3; do
    config_a=${CONFIGS[$i]}
    config_b=${CONFIGS[$((i+4))]}
    (
        run_eval $i $config_a
        run_eval $i $config_b
    ) &
done

wait
echo "All v2 factor ablation evaluations complete!"
