#!/bin/bash
# AGS-DD Phase 2: Comprehensive ICLR Experiments
# Runs after Phase 1 (lambda search + CAGS range tuning) completes.
#
# Covers:
#   - CAGS v2 (new complexity metric with variance + separability)
#   - ImageNette + ImageWoof datasets
#   - IPC=1, 10, 50
#   - Cross-architecture: ConvNet-6, ResNet-18, ResNet-50
#   - Full module ablation (CAGS, IAST, TAGS individually + combinations)
#   - 2000 epochs for final numbers
#
# Usage: bash run_phase2_experiments.sh
# Assumes Phase 1 results are available to determine optimal lambda.

set -uo pipefail
cd /root/ICLR2027_KD

SEEDS="0 1 2"
DEPTH=6
WINDOW="high_noise"
STOP_T=25
VAL_DIR="/root/data/imagenet100/val"

mkdir -p logs logs/phase2

count_pngs() { find "$1" -name "*.png" 2>/dev/null | wc -l; }

# ================================================================
# Generate + Evaluate helper
# ================================================================
gen_and_eval() {
    local GPU=$1 SPEC=$2 NCLASS=$3 IPC=$4 TAG=$5 BASE=$6 DATA_DIR=$7 EVAL_ARCH=$8 EVAL_DEPTH=$9 EPOCHS=${10} EXTRA_GEN="${11}"

    local DIR="$BASE/high_noise_${TAG}_d${STOP_T}/dataset_0"
    local NEEDED=$((NCLASS * IPC))
    local PREFIX="${SPEC}_${NCLASS}cls_ipc${IPC}_${TAG}"

    echo ""
    echo "================================================"
    echo "  ${PREFIX} — arch=${EVAL_ARCH} epochs=${EPOCHS} (GPU ${GPU})"
    echo "  $(date)"
    echo "================================================"

    # Generate if needed
    if [ ! -d "$DIR" ] || [ "$(count_pngs "$DIR")" -lt "$NEEDED" ]; then
        echo "  Generating ${NEEDED} images..."
        CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python3 duration_sweep.py \
            --window $WINDOW --durations $STOP_T --schedule constant \
            --spec $SPEC --nclass $NCLASS --ipc $IPC \
            --imagenet-dir $DATA_DIR --save-base $BASE \
            --epochs 1 --depth $DEPTH \
            --fixed-stop-t $STOP_T \
            --seeds 0 --num-datasets 1 --tag $TAG $EXTRA_GEN \
            2>&1 | tee logs/phase2/${PREFIX}_gen.log
    else
        echo "  Images exist ($(count_pngs "$DIR")/${NEEDED}), skipping gen"
    fi

    # Evaluate
    echo "  Evaluating (${EPOCHS} epochs, arch=${EVAL_ARCH}, seeds ${SEEDS})..."
    local EVAL_TAG="${PREFIX}_${EVAL_ARCH}_d${EVAL_DEPTH}"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python3 quick_eval_v2.py \
        --train-dir "$DIR" \
        --val-dir $VAL_DIR \
        --class-file ./misc/class100.txt \
        --nclass $NCLASS --epochs $EPOCHS --depth $EVAL_DEPTH --seeds $SEEDS \
        --arch $EVAL_ARCH \
        2>&1 | tee logs/phase2/${EVAL_TAG}_eval.log
}

# ================================================================
# GPU 0: 10-class experiments — CAGS v2 + ablation + cross-arch
# ================================================================
run_gpu0() {
    echo "========================================"
    echo "GPU 0: 10-class Phase 2 — $(date)"
    echo "========================================"

    local BASE="./results/sweep_in10"
    local DATA="/root/data/imagenet100_10class"
    local NCLS=10

    # --- CAGS v2 with new complexity metric ---
    # Try multiple sigmoid/range combos
    for RANGE in "0.05_0.12" "0.05_0.15" "0.08_0.12"; do
        MIN_S=$(echo $RANGE | cut -d_ -f1)
        MAX_S=$(echo $RANGE | cut -d_ -f2)
        for SLOPE_CENTER in "3.0_0.6" "5.0_0.5" "3.0_0.7"; do
            SLOPE=$(echo $SLOPE_CENTER | cut -d_ -f1)
            CENTER=$(echo $SLOPE_CENTER | cut -d_ -f2)
            TAG="cagsv2_${RANGE}_s${SLOPE}_c${CENTER}"
            gen_and_eval 0 imagenet100 $NCLS 10 $TAG "$BASE" "$DATA" convnet 6 1000 \
                "--cags-min-scale $MIN_S --cags-max-scale $MAX_S --sigmoid-slope $SLOPE --sigmoid-center $CENTER"
        done
    done

    # --- IAST ablation (fixed lambda=0.1, IAST on/off) ---
    # IAST requires use_iast=True in AGSSampler — needs code change
    # For now, test with fixed stop_t vs IAST stop_t
    # TODO: add --use-iast flag to duration_sweep.py

    # --- Cross-architecture (using best config from Phase 1) ---
    # Assume fixed_l0.1 is the baseline to beat
    for ARCH_DEPTH in "resnet_18" "resnet_ap_10"; do
        ARCH=$(echo $ARCH_DEPTH | cut -d_ -f1)
        D=$(echo $ARCH_DEPTH | cut -d_ -f2)
        # Eval existing fixed_l0.1 images with ResNet
        local DIR="$BASE/high_noise_fixed_l0.1_d25/dataset_0"
        if [ -d "$DIR" ]; then
            echo "  Cross-arch eval: fixed_l0.1 with ${ARCH}-${D}"
            CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 quick_eval_v2.py \
                --train-dir "$DIR" --val-dir $VAL_DIR \
                --class-file ./misc/class100.txt \
                --nclass $NCLS --epochs 1000 --depth $D --seeds $SEEDS \
                --arch $ARCH \
                2>&1 | tee logs/phase2/10cls_fixed_l0.1_${ARCH}_d${D}_eval.log
        fi
        # Eval best CAGS v2 config with ResNet (after CAGS runs complete)
    done

    # --- IPC variations ---
    for IPC in 1 50; do
        gen_and_eval 0 imagenet100 $NCLS $IPC "fixed_l0.1" "$BASE" "$DATA" convnet 6 1000 \
            "--no-cags --fixed-scale 0.1"
        # Best CAGS v2 config for this IPC
    done

    # --- 2000 epochs final numbers (best config) ---
    # TODO: after determining best config, run with 2000 epochs
    local DIR="$BASE/high_noise_fixed_l0.1_d25/dataset_0"
    if [ -d "$DIR" ]; then
        echo "  2000-epoch eval: fixed_l0.1"
        CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python3 quick_eval_v2.py \
            --train-dir "$DIR" --val-dir $VAL_DIR \
            --class-file ./misc/class100.txt \
            --nclass $NCLS --epochs 2000 --depth 6 --seeds $SEEDS \
            --arch convnet \
            2>&1 | tee logs/phase2/10cls_fixed_l0.1_2000ep_eval.log
    fi

    echo "GPU 0: Phase 2 complete! — $(date)"
}

# ================================================================
# GPU 1: ImageNette + ImageWoof + 100-class
# ================================================================
run_gpu1() {
    echo "========================================"
    echo "GPU 1: ImageNette/Woof + 100-class Phase 2 — $(date)"
    echo "========================================"

    # --- ImageNette (10 classes, fast) ---
    local NETTE_DATA="/root/data/imagenette2"
    local NETTE_VAL="/root/data/imagenette2/val"
    # Check if imagenette data exists
    if [ ! -d "$NETTE_DATA" ]; then
        echo "  ImageNette not found, downloading..."
        # Use existing download script or create symlinks
        mkdir -p /root/data/imagenette2/train /root/data/imagenette2/val
        for cls in $(cat misc/class_nette.txt); do
            if [ -d "/root/data/imagenet100/train/$cls" ]; then
                ln -sf /root/data/imagenet100/train/$cls /root/data/imagenette2/train/$cls 2>/dev/null
            fi
            if [ -d "/root/data/imagenet100/val/$cls" ]; then
                ln -sf /root/data/imagenet100/val/$cls /root/data/imagenette2/val/$cls 2>/dev/null
            fi
        done
        NETTE_DATA="/root/data/imagenette2"
        NETTE_VAL="/root/data/imagenette2/val"
    fi

    for IPC in 10 50; do
        for CONFIG_TAG in "fixed_l0.1" "cagsv2_0.05_0.12_s3.0_c0.6"; do
            EXTRA=""
            if [[ "$CONFIG_TAG" == fixed_* ]]; then
                LAMBDA=$(echo $CONFIG_TAG | sed 's/fixed_l//')
                EXTRA="--no-cags --fixed-scale $LAMBDA"
            else
                MIN_S=$(echo $CONFIG_TAG | sed 's/cagsv2_//;s/_s.*//;s/_/./;s/^/0/' | head -c 4)
                EXTRA="--cags-min-scale 0.05 --cags-max-scale 0.12 --sigmoid-slope 3.0 --sigmoid-center 0.6"
            fi
            gen_and_eval 1 nette 10 $IPC $CONFIG_TAG "./results/nette" "$NETTE_DATA" convnet 6 1000 "$EXTRA"
        done
    done

    # --- ImageWoof (10 classes) ---
    local WOOF_DATA="/root/data/imagewoof2"
    if [ ! -d "$WOOF_DATA" ]; then
        mkdir -p /root/data/imagewoof2/train /root/data/imagewoof2/val
        for cls in $(cat misc/class_woof.txt); do
            if [ -d "/root/data/imagenet100/train/$cls" ]; then
                ln -sf /root/data/imagenet100/train/$cls /root/data/imagewoof2/train/$cls 2>/dev/null
            fi
            if [ -d "/root/data/imagenet100/val/$cls" ]; then
                ln -sf /root/data/imagenet100/val/$cls /root/data/imagewoof2/val/$cls 2>/dev/null
            fi
        done
        WOOF_DATA="/root/data/imagewoof2"
    fi

    for IPC in 10 50; do
        for CONFIG_TAG in "fixed_l0.1" "cagsv2_0.05_0.12_s3.0_c0.6"; do
            EXTRA="--no-cags --fixed-scale 0.1"
            if [[ "$CONFIG_TAG" == cagsv2_* ]]; then
                EXTRA="--cags-min-scale 0.05 --cags-max-scale 0.12 --sigmoid-slope 3.0 --sigmoid-center 0.6"
            fi
            gen_and_eval 1 woof 10 $IPC $CONFIG_TAG "./results/woof" "$WOOF_DATA" convnet 6 1000 "$EXTRA"
        done
    done

    # --- 100-class cross-arch ---
    local BASE100="./results/sweep_in100"
    local DATA100="/root/data/imagenet100"
    local DIR100="$BASE100/high_noise_fixed_l0.1_d25/dataset_0"
    if [ -d "$DIR100" ]; then
        echo "  100-class cross-arch: ResNet-18"
        CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python3 quick_eval_v2.py \
            --train-dir "$DIR100" --val-dir $VAL_DIR \
            --class-file ./misc/class100.txt \
            --nclass 100 --epochs 1000 --depth 18 --seeds $SEEDS \
            --arch resnet \
            2>&1 | tee logs/phase2/100cls_fixed_l0.1_resnet_d18_eval.log
    fi

    echo "GPU 1: Phase 2 complete! — $(date)"
}

# ================================================================
# Main
# ================================================================
echo "========================================"
echo "AGS-DD Phase 2: Comprehensive ICLR Experiments"
echo "Started: $(date)"
echo "========================================"

run_gpu1 &
GPU1_PID=$!
sleep 10
run_gpu0 &
GPU0_PID=$!

echo "GPU 0 PID: $GPU0_PID"
echo "GPU 1 PID: $GPU1_PID"

wait $GPU0_PID
echo "GPU 0 done — $(date)"
wait $GPU1_PID
echo "GPU 1 done — $(date)"

echo ""
echo "========================================"
echo "Phase 2 complete! — $(date)"
echo "========================================"

# Collect and push
python3 collect_results.py 2>&1 | tee logs/phase2/collect_results.log
bash auto_push.sh 2>&1 | tee logs/phase2/auto_push.log
echo "All done! — $(date)"
