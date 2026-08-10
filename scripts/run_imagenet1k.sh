#!/bin/bash
# AGS-DD: Generate distilled dataset on ImageNet-1K (full scale)
# Requires multiple phases due to 1000 classes
# Usage: bash scripts/run_imagenet1k.sh [IPC] [PHASE] [CKPT_PATH]

set -e

IPC=${1:-50}
PHASE=${2:-0}
CKPT=${3:-""}
IMAGENET_DIR=${IMAGENET_DIR:-/ssd_data/imagenet/}
NCLASS_PER_PHASE=${NCLASS_PER_PHASE:-100}
SAVE_DIR="./generated/ags_dd/imagenet1k_ipc${IPC}"

EXTRA_ARGS=""
if [ -n "$CKPT" ]; then
    EXTRA_ARGS="--ckpt $CKPT"
fi

echo "=================================================="
echo "AGS-DD: Generating ImageNet-1K dataset (IPC=${IPC}, Phase=${PHASE})"
echo "=================================================="

python sample_ags.py \
    --spec imagenet1k \
    --num-samples $IPC \
    --save-dir "${SAVE_DIR}_phase${PHASE}" \
    --num-datasets 5 \
    --nclass $NCLASS_PER_PHASE \
    --phase $PHASE \
    --imagenet-dir $IMAGENET_DIR \
    --cags-alpha 0.5 \
    --cags-beta 0.5 \
    --cags-kmin 2 \
    --cags-kmax 20 \
    --guidance-scale-min 0.05 \
    --guidance-scale-max 0.5 \
    --iast-lambda 0.1 \
    --iast-min-stop 5 \
    --iast-max-stop-ratio 0.9 \
    --tags-schedule cosine \
    $EXTRA_ARGS

echo "Done! Output: ${SAVE_DIR}_phase${PHASE}"
echo "Run phases 0-9 to cover all 1000 classes"
