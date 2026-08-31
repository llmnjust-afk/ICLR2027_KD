#!/bin/bash
# AGS-DD: Generate distilled dataset on ImageWoof (fine-grained)
# Usage: bash scripts/run_imagewoof.sh [IPC] [CKPT_PATH]

set -e

IPC=${1:-10}
CKPT=${2:-""}
IMAGENET_DIR=${IMAGENET_DIR:-/ssd_data/imagenet/}
SAVE_DIR="./generated/ags_dd/woof_ipc${IPC}"

EXTRA_ARGS=""
if [ -n "$CKPT" ]; then
    EXTRA_ARGS="--ckpt $CKPT"
fi

echo "=================================================="
echo "AGS-DD: Generating ImageWoof dataset (IPC=${IPC})"
echo "=================================================="

python sample_ags.py \
    --spec woof \
    --num-samples $IPC \
    --save-dir $SAVE_DIR \
    --num-datasets 5 \
    --imagenet-dir $IMAGENET_DIR \
    --cags-alpha 0.6 \
    --cags-beta 0.4 \
    --cags-kmin 2 \
    --cags-kmax 25 \
    --guidance-scale-min 0.05 \
    --guidance-scale-max 0.6 \
    --iast-lambda 0.316 \
    --iast-min-stop 5 \
    --iast-max-stop-ratio 0.9 \
    --tags-schedule cosine \
    $EXTRA_ARGS

echo "Done! Output: $SAVE_DIR"
