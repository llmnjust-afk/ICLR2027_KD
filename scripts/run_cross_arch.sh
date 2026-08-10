#!/bin/bash
# AGS-DD: Cross-architecture evaluation
# Usage: bash scripts/run_cross_arch.sh [TRAIN_DIR] [TEST_DIR] [IPC]

set -e

TRAIN_DIR=${1:-"./generated/ags_dd/nette_ipc10/dataset_0"}
TEST_DIR=${2:-"/ssd_data/imagenet/val"}
IPC=${3:-10}
NUM_CLASSES=${NUM_CLASSES:-10}
SAVE_DIR="./results/cross_arch"

echo "=================================================="
echo "AGS-DD: Cross-architecture evaluation"
echo "  Train dir: $TRAIN_DIR"
echo "  Test dir: $TEST_DIR"
echo "  IPC: $IPC"
echo "=================================================="

python -m evaluation.cross_arch \
    --train-dir $TRAIN_DIR \
    --test-dir $TEST_DIR \
    --num-classes $NUM_CLASSES \
    --img-size 224 \
    --epochs 2000 \
    --num-seeds 5 \
    --save-dir $SAVE_DIR \
    --dataset-name imagenette \
    --ipc $IPC \
    --arch-list convnet7 resnet18 vit_tiny swin_tiny deit_tiny

echo "Done! Results: $SAVE_DIR"
