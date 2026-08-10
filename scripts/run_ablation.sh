#!/bin/bash
# AGS-DD: Run ablation study
# Usage: bash scripts/run_ablation.sh [TEST_DIR]

set -e

TEST_DIR=${1:-"/ssd_data/imagenet/val"}
SPEC=${SPEC:-nette}
IPC=${IPC:-10}

echo "=================================================="
echo "AGS-DD: Ablation Study"
echo "  Dataset: $SPEC"
echo "  IPC: $IPC"
echo "  Test dir: $TEST_DIR"
echo "=================================================="

python run_ablation.py \
    --spec $SPEC \
    --num-samples $IPC \
    --num-classes 10 \
    --output-base ./ablation \
    --test-dir $TEST_DIR \
    --arch convnet7 \
    --epochs 2000 \
    --num-seeds 5 \
    --num-datasets 5

echo "Done! Results: ./ablation/ablation_summary.json"
