#!/usr/bin/env bash
set -euo pipefail
CAGS_EXTENSION=${1:?Missing experiment directory}
CAGS_PYTHON=${2:?Missing interpreter}
CAGS_STAGE=${3:?Missing stage}
: "${SLURM_JOB_ID:?Submit through sbatch}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR=/tmp/cags-mgd3-4090-matplotlib
unset PYTHONPATH
cd "$CAGS_EXTENSION/code"
exec srun --ntasks=1 --kill-on-bad-exit=1 "$CAGS_PYTHON" "$CAGS_EXTENSION/runner.py" "$CAGS_STAGE" --task-id "${SLURM_ARRAY_TASK_ID:-0}"
