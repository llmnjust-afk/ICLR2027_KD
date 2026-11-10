#!/usr/bin/env bash
set -euo pipefail
CAGS_EXTENSION=${1:?Missing isolated extension directory}
CAGS_PYTHON=${2:?Missing Python interpreter}
: "${SLURM_JOB_ID:?Submit through the supplied launcher}"
: "${SLURM_ARRAY_TASK_ID:?Missing Slurm array task ID}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR=/tmp/cags-crossarch-matplotlib
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONPATH
cd "$CAGS_EXTENSION/code"
printf 'Cross-architecture task %s on %s; started %s\n' "$SLURM_ARRAY_TASK_ID" "$(hostname)" "$(date -Is)"
exec srun --ntasks=1 --kill-on-bad-exit=1 "$CAGS_PYTHON" "$CAGS_EXTENSION/runner.py" worker --task-id "$SLURM_ARRAY_TASK_ID"
