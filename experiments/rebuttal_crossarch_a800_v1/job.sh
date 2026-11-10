#!/usr/bin/env bash
set -euo pipefail
CAGS_EXTENSION=${1:?Missing extension directory}
CAGS_PYTHON=${2:?Missing interpreter}
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${SLURM_ARRAY_TASK_ID:?Missing array task ID}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR=/tmp/cags-a800-matplotlib
unset PYTHONPATH
printf 'A800 cross-architecture task %s on %s; started %s\n' "$SLURM_ARRAY_TASK_ID" "$(hostname)" "$(date -Is)"
exec srun --ntasks=1 --kill-on-bad-exit=1 "$CAGS_PYTHON" "$CAGS_EXTENSION/runner.py" worker --task-id "$SLURM_ARRAY_TASK_ID"
