#!/usr/bin/env bash
set -euo pipefail
CAGS_EXTENSION=${1:?Missing extension directory}
CAGS_PYTHON=${2:?Missing interpreter}
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/cags-a800-matplotlib
unset PYTHONPATH
exec srun --ntasks=1 --kill-on-bad-exit=1 "$CAGS_PYTHON" "$CAGS_EXTENSION/runner.py" summarize
