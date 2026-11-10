#!/usr/bin/env bash
set -euo pipefail
CAGS_EXTENSION=${1:?Missing isolated extension directory}
CAGS_PYTHON=${2:?Missing Python interpreter}
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/cags-crossarch-matplotlib
unset PYTHONPATH
cd "$CAGS_EXTENSION/code"
exec srun --ntasks=1 --kill-on-bad-exit=1 "$CAGS_PYTHON" "$CAGS_EXTENSION/runner.py" summarize
