#!/usr/bin/env bash
# sbatch copies this script to its spool: take the project path as an argument.
set -euo pipefail
ROOT=${1:?Missing project root}
export REBUTTAL_CONFIG=${2:?Missing configuration path}
export REBUTTAL_PYTHON=${3:?Missing Python interpreter}
: "${SLURM_JOB_ID:?This entry must be launched by sbatch}"
cd "$ROOT"
unset REBUTTAL_GPUS
printf '作业 %s，节点 %s，启动时间 %s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)"
if [[ -n ${4:-} ]]; then
    set -a
    source "$REBUTTAL_CONFIG"
    set +a
    exec srun --ntasks=1 --kill-on-bad-exit=1 "$REBUTTAL_PYTHON" -m rebuttal.slurm worker --batch "$4" --slot "${5:?Missing worker slot}"
fi
# A foreground srun step keeps the allocation alive and accounts for all workers.
exec srun --ntasks=1 --kill-on-bad-exit=1 bash "$ROOT/scripts/run_rebuttal_all.sh" --foreground
