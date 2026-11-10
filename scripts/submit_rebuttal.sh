#!/usr/bin/env bash
# Run on the login node. The allocation suite only verifies local resources.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
CONFIG=${REBUTTAL_CONFIG:-"$ROOT/rebuttal.local.env"}
if [[ -f "$CONFIG" ]]; then source "$CONFIG"; fi
PYTHON=${REBUTTAL_PYTHON:-"$ROOT/ICLR/bin/python"}
MODE=${1:-submit}
case "$MODE" in submit|--dry-run) ;; *) echo "Usage: bash $0 [--dry-run]" >&2; exit 2 ;; esac
command -v sbatch >/dev/null || { echo '缺少 sbatch；请在 Slurm 登录节点执行。' >&2; exit 1; }
[[ -z ${SLURM_JOB_ID:-} ]] || { echo '你已经在 Slurm 作业内，请直接运行 run_rebuttal_all.sh --foreground。' >&2; exit 1; }

count=${REBUTTAL_GPU_COUNT:-1}
[[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo 'REBUTTAL_GPU_COUNT 必须为正整数。' >&2; exit 2; }
partition=${REBUTTAL_SLURM_PARTITION:-rtx4090}
gres=${REBUTTAL_SLURM_GRES:-gpu:4090:$count}
logdir="$ROOT/logs/slurm"
args=(--parsable --job-name=cags-rebuttal --nodes=1 --ntasks=1
      --partition="$partition" --gres="$gres"
      --cpus-per-task="${REBUTTAL_SLURM_CPUS:-$((8 * count))}"
      --mem="${REBUTTAL_SLURM_MEM:-$((32 * count))G}"
      --chdir="$ROOT" --export=ALL
      --output="$logdir/rebuttal-%j.log" --error="$logdir/rebuttal-%j.log")
# Use the account/QOS/time defaults granted by the cluster unless explicitly set.
if [[ -n ${REBUTTAL_SLURM_ACCOUNT:-} ]]; then args+=(--account="$REBUTTAL_SLURM_ACCOUNT"); fi
if [[ -n ${REBUTTAL_SLURM_QOS:-} ]]; then args+=(--qos="$REBUTTAL_SLURM_QOS"); fi
if [[ -n ${REBUTTAL_SLURM_TIME:-} ]]; then args+=(--time="$REBUTTAL_SLURM_TIME"); fi
args+=("$ROOT/scripts/rebuttal_slurm_job.sh" "$ROOT" "$CONFIG" "$PYTHON")
if [[ "$MODE" == --dry-run ]]; then
    if [[ ${FINAL_SUITE:-full} == allocation ]]; then
        printf '只校验本地资源（不下载）：\n  REBUTTAL_CONFIG=%q REBUTTAL_PYTHON=%q bash %q --verify\n' "$CONFIG" "$PYTHON" "$ROOT/scripts/prepare_rebuttal_core.sh"
    else
        printf '先准备共享数据和模型（无需 GPU）：\n  REBUTTAL_CONFIG=%q REBUTTAL_PYTHON=%q bash %q --prepare-only\n' "$CONFIG" "$PYTHON" "$ROOT/scripts/run_rebuttal_all.sh"
    fi
    printf '资源完整后提交（未实际执行）：\n  sbatch'; printf ' %q' "${args[@]}"; printf '\n'
    exit 0
fi

export REBUTTAL_CONFIG="$CONFIG" REBUTTAL_PYTHON="$PYTHON"
if [[ ${FINAL_SUITE:-full} == allocation ]]; then
    bash "$ROOT/scripts/prepare_rebuttal_core.sh" --verify
else
    bash "$ROOT/scripts/run_rebuttal_all.sh" --prepare-only
fi
mkdir -p "$logdir"
# Do not propagate a login shell's stale device mask into the allocation.
submitted=$(env -u CUDA_VISIBLE_DEVICES -u REBUTTAL_GPUS sbatch "${args[@]}")
jobid=${submitted%%;*}
[[ "$jobid" =~ ^[0-9]+$ ]] || { echo "sbatch 返回了非预期结果：$submitted，请用 squeue 核对。" >&2; exit 1; }
printf '%s\n' "$jobid" > "$logdir/latest_job.id"
printf '已提交 Slurm 作业 %s，分区 %s，资源 %s。排队不等于已经开始训练。\n' "$jobid" "$partition" "$gres"
printf '查看队列：squeue -j %s\n查看日志：tail -f %s/rebuttal-%s.log\n' "$jobid" "$logdir" "$jobid"
