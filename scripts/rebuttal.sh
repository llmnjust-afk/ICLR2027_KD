#!/usr/bin/env bash
set -euo pipefail
REBUTTAL_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REBUTTAL_ROOT"
CONFIG=${REBUTTAL_CONFIG:-"$REBUTTAL_ROOT/rebuttal.local.env"}
if [[ -f "$CONFIG" ]]; then
    # The configuration is an explicit user-owned shell file.
    source "$CONFIG"
fi
PYTHON=${REBUTTAL_PYTHON:-"$REBUTTAL_ROOT/ICLR/bin/python"}
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${THREADS:-4}
export OPENBLAS_NUM_THREADS=${THREADS:-4}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/cags-rebuttal-matplotlib}
WORKDIR=${WORKDIR:-"$REBUTTAL_ROOT/results/rebuttal_v1"}
GPUS=${REBUTTAL_GPUS:-${GPUS:-0}}
ACTION=${1:-help}
NAME=${2:-final}

flags=()
if [[ ${LOCAL_ONLY:-1} == 1 ]]; then flags+=(--local-files-only); fi
if [[ ${DETERMINISTIC:-0} == 1 ]]; then flags+=(--deterministic); fi

require_paths() {
    : "${DATA_ROOT:?Set DATA_ROOT in rebuttal.local.env}"
    : "${DIT_CKPT:?Set DIT_CKPT in rebuttal.local.env}"
    : "${VAE_PATH:?Set VAE_PATH in rebuttal.local.env}"
    [[ -f "$DIT_CKPT" ]] || { echo "Missing DiT checkpoint: $DIT_CKPT" >&2; exit 1; }
    [[ -d "$DATA_ROOT/train" ]] || { echo "Missing train directory: $DATA_ROOT/train" >&2; exit 1; }
}

make_plan() {
    local stage=$1
    require_paths
    local common=(--workdir "$WORKDIR" --checkpoint "$DIT_CKPT" --vae "$VAE_PATH"
        --epochs "${EPOCHS:-2000}" --generation-batch-size "${GEN_BATCH_SIZE:-4}"
        --batch-size "${TRAIN_BATCH_SIZE:-128}" --num-workers "${NUM_WORKERS:-4}"
        --threads "${THREADS:-4}" --architectures "${ARCHITECTURES:-convnet6}"
        --precision "${PRECISION:-fp32}" "${flags[@]}")
    "$PYTHON" -m rebuttal.run backup
    if [[ "$stage" == tune ]]; then
        "$PYTHON" -m rebuttal.run plan "${common[@]}" --name tune --suite tune \
            --generation-seeds "${TUNE_GEN_SEEDS:-0,1,2}" --training-seeds 0
    else
        local suite=${FINAL_SUITE:-full}
        if [[ "$stage" == sensitivity || "$stage" == cfg_controls || "$stage" == factors ]]; then suite=$stage; fi
        local extra=()
        if [[ "$suite" == core || "$suite" == full ]]; then
            extra+=(--selection "$WORKDIR/runs/tune/selection.json")
        fi
        "$PYTHON" -m rebuttal.run plan "${common[@]}" --name "$stage" --suite "$suite" \
            --generation-seeds "${FINAL_GEN_SEEDS:-100,101,102,103,104}" \
            --training-seeds "${TRAIN_SEEDS:-0,1,2}" --permutation-seeds "${PERMUTATION_SEEDS:-17,29,43}" \
            --checkpoint-rule "${CHECKPOINT_RULE:-final}" \
            --report-dir "${REPORT_DIR:-$DATA_ROOT/val}" "${extra[@]}"
    fi
}

launch_stage() {
    local stage=$1
    local run=$2
    local queue_plan="$WORKDIR/runs/$run"
    local hostname_safe
    hostname_safe=$(hostname | tr -cd 'A-Za-z0-9_-')
    local logdir="$queue_plan/worker_logs"
    mkdir -p "$logdir"
    local ids=() pids=() gpu status=0
    IFS=',' read -r -a ids <<< "$GPUS"
    for gpu in "${ids[@]}"; do
        [[ "$gpu" =~ ^[A-Za-z0-9-]+$ ]] || { echo "Invalid GPU identifier: $gpu" >&2; exit 1; }
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m rebuttal.run worker \
            --plan "$queue_plan" --stage "$stage" \
            > "$logdir/${hostname_safe}_${stage}_gpu${gpu}_$(date +%Y%m%dT%H%M%S)_$$.log" 2>&1 &
        pids+=("$!")
    done
    trap 'for child in "${pids[@]}"; do kill -TERM "$child" 2>/dev/null || true; done' INT TERM
    for child in "${pids[@]}"; do
        if ! wait "$child"; then status=1; fi
    done
    trap - INT TERM
    "$PYTHON" -m rebuttal.run status --plan "$queue_plan"
    return "$status"
}

run_workers() {
    launch_stage generate "$1"
    launch_stage evaluate "$1"
}

case "$ACTION" in
    smoke)
        require_paths
        IFS=',' read -r -a smoke_ids <<< "$GPUS"
        CUDA_VISIBLE_DEVICES="${smoke_ids[0]}" "$PYTHON" -m rebuttal.run smoke \
            --workdir "$WORKDIR" --checkpoint "$DIT_CKPT" --vae "$VAE_PATH" \
            --generation-batch-size "${GEN_BATCH_SIZE:-4}" --classifier-batch-size "${TRAIN_BATCH_SIZE:-128}" \
            "${flags[@]}"
        ;;
    prepare)
        require_paths
        IFS=',' read -r -a prepare_ids <<< "$GPUS"
        prep_extra=()
        if [[ ${FINAL_SUITE:-full} == allocation ]]; then
            # Only the default VAE estimator is needed for allocation controls.
            prep_extra+=(--k-values '')
        elif [[ -n ${RESNET_FEATURE_CKPT:-} ]]; then
            prep_extra+=(--resnet-checkpoint "$RESNET_FEATURE_CKPT")
        fi
        CUDA_VISIBLE_DEVICES="${prepare_ids[0]}" "$PYTHON" -m rebuttal.run prepare \
            --workdir "$WORKDIR" --data-root "$DATA_ROOT" --vae "$VAE_PATH" \
            --class-file "${CLASS_FILE:-$REBUTTAL_ROOT/misc/class100.txt}" --nclass "${NCLASS:-100}" --ipc "${IPC:-10}" \
            --feature-batch-size "${FEATURE_BATCH_SIZE:-16}" --num-workers "${NUM_WORKERS:-4}" \
            --holdout-fraction "${HOLDOUT_FRACTION:-0.1}" --seed "${PREPARE_SEED:-2027}" \
            "${flags[@]}" "${prep_extra[@]}"
        ;;
    plan-tune) make_plan tune ;;
    plan-final) make_plan final ;;
    plan-sensitivity) make_plan sensitivity ;;
    plan-cfg-controls) make_plan cfg_controls ;;
    plan-factors) make_plan factors ;;
    workers) run_workers "$NAME" ;;
    tune)
        make_plan tune
        run_workers tune
        "$PYTHON" -m rebuttal.run select --plan "$WORKDIR/runs/tune"
        ;;
    select) "$PYTHON" -m rebuttal.run select --plan "$WORKDIR/runs/tune" ;;
    final)
        make_plan final
        run_workers final
        "$PYTHON" -m rebuttal.run summarize --plan "$WORKDIR/runs/final"
        ;;
    status) "$PYTHON" -m rebuttal.run status --plan "$WORKDIR/runs/$NAME" ;;
    summary) "$PYTHON" -m rebuttal.run summarize --plan "$WORKDIR/runs/$NAME" ;;
    retry-failed) "$PYTHON" -m rebuttal.run retry-failed --plan "$WORKDIR/runs/$NAME" ;;
    backup) "$PYTHON" -m rebuttal.run backup ;;
    test) "$PYTHON" -m unittest discover -s tests -p 'test_*.py' -v ;;
    *)
        echo "Usage: bash scripts/rebuttal.sh smoke|prepare|tune|final"
        echo "Multi-node: plan-tune -> workers tune (all hosts) -> select -> plan-final -> workers final (all hosts) -> summary final"
        echo "Other: status NAME | retry-failed NAME | summary NAME | backup | test"
        ;;
esac
