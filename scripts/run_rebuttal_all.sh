#!/usr/bin/env bash
# One command: configure once, then run the multi-GPU pipeline in the background.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
CONFIG=${REBUTTAL_CONFIG:-"$ROOT/rebuttal.local.env"}
if [[ -f "$CONFIG" ]]; then source "$CONFIG"; fi
PYTHON=${REBUTTAL_PYTHON:-"$ROOT/ICLR/bin/python"}
MODE=${1:-start}
case "$MODE" in start|--foreground|--worker|--dry-run|--check|--prepare-only) ;; *) echo "Usage: bash $0 [--foreground|--dry-run|--check|--prepare-only]" >&2; exit 2 ;; esac
# A detached child would be killed when the Slurm batch script exits.
if [[ -n ${SLURM_JOB_ID:-} && "$MODE" == start ]]; then MODE=--foreground; fi
[[ -x "$PYTHON" ]] || { echo "找不到环境解释器：$PYTHON" >&2; exit 1; }
export PYTHONUNBUFFERED=1
for key in HF_ENDPOINT REBUTTAL_DOWNLOAD_TRANSPORT REBUTTAL_DOWNLOAD_RETRIES REBUTTAL_DATASET_REPO; do
    if [[ -v "$key" ]]; then export "$key"; fi
done

# Data preparation is an explicit action for the core allocation suite.
if [[ ${FINAL_SUITE:-full} == allocation && "$MODE" == --prepare-only ]]; then
    exec env REBUTTAL_CONFIG="$CONFIG" REBUTTAL_PYTHON="$PYTHON" bash "$ROOT/scripts/prepare_rebuttal_core.sh"
fi
if [[ ${FINAL_SUITE:-full} == allocation && "$MODE" == --check ]]; then
    exec env REBUTTAL_CONFIG="$CONFIG" REBUTTAL_PYTHON="$PYTHON" bash "$ROOT/scripts/prepare_rebuttal_core.sh" --verify
fi

if [[ "$MODE" == --prepare-only ]]; then
    "$PYTHON" -m rebuttal.preflight --resources-only
else
    if [[ -n ${SLURM_JOB_ID:-} ]]; then
        [[ -n ${CUDA_VISIBLE_DEVICES:-} && ${CUDA_VISIBLE_DEVICES:-} != -1 ]] || {
            echo '当前 Slurm 作业没有可见 GPU；请使用 scripts/submit_rebuttal.sh 申请 GPU 作业。' >&2
            exit 1
        }
        SELECTED_GPUS=$CUDA_VISIBLE_DEVICES
    else
        SELECTED_GPUS=${REBUTTAL_GPUS:-${CUDA_VISIBLE_DEVICES-${GPUS:-auto}}}
    fi
    if [[ "$MODE" != --worker ]]; then
        SELECTED_GPUS=$("$PYTHON" -m rebuttal.preflight --gpus "$SELECTED_GPUS")
    fi
    export REBUTTAL_GPUS=$SELECTED_GPUS
fi
if [[ "$MODE" == --check ]]; then
    env DATA_ROOT="${DATA_ROOT:-}" REPORT_DIR="${REPORT_DIR:-}" DIT_CKPT="${DIT_CKPT:-}" VAE_PATH="${VAE_PATH:-}" \
        "$PYTHON" -m rebuttal.resources inspect --class-file "${CLASS_FILE:-$ROOT/misc/class100.txt}" \
        --nclass "${NCLASS:-100}" --ipc "${IPC:-10}"
    exit 0
fi

# Resource discovery/download precedes detached training. Missing paths no longer
# require the user to locate model files manually. Existing valid paths are reused.
if [[ ${REBUTTAL_LEGACY_GUARD:-0} == 1 && "$MODE" == --worker ]]; then
    : # The parent already checked resources before acquiring the shared-filesystem guard.
elif [[ ${FINAL_SUITE:-full} == allocation && "$MODE" != --dry-run ]]; then
    env REBUTTAL_CONFIG="$CONFIG" REBUTTAL_PYTHON="$PYTHON" bash "$ROOT/scripts/prepare_rebuttal_core.sh" --verify
    if [[ -f "$CONFIG" ]]; then source "$CONFIG"; fi
elif [[ "$MODE" != --worker && "$MODE" != --dry-run ]]; then
    echo '自动检查本地数据和模型；缺失项将下载，已有完整资源将复用。'
    resources_ready=0
    declare -A tried_endpoints=()
    for resource_endpoint in "${HF_ENDPOINT:-https://hf-mirror.com}" https://hf-mirror.com https://huggingface.co; do
        if [[ -n ${tried_endpoints[$resource_endpoint]:-} ]]; then continue; fi
        tried_endpoints[$resource_endpoint]=1
        printf '资源下载地址：%s\n' "$resource_endpoint"
        if env HF_ENDPOINT="$resource_endpoint" HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
            HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-15}" HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}" \
            DATA_ROOT="${DATA_ROOT:-}" REPORT_DIR="${REPORT_DIR:-}" DIT_CKPT="${DIT_CKPT:-}" VAE_PATH="${VAE_PATH:-}" \
            "$PYTHON" -m rebuttal.resources ensure --config "$CONFIG" \
            --class-file "${CLASS_FILE:-$ROOT/misc/class100.txt}" --nclass "${NCLASS:-100}" --ipc "${IPC:-10}"; then
            resources_ready=1
            export HF_ENDPOINT="$resource_endpoint"
            break
        else
            resource_exit=$?
            if [[ "$resource_exit" != 75 ]]; then exit "$resource_exit"; fi
            echo '下载连接失败，保留缓存并切换地址重试。' >&2
        fi
    done
    if [[ "$resources_ready" != 1 ]]; then
        echo '镜像和官方地址均无法连接。请检查该节点网络或代理；详细错误见 resources/last_error.json。' >&2
        exit 75
    fi
    if [[ -f "$CONFIG" ]]; then source "$CONFIG"; fi
    if [[ "$MODE" == --prepare-only ]]; then
        echo "资源准备完成，配置已保存：$CONFIG"
        echo "请在 GPU 节点运行：bash $ROOT/scripts/run_rebuttal_all.sh"
        exit 0
    fi
fi

need_path() {
    local key=$1 label=$2 value=${!1:-}
    if [[ -z "$value" || "$value" == /path/to/* ]]; then
        if [[ ! -t 0 ]]; then
            echo "缺少 $key。请在终端直接执行本启动命令，首次运行会询问路径；或填写 $CONFIG。" >&2
            exit 1
        fi
        read -r -p "$label（绝对路径）: " value
    fi
    [[ -n "$value" ]] || { echo "$key 不能为空" >&2; exit 1; }
    printf -v "$key" '%s' "$value"
}

need_path DATA_ROOT 'ImageNet / ImageNet-100 根目录，包含 train 和 val'
if [[ -z ${REPORT_DIR:-} || ${REPORT_DIR:-} == /path/to/* ]]; then REPORT_DIR="$DATA_ROOT/val"; fi
need_path REPORT_DIR '最终验证集目录'
need_path DIT_CKPT 'DiT-XL-2-256x256.pt 文件'
need_path VAE_PATH 'VAE 模型目录'
[[ -d "$DATA_ROOT/train" ]] || { echo "不存在 $DATA_ROOT/train" >&2; exit 1; }
[[ -d "$REPORT_DIR" ]] || { echo "不存在验证集目录：$REPORT_DIR" >&2; exit 1; }
[[ -f "$DIT_CKPT" ]] || { echo "不存在 DiT 权重：$DIT_CKPT" >&2; exit 1; }
WORKDIR=${WORKDIR:-"$ROOT/results/rebuttal_v1"}

IFS=',' read -r -a GPU_IDS <<< "$SELECTED_GPUS"
[[ ${#GPU_IDS[@]} -gt 0 ]] || { echo '未检测到 GPU。' >&2; exit 1; }
for gpu in "${GPU_IDS[@]}"; do
    [[ "$gpu" =~ ^[A-Za-z0-9-]+$ && "$gpu" != -1 ]] || { echo "无效 GPU 编号：$gpu" >&2; exit 1; }
done
export REBUTTAL_GPUS=$SELECTED_GPUS
export REBUTTAL_CONFIG=$CONFIG
export REBUTTAL_PYTHON=$PYTHON
RUN_SENSITIVITY=${RUN_SENSITIVITY:-1}
RUN_CFG_CONTROLS=${RUN_CFG_CONTROLS:-0}
if [[ ${FINAL_SUITE:-full} == allocation ]]; then
    RUN_SENSITIVITY=0
    RUN_CFG_CONTROLS=0
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
export RUN_SENSITIVITY RUN_CFG_CONTROLS

if [[ "$MODE" == --dry-run ]]; then
    printf 'GPU: %s\nWORKDIR: %s\n' "$SELECTED_GPUS" "$WORKDIR"
    if [[ ${FINAL_SUITE:-full} == allocation ]]; then
        echo '执行顺序：smoke → prepare（仅 VAE 默认配置）→ final（allocation 六组对照）'
    else
        printf '执行顺序：smoke → prepare → tune → final；sensitivity=%s，cfg_controls=%s\n' "$RUN_SENSITIVITY" "$RUN_CFG_CONTROLS"
    fi
    echo '这是预检查，没有启动实验或写入配置。'
    exit 0
fi

# Persist resolved paths so detached workers never need terminal input. Retain
# all user-specific settings already present in the configuration.
if [[ "$MODE" != --worker ]]; then
    mkdir -p "$WORKDIR/launcher_backups"
    stamp=$(date +%Y%m%dT%H%M%S)_$$
    if [[ -f "$CONFIG" ]]; then cp -- "$CONFIG" "$WORKDIR/launcher_backups/config_${stamp}.env"; fi
    config_temp=$(mktemp "${CONFIG}.XXXXXX")
    if [[ -f "$CONFIG" ]]; then cat -- "$CONFIG" > "$config_temp"; fi
    {
        printf '\n# Paths resolved by the one-command launcher\n'
        for key in DATA_ROOT REPORT_DIR DIT_CKPT VAE_PATH WORKDIR; do printf '%s=%q\n' "$key" "${!key}"; done
        printf 'GPUS=%q\n' "$SELECTED_GPUS"
        for key in ARCHITECTURES EPOCHS NCLASS IPC GEN_BATCH_SIZE TRAIN_BATCH_SIZE FEATURE_BATCH_SIZE NUM_WORKERS THREADS FINAL_SUITE CHECKPOINT_RULE PRECISION LOCAL_ONLY DETERMINISTIC TUNE_GEN_SEEDS FINAL_GEN_SEEDS TRAIN_SEEDS PERMUTATION_SEEDS CLASS_FILE RESNET_FEATURE_CKPT HOLDOUT_FRACTION PREPARE_SEED; do
            if [[ -v "$key" ]]; then printf '%s=%q\n' "$key" "${!key}"; fi
        done
    } >> "$config_temp"
    mv -- "$config_temp" "$CONFIG"
    tar -czf "$WORKDIR/launcher_backups/launcher_${stamp}.tar.gz" \
        scripts/run_rebuttal_all.sh scripts/rebuttal.sh rebuttal.example.env REBUTTAL_RUN_GUIDE.txt
    # The Python source backup also includes the Slurm submission/job scripts.
fi

run_pipeline() {
    mkdir -p "$WORKDIR"
    stage=initializing
    trap 'rc=$?; printf "exit_code=%s\nlast_stage=%s\n" "$rc" "$stage" > "$WORKDIR/pipeline_exit.txt"' EXIT
    printf '开始：%s\nGPU：%s\n工作目录：%s\n' "$(date --iso-8601=seconds)" "$SELECTED_GPUS" "$WORKDIR"
    stage=backup
    bash "$ROOT/scripts/rebuttal.sh" backup
    stage=smoke
    bash "$ROOT/scripts/rebuttal.sh" smoke
    stage=prepare
    if [[ -f "$WORKDIR/prepared/manifest.json" ]]; then
        "$PYTHON" -c 'import os,sys; from rebuttal.common import read_json; p=read_json(sys.argv[1]); assert os.path.samefile(p["root"],sys.argv[2]), "缓存对应另一数据路径，请更换 WORKDIR"' \
            "$WORKDIR/prepared/split.json" "$DATA_ROOT"
        echo '复用已有准备缓存。'
    else
        bash "$ROOT/scripts/rebuttal.sh" prepare
    fi
    if [[ ${FINAL_SUITE:-full} == core || ${FINAL_SUITE:-full} == full ]]; then
        stage=tune
        if [[ -d "$WORKDIR/runs/tune/queue" ]]; then bash "$ROOT/scripts/rebuttal.sh" retry-failed tune; fi
        bash "$ROOT/scripts/rebuttal.sh" tune
    fi
    stage=final
    if [[ -d "$WORKDIR/runs/final/queue" ]]; then bash "$ROOT/scripts/rebuttal.sh" retry-failed final; fi
    bash "$ROOT/scripts/rebuttal.sh" final
    if [[ "$RUN_SENSITIVITY" == 1 ]]; then
        stage=sensitivity
        bash "$ROOT/scripts/rebuttal.sh" plan-sensitivity
        bash "$ROOT/scripts/rebuttal.sh" retry-failed sensitivity
        bash "$ROOT/scripts/rebuttal.sh" workers sensitivity
        bash "$ROOT/scripts/rebuttal.sh" summary sensitivity
    fi
    if [[ "$RUN_CFG_CONTROLS" == 1 ]]; then
        stage=cfg_controls
        bash "$ROOT/scripts/rebuttal.sh" plan-cfg-controls
        bash "$ROOT/scripts/rebuttal.sh" retry-failed cfg_controls
        bash "$ROOT/scripts/rebuttal.sh" workers cfg_controls
        bash "$ROOT/scripts/rebuttal.sh" summary cfg_controls
    fi
    stage=completed
    echo "完成。主结果：$WORKDIR/runs/final/summary.json"
}

if [[ "$MODE" == --worker || "$MODE" == --foreground ]]; then
    if [[ ${REBUTTAL_LEGACY_GUARD:-0} != 1 ]]; then
        exec "$PYTHON" -m rebuttal.coordination --workdir "$WORKDIR" -- \
            bash "$ROOT/scripts/run_rebuttal_all.sh" --worker
    fi
    run_pipeline
else
    mkdir -p "$WORKDIR"
    log="$WORKDIR/pipeline_$(date +%Y%m%dT%H%M%S)_$$.log"
    nohup bash "$ROOT/scripts/run_rebuttal_all.sh" --worker > "$log" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" > "$WORKDIR/pipeline.pid"
    printf '已后台启动。GPU：%s；启动器 PID：%s\n日志：%s\n' "$SELECTED_GPUS" "$pid" "$log"
fi
