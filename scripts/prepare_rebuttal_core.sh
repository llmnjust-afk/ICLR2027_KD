#!/usr/bin/env bash
# Explicit CPU-only preparation. Never submits jobs or downloads model weights.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
CONFIG=${REBUTTAL_CONFIG:-"$ROOT/rebuttal.core.env"}
[[ -f "$CONFIG" ]] || { echo "找不到配置：$CONFIG" >&2; exit 1; }
source "$CONFIG"
PYTHON=${REBUTTAL_PYTHON:-"$ROOT/ICLR/bin/python"}
[[ -x "$PYTHON" ]] || { echo "找不到环境解释器：$PYTHON" >&2; exit 1; }
for key in DATA_ROOT REPORT_DIR DIT_CKPT VAE_PATH CLASS_FILE HF_ENDPOINT REBUTTAL_DOWNLOAD_RETRIES REBUTTAL_DATA_CACHE; do
    if [[ -v "$key" ]]; then export "$key"; fi
done
export PYTHONUNBUFFERED=1
exec "$PYTHON" -m rebuttal.data_setup "$@"
