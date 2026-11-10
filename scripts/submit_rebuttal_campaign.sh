#!/usr/bin/env bash
# Schedule both batches, with no more than four GPU jobs eligible at once.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON=${REBUTTAL_PYTHON:-"$ROOT/ICLR/bin/python"}
exec "$PYTHON" -m rebuttal.campaign "$@"
