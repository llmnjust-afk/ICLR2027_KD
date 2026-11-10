#!/usr/bin/env bash
# Keep the first-batch allocation study separate from the full rebuttal suite.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export REBUTTAL_CONFIG=${REBUTTAL_CONFIG:-"$ROOT/rebuttal.core.env"}
cd "$ROOT"
set -a
source "$REBUTTAL_CONFIG"
set +a
export REBUTTAL_PYTHON=${REBUTTAL_PYTHON:-"$ROOT/ICLR/bin/python"}
exec "$REBUTTAL_PYTHON" -m rebuttal.slurm submit "$@"
