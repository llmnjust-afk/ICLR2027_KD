#!/usr/bin/env bash
set -euo pipefail
CAGS_PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONNOUSERSITE=1
unset PYTHONPATH
exec "$CAGS_PROJECT/ICLR/bin/python" "$CAGS_PROJECT/experiments/rebuttal_mgd3_4090_v1/runner.py" submit "$@"
