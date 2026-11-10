#!/usr/bin/env bash
set -euo pipefail
CAGS_PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec "$CAGS_PROJECT/ICLR/bin/python" "$CAGS_PROJECT/experiments/rebuttal_missing_v1/runner.py" submit "$@"
