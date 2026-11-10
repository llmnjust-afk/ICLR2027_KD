#!/usr/bin/env bash
set -euo pipefail
CAGS_PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$CAGS_PROJECT/experiments/rebuttal_crossarch_a800_v1/submit.sh" "$@"
