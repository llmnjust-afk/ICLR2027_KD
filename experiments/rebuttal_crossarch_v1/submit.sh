#!/usr/bin/env bash
set -euo pipefail
CAGS_EXTENSION=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAGS_PROJECT=$(cd -- "$CAGS_EXTENSION/../.." && pwd)
case "${1:-}" in
  '') exec "$CAGS_PROJECT/ICLR/bin/python" "$CAGS_EXTENSION/runner.py" submit ;;
  --dry-run) exec "$CAGS_PROJECT/ICLR/bin/python" "$CAGS_EXTENSION/runner.py" submit --dry-run ;;
  --status) exec "$CAGS_PROJECT/ICLR/bin/python" "$CAGS_EXTENSION/runner.py" status ;;
  *) printf 'Usage: bash %s [--dry-run|--status]\n' "$0" >&2; exit 2 ;;
esac
