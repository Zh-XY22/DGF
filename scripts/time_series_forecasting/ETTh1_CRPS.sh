#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-train}
if [ "$#" -gt 0 ]; then shift; fi
case "$ACTION" in
  test|evaluate) COMMAND=evaluate ;;
  train|prepare|sft|initialize|prepare-modes|rl|export-config) COMMAND=$ACTION ;;
  *) echo "Usage: $0 {train|test|prepare|sft|initialize|prepare-modes|rl|export-config} [CLI flags]" >&2; exit 2 ;;
esac
exec python "$ROOT/scripts/run.py" "$COMMAND" --config "$ROOT/config/etth1_h192_crps.json" "$@"
