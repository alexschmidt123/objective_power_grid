#!/bin/bash
# Reusable diagnostic entrypoint; audit output is not a formal method result.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:---help}" in
  space) TOOL=tools/audit_mocu_space.py ;;
  master) TOOL=tools/audit_ieee9_master_search.py ;;
  preflight) TOOL=tools/mocu_preflight.py ;;
  alignment) TOOL=tools/check_mocu_alignment.py ;;
  --help|-h) echo "Usage: bash scripts/audit.sh {space|master|preflight|alignment} [arguments]"; exit 0 ;;
  *) echo "Unknown audit: $1" >&2; exit 2 ;;
esac
shift
exec python3 "$TOOL" "$@"
