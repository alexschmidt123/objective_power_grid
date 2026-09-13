#!/bin/bash
# Reset-bank design audits are retired. Numerical objective checks remain reusable.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:---help}" in
  alignment) shift; exec python3 tools/check_mocu_alignment.py "$@" ;;
  --help|-h) echo "Only alignment is retained. Reset-bank space/master audits are retired; use the reset backup for historical work." ;;
  *) echo "Retired reset-bank audit. The active project uses online non-reset experiments." >&2; exit 2 ;;
esac
