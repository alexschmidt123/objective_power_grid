#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../.." && pwd)"
source "$ROOT/run.sh"
exec python3 tools/audit_single_step_eig.py "$@"
