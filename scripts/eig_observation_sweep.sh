#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/run.sh"
exec python3 tools/audit_eig_observation_sweep.py "$@"
