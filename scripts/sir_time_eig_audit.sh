#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/run.sh"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
exec python3 tools/audit_sir_time_eig.py "$@"
