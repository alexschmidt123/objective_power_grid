#!/bin/bash
# Formal entrypoint for online IEEE9 EIG with continuous duration and state carry.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source run.sh
exec python3 -m src.objectives.eig.continuous_eig "$@"
