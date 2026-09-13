#!/bin/bash
# Online non-reset IEEE9 experiment for EIG, MSC and MOCU.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/run.sh"
exec python3 -m src.objectives.eig.continuous_eig "$@"
