#!/bin/bash
# Online non-reset IEEE9 experiment for EIG and cost_utility.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/run.sh"
exec python3 -m src.objectives.eig.continuous_eig "$@"
