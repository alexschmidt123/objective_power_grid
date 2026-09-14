#!/bin/bash
# Non-reset online sweep. Axes are explicit; no implicit publication expansion.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run.sh"
echo "[sweep_run.sh] Each cell uses its start-date folder: experiments/MMDDYYYY/"
exec python3 -m src.online_cli --sweep "$@"
