#!/bin/bash
# Non-reset online sweep. Axes are explicit; no implicit publication expansion.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run.sh"
exec python3 -m src.online_cli --sweep "$@"
