#!/bin/bash
# Plot existing results or a reference network; never train or evaluate.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:---help}" in
  results) shift; exec python3 -m src.results.plots "$@" ;;
  network) shift; exec python3 tools/render_network.py "$@" ;;
  --help|-h) echo "Usage: bash scripts/visualization.sh {results|network} [arguments]" ;;
  *) echo "Unknown visualization: $1" >&2; exit 2 ;;
esac
