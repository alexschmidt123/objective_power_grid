#!/bin/bash
# Fast repository invariants followed by the targeted regression suite.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 tools/check_repository.py
exec python3 -m unittest discover -s tools/tests -p 'test_*.py' -v
