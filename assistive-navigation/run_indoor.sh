#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="../.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python"
fi
exec "$PY" main.py --mode indoor --no-mirror-input "$@"
