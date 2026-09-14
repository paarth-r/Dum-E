#!/usr/bin/env bash
# Launch the dum-e interactive sim (PyBullet GUI, keyboard-driven, grabbable box).
# Usage:  ./run_sim.sh                # keyboard + scene (default)
#         ./run_sim.sh --noise 0.5    # extra flags pass straight through
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/dume
if [ ! -x "$PY" ]; then
  echo "dume entry point not found at $PY — run 'uv pip install -e .' (or activate the venv)." >&2
  exit 1
fi

# Defaults: keyboard control, demo scene. Append/override via "$@".
exec "$PY" sim --keyboard --scene "$@"
