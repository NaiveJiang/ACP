#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export ACP_ROBOT=roban
export PYTHONNOUSERSITE=1
PYTHON="${FOOTSTEP_PYTHON:-$ROOT/.venv_footstep/bin/python}"
exec "$PYTHON" -m supermimic.acp.footstep.play_tracker "$@"
