#!/usr/bin/env bash
# Independent offline two-foot experiment; never invokes original ACP entrypoints.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${FOOTSTEP_PYTHON:-$ROOT/.venv_footstep/bin/python}"
[[ -x "$PYTHON" ]] || { echo "Missing independent environment: $PYTHON" >&2; exit 1; }
exec "$PYTHON" -m supermimic.acp.footstep.train "$@"
