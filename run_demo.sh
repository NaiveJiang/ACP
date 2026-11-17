#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
PYTHON_BIN="${ACP_PYTHON:-/home/liujunjie/InstinctMJ/.venv/bin/python}"
exec "$PYTHON_BIN" -m supermimic.acp.demo_reference "$@"
