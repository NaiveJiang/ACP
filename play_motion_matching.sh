#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV="${ROOT}/.venv_preview/bin/uv"
if [[ ! -x "$UV" ]]; then
  UV="$(command -v uv || true)"
fi
if [[ -z "$UV" ]]; then
  echo "Install the preview environment described in README_MOTION_MATCHING.md." >&2
  exit 1
fi
cd "$ROOT"
exec "$UV" run --no-project --python "$ROOT/.venv_preview/bin/python" \
  tools/play_motion_matching.py "$@"
