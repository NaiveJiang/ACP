#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export ACP_ROBOT=roban
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$ROOT/.venv_preview/bin/uv" run --no-project \
  --python "$ROOT/.venv_blink/bin/python" python -m supermimic.acp.train_stage1 \
  --learner blink --motion_dir "$ROOT/data/roban_blink_contact/train" \
  --contacts_dir "$ROOT/data/roban_blink_contact/train_contacts" "$@"
