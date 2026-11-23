#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WBC_ROOT="${LAFAN_WBC_ROOT:-$ROOT}"
PYTHON_BIN="${FOOTSTEP_PYTHON:-$ROOT/.venv_footstep/bin/python}"
if [[ ! -f "$WBC_ROOT/supermimic/acp/reference_mimic.py" ]]; then
    printf 'WBC reference trainer not found in %s.\n' "$WBC_ROOT" >&2
    exit 1
fi

export ACP_ROBOT=roban
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-2}"

cd "$WBC_ROOT"
exec "$PYTHON_BIN" -m supermimic.acp.train_stage1 \
    --learner blink --task reference --reward supermimic --wm_only_iters 0 \
    --motion_dir "$ROOT/data/roban_lafan1_full_001" \
    --num_envs 1024 --iterations 20000 --wm_hidden 512 512 512 \
    --wm_window 8 --wm_batch 1024 --wm_batches 10 --pi_batch 512 --horizon 32 \
    --buffer_steps 1000 --pi_ema 0.995 --self_collision_weight 1.0 \
    --log_every 25 --ckpt_every 500 "$@"
