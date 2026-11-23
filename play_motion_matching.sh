#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export ACP_ROBOT=roban
export PYTHONNOUSERSITE=1
PYTHON="${FOOTSTEP_PYTHON:-$ROOT/.venv_footstep/bin/python}"

mode="${1:-}"
if [[ "$mode" == --s17-reference || "$mode" == --s17-search || "$mode" == --s17-plan ]]; then
    shift
    SOURCE="${S17_REFERENCE_SOURCE:-$HOME/Motion-Matching-main}"
    if [[ ! -f "$SOURCE/s17_controller.cpp" ]]; then
        printf 'Cannot find s17_controller.cpp in %s\n' "$SOURCE" >&2
        exit 1
    fi

    build_dir="$(mktemp -d -t roban-s17-preview.XXXXXX)"
    trap 'rm -rf -- "$build_dir"' EXIT
    prepared_args=()
    preview=""
    if [[ "$mode" != --s17-reference ]]; then
        preview="$ROOT/logs/motion_matching_upstream_roban_002"
    fi
    if [[ ${1:-} == --upstream-roban ]]; then
        shift
        preview="${preview:-$ROOT/logs/motion_matching_upstream_roban_001}"
        if [[ $# -gt 0 && $1 != -* ]]; then
            preview="$1"
            shift
        fi
    fi
    if [[ -n "$preview" ]]; then
        "$PYTHON" - "$preview" "$build_dir/mapped" <<'PY'
from pathlib import Path
import sys

from tools.motion_matching_reference import write_native_catalogue

write_native_catalogue(Path(sys.argv[1]), Path(sys.argv[2]))
PY
        prepared_args=(--prepared-motion "$build_dir/mapped/clips.txt")
    fi
    mujoco_dir="$("$PYTHON" -c 'import mujoco; from pathlib import Path; print(Path(mujoco.__file__).parent)')"
    shopt -s nullglob
    mujoco_libraries=("$mujoco_dir"/libmujoco.so*)
    if (( ${#mujoco_libraries[@]} == 0 )); then
        printf 'MuJoCo shared library not found in %s\n' "$mujoco_dir" >&2
        exit 1
    fi

    "$PYTHON" - "$build_dir/scene.xml" <<'PY'
from pathlib import Path
import sys

import mujoco

from supermimic.acp.roban import get_spec

spec = get_spec()
spec.add_texture(name="reference_grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                 builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=64, height=64,
                 rgb1=[.27, .31, .35], rgb2=[.20, .24, .28])
spec.add_material(name="reference_floor", textures=["reference_grid"], texrepeat=[1., 1.], texuniform=True)
spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                        size=[0., 0., .25], material="reference_floor")
spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1])
Path(sys.argv[1]).write_text(spec.to_xml())
PY

    if [[ "$mode" != --s17-reference ]]; then
        "${CXX:-g++}" -std=c++17 -O3 -Wno-unused-result -fPIC -shared -DS17_SEARCH_LIBRARY -I"$mujoco_dir/include" \
            "$SOURCE/s17_controller.cpp" "${mujoco_libraries[0]}" -Wl,-rpath,"$mujoco_dir" -lglfw \
            -o "$build_dir/s17_search.so"
        module="supermimic.acp.footstep.motion_matching"
        if [[ "$mode" == --s17-plan ]]; then
            module="supermimic.acp.footstep.reference_planner"
        fi
        "$PYTHON" -m "$module" --upstream-roban "$preview" \
            --s17-search-library "$build_dir/s17_search.so" "$build_dir/scene.xml" "$build_dir/mapped/clips.txt" "$@"
        exit
    fi

    cat > "$build_dir/gamepads.txt" <<'MAPPINGS'
030003e5bc2000007e50000011010000,BEITONG A1N3 BFM GAMEPAD,a:b0,b:b1,x:b3,y:b4,back:b10,guide:b12,start:b11,leftstick:b13,rightstick:b14,leftshoulder:b6,rightshoulder:b7,dpup:h0.1,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,leftx:a0,lefty:a1,rightx:a2,righty:a3,lefttrigger:a5,righttrigger:a4,platform:Linux,
MAPPINGS
    "${CXX:-g++}" -std=c++17 -O3 -Wno-unused-result -I"$mujoco_dir/include" \
        "$SOURCE/s17_controller.cpp" "${mujoco_libraries[0]}" -Wl,-rpath,"$mujoco_dir" -lglfw \
        -o "$build_dir/controller"
    cd "$SOURCE"
    "$build_dir/controller" --motion-dir "$HOME/roban数据/lafan1" --scene "$build_dir/scene.xml" \
        --gamepad-db "$build_dir/gamepads.txt" "${prepared_args[@]}" "$@"
else
    exec "$PYTHON" -m supermimic.acp.footstep.motion_matching "$@"
fi
