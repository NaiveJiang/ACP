#!/usr/bin/env bash
# Install the current RoBan contact/planner/Blink environment, not Isaac Lab.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$ROOT/.venv_footstep"
DEVICE=cuda
SYSTEM_DEPS=0
CHECK_ONLY=0
INDEX="${INSTALL_PYPI_INDEX:-https://pypi.org/simple}"
UV_VERSION=0.12.9

usage() {
  cat <<'EOF'
用法：bash install_env.sh [选项]
  --system-deps      同时用 apt 安装编译器、EGL/OpenGL 等系统库（需要 sudo/root）
  --env PATH         环境目录，默认项目中的 .venv_footstep
  --cpu              安装 CPU 版 PyTorch，用于离线规划器/CPU 播放，不支持 GPU 快速训练
  --check-only       只检查已有环境，不安装包或系统库
  --index-url URL    普通 Python 包索引；PyTorch 后端仍使用官方索引
  -h, --help         显示帮助

自动准备 uv、Python 3.11，并安装固定版本依赖；不安装 NVIDIA 驱动或启动训练。
EOF
}
die() { echo "错误：$*" >&2; exit 1; }
while (($#)); do
  case "$1" in
    --system-deps) SYSTEM_DEPS=1; shift ;;
    --cpu) DEVICE=cpu; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --env|--index-url)
      (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || die "$1 缺少参数"
      if [[ "$1" == --env ]]; then ENV_DIR="$2"; else INDEX="$2"; fi
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "未知参数 $1（使用 --help 查看帮助）" ;;
  esac
done
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "当前固定环境支持 Linux x86_64；Windows 请使用 WSL2 Linux。"
ENV_DIR="$(realpath -m -- "$ENV_DIR")"
ENV_PYTHON="$ENV_DIR/bin/python"
export PYTHONNOUSERSITE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export ACP_ROBOT=roban

if ((CHECK_ONLY)); then
  [[ -f "$ENV_DIR/pyvenv.cfg" && -x "$ENV_PYTHON" ]] || die "找不到虚拟环境 $ENV_DIR"
  if command -v uv >/dev/null 2>&1; then uv pip check --python "$ENV_PYTHON"; fi
  exec "$ENV_PYTHON" "$ROOT/tools/check_footstep_env.py" --device "$DEVICE"
fi

if ((SYSTEM_DEPS)); then
  command -v apt-get >/dev/null 2>&1 || die "--system-deps 目前支持 Ubuntu/Debian 的 apt-get。"
  APT_PREFIX=()
  if ((EUID != 0)); then
    command -v sudo >/dev/null 2>&1 || die "安装系统库需要 sudo 或 root。"
    APT_PREFIX=(sudo)
  fi
  "${APT_PREFIX[@]}" apt-get update
  "${APT_PREFIX[@]}" apt-get install -y curl ca-certificates git build-essential \
    libegl1 libgl1 libglfw3 libglib2.0-0 libgomp1
fi
if [[ "$DEVICE" == cuda ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || die "找不到 NVIDIA 驱动；请先安装可用驱动，或用 --cpu 配置 CPU 环境。"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || die "NVIDIA 驱动不可用；Docker 内还需暴露 GPU。"
  command -v g++ >/dev/null 2>&1 || die "Torch 编译需要 g++；可使用 --system-deps 自动安装。"
fi
if [[ -e "$ENV_DIR" && ! -f "$ENV_DIR/pyvenv.cfg" ]]; then
  die "$ENV_DIR 已存在且不是虚拟环境；请用 --env 指定新目录。"
fi

UV="$(command -v uv || true)"
# Use an existing capable uv, otherwise install a pinned project-local binary.
if [[ -z "$UV" ]] || ! "$UV" pip install --help | grep -- '--torch-backend' >/dev/null; then
  UV="$ROOT/.cache/installer/uv/uv"
  if [[ ! -x "$UV" ]]; then
    INSTALLER_TMP="$(mktemp)"
    trap 'rm -f -- "$INSTALLER_TMP"' EXIT
    URL="https://astral.sh/uv/$UV_VERSION/install.sh"
    if command -v curl >/dev/null 2>&1; then
      curl --fail --location --retry 3 "$URL" --output "$INSTALLER_TMP"
    elif command -v wget >/dev/null 2>&1; then
      wget --tries=3 --output-document="$INSTALLER_TMP" "$URL"
    else
      die "需要 curl 或 wget；Ubuntu/Debian 可用 --system-deps 安装。"
    fi
    UV_UNMANAGED_INSTALL="$ROOT/.cache/installer/uv" sh "$INSTALLER_TMP"
  fi
fi
"$UV" --version
if [[ ! -f "$ENV_DIR/pyvenv.cfg" ]]; then
  echo "创建 Python 3.11 环境：$ENV_DIR"
  "$UV" venv --python 3.11 "$ENV_DIR"
fi
"$ENV_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11), "请使用 Python 3.11；现有环境未修改，请指定新的 --env 目录"'
BACKEND=cu128
[[ "$DEVICE" != cpu ]] || BACKEND=cpu
echo "安装固定依赖，PyTorch 后端：$BACKEND（首次下载可能需要较长时间）"
"$UV" pip install --python "$ENV_PYTHON" --torch-backend "$BACKEND" \
  --default-index "$INDEX" \
  --requirement "$ROOT/requirements-install.txt" \
  --constraint "$ROOT/requirements-footstep-tracker.lock"
"$UV" pip check --python "$ENV_PYTHON"
"$ENV_PYTHON" "$ROOT/tools/check_footstep_env.py" --device "$DEVICE" \
  --report "$ENV_DIR/install-report.json"
echo "环境安装完成：$ENV_PYTHON"
if [[ "$ENV_DIR" != "$ROOT/.venv_footstep" ]]; then
  printf '使用自定义环境运行接触任务前执行：export FOOTSTEP_PYTHON=%q\n' "$ENV_PYTHON"
fi
echo "未启动训练；训练数据、机器人资产和 checkpoint 不由此脚本下载。"
