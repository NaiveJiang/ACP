# RoBan ACP 环境配置

> 本文记录早期分发包的 Python 3.12 / Torch 2.10 环境。当前接触控制器和 Blink 快速训练使用 Python 3.11 / Torch 2.9，新增一键入口 `bash install_env.sh --system-deps`，以 [README_INSTALL.md](README_INSTALL.md) 为准；以下旧环境记录保留。

此分发包仅包含仓库代码、Roban 资产与数据，以及 Stage 2 第 22000 次权重。**不包含虚拟环境、依赖安装目录或 mjlab 源码。** 环境需要在目标机器重新配置，首次安装需要联网。

## 已验证的版本

| 项目 | 版本 / 要求 |
|---|---|
| 操作系统 | Linux x86_64 |
| Python | 3.12；验证版本为 3.12.3 |
| PyTorch | 2.10.0，CUDA 12.8 构建 |
| MuJoCo / MuJoCo-Warp | 3.11.0 / 3.11.0 |
| Warp | 1.14.0，来自 NVIDIA Python 索引 |
| mjlab | 1.6.0，固定源码提交 `0fb8a681136be94ffc636a3dd423cabb97d91f10` |
| NumPy / SciPy | 2.4.2 / 1.17.1 |

其余依赖固定在 `requirements-runtime.txt`，针对 Linux / Python 3.12。GPU 批量仿真和训练需要支持 CUDA 12.8 的 NVIDIA 驱动；单机器人 `--device cpu` 演示可以在没有 NVIDIA GPU 的机器上运行。图形界面需要桌面显示和可用的 OpenGL；无桌面时使用 `--headless`。

## 安装

先安装系统的 Python 3.12、venv 支持和 Git，然后进入本 README 所在目录。下面两种安装方式任选一种。

### 使用 uv

```bash
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python --index-url https://pypi.nvidia.com --no-deps 'warp-lang==1.14.0'
uv pip install --python .venv/bin/python --index-url https://pypi.org/simple -r requirements-runtime.txt
uv pip install --python .venv/bin/python --no-deps \
  'mjlab @ git+https://github.com/mujocolab/mjlab.git@0fb8a681136be94ffc636a3dd423cabb97d91f10'
uv pip check --python .venv/bin/python
```

### 使用 pip

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --index-url https://pypi.nvidia.com --no-deps 'warp-lang==1.14.0'
.venv/bin/python -m pip install --index-url https://pypi.org/simple -r requirements-runtime.txt
.venv/bin/python -m pip install --no-deps \
  'mjlab @ git+https://github.com/mujocolab/mjlab.git@0fb8a681136be94ffc636a3dd423cabb97d91f10'
.venv/bin/python -m pip check
```

先安装 NVIDIA 索引中的 Warp，再安装其他依赖，与验证环境保持一致。`mjlab` 使用固定源码提交，避免仅凭相同版本号安装到不同源码。源码构建的 Python 后端由安装工具自动安装。

## 验证与启动

默认启动脚本使用解压目录中的 `.venv/bin/python`，也可以通过 `ACP_PYTHON=/path/to/python` 指定另一个兼容环境。

```bash
# 查看核心版本及 CUDA 可用性
.venv/bin/python -c 'import torch, mujoco, mjlab; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), mujoco.__version__, mjlab.__file__)'

# 使用随包的 22000 权重，在 CPU 上执行两秒无窗口 planner-tracker 闭环
./run_roban_interactive.sh --headless --device cpu --seconds 2

# 在有桌面环境的机器上运行交互界面
./run_roban_interactive.sh

# 在 NVIDIA GPU 上验证资产、控制器、世界模型梯度和小批量物理仿真
./run_roban.sh smoke_robot --device cuda:0 --steps 8 --out artifacts/package_smoke
```

以上 CPU 闭环、GPU 仿真与梯度检查已在仓库外新建的独立虚拟环境中通过，安装依赖检查也通过。验证使用的 mjlab 来自该固定提交的干净本地源码副本；上述 Git 远程安装方式未在第二个新环境重复执行。

批量闭环测评命令见 `RUN_PACKAGE.txt`。编号 checkpoint 包含 tracker、世界模型和 planner，不含完整优化器与 critic 状态，不能当作完整 `last.pt` 使用 `--resume` 续训。

原始仓库 `README.md` 同时介绍 G1 / Isaac Lab 等路线；本分发包的 Roban 环境以本文为准。`data_collect_repo` 未包含，上游数据采集流程需要另行获取其仓库。
