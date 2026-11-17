# 一键安装当前训练环境

本入口服务于当前 RoBan 接触控制器、离线规划器和 Blink 加速训练，支持 Linux x86_64（Ubuntu/Debian 可自动安装系统库）。不需要先安装 Python、Conda 或本地 CUDA Toolkit；GPU 机器需要先有可用的 NVIDIA 驱动，容器内还需要暴露 GPU。

## 新设备安装

把仓库克隆到新机器后，在仓库根目录执行：

```bash
bash install_env.sh --system-deps
```

`--system-deps` 使用 apt 安装编译器、Git、curl、EGL/OpenGL 等系统库，普通用户可能需要输入 sudo 密码。已装好这些系统库的机器只需：

```bash
bash install_env.sh
```

脚本自动准备 uv、下载缺失的 Python 3.11、在项目中创建 `.venv_footstep`，并安装固定依赖。已有虚拟环境会复用并补齐/调整依赖，不删除目录；Python 版本不是 3.11 时停止，要求另选目录。不要对正在执行训练的环境运行安装，可改用新目录。

核心版本与当前接触控制训练对应：PyTorch 2.9.0 / torchvision 0.24.0（默认 CUDA 12.8 构建）、mjlab 1.6.0、MuJoCo/MuJoCo-Warp 3.11.0、Warp 1.14.0、NumPy 2.3.4。直接依赖位于 `requirements-install.txt`，传递依赖使用 `requirements-footstep-tracker.lock` 约束，不混用早期 Python 3.12 / Torch 2.10 的环境。

PyTorch 后端通过 uv 的 `--torch-backend cu128` 单独选择官方 wheel，普通依赖来自 PyPI；不会拿普通镜像中的 CPU PyTorch 替代 CUDA 构建。首次安装会下载较大的 GPU 包，可重复运行，已下载的包复用缓存。

安装结束会检查依赖一致性、关键版本和项目导入、CUDA BF16 前后向、CUDA Graph、Warp GPU 和无窗口 EGL 渲染，并保存 `.venv_footstep/install-report.json`。若机器人资产在场，也会编译验证模型。不会启动训练。

## 其他选项

```bash
# 只检查，不安装或改动依赖
bash install_env.sh --check-only

# 指定新环境，不调整当前训练环境
bash install_env.sh --env .venv_footstep_new
export FOOTSTEP_PYTHON="$PWD/.venv_footstep_new/bin/python"

# 普通依赖使用指定镜像，PyTorch 仍使用官方后端索引
bash install_env.sh --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 没有 GPU：安装 CPU PyTorch，仅用于离线规划器和显式 CPU 播放
bash install_env.sh --cpu --env .venv_footstep_cpu
bash install_env.sh --cpu --env .venv_footstep_cpu --check-only
```

CPU 配置不能启动默认 CUDA/BF16 快速训练。当前验证配置为 Linux x86_64；其他系统应另建环境验证兼容性。

## 数据和资产

Git 忽略了 `data/`、`logs/` 和录制数据。依赖安装不会下载它们。从旧设备另外复制：

- `data/roban_description/`：机器人 XML 和 mesh，运行接触控制器必须有。
- 三组采集目录：`full_stop_001`、`forward_continuous_001`、`backward_continuous_001`，需要保留原始记录及 `future_contact_v2` 标签。
- 需要播放或续训时，复制对应的 checkpoint。

没有机器人资产时，检查脚本会明确提示，但依赖安装本身可以完成。恢复资产后重新执行 `--check-only`。训练时用 `--data` 指定新机器上的真实路径，不要照抄 `/home/waao/`。

`.venv` 等目录不能直接复制到其他设备，里面的 Python 路径和二进制依赖可能不匹配；应使用本脚本重建。旧 ACP 启动入口如使用 `ACP_PYTHON`，可显式设置为 `"$PWD/.venv_footstep/bin/python"`；接触任务入口默认已使用 `.venv_footstep`。

v3 训练和播放见 [README_STABLE_CONTACT.md](README_STABLE_CONTACT.md)，快速引擎参数见 [README_FAST_CONTACT.md](README_FAST_CONTACT.md)。原始 Isaac Lab/Docker 路线继续使用 `tools/setup.sh`，不由本脚本安装。

安装机制参考 [uv 安装文档](https://docs.astral.sh/uv/getting-started/installation/) 和 [uv PyTorch 后端文档](https://docs.astral.sh/uv/guides/integration/pytorch/)。

## 本次验证

已在 `/tmp/acp-contact-install-validation` 新建独立 Python 3.11 环境，完成依赖解析/安装、`uv pip check`、CUDA BF16 前后向与图重放、Warp、机器人编译和 EGL 渲染检查；安装报告保存在该环境的 `install-report.json`。另在 PATH 中排除已有 uv，验证了项目本地 uv 的自动下载和重复安装流程，原 `.venv_footstep` 未调整。

新环境完成 8 个物理环境、4 次更新的 v3 控制器启动检查（CUDA Graph/BF16 开启，Torch 编译关闭），记录在 `logs/contact_install_smoke_001/`，没有非有限梯度拒绝。这是安装验证，不是正式训练。系统库在本机已经齐备，因此没有实际执行 apt 安装分支；CPU 安装分支未另建环境验证。
