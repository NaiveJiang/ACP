# 双脚真实接触目标控制任务

本页描述保留的 **16 维速度＋接触目标版本**。新增的 **13 维纯接触控制版本**及其训练、播放命令见 [README_CONTACT_ONLY.md](README_CONTACT_ONLY.md)。

新增入口为 `run_footstep_tracker.sh`，实现位于 `supermimic/acp/footstep/{reference,task,env,train_tracker,evaluate_tracker}.py`。原版 ACP 的九部位任务、网络、环境、Stage 训练入口及启动脚本均未修改。新任务复用原版 mjlab 物理后端、世界模型实现、回放缓冲区与数值检查，注入独立的双脚参考库、观察、指令和奖励。

这是 **oracle 接触控制器训练**：使用数据里的真实未来接触目标，不加载学习出来的规划器。训练不是离线动作回归，也不需要原控制器的网络动作标签。

## 环境与模型

使用独立 `.venv_footstep`，已扩展安装并验证 PyTorch 2.9.0/CUDA 12.8、mjlab 1.6.0、MuJoCo 3.11.0、MuJoCo-Warp 3.11.0、Warp 1.14.0。新增模拟器依赖的完整锁定文件为 `requirements-footstep-tracker.lock`；原来规划器的最小依赖文件保留。重建时使用：

```bash
uv venv --python 3.11 .venv_footstep
uv pip install --python .venv_footstep/bin/python -r requirements-footstep-tracker.lock
```

从宿主机启动，不需要采集 Docker 或 GUI。这里训练的物理资产是仓库既有 RoBan 21 关节资产：显式 PD、平滑干摩擦、无指令延迟，50 Hz 控制、每步 4 次 5 ms 物理子步。采集使用的 WAQ14 资产、动力学和 2 ms 物理步长与其不同。

训练启动会对每个片段最多抽查 32 帧，比较训练资产 FK 与记录脚底参考点位置，并输出 `geometry.json`；默认最大偏差超过 5 cm 就停止。三组数据全部训练片段的抽查最大偏差约 2.67 cm。此检查只证明近似运动学兼容，不代表动力学相同，也不证明实机可用。脚底参考点沿用 `leg_l6_link/leg_r6_link` 下 `[0.0486,0,-0.04]`。

## 输入、输出和 oracle 语义

策略每帧观察 69 维：重力方向 3、机身角速度 3、21 关节位置、21 关节速度、上一时刻本策略动作 21，默认 5 帧历史。没有当前接触状态、世界位置、参考姿态输入；参考动作仅用于重置、oracle 指令、奖励和特权 critic。目标编码器和动作解码器均为 MLP，无 MHA。

任务指令为 16 维，与规划器约定一致：

```text
[vx, vy, vyaw, Δt, 左接触, 右接触,
 左x, 左y, 左z, 右x, 右y, 右z,
 左yaw_sin, 左yaw_cos, 右yaw_sin, 右yaw_cos]
```

每个环境在重置时随机选择一个 Δt，默认在 0.1、0.2、0.3、0.5、0.8、1.0 秒中选择。每步根据当前参考时刻查询未来 Δt 的真实目标，所以这是**滚动未来查询**，不是锁定某个绝对落地时刻的倒计时。目标世界位置转换到**仿真机器人当前实际 base** 的位置及偏航坐标系中，z 向上；不是直接照搬预先存储的参考 base 局部坐标。摆动脚的支撑位置、朝向填零。

输出为 21 个归一化关节动作，转为 `default_q + joint_offset + action_scale * action`，由 PD 执行。想象和仿真中的策略动作都限制在 [-5,5]；仿真探索噪声也在此范围内。

参考库从 v2 labels 文件读取测量的 qpos/qvel、脚底位姿、接触力/点数和速度，不对速度重新差分。它保留每个参考块的未来查询尾部供目标使用，提前结束可执行区间，确保目标不越界。控制器训练允许跨速度切换的真实目标，不使用规划器的 command-constant 筛选；没有任何记录的速度/状态时拒绝该片段。

## 奖励与训练路径

奖励由原 SuperMimic 全身参考跟踪奖励，加四项组成：当前接触状态匹配、当前支撑脚位置、当前支撑脚 yaw、速度跟踪。新增权重依次为 1、1、0.5、0.5；位置尺度 0.08 m、yaw 尺度 0.4 rad、速度尺度 0.5。动作变化和关节限制沿用底层跟踪奖励。

**奖励按当前参考相位计算，不会要求摆动脚立即到达未来支撑位置。** 参考接触标签用记录的竖直力和接触点数判断；可微奖励用脚碰撞几何到平地的距离代理接触（15 mm 阈值、5 mm sigmoid 温度），不是可微接触力，也不将当前接触标记输入策略。只对真实参考支撑脚计算位置/yaw 项。

训练顺序：

1. 策略在 mjlab 执行，获得 `(state, action, next_state)`，放入回放缓冲区。
2. 对真实仿真转移进行多步回归，训练世界模型。
3. 前 500 次迭代只更新世界模型；之后冻结其参数，在模型中展开 16 步，通过可微奖励和 critic 反向更新策略，再更新 critic。

critic 可以使用参考误差、动力学上下文、地形扫描和 oracle 指令；这些特权量不直接进入 actor 的自身观察。世界模型暂时保留原后端的地形扫描输入，地形为平面。普通任务失败保留 terminal transition；数值无效转移被隔离，回放窗口和想象起点不跨重置。

## 正式训练命令

```bash
cd /home/waao/Acp_supermimic_vibed
bash run_footstep_tracker.sh \
  --data \
    /home/waao/lejulab_platform/data/locomotion/full_stop_001 \
    /home/waao/lejulab_platform/data/locomotion/forward_continuous_001 \
    /home/waao/lejulab_platform/data/locomotion/backward_continuous_001 \
  --log-dir logs/footstep_tracker_train_001 \
  --device cuda:0 --num-envs 512 --iterations 20000 \
  --hist 5 --horizons 0.1 0.2 0.3 0.5 0.8 1.0 \
  --wm-only-iters 500 --wm-window 8 --horizon 16 \
  --wm-batch 512 --pi-batch 512 --wm-batches 4 \
  --collect-steps 3 --ckpt-every 500 --log-every 25
```

`--iterations` 为“采样 + 世界模型更新 + 策略/critic 更新”外循环次数，不是规划器的监督训练 steps。默认先做 64 个控制步 warmup，每次迭代再收集 3 步；`sim_steps` 计所有环境的转移总数。eager PyTorch 版本优先验证正确性，尚未做 torch.compile 加速。

默认依据数据组/类别按完整片段划分 80/20；单个长片段按不重叠时间块划分。只将训练块装入仿真参考库，未来查询不能跨到验证块。当前三组数据共得到 690 个训练参考块、415280 帧（包含预留的查询尾部）。

日志目录必须新建，输出：

- `args.json`：参数。
- `task.json`：任务接口、数据划分、参考块、机器人控制指纹和奖励定义。
- `geometry.json`：脚底运动学抽查。
- `train.csv`：WM/策略/critic 损失、策略梯度、想象存活率、仿真奖励及已完成 episode 统计。
- `last.pt`：策略、世界模型、critic、优化器、归一化、数据划分及当前迭代。

想象存活率是世界模型内部的估计，不是实际行走成功率。

续训：在同一命令上增加 `--resume logs/footstep_tracker_train_001/last.pt`，更换 `--log-dir` 为新目录，`--iterations` 指定要达到的总迭代数。数据、hist、查询时间和网络尺寸须一致。回放缓冲区和正在运行的物理 episode 不保存，续训会重置环境并重新 warmup，不是逐位复现中断现场。

## 可视化播放策略

在宿主机桌面终端运行：

```bash
cd /home/waao/Acp_supermimic_vibed
bash play_footstep_tracker.sh \
  --checkpoint logs/footstep_tracker_train_001/last.pt \
  --category forward --seconds 0
```

默认读取验证集中的第一个匹配片段，Δt=0.3 s，按照真实时间播放；摔倒或参考结束时自动重置并重复所选片段。窗口显示的是策略实际驱动的 mjlab 机器人，通过 CPU MuJoCo viewer 显示实际仿真状态，不是参考动画。红色和蓝色标记分别是未来左、右脚支撑位置，箭头表示支撑 yaw；目标接触为 0 的脚不显示标记。

按键：Space 暂停/继续，R 重置，F 切换相机跟随，P 显示/隐藏目标，Esc/Q 退出。鼠标操作沿用 MuJoCo viewer。

先列出可选片段，再用列表中的编号选择：

```bash
bash play_footstep_tracker.sh --checkpoint logs/footstep_tracker_train_001 --category forward --list-clips
bash play_footstep_tracker.sh --checkpoint logs/footstep_tracker_train_001 --category forward --clip-index 2 --seconds 0
```

`--checkpoint` 也接受运行目录，或 `latest`（默认，选择 `logs/footstep_tracker_train*/last.pt` 中最新的权重）。`--split train` 切换到训练片段，`--category backward` 等选择类别，`--phase 0.3` 从可执行参考块的 30% 开始，`--query-dt 0.5` 查询不同未来时间，`--speed 0.5` 半速播放。query-dt 必须出现在 checkpoint 的 horizons 中。默认播放 60 个仿真秒；`--seconds 0` 持续到关闭窗口，`--no-loop` 在第一次结束/终止后退出。

这里仍是 **真实接触目标 + 控制策略**，不是摇杆自由控制，也不使用学习的规划器。脚本不加载世界模型。窗口中模拟器自身的 Run/Pause 状态不负责推进后端，请使用上面的 Space 快捷键。

无窗口检查：

```bash
bash play_footstep_tracker.sh --checkpoint logs/footstep_tracker_train_001/last.pt \
  --headless --seconds 2 --no-pace
```

## 仿真评估

```bash
ACP_ROBOT=roban .venv_footstep/bin/python \
  -m supermimic.acp.footstep.evaluate_tracker \
  --checkpoint logs/footstep_tracker_train_001/last.pt \
  --out logs/footstep_tracker_eval_001 \
  --num-envs 32 --steps 500 --device cuda:0
```

评估使用 checkpoint 中留出的参考块、无 reset/观察/动作噪声的确定性物理设置，但从参考块中随机采样状态开始，仍使用 oracle 接触目标。报告终止次数、完成 episode 长度、参考支撑位置误差、几何接触匹配率及数值无效转移；未结束的 episode 是截尾样本，不计入已完成长度均值。此入口不运行学习出来的规划器，也不导出实机策略。

## 已完成的检查

```bash
ACP_ROBOT=roban .venv_footstep/bin/python -m unittest \
  tools.test_footstep_planner tools.test_footstep_tracker -v
```

单元测试覆盖 oracle 时间、当前实际 base 坐标变换、观察不含接触标签、奖励不提前追未来目标，以及梯度穿过冻结世界模型到策略。GPU 小规模检查已覆盖三组数据、全部六种查询时间、WM 更新、actor/critic 更新、保存/重载续训、留出块上的物理评估。小规模权重仅用于检查链路，不代表已训练出稳定行走控制器。
