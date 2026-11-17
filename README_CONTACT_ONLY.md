# 13 维接触控制器：不输入速度命令

此版本的 actor 只接受本体观察历史和接触计划。速度命令仍可输入上层规划器，由规划器生成接触计划；绕过规划器时，外部也可提供相同的 13 维指令。

旧版 16 维任务、入口及 `logs/footstep_tracker_train_001/last.pt` 保留。新增训练入口为 `run_contact_tracker.sh`，新增播放入口为 `play_contact_tracker.sh`，继续使用独立 `.venv_footstep` 环境。共享训练引擎、评估和播放按 checkpoint format 显式选择任务，不将新旧权重混用。

## 输入与输出

本体观察仍为每帧 69 维、默认 5 帧历史：重力方向 3、机身角速度 3、关节位置 21、关节速度 21、上一时刻动作 21。这里去掉的是**速度命令**，本体的关节速度、角速度仍保留。

任务输入从 16 维变为 13 维，顺序如下：

```text
[Δt, 左接触, 右接触,
 左x, 左y, 左z, 右x, 右y, 右z,
 左yaw_sin, 左yaw_cos, 右yaw_sin, 右yaw_cos]
```

位置和朝向在机器人当前实际 base 的位置及偏航坐标系中表达，z 向上。目标接触为 0 的脚，其支撑位置/朝向置零。输出仍是 21 维归一化关节动作，经 PD 执行。网络仍为 MLP，无 MHA。

现有三组 `future_contact_v2` 数据无需重新转换：训练从记录的脚底位姿/接触状态直接构造目标，不使用其 16 维 task_command。数据文件内的速度记录仍保留作规划器训练或评估诊断，但不进入本版 actor 或速度奖励。

## 训练目标

这不是将旧输入中的速度填零。新任务已经移除：

- 显式速度命令跟踪奖励。
- 参考身体线速度、角速度的跟踪奖励。
- 参考 base 的水平位置和全局 yaw 轨迹跟踪。
- 参考下肢关节姿态/全身参考姿态的强模仿奖励。

主要奖励为：

| 项 | 权重/参数 |
|---|---|
| 当前相位的接触状态匹配 | 2 |
| 当前参考支撑脚位置 | 4，位置尺度 0.08 m |
| 当前参考支撑脚 yaw | 1.5，角度尺度 0.4 rad |
| 机身直立 | 0.5，不约束全局 yaw |
| 机身高度 | 0.25，目标 0.77 m、尺度 0.15 m |
| 腰和双臂的弱姿态先验 | 0.05，不包含下肢关节 |
| 动作变化、关节限制、实际关节速度正则 | -0.01、-0.5、-0.0001 |

奖励最后乘以 0.02 s。位置和朝向只对参考支撑脚计算。参考接触来自记录的接触力；可微接触奖励仍使用碰撞几何距离代理。终止条件改为实际 base 高度低于 0.45 m 或竖直方向余弦低于 0.5，不再依赖参考 base 姿态。

actor 不看到当前接触标签。critic 使用当前实际状态、物理上下文、地形、13 维计划和当前参考接触目标作为特权信息，不使用参考身体速度、根轨迹或下肢姿态误差。世界模型仍需要实际机器人速度来学习动力学，这与把速度命令输入 actor 不同。

仍然使用当前参考接触相位提供过程奖励，未来 Δt 目标作为策略输入；不会从起脚开始就把摆动脚强拉到未来终点。这一阶段属于 oracle 接触跟踪预训练，尚未加入任意单脚落足的程序化目标或终端到达任务，因此不保证直接完成后移 25 cm、单脚转 90° 等训练数据未覆盖的动作。

## 从头训练

```bash
cd /home/waao/Acp_supermimic_vibed
bash run_contact_tracker.sh \
  --data \
    /home/waao/lejulab_platform/data/locomotion/full_stop_001 \
    /home/waao/lejulab_platform/data/locomotion/forward_continuous_001 \
    /home/waao/lejulab_platform/data/locomotion/backward_continuous_001 \
  --log-dir logs/contact_tracker_train_002 \
  --num-envs 4096 --iterations 20000
```

默认前 500 次迭代仅训练世界模型，之后更新 actor/critic。默认使用 5 帧观察、六种查询时间 0.1/0.2/0.3/0.5/0.8/1.0 s、16 步想象、8 步 WM 回归、WM/策略 batch 512、每 500 次保存 last.pt。完整参数可以用 `bash run_contact_tracker.sh --help` 查看。

每组运行目录必须新建。新 checkpoint format 为 `two_foot_contact_only_tracker_v2`，策略 goal normalizer 和目标编码器输入均为 13 维。默认从头训练，不会自动导入旧 16 维策略。续训只接受相同 format 的 `--resume`，其余规则同旧训练引擎。

助手提前启动的 `contact_tracker_train_001` 已按用户要求停止，日志和启动记录保留。正式训练由用户亲自执行，上面命令使用新的 `contact_tracker_train_002`，直接在当前终端输出进度。另一个终端可查看指标：

```bash
tail -f logs/contact_tracker_train_002/train.csv
```

## 播放与评估

```bash
bash play_contact_tracker.sh \
  --checkpoint logs/contact_tracker_train_002/last.pt \
  --category forward --seconds 0
```

Space 暂停/继续，R 重置，F 相机跟随，P 目标显示，Esc/Q 退出；`--list-clips`、`--clip-index`、`--split`、`--query-dt` 等与旧播放入口相同。默认 latest-contact 只查找 `logs/contact_tracker_train*/last.pt`，不会自动选择旧 16 维权重。必须先产生 checkpoint 才能播放。

评估入口按 format 自动选择 13 维任务：

```bash
ACP_ROBOT=roban .venv_footstep/bin/python \
  -m supermimic.acp.footstep.evaluate_tracker \
  --checkpoint logs/contact_tracker_train_002/last.pt \
  --out logs/contact_tracker_eval_002 --num-envs 128 --steps 1000
```

报告中的速度对比只是相对于记录命令的诊断，不能当成本任务的训练目标。当前播放/评估仍使用真实接触计划；上层规划器联训及外部指定落足的交互入口尚未接入。

## 验证

```bash
ACP_ROBOT=roban .venv_footstep/bin/python -m unittest \
  tools.test_footstep_planner tools.test_footstep_tracker tools.test_contact_only_tracker -v
```

新增测试覆盖：13 维顺序、改变速度记录不影响输入或奖励、改变参考身体速度/根轨迹/下肢姿态不影响奖励、支撑位置影响奖励及有限梯度，以及旧任务 dispatch 不变。另有 GPU 新旧版训练、13 维续训、评估和播放检查。
