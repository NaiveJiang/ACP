# v5：落足控制与自然步态软约束

> 双腿自碰撞已新增为 v6，入口 `run_contact_tracker_collision.sh`，见 [README_LEG_COLLISION.md](README_LEG_COLLISION.md)。v5 仍使用原来的地形碰撞设置，不会自动启用双腿碰撞。

新入口 `run_contact_tracker_gait.sh` 在 v4 上身直接监督的基础上，补上下肢的运动约束，避免只因脚端到位就接受扭髋、歪骨盆和怪异摆腿。v1–v4 的入口、任务和 checkpoint 播放方式保留，不会修改已经保存的权重。不需要新增环境依赖。

## 本次修改

接触、位置、朝向奖励继续为 2/4/1.5，增加以下代价：

| 项目 | 系数 | 含义 |
|---|---:|---|
| 下肢关节姿态 | 2.0 | 十二个关节与当前参考相位的姿态误差，引导屈膝、髋部与踝部协调 |
| 下肢关节速度 | 0.2 | 参考关节速度误差，引导连续摆腿与停步 |
| 骨盆倾斜 | 0.5 | 与参考骨盆 roll/pitch 对应的重力方向误差 |
| 支撑脚倾斜 | 0.5 | 支撑脚相对参考的 roll/pitch 误差，允许数据中的脚掌滚动 |
| 膝过伸 | 0.25 | 膝角低于 0 rad 的软代价 |

均使用 Smooth L1（Huber）代价：偏差小时平滑，偏差大时仍有纠正梯度。姿态按每条腿六个关节的尺度 `(0.35,0.25,0.25,0.45,0.30,0.20)` rad 归一化，关节速度尺度为 `(3,2,2,4,3,2)` rad/s；骨盆/脚底倾斜尺度 0.2，膝过伸尺度 0.1 rad。系数只与这些误差定义配套使用，不能单独通过数值大小比较任务重要性。

支撑脚倾斜只在当前参考接触为 1 时计分，不会在摆动期强行压平脚底。关节与姿态约束都对齐当前物理控制帧；未来查询的脚端目标仍由原接触任务提供。

这些是加权软约束，允许腿部偏离参考以完成落足，不是锁死参考动作，也不保证数学上的严格任务优先级。仍需训练后验证落足误差与运动自然性的实际权衡。没有增加世界坐标根部轨迹或身体速度跟踪；骨盆/脚底倾斜的定义对全局 yaw 旋转不敏感。

策略与 v4 相同：69 维本体历史 × 5 帧，加 13 维接触计划；不加入期望速度、参考腿部姿态或当前接触测量。新的参考误差只用于奖励和特权 critic，上身继续采用 v4 的独立分支直接监督。

## 手动训练

可以从现有 v3 初始化腿部和世界模型，上身分支新建，critic/优化器重新开始：

```bash
cd /home/waao/Acp_supermimic_vibed
bash run_contact_tracker_gait.sh \
  --data \
    /home/waao/lejulab_platform/data/locomotion/full_stop_001 \
    /home/waao/lejulab_platform/data/locomotion/forward_continuous_001 \
    /home/waao/lejulab_platform/data/locomotion/backward_continuous_001 \
  --init logs/contact_tracker_train_stable_001/last.pt \
  --log-dir logs/contact_tracker_train_gait_001 \
  --num-envs 4096 --iterations 20000
```

若已训练 v4，把 `--init` 路径换为该 v4 的 `last.pt`，会保留完整策略和世界模型，重新建立 critic/优化器；也可删除 `--init` 从头训练。初始化要求数据、机器人和网络配置匹配，归一化统计会重新允许适应新分布。旧 v3/v4 不能用 `--resume` 直接跨任务续训。

入口默认启用编译、CUDA Graph 和 BF16，`--wm-only-iters 0`。初始采集 64 步后，每轮交替更新世界模型、策略和 critic。正式训练由用户手动启动。

调节参数可添加 `--leg-pose-weight`、`--leg-velocity-weight`、`--stance-tilt-weight`，或继承的 `--upper-supervision-weight`。v5 完整续训用 `--resume <v5>/last.pt` 和新日志目录；这些权重若变化，应通过 `--init` 开始新实验。

## 播放和诊断

```bash
bash play_contact_tracker.sh \
  --checkpoint logs/contact_tracker_train_gait_001/last.pt \
  --category forward --seconds 0
```

v5 格式为 `two_foot_gait_style_tracker_v5`，按格式自动加载。当前播放仍使用数据中的真实接触目标，不是速度规划器闭环。

`train.csv` 保留总奖励、上身监督等字段，并增加 `leg_pose_cost`、`leg_velocity_cost`、`pelvis_tilt_cost`、`stance_tilt_cost`、`knee_cost`、`contact_reward`、`position_reward`、`yaw_reward`。这些新增列是每个日志时刻当前物理环境的平均贡献，均已乘系数、未乘 0.02 s；cost 为正数，进入总奖励时被减去。它们与原有跨多个采集步累计的 `sim_reward` 统计窗口不同。

应在相同参考、初态、查询时间下比较落足位置/朝向误差、接触时序、骨盆倾斜、腿部姿态、动作变化及关节速度，不能只看总奖励（跨版本定义不同）。关节参考是风格引导，外部任意大角度落足仍受训练数据覆盖限制。

## 验证命令

```bash
.venv_footstep/bin/python -m unittest \
  tools.test_gait_style tools.test_upper_supervised tools.test_stable_contact \
  tools.test_contact_only_tracker tools.test_footstep_tracker tools.test_fast_contact -v
```

测试覆盖参考步态零代价、大偏差纠正梯度、脚端已经匹配时仍有腿部风格约束、摆动脚不受支撑倾斜约束、世界平移/yaw 不变性、期望速度不进入输入或奖励、膝过伸梯度、特权 critic、编译和 v4 策略兼容性，并回归旧版本。短测试不能证明正式训练后的自然性或落足精度。

本次 32 项测试通过。`logs/contact_gait_compile_smoke_001/` 完成从 v4 初始化后的编译、CUDA Graph、BF16 检查（8 环境、4 次更新）；`contact_gait_init_v3_smoke_001/` 验证从实际 v3 权重初始化；`contact_gait_resume_smoke_001/` 验证 v5 续训。播放执行 10 个物理步，无终止和无效转移，新增日志列均正常输出。未启动正式长训练。
