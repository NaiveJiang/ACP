# v7：速度规划器与接触控制器联合微调

**更新：** 已有 v7 收敛模型出现贴地挪动和规划目标失真，后续训练请使用 [v8 修正版](README_CONSISTENT_JOINT.md)，从 v6 重新初始化。本文保留旧任务说明。

入口 `run_contact_joint.sh`。按用户选择，主规划器和主控制器都参与优化，世界模型继续用真实物理交互更新；保留 v6 双腿碰撞、v5 步态软约束、v4 上身直接监督和原版各任务。

## 模型与输入

```text
本体历史前 48 维 + 速度指令(vx,vy,vyaw) + 查询时间
  → FootstepPlanner
  → 13 维双脚未来接触目标
  → 接触控制器 + 69 维本体历史
  → 21 维关节动作
```

速度只进入规划器，低层控制器仍接收 13 维接触目标，没有增加速度指令。高层命令共 4 维，因此启动打印 `Fast 4D`；这不是把低层接触输入改成 4 维。状态历史取现有控制器的前 48 维（重力、机身角速度、测得的原始关节角和关节速度），与离线规划器的数据顺序/单位一致；不额外加默认站姿偏置。

规划器输出接触 logits、位置和 sin/cos 朝向。训练使用 straight-through 接触解码：前向仍为 0/1，摆动脚位置/朝向为零，反向用 sigmoid 导数，避免硬阈值截断规划器梯度；播放同样使用二值前向值。朝向归一化，零向量回退到 sin/cos=(0,1)。规划器与控制器一起通过世界模型 rollout 优化，并维护 EMA。

## 联合目标与防遗忘

新奖励跟踪实际机身偏航坐标系下的水平速度（权重 4、尺度 0.35 m/s）及世界竖直轴角速度（权重 1、尺度 0.5 rad/s），保留直立、高度、自然步态、上身姿态、平滑、关节限位和双腿避碰。取消“追随参考脚的世界位置/朝向轨迹”的奖励，允许规划器调整落足来达到速度。

同时增加两项辅助监督：

- 离线规划器损失：原数据中有效、未来命令保持不变的接触/位置/朝向标签，损失系数 1。只从控制器保存的训练划分采样，不取验证数据。
- 原控制器回放约束：额外保留一份冻结的 v6 控制器，在真实回放状态/历史和真实接触目标上提供动作参考，MSE 系数 0.1。它只作为防遗忘参考，**主控制器仍更新**。v4 上身监督也继续作用于主控制器。

总 actor 损失为模型式目标加这些辅助项。辅助约束帮助保持接触输出的语义，不构成严格的落足可达性保证。

采集初期每个 episode 有 25% 概率使用规划器目标，其余使用真实目标；前 2000 次更新线性过渡至 100%，现有 episode 在下次重置时采用新的概率。想象 rollout 始终走规划器链路。参数为 `--planner-mix-start`、`--planner-mix-iters`；若要真实采集立即完全使用规划器，设 `--planner-mix-start 1`。

想象 rollout 内保持起点速度指令不变，不能读取后续参考命令。未来查询/rollout 会跨命令切换的 actor 起点被排除；切换后命令稳定的过渡状态保留。物理世界模型仍可学习真实命令切换时的动力学转移，因为它使用真实动作而非未来速度标签。

## 正式训练：由用户启动

已经找到并检查的初始化权重：

- 控制器：`logs/contact_tracker_train_collision_001/last.pt`，v6，20000 轮。
- 规划器：`logs/footstep_train_001/best.pt`，离线训练第 7000 步保存；其训练/验证划分与控制器一致。已有离线验证接触准确率约 98.2%、支撑位置平均误差约 1.67 cm；这些不代表闭环速度跟踪效果。

```bash
cd /home/waao/Acp_supermimic_vibed
bash run_contact_joint.sh \
  --data /home/waao/lejulab_platform/data/locomotion/{full_stop_001,forward_continuous_001,backward_continuous_001} \
  --init logs/contact_tracker_train_collision_001/last.pt \
  --planner-init logs/footstep_train_001/best.pt \
  --log-dir logs/contact_tracker_train_joint_001 \
  --num-envs 4096 --iterations 20000
```

默认编译、CUDA Graph、BF16、EMA 0.995；无世界模型单独预训练，初始采集 64 步用于回放缓冲。主控制器学习率 2e-4，规划器为其 0.25 倍（`--planner-lr-mult`）。当前联合入口支持 Adam，不支持 Muon。世界模型初始化自 v6 并在线更新，critic/优化器重新开始。

初始化会检查规划器格式、关节顺序、状态定义、历史长度、查询时间以及完整训练/验证划分和数据指纹，防止原控制器的验证数据泄漏进联合监督。规划器配置保存在 joint checkpoint 内。v7 格式为 `two_foot_joint_planner_controller_v7`，包含主规划器、主控制器、冻结参考控制器、世界模型、critic、EMA 和优化器，无需播放时另外提供旧权重。

续训同一 joint 任务用 `--resume <v7>/last.pt` 和新日志目录，去掉 `--init`、`--planner-init`。保持网络、EMA、优化器、规划器学习率比例和混合采集日程一致。回放缓冲/物理现场不保存，续训会重新填充。

## 只输入速度进行闭环播放

```bash
bash play_contact_tracker.sh \
  --checkpoint logs/contact_tracker_train_joint_001/last.pt \
  --category forward --velocity 0.2 0 0 --query-dt 0.3 --seconds 0
```

`--velocity vx vy vyaw` 的单位为 m/s、m/s、rad/s。删除它则按所选数据片段的命令时序输入速度，接触目标仍由规划器生成。P 显示的是实际规划器预测目标，输出明确标记 `oracle_targets=false`。仅支持在已训练查询时间中选择 `--query-dt`。

播放仍使用参考片段初始化，并在参考片段结束或摔倒后重置；它是现有仿真中的速度闭环检查，不是无限时长的自由遥控环境。实际策略前向本身只需要历史和速度/查询时间，不需要参考轨迹。应先在现有数据覆盖的速度/组合范围验证。保持外部直接接触控制时，可调用 joint 模型的 `controller_action(history, contact13)` 绕过规划器。

## 指标与验证

`train.csv` 增加 `planner_supervision_loss`（离线规划器损失与旧控制器回放约束之和）、`planner_collection_fraction`、`usable_actor_fraction`、实际水平速度/角速度误差和避碰诊断。`upper_supervision_loss` 单独保留。辅助损失已按上述系数进入 actor 总损失，不能与不同统计窗口的物理平均奖励直接相减。

评估：

```bash
.venv_footstep/bin/python -m supermimic.acp.footstep.evaluate_tracker \
  --checkpoint logs/contact_tracker_train_joint_001/last.pt \
  --out logs/contact_joint_eval_001 --num-envs 32 --steps 500
```

v7 评估使用规划器闭环并报告速度误差；参考脚位置/接触比较明确标为 reference diagnostics，不能当作实际规划器目标的跟踪精度。

```bash
.venv_footstep/bin/python -m unittest \
  tools.test_joint_contact tools.test_footstep_planner tools.test_leg_collision \
  tools.test_gait_style tools.test_upper_supervised tools.test_stable_contact \
  tools.test_contact_only_tracker tools.test_footstep_tracker tools.test_fast_contact -v
```

49 项测试通过，覆盖高/低层梯度、冻结参考控制器、13 维解码、命令窗口与保持指令、离线历史/目标、移除世界脚轨迹奖励，以及原功能回归。短训练检查中主控制器 22 个权重/偏置张量、规划器 12 个权重/偏置张量发生更新，参考控制器完全不变。编译/Graph/BF16/EMA、v6 实际权重初始化、v7 续训和评估入口均通过。初始化后额外播放 3 秒无摔倒/无效状态，不代表速度跟踪已经达标。

开发只做 4 次更新等小规模检查，未启动正式长训练。记录位于 `logs/contact_joint_*smoke_001/`，3 秒闭环检查日志为 `/tmp/contact_joint_closed_loop_baseline.log`。
