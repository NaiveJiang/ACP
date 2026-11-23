# BlinkMimic 加速迁移版

ACP 双脚落足事件与联合续训见 [README_ACP_NEXT_CONTACT.md](README_ACP_NEXT_CONTACT.md)。

联合训练修正版 v8 使用 `run_contact_joint_consistent.sh`，默认 EMA 0.995、无需世界模型单独预训练；见 [README_CONSISTENT_JOINT.md](README_CONSISTENT_JOINT.md)。

速度规划器与控制器联合微调 v7 使用 `run_contact_joint.sh`，两者共同优化，见 [README_JOINT_CONTACT.md](README_JOINT_CONTACT.md)。

双腿自碰撞 v6 使用 `run_contact_tracker_collision.sh`（`--controller collision`），默认 EMA 0.995、无世界模型单独预热，见 [README_LEG_COLLISION.md](README_LEG_COLLISION.md)。

最新下肢自然步态软约束使用 `run_contact_tracker_gait.sh`（`--controller gait`），见 [README_GAIT_STYLE.md](README_GAIT_STYLE.md)。

最新上身直接动作监督 v4 使用 `run_contact_tracker_upper.sh`（`--controller upper`），见 [README_UPPER_SUPERVISED.md](README_UPPER_SUPERVISED.md)。该入口默认不做世界模型单独预热，旧入口默认设置保留。

新入口 `run_contact_tracker_fast.sh`，实现位于 `supermimic/acp/footstep/fast/`。使用现有 `.venv_footstep`，无需安装或切换环境。原 ACP、13/16 维训练入口和已有权重均保留，正式长训练由用户亲自启动。

已接入：Torch 编译、固定形状且隔离无效 episode 后缀的世界模型损失、CUDA Graph 下的世界模型/actor/critic/优化器更新、RoBan 固定连杆及关节锚点兼容的融合 FK、批量多步奖励、GPU 归一化和有限梯度门控、BF16/TF32、融合 Adam、异步原子 checkpoint 写入。

此入口默认保持原有 13 维接触接口、网络尺寸、任务奖励、500 次世界模型预热和 Adam 超参数。上身姿态、动作抖动修正版使用独立入口 `run_contact_tracker_stable.sh`（或 `--controller stable`），详见 [README_STABLE_CONTACT.md](README_STABLE_CONTACT.md)。提速不等于动作质量已改善。

## 手动训练

```bash
cd /home/waao/Acp_supermimic_vibed
bash run_contact_tracker_fast.sh \
  --data \
    /home/waao/lejulab_platform/data/locomotion/full_stop_001 \
    /home/waao/lejulab_platform/data/locomotion/forward_continuous_001 \
    /home/waao/lejulab_platform/data/locomotion/backward_continuous_001 \
  --log-dir logs/contact_tracker_train_fast_001 \
  --num-envs 4096 --iterations 20000
```

默认编译、CUDA Graph、BF16 均开启。第一次会编译多个前后向图，冷启动可能需要数分钟，参数/形状相同的后续进程可复用磁盘缓存。进度长时间停在首轮时先检查进程是否仍在编译，而不要重复启动相同日志目录。

`--no-compile`、`--no-cuda-graph`、`--amp none` 分别关闭对应优化，便于排查。该入口面向 CUDA GPU 和平地数据，不支持任意地形。

### 与 WBC 分支对照

`--profile wbc` 以 `origin/wbc` 的 RoBan **参考跟踪**训练配置为基准，将可共用的优化设置应用于 v6 双脚接触任务。

```bash
bash run_contact_tracker_collision.sh --profile wbc \
  --data /home/waao/lejulab_platform/data/locomotion/full_stop_001 \
         /home/waao/lejulab_platform/data/locomotion/forward_continuous_001 \
         /home/waao/lejulab_platform/data/locomotion/backward_continuous_001 \
  --log-dir logs/contact_tracker_wbc_001 --iterations 20000
```

| 项目 | WBC 对照配置 |
|---|---|
| 采集 / 想象 | 1024 环境，回放 1000 步，填充 64 步，平均每次更新采 2.5 步；本体历史 10 帧，想象 32 步 |
| 世界模型 | 结构化半隐式积分；每轮 10 批，每批 1024 个 8 步窗口，镜像其中一半；专门预训练 0 轮 |
| 优化 | Muon + Adam，基准学习率 4e-4；策略初始为 4 倍；采集奖励达到上限的 75% 或到第 1400 次更新时，开始用 300 次更新降到 1 倍 |
| 策略 / critic | 策略每批 512 个起点；critic 5 个 epoch，每个 epoch 分 4 批；策略 EMA 0.995 |

此实验继续使用现有 RoBan 平地资产、双脚接触标签、碰撞处理和奖励。学习率阈值按此任务的奖励上限归一化。WBC 的完整参考关节窗口、动作残差 skip 和 S17 专用执行器属于另一套训练接口。新配置默认的 10 帧历史需要从头训练，也可与同结构 checkpoint 配套续训。

失败重置按 50 帧分段，在同一参考片段内提高真实摔倒前一段的重置权重（`--failure-reset-share 0.30`），并保留 `--start-reset-fraction 0.30` 的片首重置。正常超时和数值无效状态没有失败权重。EMA 从第 1000 次更新起，每 125 次在 16 个单独的评估环境中跑参考接触，标记失败所在段及前两段（`--probe-share 0.15`）。设置 `--probe-share 0` 可关闭探测。采样统计随 checkpoint 保存，旧 WBC 配置的 checkpoint 从空统计开始累积。

### 参考步态对照

在相同命令上加 `--motion-style-weight 1.0`，并使用新日志目录。训练奖励额外比较抬脚和落地前后两帧的接触状态、机身偏航坐标系下的摆动脚轨迹，以及骨盆相对支撑脚的位置和机身速度。重心转移用这些运动量约束。CSV 分别记录 `lift_timing_cost`、`swing_motion_cost` 和 `weight_shift_cost`。不传这一选项时，这三项权重为零。

### 机身方向与航向对照

在参考步态训练的命令上再加 `--body-goal`，并另开日志目录。控制器的指令是 13 维接触目标加 `[dir_x, dir_y, Δyaw]`。训练和参考回放使用同一片段 0.8 s 后的机身位置与航向，水平位移用当前机身偏航坐标系表达并按 0.3 m 半径限幅，偏航角误差取 `[-π, π)`。接触目标查询仍独立使用原有时间窗口。

该选项使用独立 checkpoint 格式，从头训练后可用原播放器和批量评估脚本。首轮对照使用记录接触目标和参考机身目标，奖励与参考步态版本保持一致。

### 下一次接触的 Stage 2

```bash
bash run_contact_tracker_collision.sh --profile wbc --event-stage2 \
  --init logs/contact_tracker_wbc_style_001/last.pt \
  --data /home/waao/lejulab_platform/data/locomotion/full_stop_001 \
         /home/waao/lejulab_platform/data/locomotion/forward_continuous_001 \
         /home/waao/lejulab_platform/data/locomotion/backward_continuous_001 \
  --log-dir logs/contact_tracker_event_stage2_001 --iterations 10000
```

控制器从 Stage 1 的 EMA 权重开始继续更新。规划器每隔 50 轮用独立环境中的成功闭环补充数据。接触有无取自 MuJoCo 求解力，位姿取实际落地帧的脚底中心和朝向。规划器用独立优化器监督学习未来 1.5 s 内各脚下一次新接触的位姿和时间，接触至少持续 10 帧；跨速度指令变化的标签窗口排除。首轮只调节双脚接触规划，保留已有的步态正则及上身监督。

默认第 1000 轮开始把预测目标逐步交给控制器，第 3000 轮开始在目标到期时替换接触项，两个过渡各 4000 轮。控制器通过世界模型学习，梯度不回传给规划器。控制器接口保留 13 维，Stage 2 使用 0.3 s、0.5 s 查询。下一次接触到查询目标的转换会保留查询时仍应支撑的脚。

日志记录成功闭环数、事件样本数、规划器损失和两个交接比例。保存点包含两侧权重及优化器。续训按保存轮次重新采集闭环样本，可用同一保存点指定 `--checkpoint` 和 `--planner-model`，加 `--velocity 0.5 0 0 --category forward --query-dt 0.3` 回放规划驱动。此入口的回放使用固定速度。

## 播放与续训

```bash
bash play_contact_tracker.sh \
  --checkpoint logs/contact_tracker_train_fast_001/last.pt \
  --category forward --seconds 0
```

保存的 actor 保持原 13 维格式，现有播放和评估入口可读取。完整 checkpoint 额外包含 fast 引擎、世界模型归一化及优化器信息。

续训用同一 fast 命令增加 `--resume <fast运行>/last.pt`，更换日志目录，`--iterations` 为总迭代数。须保持任务、数据、网络结构、优化器、AMP 和 EMA 设置一致。回放缓冲区和物理现场不保存，重新启动会 refill。旧训练入口不用于续训 fast checkpoint。

`--init <旧13维权重>` 可初始化同接口模型并重新开始优化器；若切换世界模型积分结构，则只初始化策略。16 维任务用 `--controller velocity`，只能与 16 维权重配套。

## 可选方法变化

提供 `--structured-wm`（速度积分＋位置/姿态残差）、`--optimizer muon`（隐藏矩阵 Muon、其余 Adam）、`--ema 0.995`。默认均不启用，以便先隔离工程提速；本次速度基准不代表这些选项的收敛效果已在 RoBan 上验证。

未直接迁移依赖完整参考关节姿态的 `pi_ref_skip`，否则会破坏只给接触计划的接口。G1 镜像映射也不能直接用于 RoBan，因此未启用。整个物理采集/reset 的 CUDA Graph、异步评估引导重置尚未迁移；当前仍保留原物理后端及即时数值隔离。

## 检查与性能记录

```bash
.venv_footstep/bin/python -m unittest tools.test_fast_contact -v
```

检查包括 FK 数值/梯度、归一化等价性、结构化积分往返、无效后缀含 NaN 时的掩码隔离，以及图预热不会偷偷多更新、坏梯度不会修改参数/优化器状态。

4096 环境、相同网络、WM batch 512×4、actor batch 512、16 步想象的 60 次更新短基准见 `logs/fast_contact_bench_002/comparison.json`。稳定迭代约 65 ms，原入口约 0.5 s，具体以 JSON 数值为准。此结果是吞吐检查，不是完整收敛或动作质量评估。

`performance.json` 同时记录启动/预热、包括编译的总更新耗时和排除首次图捕获的稳定窗口。首次冷缓存基准启动与编译累计约 168 s；不能只看稳定迭代时间估算短任务耗时。

GPU 在更新内跳过非有限梯度，参数和优化器计数均不更新；主机在日志边界读取计数，若出现连续三次拒绝会保存 checkpoint 并停止。物理仿真的无效状态仍由原 `TransitionGuard` 每步隔离。CUDA Graph 预热会恢复参数、优化器和 RNG，不计作额外训练更新。

优化器及部分数学/图执行设计来自本地 BlinkMimic。MIT 许可和第三方声明保留在 `supermimic/acp/footstep/fast/vendor/`。
