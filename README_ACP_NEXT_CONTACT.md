# ACP 双脚落足事件实验

控制器每步接收 10 帧本体观测，以及左右脚各自下一次新接触的位姿与有效标记。12 维接触向量依次为左右脚有效标记、左右脚的 XYZ、左右脚朝向的 sin/cos。坐标原点在当前骨盆，XY 随骨盆偏航转动。控制器没有速度、航向、接触倒计时或未来支撑状态输入。双脚一直踩地且没有新接触时，两脚标记均为 0。

规划器接收 10 帧本体历史和速度指令 `(vx, vy, vyaw)`。它复用 ACP 的 `ContactPlanner` 和软标签损失，在平地候选网格上预测左右脚的落足概率分布以及各自的事件有效性。每步解码每只脚至多一个目标，额外预测脚底相对骨盆的 Z 和脚朝向。1.5 秒仅定义监督范围。指令采用采集脚本记录的 `requested`，单位为 m/s、m/s、rad/s，模拟现场跟踪的实际速度仍用实测值核对。

## 第一阶段

从已训练的 WBC 控制器与世界模型初始化。接触编码器去掉查询时间输入对应的列，保留其余权重，再用记录中各脚真实的下一次落地重新训练。保留步态、上身和自碰撞约束，同时更新世界模型，无独立预训练阶段。

```bash
cd ~/Acp_supermimic_vibed
DATA=(~/lejulab_platform/data/locomotion/{full_stop_001,forward_continuous_001,backward_continuous_001})

bash run_contact_tracker_collision.sh --profile wbc --next-contact \
  --init logs/contact_tracker_wbc_style_001/last.pt \
  --data "${DATA[@]}" --log-dir logs/contact_tracker_next_001 --iterations 10000

bash play_contact_tracker.sh --checkpoint logs/contact_tracker_next_001/last.pt \
  --recorded-plan --category forward --seconds 0
```

用参考落脚点检查起步、步态与落足跟踪，再进入第二阶段。播放默认使用验证集。可用 `--list-clips` 和 `--clip-index` 选出同一片段用于比较。

## 联合续训

```bash
bash run_contact_tracker_collision.sh --profile wbc --next-contact-stage2 \
  --init logs/contact_tracker_next_001/last.pt \
  --data "${DATA[@]}" --log-dir logs/contact_tracker_next_joint_001 --iterations 10000

bash play_contact_tracker.sh --checkpoint logs/contact_tracker_next_joint_001/last.pt \
  --category forward --velocity 0.2 0 0 --seconds 0
```

控制器全程继续训练。规划器使用独立优化器，周期性用 EMA 控制器在独立 MuJoCo 环境采集闭环轨迹。从求解器接触力识别新落地，从实际脚底位姿生成监督。落地须持续至少 0.2 秒。指令变化前已完整观测的落地可作为正样本，负样本要求完整观测。发生摔倒、明显偏离参考或移动进度不足的轨迹不进入规划器回放。

前期控制器接收参考目标，规划器积累监督。接管从少量规划器目标开始，闭环移动通过率达标后再提高比例。采集始终保留一部分参考目标驱动的 rollout。接管后期逐渐在触地时加入规划落点的位姿奖励。奖励在 BlinkMimic 想象轨迹里用脚底间隙变化估计触地，接触力只用于真实仿真中的规划器监督。训练 CSV 中的 `event_loss`、`event_samples`、`event_successes`、`eta_obs` 和 `eta_reward` 分别记录规划器损失、回放样本数、累计成功 rollout 数、规划输入接管比例和规划奖励权重。

诊断阶段二控制器时，给同一 checkpoint 加 `--recorded-plan` 并移除 `--velocity`。也可在第一阶段 checkpoint 上加 `--planner-model logs/contact_tracker_next_joint_001/last.pt --velocity 0.2 0 0`，单独比较换用规划目标的影响。窗口里的红蓝标记对应左右脚。用 `--headless --seconds 8 --no-loop --contact-trace /tmp/next_contact_trace.json` 可保存逐帧脚底和落点。该实验面向平地，速度和切换能力需在完成训练后用闭环评估确认。
