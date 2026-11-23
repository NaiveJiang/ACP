# v4：接触控制与上身直接动作监督

> 下肢自然步态软约束已新增为 v5，使用 `run_contact_tracker_gait.sh`，见 [README_GAIT_STYLE.md](README_GAIT_STYLE.md)。本 v4 入口保留用于原任务续训和对照。

新增入口 `run_contact_tracker_upper.sh`。保留 v1/v2/v3 入口和权重，使用相同 `.venv_footstep`，不需要新增依赖。正式训练仍由用户启动。

## 为什么继续修正

v3 的前进回放中，上身姿态误差从 v2 的约 47.7° 降到 23.0°，但左臂旋转关节仍长期正向饱和，双肘偏直。该次对比来自约 15500 轮的 v3 权重，见 `logs/contact_tracker_diagnosis_v3_001/`，不是最终 20000 轮的完整评估。仅靠世界模型传递的状态奖励没有解决问题，不能由此断言唯一原因是世界模型误差。

v4 增加不经过世界模型的上身监督：策略拆为十二个腿部关节的分支和腰部、双臂九个关节的独立分支。二者都只看本体历史和 13 维接触计划。上身监督只更新上身分支，不更新腿部分支；原接触任务的模型式训练仍更新两个分支，以学习平衡和接触控制。

## 监督目标

从真实回放缓冲中采样 `s_t`、本体历史、查询时间和零偏，使用数据下一控制帧 `f+1` 的上身关节位置、速度构造名义 PD 目标：

```text
q_des = q_ref[f+1] + (kd / kp) * qd_ref[f+1]
a_target = (q_des - default_q - encoder_offset) / action_scale
```

这不是原采集策略的真实电机指令，而是由参考运动构造的辅助目标，速度项补偿 PD 阻尼；没有把它称为完整逆动力学，重力、惯性及模型差异仍由原物理姿态跟踪奖励和 RL 处理。

目标限制在每个关节动作范围的 98% 内，再通过逆 tanh 映射到网络未限幅的输出空间，用 Smooth L1 损失监督。这样错误饱和到最大值时仍有纠正梯度，不会被 tanh 的零梯度锁住。腰部监督权重额外乘 3。总 actor 损失为原模型式目标加 `0.5 * upper_supervision_loss`；可用 `--upper-supervision-weight` 调整，日志单独记录 `upper_supervision_loss`。

监督只使用真实回放样本，输入与目标均切断对状态/世界模型的梯度；无效样本被屏蔽。标签固定为下一控制帧，不把可查询的 0.1–1 秒落足时刻当作上身下一动作时刻。

部署输入不增加：本体观测 69 维 × 历史 5 帧，接触目标 13 维。播放时不需要上身参考姿态、期望速度或当前接触测量。状态奖励、PD 范围限制和动作平滑沿用 v3。

## 用户手动训练

推荐从已经训练的 v3 初始化腿部及世界模型，新上身分支随机初始化，重新优化：

```bash
cd /home/waao/Acp_supermimic_vibed
bash run_contact_tracker_upper.sh \
  --data \
    /home/waao/lejulab_platform/data/locomotion/full_stop_001 \
    /home/waao/lejulab_platform/data/locomotion/forward_continuous_001 \
    /home/waao/lejulab_platform/data/locomotion/backward_continuous_001 \
  --init logs/contact_tracker_train_stable_001/last.pt \
  --log-dir logs/contact_tracker_train_upper_001 \
  --num-envs 4096 --iterations 20000
```

初始化会精确复制旧编码器和腿部输出对应的权重，不继承异常的上身输出；沿用已有归一化均值/方差，并降低累计统计权重，让它们适应新的上身运动。优化器、critic 和迭代计数重新开始。数据指纹、机器人及网络配置必须与初始化权重匹配。也可以删除 `--init` 从头训练。

此新入口默认 `--wm-only-iters 0`，即初始采集 64 步后，世界模型、策略和 critic 从首轮开始交替更新；默认仍开启编译、CUDA Graph 和 BF16。第一次编译需要等待。

v4 保存格式为 `two_foot_upper_supervised_tracker_v4`。续训 v4 用 `--resume <v4>/last.pt` 和新日志目录；旧 v3 只能通过上述 `--init` 迁移，不能直接 `--resume`。现有 v3 权重不会因代码新增自动变成 v4。

## 播放

```bash
bash play_contact_tracker.sh \
  --checkpoint logs/contact_tracker_train_upper_001/last.pt \
  --category forward --seconds 0
```

当前播放仍用真实参考接触计划。上身也有多种合理运动，接触输入不唯一确定参考摆臂，因此此监督提供动作风格约束，不能保证逐帧与原始数据相同。动作质量需正式训练后与 v3 在相同参考/初态下回放比较。

## 验证

```bash
.venv_footstep/bin/python -m unittest \
  tools.test_upper_supervised tools.test_stable_contact \
  tools.test_contact_only_tracker tools.test_footstep_tracker tools.test_fast_contact -v
```

测试覆盖错误饱和关节仍有纠正梯度、单独的监督更新不改变腿部输出、v3 初始化保留腿部映射、下一帧/零偏/PD 目标一致性、无效样本屏蔽、编译及权重加载，并检查旧版本回归。开发仅运行数次更新的小规模检查，不启动正式长训练。

本次 26 项测试通过。`logs/contact_upper_compile_smoke_001/` 完成编译、CUDA Graph、BF16 的 8 环境/4 次更新检查，`contact_upper_init_smoke_001/` 验证从实际 v3 权重初始化，`contact_upper_resume_smoke_001/` 验证 v4 优化器续训。新播放入口运行 10 个物理步，无无效转移。这些只证明实现和入口可运行，尚未证明正式训练后的动作质量。

接触/位置/朝向奖励仍为 2/4/1.5，未与本次上身修正同时调整。过强的落足约束可能影响动作平滑，但需要结合误差尺度、触地切换附近的动作突变、实际落足误差和关节速度判断；不能仅凭权重数值大小确认原因。
