# 上身与抖动修正版（v3）

> 最新上身直接动作监督使用 `run_contact_tracker_upper.sh`，见 [v4 说明](README_UPPER_SUPERVISED.md)。本页和 v3 入口保留用于原权重播放、续训和对照。

入口 `run_contact_tracker_stable.sh` 使用 Blink 快速训练引擎，默认编译、CUDA Graph、BF16。旧 16 维、旧 13 维入口及权重保持原行为。正式训练由用户自行启动；开发验证只运行了 4 次更新的小规模检查。

## 修正内容

策略仍然接收 5 帧本体观测和 13 维接触目标：查询时间、双脚接触、位置、朝向。不加入期望速度、参考关节姿态或当前接触测量。下肢保持接触优先，没有恢复整条腿的关节轨迹模仿。

上身用数据中的腰部和双臂 9 个关节姿态、关节速度作为训练监督，保留参考摆臂。原来权重 0.05 的指数姿态奖励，在严重偏离时奖励及梯度都很小；现在改成持续提供纠正梯度的平方误差代价，权重 2，腰部姿态额外乘 3。上身关节速度误差权重 0.03。参考误差只提供给奖励和特权 critic，播放时 actor 不需要参考上身动作。

动作变化代价从 `-0.01 * sum((a-a_prev)^2)` 提高为 `-0.1`；全身关节速度平方代价从 `-0.0001` 提高为 `-0.0005`。直立/高度奖励分别从 0.5/0.25 提高为 1/0.5。接触、位置、朝向权重仍为 2/4/1.5。

每个关节的动作范围根据物理关节范围的 95% 软限位、默认角度和 action scale 计算，并为仿真的 ±0.01 rad 关节零偏预留空间。策略使用非对称 tanh，零输出仍对应默认姿态。在世界模型想象和真实物理执行中采用相同的归一化动作含义。采集加入探索噪声后再次按同一范围限幅，回放缓冲和世界模型记录的都是实际下发动作；没有在播放阶段单独添加平滑滤波。

这些是针对已发现原因的修正，尚未用完整训练证明动作质量。旧诊断位于 `logs/contact_tracker_diagnosis_001/`。接触目标未完全约束上身，摆臂也存在多种可行动作，因此仍需训练后回放对比；现有数据不覆盖任意大转角落足。

## 正式训练

建议从头训练新版本，不使用旧异常姿态权重续训。v3 的动作参数化、critic 和奖励发生变化，格式独立，旧 v1/v2 checkpoint 不接受直接 `--resume` 或 `--init`。

```bash
cd /home/waao/Acp_supermimic_vibed
bash run_contact_tracker_stable.sh \
  --data \
    /home/waao/lejulab_platform/data/locomotion/full_stop_001 \
    /home/waao/lejulab_platform/data/locomotion/forward_continuous_001 \
    /home/waao/lejulab_platform/data/locomotion/backward_continuous_001 \
  --log-dir logs/contact_tracker_train_stable_001 \
  --num-envs 4096 --iterations 20000
```

第一次编译需要等待；日志目录必须尚不存在。入口为 fast 引擎增加 `--controller stable`，其余参数见 `README_FAST_CONTACT.md` 或 `bash run_contact_tracker_stable.sh --help`。奖励配置和动作上下限随 checkpoint 保存。

## 播放、续训及验证

```bash
bash play_contact_tracker.sh \
  --checkpoint logs/contact_tracker_train_stable_001/last.pt \
  --category forward --seconds 0
```

续训使用上述训练命令，换新日志目录并增加 `--resume logs/contact_tracker_train_stable_001/last.pt`。保存格式为 `two_foot_stable_contact_tracker_v3`，播放和评估按格式自动选择策略。

```bash
.venv_footstep/bin/python -m unittest \
  tools.test_stable_contact tools.test_contact_only_tracker \
  tools.test_footstep_tracker tools.test_fast_contact -v
```

针对问题的检查包括：严重上身偏差仍有纠正梯度；上身监督不约束腿部关节姿态；极端输出与探索噪声在零偏两端均不产生越界 PD 目标；速度命令不影响策略输入/奖励；编译梯度及新 checkpoint 读写。另保留旧版本和加速引擎的回归检查。
