# v8：未来接触目标与实际运动一致的联合训练

入口 `run_contact_joint_consistent.sh`，使用相同 `.venv_footstep`，无需新增依赖。原版及 v1–v7 入口和权重保留，正式训练由用户启动。

## 原因与修正

v7 移除参考脚世界轨迹奖励后，缺少“规划目标到查询时刻是否真正实现”的约束。离线监督和少量控制器回放不足以阻止两个网络共同偏离接触语义。

已有 v7 模型的 4 秒物理诊断：前进命令 0.2 m/s，实际平均 0.208 m/s，但规划接触状态不切换，两脚离地间隙的 95 分位约 3 mm，未来支撑位置误差约 46 cm。原 v6 控制器配真实目标时，该位置误差约 6 mm。记录见 `logs/joint_gait_repair_probe_001/report.json`；这是单片段诊断，不代表全数据指标。

v8 增加以下约束，保留速度奖励、上身监督、自然步态软约束及双腿避碰：

- 平地支撑高度投影：用测得关节角、重力方向及脚底碰撞几何推断相对地面高度，替代规划器自由输出的支撑 z，策略不读取世界 root 高度。假设平地行走、至少一脚接地，不支持腾空或台阶。
- 真实未来反馈监督：按查询时间从物理回放取真实未来双脚位置、朝向及几何接触标签，转换到起点 base 偏航坐标系监督规划器。跨 episode、跨未来命令切换、未来长度不足的样本排除。
- 想象轨迹的目标实现损失：按真正的查询时刻检查未来脚位置、接触、朝向。超出 rollout 的查询不截断到末帧，仍有真实未来反馈和离线监督。默认 H=16、50 Hz，覆盖 0.1/0.2/0.3 秒查询。
- 参考相位的支撑/摆动约束、摆动离地不足代价、近地脚切向防滑。使用脚底最低碰撞点速度，允许脚掌滚动；标签是几何近似，不是接触力测量。
- 防遗忘：原控制器在真实目标上的回放权重从 0.1 提至 1，另在规划目标上增加 0.5 的动作回放。教师冻结，主控制器和规划器均更新。

速度只进入规划器；低层仍接收本体历史和 13 维双脚未来接触目标，不直接接收速度。

## 正式训练

从原 v6 和离线规划器重新初始化，避免继承 v7 已退化的步态。使用尚不存在的新日志目录。

```bash
cd /home/waao/Acp_supermimic_vibed
bash run_contact_joint_consistent.sh \
  --data /home/waao/lejulab_platform/data/locomotion/{full_stop_001,forward_continuous_001,backward_continuous_001} \
  --init logs/contact_tracker_train_collision_001/last.pt \
  --planner-init logs/footstep_train_001/best.pt \
  --log-dir logs/contact_tracker_train_joint_consistent_001 \
  --num-envs 4096 --iterations 20000
```

默认编译、CUDA Graph、BF16、EMA 0.995，无世界模型单独预训练；复用 v6 世界模型并在线更新。最初采集 64 步填充缓冲，规划器/真实目标混合采集日程沿用 v7。主控制器和规划器一起微调。

续训用同一入口，将两项初始化参数替换为 `--resume <v8>/last.pt`，换新日志目录；保持网络、优化器、EMA、rollout horizon 和混合日程一致。格式为 `two_foot_consistent_joint_controller_v8`。

```bash
bash play_contact_tracker.sh \
  --checkpoint logs/contact_tracker_train_joint_consistent_001/last.pt \
  --category forward --velocity 0.2 0 0 --query-dt 0.3 --seconds 0
```

播放使用预测目标，环境仍通过参考片段初始化，并在片段结束/摔倒时重置。外部直接落足控制仍可调用 `controller_action(history, contact13)`。

## 指标与验证范围

`train.csv` 新增 `trajectory_consistency_loss`、`hindsight_valid_fraction`、`in_model_query_fraction`、`phase_contact_cost`、`swing_clearance_cost`、`slip_cost`。规划器辅助损失包含真实未来反馈及控制器回放。应同时检查抬脚、接触切换和目标实现，不能只看速度奖励。

单元测试和短训练检查用于验证采样、梯度、编译和模型保存加载。尚未完成 v8 收敛训练，步态改善需要重训后物理验证。

当前 54 项测试通过；实际 v6 权重初始化的 4 次更新、v8 续训、无界面播放及评估通过。真实权重短训练中主控制器与规划器均更新，冻结教师与原权重完全一致，模型参数和损失有限。记录在 `logs/contact_consistent_*smoke_001/` 和 `logs/contact_consistent_smoke_002/`。这几次更新不用于判断训练收敛。

编译 + CUDA Graph + BF16 + EMA 的 4 次更新也通过，见 `logs/contact_consistent_compile_smoke_001/`。首次编译需要数分钟；改动网络或批量形状后可能重新编译。
