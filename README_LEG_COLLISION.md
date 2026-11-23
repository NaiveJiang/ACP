# v6：双腿物理自碰撞与可微避碰代价

> v6 训练完成后可接入规划器联合微调，使用 `run_contact_joint.sh`，见 [README_JOINT_CONTACT.md](README_JOINT_CONTACT.md)。

入口 `run_contact_tracker_collision.sh`。在 v5 步态软约束和 v4 上身监督基础上，补上左右腿之间的碰撞响应与避碰训练信号。旧入口、旧 checkpoint 的物理语义保留；已有 v5 训练不会自动切换。

## 为什么需要同时修改物理与奖励

此前 RoBan 的 `get_spec()` 将机器人碰撞几何设置为 `contype=0, conaffinity=1`，只与地形交互，因此双腿之间没有碰撞力。姿态模仿或关节限位都不能代替自碰撞检查。

v6 对左右腿的现有碰撞几何分别设置 `(contype,conaffinity)=(2,5)` 和 `(4,3)`，保留地形位 1。左右腿互相碰撞，同侧相邻几何和其他身体部件保持旧行为，视觉 mesh 不参与碰撞。实际求解使用原模型的胶囊、圆柱和盒子，而非渲染 mesh。这次覆盖左右腿之间的现有碰撞几何，不是全身所有部件的自碰撞系统。

`LegCollisionSim` 通过单独的场景配置启用这些掩码，不更改原始机器人资产。训练、播放、评估和对比回放均按 v6 格式选择同一个环境，避免训练和播放使用不同碰撞设置。原通用 `TrackingSim` 只新增可选场景参数，默认配置保持原行为。

## 世界模型中的避碰信号

物理碰撞力不能直接作为世界模型想象过程的解析奖励，因此另用可微 FK 和几何代理计算两腿间距：原胶囊保留精确中心线/半径，脚底薄圆柱和盒子用保守覆盖球近似。当前每条腿 12 个代理，检查全部 144 对左右腿组合，涵盖两侧大腿、小腿、膝踝附近与脚部原有碰撞几何。

```text
gap = 两中心线最近距离 - 左半径 - 右半径
cost = sum(relu(0.01 - gap)^2 / 0.03^2)
reward -= 0.02 * 2.0 * cost
```

默认安全间距 1 cm、尺度 3 cm、系数 2。间距不足时开始惩罚，穿透时继续增大，梯度可以通过 FK 传回策略；真实仿真奖励和想象奖励使用相同计算。`--self-collision-weight` 可调整惩罚系数，即使设为 0，v6 的物理碰撞仍然开启。

代理间距不是精确 mesh 距离，薄脚底的球代理会偏保守。它约束空间重叠，不用固定步宽或简单的左右脚 y 坐标顺序来禁止所有交叉步。物理离散步长、求解器及碰撞几何近似仍有误差，不能承诺视觉模型绝对零穿透。

## 用户手动训练

建议在新的物理设置下从头训练：

```bash
cd /home/waao/Acp_supermimic_vibed
bash run_contact_tracker_collision.sh \
  --data /home/waao/lejulab_platform/data/locomotion/{full_stop_001,forward_continuous_001,backward_continuous_001} \
  --log-dir logs/contact_tracker_train_collision_001 \
  --num-envs 4096 --iterations 20000
```

新入口默认 **EMA 0.995、世界模型单独预热 0 轮**，编译、CUDA Graph、BF16 开启。初始采集 64 步后，世界模型、策略和 critic 交替训练。

可选 `--init <v5>/last.pt` 迁移完整策略和世界模型（v4 也支持；v3 迁移会新建上身分支），critic、优化器和计数重新开始，统计允许继续适应新分布。物理碰撞动力学改变，旧世界模型需要在线学习新响应，初始化不代表已适配。数据指纹和网络配置须一致。旧版本不能直接 `--resume` 跨任务续训；v6 续训用 v6 权重与新日志目录，保持 EMA 等设置一致。

输入仍是本体历史和 13 维接触计划；没有新增速度指令或接触测量，也无需重新制作数据标签。

## 播放与日志

```bash
bash play_contact_tracker.sh \
  --checkpoint logs/contact_tracker_train_collision_001/last.pt \
  --category forward --seconds 0
```

格式 `two_foot_leg_collision_tracker_v6` 自动选择双腿碰撞环境。EMA 开启时播放保存的平均策略 `pi`，实时策略保存在 `pi_live`。

`train.csv` 在 v5 诊断列后增加 `self_collision_cost`（乘系数、未乘控制周期的平均代价）、`min_leg_clearance_m`（各环境最小代理间距的平均值）、`leg_proxy_overlap_fraction`（至少一对代理重叠的环境比例）。均为日志时刻当前物理状态快照，不是精确接触数或真实 mesh 穿透率。

## 验证

```bash
.venv_footstep/bin/python -m unittest \
  tools.test_leg_collision tools.test_gait_style tools.test_upper_supervised \
  tools.test_stable_contact tools.test_contact_only_tracker \
  tools.test_footstep_tracker tools.test_fast_contact -v
```

37 项测试通过，包括真实 MuJoCo 强制交叉时产生接触和法向力、原物理配置不产生该接触、视觉/地形/同腿过滤保持预期、线段距离的退化与平行情况、接近碰撞时的分离梯度、编译和旧版回归。抽样 20 个训练片段中的 1000 个参考姿态，最小代理间距约 6.1 cm，没有触发 1 cm 安全间距惩罚；此抽查不等于全数据验收。

开发仅运行小规模安装/训练检查，不启动正式长训练。实际自碰撞频率、落足精度与动作自然性仍需用户正式训练后回放验证。

本次 `logs/contact_collision_compile_smoke_001/` 完成从 v5 初始化后的 8 环境、4 次更新检查（编译/CUDA Graph/BF16/EMA 0.995）；`contact_collision_resume_smoke_001/` 验证 v6 优化器和 EMA 续训。新碰撞环境播放执行 10 步、评估执行 2 环境各 4 步，入口均正常。日志中的安全参考状态碰撞代价为 0。首次新碰撞对的 Warp 内核和奖励图编译会增加启动耗时。
