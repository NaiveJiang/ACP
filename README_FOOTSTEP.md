# 双脚接触规划器实验

> 规划器与 v6 接触控制器的联合微调已接入 v7，见 [README_JOINT_CONTACT.md](README_JOINT_CONTACT.md)。本页继续保留离线规划器训练与数据说明。

新增的真实接触目标控制器任务见 [README_FOOTSTEP_TRACKER.md](README_FOOTSTEP_TRACKER.md)；本页仍介绍离线规划器。

此实验位于 `supermimic/acp/footstep/`，使用独立 Python 环境 `.venv_footstep`、启动脚本 `run_footstep.sh` 和输出目录。原版九部位 ACP 的 `planner.py`、`task.py`、`env.py`、各 Stage 训练及启动脚本保持不变。此阶段是离线监督训练，不运行仿真，也不修改接触跟踪器或世界模型。

## 独立环境

已在当前机器创建 Python 3.11 环境，安装 PyTorch 2.9.0（CUDA 12.8）、NumPy 2.3.4，完整依赖见 `requirements-footstep.lock`。不需要激活其他仓库的环境。

```bash
cd /home/waao/Acp_supermimic_vibed
source .venv_footstep/bin/activate
python -c 'import sys, torch; print(sys.prefix, torch.__version__, torch.cuda.is_available())'
```

在新机器重建：

```bash
uv venv --python 3.11 .venv_footstep
uv pip install --python .venv_footstep/bin/python -r requirements-footstep.lock
```

## 数据与输入

使用 `/home/waao/lejulab_platform/data/locomotion/` 下的三组 `future_contact_v2` 数据；`--data` 可传采集根目录或 v2 目录。读取同组原始 `metadata.json`，按关节名称匹配位置/速度地址，避免腰关节与腿关节错序。

输入为 5 帧状态历史（50 Hz，相邻帧间隔 20 ms，首尾跨度 80 ms）、当前物理速度 `[vx,vy,vyaw]` 和查询时间 Δt。每帧 48 维状态的顺序：

```text
机身坐标系重力方向[3]
机身坐标系角速度[3]（MuJoCo free-joint qvel 的角速度分量已在机身坐标系）
21 关节位置[21]
21 关节速度[21]
```

关节顺序：腰 yaw、左腿 1～6、右腿 1～6、左臂 1～4、右臂 1～4。无当前接触、无世界位置、无未来状态输入。第一版规划器不输入上一时刻动作；原数据的 `ctrl` 不是策略网络动作。将来接入跟踪器时，跟踪器自己的动作历史仍可使用。

网络为 MLP `244 → 256 → 256 → 128` 加三个输出头：接触 logits 2、支撑位置 6、朝向 sin/cos 4。位置和朝向均在**当前** base 位置及偏航坐标系中，z 轴向上。只预测指定时刻接触，不预测这段时间内完整的接触顺序。

## 加载与训练

数据加载只选择 `planner_training_valid=True` 的 `(当前帧, horizon)`。历史不足的帧被跳过，不借用上一片段。目标时间范围由数据提供，默认 0.1、0.2、0.3、0.5、0.8、1.0 秒。窗口内再切换命令的样本不用于规划器，但切换后保持当前命令的过渡仍保留。

同一数据组/类别有多个片段时，按完整片段随机留出 20% 验证；只有一段的连续数据按前 80%、后 20% 划分。每条样本的完整历史和未来目标都必须位于本方时间块，边界样本自动排除。不会把相邻重叠窗口随机分到两侧。

首次运行将所需数组解压至 `data/cache/footstep_v2/` 的按指纹区分的 `.npy` 缓存。后续训练使用 mmap 和最多 16 个片段的映射缓存，不逐样本重新解压 NPZ。输入文件大小、mtime、元数据和特征版本参与缓存标识。

训练按“类别 × 查询 horizon”平衡采样，状态及命令归一化仅使用训练集样本。接触损失为双脚 BCE；位置为按 0.2 m 缩放后的 Smooth L1；朝向为 `1-dot(sincos_pred,sincos_true)`，三个损失相加。位置和朝向只按**真实支撑标签**计算，不按网络预测的支撑状态屏蔽。

正式训练命令（使用新日志目录）：

```bash
bash run_footstep.sh \
  --data \
    /home/waao/lejulab_platform/data/locomotion/full_stop_001 \
    /home/waao/lejulab_platform/data/locomotion/forward_continuous_001 \
    /home/waao/lejulab_platform/data/locomotion/backward_continuous_001 \
  --log-dir logs/footstep_train_001 \
  --device cuda:0 --steps 10000 --batch-size 512 \
  --val-every 500 --val-samples 8192 --seed 42
```

先检查流程时可改为 `--steps 200 --batch-size 256 --val-every 100 --val-samples 2048 --norm-samples 4096`。`--steps` 是梯度更新次数，并非完整遍历数据的 epoch。CPU 可用 `--device cpu`。默认拒绝复用日志目录，防止覆盖权重。

输出包括：`args.json`、`dataset.json`（划分、样本数、缓存指纹、关节顺序）、`train.csv`、初始及定期验证 JSON、`best.pt`、`last.pt`。权重包含网络、归一化、优化器、划分和输入定义；当前入口没有实现 resume。仅加载可信来源的权重。

## 评估与后续接入

从日志目录权重重新评估：

```bash
bash run_footstep.sh \
  --data \
    /home/waao/lejulab_platform/data/locomotion/full_stop_001 \
    /home/waao/lejulab_platform/data/locomotion/forward_continuous_001 \
    /home/waao/lejulab_platform/data/locomotion/backward_continuous_001 \
  --eval-only logs/footstep_smoke_001/best.pt \
  --log-dir logs/footstep_eval_001 --val-samples 2048 --batch-size 256
```

评估复用权重中的划分和归一化，并检查数据指纹。报告左右脚 precision/recall/F1、接触准确率、真实支撑脚上的位置平均误差/p95、朝向平均误差，以及各 horizon 的结果。这是离线预测指标，不代表闭环行走能力；现有数据中速度跟踪较差的动作尚未过滤。

`FootstepPlanner.command()` 可将预测与当前速度、Δt 拼成约定的 16 维指令，预测接触概率阈值为 0.5，摆动脚位置及朝向置零。尚未连接原版九部位跟踪器：后续需另建兼容这 16 维接口的双脚跟踪任务与奖励，再进行闭环验证，原版 ACP 权重不能直接替代。

单元测试：

```bash
.venv_footstep/bin/python -m unittest tools.test_footstep_planner -v
```
