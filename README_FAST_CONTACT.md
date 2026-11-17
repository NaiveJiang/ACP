# BlinkMimic 加速迁移版

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
