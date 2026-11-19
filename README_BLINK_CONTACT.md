# BlinkMimic 全身接触控制器

这是 ACP Stage 1 的 BlinkMimic 算法适配，不是平地双脚 fast 入口。
actor 使用本体历史、九部位未来接触目标、目标方向和局部地形扫描，
不直接读取参考关节姿态。脚、手、膝、肘和骨盆均参与接触标签。

## 数据

使用已预览的三类长时程序列及镜像。`data/roban_blink_contact/train`
包含 0°、+20° 的 12 条动作，`validation` 包含 -20° 的 6 条。
验证集只是留出的箱子摆放条件，复用了相同底层动作，不能当作独立动作泛化测试。
各动作的箱体几何与预览文件逐字节一致。

RoBan XML 和 meshes 从预览使用的资产复制到 `data/roban_description`。
原资产不修改。支持把独立 handball mesh 映射为手接触部位，保留其
几何、质量和运动学变换；标签和仿真共用这一模型。

## 方法

- 世界模型采用 BlinkMimic 的半隐式结构积分及位置/姿态残差，积分权重 0.625。
- 世界模型和策略使用仓库已归属许可的 Muon + Adam 优化器。
- 保留 BF16、可选 Torch compile，加入策略 EMA 并保存 `pi_ema`。
- 多步世界模型损失以完整状态增量的统计归一化，不能误用结构残差的尺度。
- 保留 ACP 的真实地形扫描、九部位接触命令和 Stage 1 奖励。
- 不启用依赖参考关节姿态的 reference-action skip，避免泄漏到接触 actor。
- 目前未移植全身任务的 CUDA Graph 更新、在线 WM 镜像和评估探针。
  这不是原版 BlinkMimic 所有加速/采样功能的完整复刻。

checkpoint 的 `args.learner` 为 `blink`，完整文件还带
`training_engine=blink_whole_body_contact_v1`。不允许和普通 SuperMimic
checkpoint 交叉续训。EMA 是另存的 `pi_ema`，默认 `pi` 仍为在线策略，
现有只加载策略的评估入口可以读取；Stage 2 的世界模型加载尚未适配。

## 接触标签

在 ACP 根目录使用独立 `.venv_blink`，通过 uv 运行：

环境的核心依赖固定在 `requirements-blink.txt`，使用 Python 3.11。
其中固定 mjviser 0.0.16，避免新版引入 MuJoCo 4 core 并覆盖 3.11 的绑定。

```sh
ACP_ROBOT=roban .venv_preview/bin/uv run --no-project \
  --python .venv_blink/bin/python python -m supermimic.acp.make_labels \
  --motion_dir data/roban_blink_contact/train \
  --contact-profile roban_knee_caps_v1 --workers 4
```

标签基于 MuJoCo 几何距离，不是实测接触力。几何接触、穿透情况和
不允许接触的身体部位须结合 `train_contacts/summary.json` 检查。
参考动画看起来合理不代表它已经满足真实动力学。

## 训练

正式训练入口如下，日志目录必须是新目录。
先前的 `logs/blink_contact_train_001` 已按用户要求停止，日志和权重保留。
请手动启动，并使用新的日志目录：

```sh
bash run_blink_contact.sh \
  --log_dir logs/blink_contact_train_002 \
  --num_envs 1024 --iterations 20000 \
  --wm_hidden 512 512 512 --horizon 16 \
  --wm_batch 512 --pi_batch 512 --wm_batches 4 \
  --buffer_steps 512 --wm_only_iters 500 \
  --log_every 25 --ckpt_every 500
```

前 500 次只训练世界模型；随后开始策略/critic 更新。首次编译会有延迟。
`--no-compile --amp none` 用于定位问题；`--resume PATH` 要保持网络、
数据、任务及 learner 配置一致。日志保存 `train_log.csv`、`last.pt`
和周期 checkpoint。想象回报、想象存活率不是实际任务成功率。

小规模验证可改用新日志目录、`--num_envs 8 --iterations 4`
`--wm_hidden 32 32 --horizon 4 --wm_batch 8 --pi_batch 8`
`--wm_batches 1 --wm_window 2 --buffer_steps 32 --warmup_steps 16`
`--wm_only_iters 1 --critic_epochs 1 --critic_minibatches 1`
`--log_every 1 --ckpt_every 4 --no-compile --amp none`。

## 本机验证记录

8 项测试和 2 项子测试通过；8 环境、4 次更新的 GPU 冒烟测试完成，
世界模型、策略、critic 和 EMA checkpoint 均为有限值。
结果位于 `logs/blink_contact_smoke_001`，不是收敛权重。

训练集 12 条参考的接触标签已生成，共 11820 帧；接触部位最大几何
穿透未超过 2 cm。逐片段结果保存在 `data/roban_blink_contact/contact_audit.json`。
这是初步几何检查，不是闭环任务成功率验证。

训练控制台：

```sh
tail -f logs/blink_contact_train_001.console.log
```

## 播放训练策略

以已完成的 `blink_contact_train_003` 为例：

```sh
bash play_blink_contact.sh logs/blink_contact_train_003/last.pt --ema
```

先在 GPU 上运行确定性仿真评估并录制，再打开 Viser 服务。
浏览器访问 http://127.0.0.1:8080 ，用 `Sequence` 选片段，`Play` 暂停，
`Frame` 拖动进度。这是实际策略仿真结果的回放，不是参考动作，
也不是实时推理控制。当前仍由参考轨迹提供接触计划，尚未接上层规划器。
失败时只播放存活前缀，不把失败后的停放状态当作有效动作。

不带 `--ema` 时使用在线策略 `pi`。端口可在末尾加 `--port 8081`。
每次录制保存在独立的 `logs/blink_play_<variant>_*` 目录中，包含
`evaluation.json` 和 `rollout.npz`。已有录制可直接播放：

```sh
bash play_motion_matching.sh --rollout logs/blink_play_ema_XXXXXX/rollout.npz
```

无窗口检查使用 `--ema --check`。普通 `bash play_motion_matching.sh`
仍然只播放拼接参考轨迹，不会加载训练权重。
