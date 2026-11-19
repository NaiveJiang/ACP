# RoBan 长时程参考轨迹预览

这是运动学拼接实验，不是策略执行，不代表已经能在物理仿真中完成。
原始动作不修改，输出在 `data/motion_matching_preview/`。

## 播放

已配置本机独立环境 `.venv_preview`，在 ACP 根目录执行：

```sh
bash play_motion_matching.sh
```

浏览器打开 http://127.0.0.1:8080 。默认播放完整原版序列。
`Sequence` 选择轨迹，`Play` 暂停/继续，`Frame` 拖动帧，`Speed` 调速。
播放到末尾会回到第 0 帧，这个循环重置不是轨迹内部的动作接缝。
鼠标拖动/滚轮控制视角。左侧显示当前动作阶段和接缝帧。
绿色线是根位置轨迹；实体机器人由关节角经 MuJoCo 正向运动学生成。
如需其他端口，增加 `--port 8081`；远程访问需显式 `--host 0.0.0.0`
或使用 SSH 端口转发。

## 24 条轨迹

四种序列各有原版和左右镜像，以及三种箱子摆放。

| 名称前缀 | 阶段 | 时长 |
| --- | --- | --- |
| `ground_up` | 地面走路 → 上箱 | 约 15.08 s |
| `top_down` | 箱顶走路 → 下箱 | 约 14.44 s |
| `stand_down` | 箱顶静止 1 s → 从原下箱入口起步 | 约 10.86 s |
| `full` | 地面走路 → 上箱 → 箱顶走路 → 下箱 | 约 29.52 s |

在已有实验环境中，新增六条 `stand_down` 的预览可用以下命令打开。

```sh
bash play_motion_matching.sh --directory data/motion_matching_preview_stand_down
```

`stand_down` 从原下箱入口姿态开始，静止段之后对下箱参考的前 0.4 s
做缓启动重定时。后续下箱姿态及其与箱沿的相对位置保持一致。

箱顶站起阶段的贴地修正版可用以下命令打开。

```sh
bash play_motion_matching.sh --directory data/motion_matching_preview_flat_soles
```

修正版使用碰撞脚底网格和箱顶平面，只对进入箱顶支撑区的脚做腿部逆运动学。
修正权重随脚底高度、倾角和距箱沿的距离平滑变化，保留跨箱沿的抬脚动作。
生成时可通过 `--align-ascent-soles` 启用。

箱体中心 XY 分别设为 `(3, 0)`、`(3, 0.6)`、`(3, -0.6)` m，
对应 yaw 为 `0°`、`+20°`、`-20°`。宽度设为 2 m，箱顶高度约 0.547 m。
长度按各序列的实际箱顶行走跨度生成，不能混用不同序列的地形。
偏转场景的地面接近段初始朝向保持世界 X 正向，再平滑转向箱子朝向。
默认预览的箱顶行走整体沿箱子的长轴。

### 箱顶偏角下箱预览

```sh
bash play_motion_matching.sh --directory data/motion_matching_preview_angled_descent
```

选择 `top_down_original_yaw+20` 或 `top_down_original_yaw-20`。
通过 `--turn-top-approach` 生成的这两种箱体角度及各自镜像，从世界 +X
方向起步，在箱顶行走过程中平滑转到原下箱入口。用箱顶走路完成转向，
`stand_down` 继续使用静止起步的原下箱参考。预览沿用贴地版的上箱数据。

### 静止起步的上下箱预览

```sh
bash play_motion_matching.sh --directory data/motion_matching_preview_stand_entries
```

- `stand_up_original_yaw+00` 从原上箱入口姿态静止 1 s，随后缓启动进入
  贴地版上箱动作。使用 `--include-stand-up` 生成。
- `stand_down_original_yaw+00` 在箱顶静止 1 s，再从原下箱入口起步。
- 两类静止起步各保留 `yaw+00` 的原版和镜像。走路接上下箱与完整序列
  各保留三种箱体摆放。

预览共 22 条。重建时同时使用 `--align-ascent-soles --turn-top-approach
--include-stand-up`。

## 方法及范围

- 使用相邻 `php_parkour` 的 `motion_matching_search_fixed` 搜索函数。
- 将 PHP 的 27 维匹配特征适配到 RoBan：双脚位置/速度、根速度、
  三个未来时刻的位置和朝向；统一使用根偏航坐标系和 XY 水平面。
- 对特征做分组加权标准化，加 0.05 数值下限，避免静止特征除零。
- 直接调用 PHP 的向量和四元数惯性化函数，半衰期为 0.1 s。
- 通过 RoBan 的关节位置和根姿态重新计算 body 轨迹，避免直接插值
  所有 body 导致骨架不一致。速度从最终轨迹重新差分。
- 上下箱使用 `supermimic_teacher_hand_clearance` 的两条参考及镜像。
  平地使用清理后的前向直线动作库，在候选窗口中匹配接入相位。
- 下箱段随平台远端沿 X 平移，保持它与下箱边缘的相对位置。
- 偏转接近段使用显式平滑偏航/路径变形，不是从转弯动作库检索出的
  真实转弯，也没有执行足端锁定 IK。可能存在滑脚，需在预览中检查。

这不是对 PHP G1 完整运行时的移植；当前是带阶段约束的离线窗口搜索。
箱子位姿是预设场景参数，不是视觉识别结果。
`manifest.json` 记录源文件哈希、匹配窗口、PHP 提交号和连续性检查。
脚部支撑检查仅用脚踝原点减 35 mm 近似，不能代替精确碰撞、接触力、
平衡和闭环控制验证。未生成 ACP 接触标签，暂不要直接用于正式训练。

## 重建与检查

三个相邻仓库的位置默认如下：

```text
ACP/
php_parkour/
kuavo_amp_mjlab/
```

新的环境可使用 `uv venv .venv_preview` 和
`uv pip install --python .venv_preview/bin/python -r requirements-preview.txt`
安装。运行时需要 `uv` 在 PATH 中，或已有本机环境内的 uv。

```sh
uv run --no-project --python .venv_preview/bin/python \
  tools/roban_motion_matching.py --output data/motion_matching_preview_new
uv run --no-project --python .venv_preview/bin/python \
  tools/play_motion_matching.py --directory data/motion_matching_preview_new --check
```

输出目录必须不存在，不覆盖旧实验。生成器支持 `--php`、`--motions`、
`--robot` 路径参数，以及 `--walk-seconds`、`--top-seconds` 时长参数。
播放目录中的 manifest 保留模型 XML 的绝对路径；移动仓库后请重新生成。
