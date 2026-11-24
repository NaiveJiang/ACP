# 先训练 Mimic，再独立采集真实接触

这是一条与接触控制器分开的参考跟踪基线。不会读取 `train_contacts`，
不会在训练结束后自动采集，也不会自动启动后续接触控制器训练。
训练内部仍需采样 `(state, action, next_state)` 学习世界模型；这不等于
向磁盘导出 rollout 接触数据集。

## 1. 手动训练

```sh
cd /home/waao1405/ACP
bash run_blink_mimic.sh \
  --log_dir logs/blink_mimic_train_001 \
  --num_envs 1024 --iterations 20000 \
  --wm_hidden 512 512 512 --horizon 32 \
  --wm_batch 1024 --pi_batch 512 --wm_batches 10 \
  --buffer_steps 1000 --wm_only_iters 0 --pi_ema 0.995 \
  --log_every 25 --ckpt_every 500
```

日志目录必须不存在。默认采用独立的 `.venv_blink` 环境，标签路径不参与
训练。输入动作目录仍是 `data/roban_blink_contact/train`，这里只使用
其中的动作姿态、速度和配套地形，不使用以前生成的几何接触目标。

Actor 输入本体历史和参考窗口，参考偏移默认为 1/5/10/20/30/50 帧：
每帧包括机器人偏航坐标系中的参考根位置差、参考朝向、根线/角速度、
21 个参考关节角和速度；另有当前局部高度扫描。未来窗口不跨片段边界。
模型采用结构化世界模型、Muon + Adam、EMA 和可选 Torch compile，
默认没有世界模型独占预热。64 个初始采样步只用于填充内部回放缓冲。

默认启用参考动作 skip，学习参考关节目标上的残差；物理执行和想象
都使用相同的关节限位投影，探索噪声后再次投影。reset 时按实际 reset
关节目标初始化上一动作，避免从默认姿态人为制造 action-rate 脉冲。
奖励使用现有全身参考位置、姿态和速度跟踪，不含接触标签奖励。
CSV 中接触精度列对这个基线不适用，保留为 0。

checkpoint 使用 `task_interface=reference_mimic_v1` 和独立的
`reference_cfg`。不能从以前的 contact checkpoint 直接续训。
续训时保持参数一致，增加 `--resume PATH`，并使用新的日志目录。

## 2. 独立摩擦与膝角跟踪实验

原版上箱的承重膝跟踪实验可在新训练命令中加入以下选项。

```sh
--ground_friction_range 0.3 1.2 --box_friction_range 0.1 0.6 \
--knee_tracking_weight 1.0
```

两种滑动摩擦各自均匀采样一次，并在每个并行环境内固定。世界模型和
critic 都接收两种数值，策略仍根据观测与参考动作输出控制。为使求解器
使用目标表面摩擦，机器人碰撞体使用两种采样系数的较小值。
`--knee_tracking_weight` 给两侧膝角增加平滑 L1 跟踪代价。`1.0` 是实验
初值，效果需要用原版、镜像和多摩擦系数的闭环回放比较。
独立摩擦配置增加了世界模型和 critic 的输入维度，请使用新日志目录从头训练。

### 箱顶贴地修正版

训练集位于 `data/roban_blink_mimic_flat_soles/train`，有 16 条动作，
覆盖完整序列、上箱、箱顶行走后下箱，以及静止下箱。另留出 `yaw-20`
的 8 条动作做箱体偏转验证。使用下面的命令沿用独立摩擦和膝角跟踪配置。

```sh
cd /home/waao1405/ACP
bash run_blink_mimic.sh \
  --motion_dir "$PWD/data/roban_blink_mimic_flat_soles/train" \
  --log_dir logs/blink_mimic_flat_soles_001 \
  --num_envs 1024 --iterations 20000 \
  --wm_hidden 512 512 512 --horizon 32 \
  --wm_batch 1024 --pi_batch 512 --wm_batches 10 --buffer_steps 1000 \
  --wm_only_iters 0 --pi_ema 0.995 \
  --ground_friction_range 0.3 1.2 --box_friction_range 0.1 0.6 \
  --knee_tracking_weight 1.0 --log_every 25 --ckpt_every 500
```

### S17 执行器对照

在箱顶贴地训练命令中加入 `--s17_actuators`，将日志目录改为
`logs/blink_mimic_s17_001`，从头训练。该配置采用
`kuavo_amp_mjlab` 的 S17 PD 增益、关节侧转动惯量、力矩–转速曲线和
统一的动作比例。每个 5 ms 物理步独立抽取 0–10 步目标延迟，
历史目标进入世界模型和 critic。该配置的世界模型和 critic 输入维度
随之改变，需要使用同配置的 checkpoint 续训。

回放和接触采集会从 checkpoint 读取执行器配置。若训练启用了独立摩擦，
默认使用各自训练范围的中点，回放仍可显式覆盖摩擦值。

### 自碰撞避让

训练时加入 `--self_collision_weight 1.0` 可启用机器人自碰撞，沿用
XML 的碰撞排除规则。物理跟踪和世界模型想象都对超过 1 mm 的近似
自重叠施加二次惩罚，权重、容差和尺度沿用 Kuavo 教师设置。
使用新日志目录从头训练，回放和接触采集会自动沿用 checkpoint 的配置。
训练 CSV 记录重叠帧比例和峰值，回放报告按轨迹记录峰值及超容差帧数。
接触采集为每次尝试报告峰值，接触目标仍来自机器人与地形的求解接触。

### 圆柱与盒体组合脚底

在训练命令中加 `--primitive_soles`，即可加载
`supermimic/acp/assets/biped_s17_verified_handball_fixedhead_primitive_soles.xml`。
每只脚使用后跟圆柱、
前掌圆柱和脚掌中部盒体，中心高度统一为相对脚踝 −30 mm，最低点为 −35 mm。
支持与 `--s17_actuators --self_collision_weight 1.0` 一起使用。回放和
采集从 checkpoint 读取脚底配置，采集将每只脚的三个碰撞体归到同一个脚接触标签。

### 失败片段重采样

参考 BlinkMimic 默认使用 Kuavo 式 reset 混合。它在每条轨迹内按 50 帧
划分采样区间，结合回报、失败位置和 EMA 探测结果。训练失败会提高前一个
区间的 reset 权重。EMA 从第 1000 次迭代起每 125 次在独立的 16 个环境里
探测，重点采样探测失败所在区间及之前两个区间。

默认失败分支权重为 0.30，探测分支为 0.15，另有 0.30 的 reset 强制从
片首开始。可用 `--failure_reset_share`、`--probe_share` 和
`--start_reset_fraction` 调整。CSV 末尾记录失败和探测权重，`phase_*`
表示活动环境这一轮实际的起始进度。使用 `last.pt` 续训会恢复采样统计。
旧 checkpoint 可以续训并开始累积统计。以 `--backward_curriculum` 显式
选择旧反向课程，或用 `--start_from_beginning` 指定始终从片首开始。

### 世界模型窗口与策略学习率

RoBan 参考训练默认使用 8 步世界模型窗口，每轮 10 批，每批 1024 个
窗口。回放容量为 1000 步，每批策略训练使用 512 个起点。世界模型每批
对半数窗口同步镜像状态、动作、关节偏置、目标延迟队列和地形箱体。
策略想象继续用 32 步，未来参考继续用第 1、5、10、20、30、50 帧。

基准学习率为 `4e-4`。策略初始倍率为 4，归一化采样奖励达到 0.75
或到第 1400 轮时，开始用 300 轮将倍率线性降到 1。
CSV 的 `lr` 和 `pi_lr` 分别记录基准与策略值，续训会恢复衰减进度。
可用 `--wm_mirror 0` 关闭回放镜像，或显式覆盖批量、基准学习率和倍率。

## 3. 策略回放

```sh
bash play_blink_contact.sh logs/blink_mimic_train_001/last.pt --ema
```

通用播放入口现在会自动识别参考 Mimic 或接触控制器。先执行闭环仿真，
再通过 http://127.0.0.1:8080 回放。这一步只做可视化评估，不生成
供接触控制器训练的接触力数据集。未确认完整上下箱成功前，不建议
直接制作第二阶段训练集。

需要固定地面摩擦并评估较滑箱面时，在回放命令末尾加
`--ground-friction 1.0 --box-friction 0.6`。

## 4. 确认后另行采集

**这是独立命令，训练脚本不会调用它。** 输出目录必须不存在。

```sh
bash collect_blink_mimic.sh \
  --ckpt logs/blink_mimic_train_001/last.pt \
  --output data/blink_mimic_rollouts_001 --ema
```

采集只加载 actor，不加载世界模型进行虚拟 rollout。使用一个真实
MuJoCo-Warp 世界逐条执行，从每个源片段第 0 帧开始，关闭 reset 和
观测噪声。默认动作噪声为 0、每条一次；重复确定性采集不能视为新数据。
可用 `--motion-dir` 指定独立验证动作目录，但不要混入训练集。

每个控制步为 20 ms，包含四个 5 ms 物理子步。采集直接读取各子步
求解后的 `mujoco_warp.contact_force`，不在 CPU 上重放参考并重新求解。
记录 contact 点、法向、法向力、世界坐标力/力矩、碰撞几何和地形归属。
每个部位的接触标记来自四子步平均法向力，默认阈值 5 N，可用
`--force-threshold` 调整。代表点选取最强实际接触点，不将离面平均点
伪装成真实接触点；原始多接触记录完整保留。

只有完整走到参考末尾、没有物理无效/跟踪终止，且最终 XY 误差不超过
0.3 m、高度误差不超过 0.15 m、姿态/朝向匹配、关节 RMS 误差不超过
0.4 rad、根线速度误差不超过 0.75 m/s 的 episode 才会进入成功集。
人工 `--max-steps` 截断的片段不会作为成功数据导出。

## 输出契约

```text
output/
  manifest.json           每次尝试、拒绝原因、checkpoint 哈希及机器人配置
  attempts/<group>/*.npz  成功和失败的原始状态、动作、原始物理接触记录
  motions/<group>/*_motion.npz
  motions/<group>/*_terrain.npy
  contacts/<group>/*.npz
```

`motions` 和 `contacts` **只包含成功 episode**。若一个成功样本都没有，
会保留 manifest 和失败记录，但不要继续启动接触控制器训练。

导出的根/关节状态是策略实际执行状态，不是原始参考姿态。速度保留
求解器测量值；`joint_vel` 的根角速度采用 MuJoCo free-joint 的根局部
坐标约定，body 速度采用世界坐标。`measured_velocities=True` 告诉
MotionLib 不要用差分替换测量速度。地形保存实际执行时的箱体中心、
姿态和尺寸。

初始 reset 状态没有前一控制区间的实测接触力，所以它只留在原始
attempt 中，不放进训练数组。导出状态第 0 帧是 episode 的 t=0.02 s，
接触力对应其前一个 20 ms 区间。`previous_action` 与状态等长，
`action` 是这些状态之间的实际下发动作，长度比状态少 1。
原始接触事件的帧号同步平移到导出状态的 0-based 索引。

接触标签包含实测力、原始/消抖接触状态和最近实际接触点；另外提取
未来 1.5 s 内持续至少 10 帧的接触起点作为目标。所有目标与 rollout
自身时间轴对齐，不再搭配原始动作的时间轴。标签可由现有 ContactTask
读取，不要再对这个目录运行几何 `make_labels` 覆盖实测标签。
后续接触控制器的可微接触奖励仍是几何近似；实测力用于建立更可信的
参考接触标签，并不意味着现有世界模型已经能预测接触力。

## 验证

`tools/test_reference_mimic.py` 覆盖参考边界、reference skip、动作限位、
实测速度保留、子步力聚合、短接触过滤、成功筛选和导出数据重载。
GPU 冒烟验证仅使用极小网络和少量更新，不代表已经训练出成功策略。
## RoBan 平地跟踪器与 LMM

先用 `bash install_env.sh` 创建 `.venv_footstep`，再准备已经重映射到 RoBan 的 LaFAN1 CSV。全身跟踪器使用完整 LaFAN1，LMM 使用由原版 Motion Matching 映射出的站立、走路、跑步和高抬腿跑步片段。复现 WBC 训练时，将 `ACP_ROBAN_DESCRIPTION` 指向训练所用的 RoBan 资产根目录，其中包含 `xml/` 和 `meshes/`。

```bash
py=.venv_footstep/bin/python
retargeted="$HOME/roban数据/lafan1"
library=logs/motion_matching_upstream_roban_002
export ACP_ROBAN_DESCRIPTION="$HOME/kuavo_amp_mjlab/src/mjlab/asset_zoo/robots/biped_s17_ver"

"$py" -m tools.export_lafan_mimic --source "$retargeted" --out data/roban_lafan1_full_001
"$py" -m tools.motion_matching_reference \
  --database "$HOME/Motion-Matching-main/resources/database.bin" \
  --csv-dir "$retargeted" --add-high-step-run --out "$library"
bash run_lafan_mimic.sh --log_dir logs/roban_lafan1_mimic_001
```

用保留的参考片段验证跟踪器。将参考整理为与训练相同的 `*/*_motion.npz` 和对应的地形文件，再指定该目录。第二条命令把 MuJoCo 闭环执行结果和参考动作渲染到同一视频，需要本地 FFmpeg。

```bash
ACP_ROBOT=roban "$py" -m supermimic.acp.evaluate \
  --ckpt logs/roban_lafan1_mimic_001/last.pt --ema --motion-dir path/to/validation_motions \
  --out logs/roban_tracking_eval.json --record logs/roban_tracking_eval.npz
ACP_ROBOT=roban MUJOCO_GL=egl "$py" -m supermimic.acp.render \
  logs/roban_tracking_eval.npz logs/roban_tracking_eval.mp4
```

跟踪通过验收后，依次训练编解码器、Stepper 和 Projector。下列命令使用新输出目录。

```bash
codec=logs/roban_lmm_codec_001
stepper=logs/roban_lmm_stepper_001
lmm=logs/roban_lmm_001

ACP_ROBOT=roban "$py" -m supermimic.acp.footstep.motion_codec \
  --upstream-roban "$library" --out "$codec"
ACP_ROBOT=roban "$py" -m supermimic.acp.footstep.motion_stepper \
  --codec "$codec" --window 62 --out "$stepper"
ACP_ROBOT=roban "$py" -m supermimic.acp.footstep.learned_motion \
  --upstream-roban "$library" --codec "$codec" --stepper "$stepper/model.pt" --out "$lmm"
ACP_ROBOT=roban "$py" -m supermimic.acp.footstep.reference_planner \
  --lmm "$lmm" --out logs/roban_lmm_eval_001 --seconds 6 \
  --switch 1 0.65 0 0 --switch 3 1.5 0 0 --switch 5 0 0 0
```

`forecast.npz` 含 20、100、200、400、600、1000 ms 的参考。把 `ReferenceWindow.components()` 的六类数组加上 batch 维度，交给 `ReferenceTask.command_from_reference()`，再用跟踪器生成 21 个关节指令。

## Frax S17 数据与方案 1

在 Frax 仓库用 `examples/ik/lafan1/retarget_s17.py` 导出 `data/roban_lafan1_frax_s17_001`。该工具读取原始 LaFAN1 BVH 和 `biped_s17_ver` 的 URDF 与 MJCF。将动作目录放在 ACP 根目录下，用下面的命令按原版 Motion Matching 的选段与时序生成 60 Hz 参考库。指定新的输出目录即可保留已有参考库。

```bash
export ACP_ROBAN_DESCRIPTION="$HOME/kuavo_amp_mjlab/src/mjlab/asset_zoo/robots/biped_s17_ver"
ACP_ROBOT=roban .venv_footstep/bin/python -m tools.motion_matching_reference \
  --database "$HOME/Motion-Matching-main/resources/database.bin" \
  --frax-data data/roban_lafan1_frax_s17_001 --add-high-step-run \
  --out logs/motion_matching_upstream_roban_frax_001

bash run_lmm_frax.sh --check-only
```

全身跟踪器训练完成后，执行 `bash run_lmm_frax.sh`。脚本依次训练编解码器、Stepper 和 Projector，最终将三组推理权重保存到 `logs/roban_lmm_frax_s17_001/model/`。使用不同的参考库位置时传入 `--upstream-roban DIRECTORY`，开始新一轮训练时用 `--out NEW_DIRECTORY` 指定输出。

训练好 LMM 后，运行 `bash play_lmm_tracker.sh --gamepad`，在 MuJoCo 中
用手柄输入实时生成参考，再让全身跟踪器闭环执行。默认使用上述 LMM
和 `logs/roban_lafan1_frax_s17_mimic_002/last.pt` 的 EMA，可用
`--lmm` 和 `--checkpoint` 覆盖路径。手柄沿用 C++ 检索播放器的输入
规则，左摇杆指定移动方向和速度，右摇杆指定航向，LT 锁航向后允许
后退或侧移，A 慢走，Start 重置，RT 配合右摇杆转镜头，空格暂停。
后拉时立即更新目标朝向，移动速度幅值随左摇杆保持。LMM 按新航向
的平滑预测轨迹生成动作，仿真跟踪器通过六个未来参考时刻执行。

#### 方案 1 联合续训

在平地手柄控制中先训练局部根参考的全身跟踪器。每个窗口的未来根部
XY 和 yaw 都相对 LMM 或数据轨迹的当前帧编码。姿态、速度和机身离地
高度奖励在同一局部接口下计算，累计世界 XY 和 yaw 偏差不会拉回原轨迹。
原有 reference checkpoint 保留其训练设置。实时播放器可显式传入
`--root-reference local` 或 `--root-reference world` 做诊断。

```bash
export ACP_ROBAN_DESCRIPTION="$HOME/kuavo_amp_mjlab/src/mjlab/asset_zoo/robots/biped_s17_ver"
bash run_lafan_mimic.sh \
  --reference_root local \
  --motion_dir data/roban_lafan1_frax_s17_001 \
  --log_dir logs/roban_lafan1_frax_s17_local_001
```

使用已训练的 LMM 和全身跟踪器作为初始权重。MuJoCo 闭环执行与手柄播放
共用方向映射及六个未来参考时刻。PPO 更新 Projector 和全身跟踪器，冻结
Codec 和 Stepper。Projector 的探索只加在实际采用的重投影上，前视预测
本身采用确定性推理。速度和航向误差以指令为目标，全身跟踪奖励用 LMM
参考计算，并对倒地扣分。两个训练模块都有初始策略约束。

```bash
bash run_joint_lmm.sh \
  --checkpoint logs/roban_lafan1_frax_s17_local_001/last.pt \
  --out logs/roban_lmm_joint_local_001 --iterations 120
```

默认每轮采 4 个最长 6 秒的回合，覆盖起停、变速、转向和掉头。验证使用
单独的指令序列，每 10 轮比较倒地次数和回报。输出目录的 `last/` 保存
最新配对权重，`best/` 保存通过该验证选择的权重，`train.jsonl` 记录
基线和续训结果。全身跟踪器的部署权重使用 EMA。

```bash
bash play_lmm_tracker.sh --gamepad \
  --lmm logs/roban_lmm_joint_local_001/best/model \
  --checkpoint logs/roban_lmm_joint_local_001/best/controller.pt
```

## Frax upstream 闭环采集与方案 2

全身跟踪器训练完成后，直接指定参考跟踪器的 checkpoint 和新的输出目录。

```bash
bash collect_frax_upstream.sh \
  --ckpt logs/roban_lafan1_frax_s17_mimic_002/last.pt \
  --output data/roban_frax_upstream_contacts_001
```

默认使用 EMA、零动作噪声及 `logs/motion_matching_upstream_roban_frax_001`。
可用 `--upstream-roban DIRECTORY` 指定其他 Frax S17 upstream 库。
默认执行库里的全部站、走、跑及镜像片段。默认把长轨迹分成 8 秒的独立
episode，每段从本段第一帧重置。内部切段会保留原轨迹的后续参考供 WBC
前视，实际执行和落足标签仍止于该段末尾。需要按源片段连续执行时用
`--window-seconds 0`。
先冒烟运行时可加 `--case run_high_step_5731 --window-seconds 4 --max-clips 1`。
多个 `--case` 可以同时指定。`--max-steps` 截断的段只用于诊断。

仿真中全身跟踪器使用自己的本体历史和六个未来参考时刻，在一个
MuJoCo-Warp 世界闭环执行。原始参考从 60 Hz 对齐到控制频率 50 Hz，
保留 upstream 帧号和 LaFAN1 源行号。每个 20 ms 控制步记录实际状态、
动作和左右脚脚底中心位姿，接触力取前四个求解子步。默认以平均法向力
达到 5 N、经消抖后持续至少 5 帧作为承重确认。落足事件取这段承重的
第一帧，位置使用该帧实测脚底中心，朝向使用脚的世界 yaw。可用
`--force-threshold` 和 `--min-support-steps` 调整这两个阈值。

输出目录包含以下内容。

| 路径 | 内容 |
| --- | --- |
| `references/` | 策略实际使用的 50 Hz Frax 参考及 upstream 来源索引 |
| `attempts/` | 所有尝试的仿真状态、时间对齐参考、脚底位姿、求解力和原始碰撞记录，含 reset 帧 |
| `motions/`、`contacts/` | 通过既有动作跟踪验收的物理轨迹和兼容 `ContactTask` 的接触标签 |
| `samples/` | 通过验收的方案 2 样本，已排除无接触力测量的 reset 帧 |
| `manifest.json` | checkpoint、来源、每段结果、拒绝原因和可用窗口统计，`completed` 标明是否已处理全部选定尝试 |

每个 `samples/<case>/*.npz` 分片的字段如下。

- `qpos`、`qvel`、`observation`、`previous_action` 是实际状态、WBC 本体观察和上一动作。训练时从同一分片取本体历史。`reference_qpos`、`reference_qvel` 与每一帧一一对齐，可用于动作跟踪或姿态奖励。
- `sole_pos_w`、`sole_quat_w`、`foot_normal_force_n`、`foot_force_w`、`foot_contact_raw`、`foot_support` 均按左脚、右脚排序。位姿采用世界系和 `wxyz` 四元数，力以 N 为单位。
- `event_frame`、`event_foot`、`event_pos_w`、`event_yaw_w` 是按时间排序的落足事件表。对第 `t` 帧，用 `lo, hi = future_event_bounds[t]` 取事件表的 `[lo:hi]`，覆盖严格晚于当前帧、距当前帧最多 1.5 秒的全部事件。相对时间是 `(event_frame[lo:hi] - t) * 0.02`。
- `next_pos_base`、`next_heading_base`、`next_time_s` 提供每只脚的第一个落足目标，基底坐标使用当前实测根位置及仅含 yaw 的坐标轴。朝向编码为 `[sin(yaw), cos(yaw)]`，通过 `next_event_valid` 识别目标是否存在。
- `sample_valid` 标记有完整 1.5 秒窗口且留足承重确认和消抖余量的帧，训练完整序列时使用该掩码。`next_event_known` 可用于逐脚监督，允许尾部已确认的正样本，同时屏蔽未来未观测到的负样本。

该库没有同步手柄输入，样本没有速度命令字段。若正式采集得不到任何完整
窗口，命令会以非零状态退出，可从 manifest 查看哪些片段未通过验收。

### 训练方案 2 接触控制器

全身跟踪器采集完成后，把输出目录直接传给独立训练入口。

```bash
bash run_rollout_contact_tracker.sh \
  --data data/roban_frax_upstream_contacts_001 \
  --log-dir logs/roban_frax_contact_controller_001 \
  --iterations 20000
```

训练默认采用 WBC 的 BlinkMimic 配置，包括本体历史 10 帧、EMA 和训练期间
同步更新的世界模型。策略接收本体历史和未来 1.5 秒内全部有序落足事件，
输出 21 个有界的 PD 关节动作。每个事件槽有 9 个值，依次为有效位、左脚位、
右脚位、剩余秒数、局部 XYZ 以及相对偏航的正弦和余弦。事件槽数由样本
最大值确定并保存到 checkpoint，可用 `--event-slots` 增大固定容量。
`encode_event_plan` 是后续 LMM 接入时应共用的局部坐标编码。

仿真和想象中均按机器人当下根位置和 yaw 编码固定的世界落点。全身走姿奖励
使用 `motions/` 中的实测 rollout，载入时会与 `samples/` 中的实际状态核对。
支撑时序、落地位置和朝向也来自同一次 rollout。使用实际支撑标签奖励触地，
在落地后约 60 ms 内评分落点和偏航，摆动和换重心由局部走姿奖励约束。
支持双脚腾空，另外惩罚脚滑、过快下落、两腿自碰撞和动作变化。原始 Frax
全身参考用于对照采集质量。Actor 的输入是本体历史和落足计划。

训练和验证按原始动作分组，同一条动作的切段和镜像都进入同一侧。可用
多个 `--data` 目录补充不同来源的成功 rollout。采集目录应有 `completed`
为真的 manifest，且包含足够长的样本和至少两条不同的原始动作。

```bash
ACP_ROBAN_DESCRIPTION="$HOME/kuavo_amp_mjlab/src/mjlab/asset_zoo/robots/biped_s17_ver" \
  .venv_footstep/bin/python -m supermimic.acp.footstep.evaluate_tracker \
  --checkpoint logs/roban_frax_contact_controller_001/last.pt \
  --out logs/roban_frax_contact_eval_001
```

当前验证入口在保留的轨迹上计算闭环生存、支撑位置误差和几何接触匹配。
几何指标用于首轮对比，触地力、时间和脚滑仍需要求解器接触记录评估。

### 方案 2 的 LMM 落足规划器

第一阶段冻结 LMM。训练输入使用实测本体历史和从对应 upstream 潜变量
自回归展开的 LMM 姿态，监督来自 WBC 实测的双脚承重及落足。预测头逐帧
预测左右脚支撑与脚底位姿修正，确认承重后输出 1.5 秒内按时间排序的事件。
位置和朝向标签使用查询时的实测根位置和 yaw。LMM 的姿态只提供相对足底
运动，新落点以控制器的实际根节点为锚，LMM 世界位置漂移不会加到落点上。
选定的临近落点会保存在地面坐标中，持续跟踪到落地。
验证按源动作划分，报告和几何基线相比的落足精确率、召回率、配对后的
时间、平面位置及 yaw 误差。配对需要脚别一致且落地时间相距不超过 0.2 秒。
留出划分用于检验落足头。冻结的 LMM 可能在此前训练见过留出片段的参考动作。
旧版 v1 落足头使用 LMM 根坐标标签，需要从已有实测数据重新训练第一阶段。

```bash
# 现有兼容 LMM 可直接验证。需要按较长时域训练 LMM 时可选 100 帧。
bash run_lmm_frax.sh --stepper-window 100 --out logs/roban_lmm_frax_s17_001

bash run_lmm_contact_planner.sh stage1 \
  --lmm logs/roban_lmm_frax_s17_001/model \
  --codec logs/roban_lmm_frax_s17_001/codec \
  --upstream-roban logs/motion_matching_upstream_roban_frax_001 \
  --data data/roban_frax_upstream_contacts_001 \
  --out logs/roban_lmm_foothold_stage1_001
```

第二阶段先生成起停、走跑切换和左右转向的 LMM 参考，再运行 WBC 教师采集
带指令的实测接触。传入已经训好的方案 2 接触控制器，用 MuJoCo 闭环数据
做 PPO 微调，落足头同时用教师样本及通过验收的实际闭环样本继续做监督。
LMM 的三组权重保持冻结。计划用世界落点执行，重规划时保留即将落地的
目标。发生后续指令变化的预测窗口不进入落足监督。

```bash
bash run_lmm_contact_planner.sh stage2 \
  --lmm logs/roban_lmm_frax_s17_001/model \
  --head logs/roban_lmm_foothold_stage1_001/best.pt \
  --teacher logs/roban_lafan1_frax_s17_mimic_002/last.pt \
  --controller logs/roban_frax_contact_controller_001/last.pt \
  --out logs/roban_lmm_foothold_stage2_001
```

`--scenarios` 可传入 JSON，键为场景名，值为从 0 秒开始的
`[秒数, vx, vy, vyaw]` 行列表。`--prepared` 可复用上一次输出中的
`prepared/`，其中包含场景参考和 WBC 回放，使用时可省略 `--teacher`。
场景参考在配置时长后保持最后一个指令，为末尾的落足确认留出余量。
接管比例按完整回合选择，初始默认 0.15。每次提高前至少检查 3 个有运动的
计划回合，最近最多 12 个回合中至少 60% 要通过存活、进度和实测落足验收。
落足匹配要求同脚且在
0.12 秒、0.12 米及 0.45 弧度内。训练日志单独记录计划回合数及通过数。
`head.pt` 为落足头，`last.pt` 含更新后的控制器与落足头。
`latest_rollouts/` 保存最近一轮的实际状态、测得的法向力、支撑点速度、
已下发的目标和实测落足事件，可用于排查接管之后的动作。
第二阶段的 PPO checkpoint 可与同一套 LMM 权重复用。采用旧落足坐标系的
第二阶段权重需回到原始接触控制器重新做联合微调。通用
`evaluate_tracker` 使用实测计划评估控制器，联合闭环结果见第二阶段
`train.jsonl`。BlinkMimic 第一阶段入口使用其自己的 checkpoint。
