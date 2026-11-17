"""Write the repository comparison report and an exportable terrain figure."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('directory', type=Path)
    ap.add_argument('--videos', type=Path, required=True); args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]; out = args.directory.resolve(); videos = args.videos.resolve()
    s = json.loads((out/'summary.json').read_text()); manifest = json.loads((out/'manifest.json').read_text())
    v = json.loads((videos/'index.json').read_text()); assert not v['pending']
    rel = lambda p: '../' + str(Path(p).relative_to(root))
    g, cats = s['groups'], s['per_category']
    score = lambda x: f"{x['completed']}/{x['episodes']}（{x['completion_rate']:.1%}）"
    pair = lambda split, mode, driver: [g[f'{m}/{split}/{mode}/{driver}'] for m in ('pre','latest')]
    pc, lc = pair('train','nominal','planner'); pr, lr = pair('train','training_randomization','planner')
    lines = ['# RoBan Stage 2 失稳前与最新模型闭环测评', '',
             '日期：2026-11-12。固定比较第 **22,000 次**（第 22,125 次异常前最后一个保存点）和第 **31,000 次**权重。', '',
             f"最新模型的总体完成率下降，翻箱退化最明显。纯 planner 闭环在全部训练参考上的无扰动完成数为 **{score(pc)} → {score(lc)}**，训练范围随机化下为 **{score(pr)} → {score(lr)}**。参考目标驱动的对照也出现下降，说明退化涉及 tracker 在参考命令下的执行能力，不能只归因于 planner 落点预测。", '',
             f"[完整同步对比视频]({rel(videos/'all.mp4')}) · [翻箱合集]({rel(videos/'boxes.mp4')}) · [退化案例]({rel(videos/'regressions.mp4')}) · [视频索引]({rel(videos/'index.json')})。左侧为 22,000 次，右侧为 31,000 次。", '',
             '## 1. 测试范围和可比性', '',
             f"共 **{s['episodes']:,} 回合**：260 条训练参考（21 类，含 60 条翻箱）及 20 条独立翻箱 holdout；两份模型分别运行纯 planner 接触目标与参考目标对照。无扰动用 seed 0，随机化用 seed 0/1/2，全部从参考第一帧开始。holdout 的片段和来源分组与训练集无重叠。", '',
             '两组每个条件的初始根状态、关节状态、速度、摩擦、COM 偏移、关节偏置、地形 variant 和目标前瞻均通过哈希及逐字段校验。参考对照也执行 planner 前向与相同随机采样调用，再强制使用参考目标。GPU 物理求解不能保证逐位确定。', '',
             '随机化包含现有训练范围的 reset 位姿/速度/关节噪声、观测/指令/动作噪声、摩擦系数、COM 和关节偏置；目标前瞻按现有训练分布采样。无扰动使用摩擦 1.0、参考状态重置、1 秒前瞻。两者均保留 reset 穿透抬升修正。', '',
             '纯 planner 表示接触目标 100% 来自各自模型的 planner；根目标仍来自参考，地形扫描来自仿真真值。测评期间不更新模型。这里验证的是参考引导的接触规划执行，不是自由导航或实机泛化。', '',
             '## 2. 总体结果', '',
             '| 数据 / 条件 | 驱动 | 22,000 次完成率 | 31,000 次完成率 | 22,000 / 31,000 完成且终点 ≤20 cm |',
             '|---|---|---:|---:|---:|']
    labels = {'train':'训练参考','holdout':'独立翻箱 holdout'}
    for split in ('train','holdout'):
        for mode, label in [('nominal','无扰动'),('training_randomization','训练随机化，3 seeds')]:
            for driver in ('planner','reference'):
                a,b = pair(split,mode,driver)
                lines.append(f"| {labels[split]} / {label} | {'纯 planner' if driver=='planner' else '参考目标对照'} | {score(a)} | {score(b)} | {a['completed_endpoint_20cm']}/{a['episodes']} / {b['completed_endpoint_20cm']}/{b['episodes']} |")
    lines += ['', '完成判据沿用代码：整条参考执行到末尾，且没有触发根高度差 >0.25 m 或倾斜 R[2,2] 差 >0.8 的终止条件，term_window=0。**终止不一定代表已经摔倒**；动作进度滞后也会触发高度差。终点 ≤20 cm 是额外诊断，未改变原完成判据。', '',
              '随机化三个种子的纯 planner 完成数（每次 260 条训练参考）：', '',
              '| seed | 22,000 次 | 31,000 次 |', '|---|---:|---:|']
    for seed in range(3):
        cells=[s['trials'][f'{m}_train_training_randomization_planner_seed{seed}']['completed'] for m in ('pre','latest')]
        lines.append(f'| {seed} | {cells[0]}/260 | {cells[1]}/260 |')
    lines += ['', '配对统计中，无扰动纯 planner 有 30 条由成功变为失败、1 条由失败变为成功；随机化有 123 个回合由成功变为失败、4 个由失败变为成功。统计按相同片段与 seed 配对，未将净差额当成退化片段数量。', '',
              '## 3. 各类地形的纯 planner 闭环', '',
              '| 类别 | 无扰动：22k | 无扰动：31k | 随机化：22k | 随机化：31k |', '|---|---:|---:|---:|---:|']
    categories = sorted(cats['pre/train/nominal/planner'])
    conditions = [('pre','nominal'),('latest','nominal'),('pre','training_randomization'),('latest','training_randomization')]
    for category in categories:
        cells = [cats[f'{m}/train/{mode}/planner'][category] for m,mode in conditions]
        lines.append('| `'+category+'` | '+' | '.join(score(x) for x in cells)+' |')
    heat = np.array([[cats[f'{m}/train/{mode}/planner'][c]['completion_rate']*100 for m,mode in conditions] for c in categories])
    fig,ax = plt.subplots(figsize=(10,10),layout='constrained')
    im = ax.imshow(heat,vmin=0,vmax=100,cmap='RdYlGn',aspect='auto')
    ax.set_yticks(range(len(categories)),categories);ax.set_xticks(range(4),['22k clean','31k clean','22k randomized','31k randomized'])
    for i,c in enumerate(categories):
        for j,(m,mode) in enumerate(conditions):
            x=cats[f'{m}/train/{mode}/planner'][c]
            color = 'white' if heat[i,j] >= 85 or heat[i,j] <= 15 else '#111111'
            ax.text(j,i,f"{x['completed']}/{x['episodes']}",ha='center',va='center',color=color,fontsize=9)
    ax.set_title('Full planner contact targets: completion from reference frame zero')
    fig.colorbar(im,ax=ax,label='Completion (%)',shrink=.5)
    fig.savefig(out/'completion_by_terrain.png',dpi=180);plt.close(fig)
    lines += ['', '各类样本数不均：翻箱为 60 条，其余各类为 10 条，因此总体完成率按回合计数，不能视为各地形等权平均。`flat` 目录含 9 条纯平地和 1 条含箱体的混合片段；两版模型的纯平地均为无扰动 9/9、随机化 27/27。窄坡的参考片段是路线窗口，完成片段不等于穿越源仓库的完整路线。', '',
              f"![分地形完成率]({rel(out/'completion_by_terrain.png')})", '',
              '## 4. 跟踪精度、翻箱与误差解释', '',
              '| 纯 planner / 条件 | 根 XY RMSE cm：22k / 31k | 关节 RMSE deg：22k / 31k | 左脚局部 RMSE cm：22k / 31k |',
              '|---|---:|---:|---:|']
    for split in ('train','holdout'):
        for mode in ('nominal','training_randomization'):
            a,b=pair(split,mode,'planner')
            cells=[]
            for key,factor in [('root_xy_error_m',100),('joint_angle_rmse_rad',180/np.pi),('left_foot_link_local_error_m',100)]:
                cells.append(' / '.join(f"{x['tracking'][key]['rmse']*factor:.2f}" for x in (a,b)))
            lines.append(f"| {labels[split]} / {mode} | "+' | '.join(cells)+' |')
    lines += ['', 'RMSE 按有效活跃帧合并，包含失败转移；失败更早的模型也更早停止计分，因此误差必须与完成率、终点诊断一起看。接触 precision/recall 衡量与参考的触地时序一致性；对自身 planner 指令的落点误差不等价于跟随参考的误差。', '',
              '翻箱训练参考的独立统计：', '',
              '| 条件 | 驱动 | 22,000 次 | 31,000 次 |', '|---|---|---:|---:|']
    for mode,label in [('nominal','无扰动'),('training_randomization','训练随机化')]:
        for driver in ('planner','reference'):
            a,b=[cats[f'{m}/train/{mode}/{driver}']['box_climb'] for m in ('pre','latest')]
            lines.append(f'| {label} | {driver} | {score(a)} | {score(b)} |')
    lines += ['', '失稳前模型在参考目标下的翻箱完成数明显高于纯 planner，因此其 planner 翻箱能力本来也有限；最新模型在两种目标来源下均未完成本次翻箱回合。这个对照支持 tracker 执行能力也已退化，但不能单独给出 planner 与 tracker 变化各自的因果贡献，进一步分离需要交叉交换两版模块。', '',
              f"物理无效回合：**{len(s['invalid_physics'])}**。它们保留原始记录，保守计为未完成，整回合不参与跟踪与接触误差汇总，也不作为普通策略失败归因。具体原因见 summary.json 的 invalid_physics。", '',
              '## 5. 典型视频', '',
              '代表样本按各类参考时长中位数选择，与模型结果无关；纯平地排除含箱体的混合片段。另选部分随机扰动样本，并明确标注“失稳前成功、最新失败”的退化补充。视频比例不代表整个测试集的成功率。', '',
              '视频为 1920×720、25 fps 原速播放，使用记录的真实 post-step 状态和精确终止帧，未重新生成轨迹。相同相机逐帧同步；失败侧的实体和参考在终止时定格，另一侧继续。彩色点是实际输入策略的接触目标（包含相应指令噪声），不是接触力。', '',
              '| 视频 | 22k / 31k |', '|---|---|']
    for x in v['clips']:
        lines.append(f"| [{x['label']}]({rel(videos/x['file'])}) | {'完成' if x['pre_completed'] else '失败'} / {'完成' if x['latest_completed'] else '失败'} |")
    lines += ['', '## 6. 记录与复现', '',
              f"- [测评配置、数据和源码哈希]({rel(out/'manifest.json')})。",
              f"- [完整统计及配对成功/失败清单]({rel(out/'summary.json')})、[失败详情]({rel(out/'failures.json')})。",
              f"- [视频解码验证]({rel(videos/'verification.json')})。",
              '- 原始逐回合结果和轨迹位于本次 artifacts 目录的各条件子目录；源码快照位于 source/。', '',
              '测评前暂时暂停训练以释放 GPU，并保存第 31,000 次完整模型和优化器状态。训练恢复记录另存 training_resume.json；恢复沿用原训练目标与配置，仿真及 FIFO 回放缓存由现有恢复逻辑重新填充。', '']
    path=root/'docs/roban_stage2_iter022000_vs031000_evaluation_zh.md';path.write_text('\n'.join(lines))
    print(path)


if __name__=='__main__':
    main()
