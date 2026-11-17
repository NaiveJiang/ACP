"""Produce the repository report and standalone completion figure from verified results."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def pct(x):return f'{100*x:.1f}%'
def rate(r):return f"{r['completed']}/{r['episodes']} ({pct(r['completion_rate'])})"
def num(x,scale=1):return '—' if x is None else f'{x*scale:.2f}'


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path);args=ap.parse_args()
    out=args.directory.resolve();root=Path(__file__).resolve().parents[1];s=json.loads((out/'summary.json').read_text());meta=json.loads((out/'selection.json').read_text());manifest=json.loads((out/'manifest.json').read_text());video=json.loads((Path(meta['video_dir'])/'index.json').read_text());assert not video['pending']
    selection=json.loads((out/'video_selection.json').read_text());labels={i['category']:i['label'] for i in selection['clips'] if i['id']==i['category']};labels['flat']='平地目录（含箱体样本）';labels['box_climb']='箱体翻越'
    categories=list(labels);g=s['groups'];pc=s['per_category'];mainkey='train/training_randomization/phase0'
    def link(path):return '../'+str(Path(path).relative_to(root))
    mode_names={'nominal':'无扰动','training_randomization':'训练范围扰动','additional_reset_noise':'训练扰动＋额外初态噪声','friction_0.3':'固定摩擦 0.3，其余无扰动','friction_1.2':'固定摩擦 1.2，其余无扰动','goal_0.5':'前瞻 0.5 s，其余无扰动','goal_1.5':'前瞻 1.5 s，其余无扰动'}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import numpy as np
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plot_keys=['train/nominal/phase0',mainkey,'train/additional_reset_noise/phase0']
    values=np.array([[pc[k][cat]['completion_rate'] for k in plot_keys] for cat in categories])
    fig,ax=plt.subplots(figsize=(9.8,11.4));im=ax.imshow(values,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(3),['无扰动','训练范围扰动','额外初态噪声'],fontproperties=font,fontsize=11)
    ax.set_yticks(range(len(categories)),[labels[c] for c in categories],fontproperties=font,fontsize=10)
    for i,cat in enumerate(categories):
        for j,k in enumerate(plot_keys):
            r=pc[k][cat];ax.text(j,i,f"{r['completion_rate']*100:.1f}%  ({r['completed']}/{r['episodes']})",ha='center',va='center',color='white' if values[i,j]>.65 else 'black',fontsize=9)
    ax.set_title(f"RoBan tracker 第 {s['iteration']} 次迭代\n从第 0 帧开始的全程完成率",fontproperties=font,fontsize=15,pad=17)
    fig.colorbar(im,ax=ax,fraction=.025,pad=.025,label='Completion rate');fig.tight_layout()
    fig.savefig(out/'completion_matrix.png',dpi=180);fig.savefig(out/'completion_matrix.pdf');plt.close(fig)
    main=g[mainkey];hold=g['holdout/training_randomization/phase0'];nom=g['train/nominal/phase0']
    lines=[f"# RoBan tracker 第 {s['iteration']} 次全面测评",'',
        f"2026-11-11 固定权重的真实物理闭环测评。训练集从第 0 帧开始，在训练范围扰动下完成 **{rate(main)}**；无扰动完成 **{rate(nom)}**。独立翻箱保留验证集在训练扰动下完成 **{rate(hold)}**。全部 {s['episodes']} 个正式回合中，数值异常或活动环境接触溢出为 **{s['invalid_physics']}**。",'',
        '主要短板是翻箱对初态和低摩擦条件的敏感性，以及部分坡面的重复失败。相同翻箱协议下，本次权重低于重新测试的第 73500 次权重；不能只按迭代编号选择用于翻箱的模型。', '',
        '## 测评范围与口径','',
        f"- 固定权重：`{Path(s['checkpoint']).relative_to(root)}`；选择时间 `{meta['selected_at']}`。训练继续运行，整套测评使用同一权重。",
        f"- SHA-256：`{s['checkpoint_sha256']}`。",
        '- 训练参考集：21 类、260 条，其中翻箱 60 条，其余 20 类各 10 条。全程无扰动各运行一次；训练范围扰动每条运行 5 个种子（0–4）。',
        '- 独立翻箱验证集：20 条，按采集时几何分组留出；分组与训练集无重叠，动作、地形、接触标签哈希已校验。常规 12 条、边缘 4 条、初态变化 4 条。无扰动一次，训练范围扰动 5 个种子。',
        '- 真实物理仿真由 tracker 输出驱动，接触目标来自参考标签。该结果评估参考指令 tracker；不代表 planner 自主规划效果。其他 20 类尚无独立保留集。',
        '- 初始状态使用对应参考帧的姿态和速度，历史观测由首帧重复填充。默认初态噪声关闭，保留穿透修正。额外初态噪声测试单独汇总。',
        '- 训练扰动包括观测、接触目标与扫描、动作噪声，摩擦、关节偏置、质心偏移和每回合 0.5–1.5 s 的目标前瞻。噪声范围沿用 checkpoint；固定摩擦和固定前瞻测试均在其余无扰动条件下进行。',
        '- 严格完成：未触发失败且到达参考末帧，活动物理状态与指标有限、无接触溢出。参考高度差 >25 cm 或 Rzz 差 >0.8 会终止；该判据不等同于肉眼摔倒。水平误差不直接触发此终止。完成率以全部尝试为分母，无效仿真也保守计为未完成；另在 JSON 中提供仅有效回合完成率。',
        '- 附加精度判据：严格完成且末端根位置水平误差 ≤20 cm；另记录同时满足末端朝向误差 ≤15° 的数量。这些是测评诊断指标，不改变训练奖励或终止条件。',
        '- 误差与接触统计排除整条仿真无效回合；其余回合包含失败转移，按有效帧数合并；失败后的停车状态不计入。提前失败片段较短，因此误差均值必须结合完成率阅读。分位数仅保留逐回合值，不把逐回合 p95 的平均冒充总体 p95。',
        '- 原始 GPU 仿真重复运行也存在数值非确定性；随机种子固定并不保证逐位相同。新增统计已验证不改变输入状态、不消耗随机数，参考状态自比较的位置和关节误差为零。',
        f"- [实验配置与源文件快照]({link(out/'manifest.json')})、[指标校验]({link(out/'metrics_verification.json')})、[机器可读总表]({link(out/'summary.json')})。",'',
        '## 完成率与终点精度','',
        '| 数据与条件 | 全程完成 | 完成且终点 ≤20 cm | 根水平 RMSE cm | 关节 RMSE 度 |',
        '|---|---:|---:|---:|---:|']
    for key,r in g.items():
        split,condition,phase=key.split('/')
        if split=='train_box':continue
        title=('独立翻箱验证' if split=='holdout' else '训练集')+' / '+mode_names[condition]
        lines.append(f"| {title} | {rate(r)} | {r['completed_endpoint_20cm']}/{r['episodes']} ({pct(r['endpoint_20cm_rate'])}) | {num(r['tracking']['root_xy_error_m']['rmse'],100)} | {num(r['tracking']['joint_angle_rmse_rad']['rmse'],180/math.pi)} |")
    lines+=['',f"训练扰动下，按回合计算的总体完成率为 {pct(main['completion_rate'])}；各类别等权平均为 {pct(main['category_macro_completion_rate'])}。前者中翻箱占 60/260，后者每类权重相同。",'',
        '## 全部类别','',f"![各类完成率]({link(out/'completion_matrix.png')})",'',
        '| 类别 | 无扰动 | 训练扰动 5 次 | 额外初态噪声 3 次 | 训练扰动水平 RMSE cm |',
        '|---|---:|---:|---:|---:|']
    for cat in categories:
        rs=[pc[k][cat] for k in plot_keys]
        lines.append(f"| {labels[cat]} `{cat}` | {rate(rs[0])} | {rate(rs[1])} | {rate(rs[2])} | {num(rs[1]['tracking']['root_xy_error_m']['rmse'],100)} |")
    f=s['flat_breakdown']['training_randomization']
    lines+=['',f"`flat` 目录保留原有全部 10 条参与总分。其中 9 条无箱体的样本为 {rate(f['pure_flat'])}；`flat/core_low_01_ep_000082` 含 5 个地形箱体，为 {rate(f['mixed_terrain'])}。箱体包括平台等地形表达，不能仅凭文件中有箱体就断言机器人在跨障碍；此处单列是为了区分数据组成，不从总体结果中移除失败样本。",'',
        '## 翻箱验证与中途重置','',
        '| 独立翻箱验证分层 | 训练扰动完成率 | 水平 RMSE cm |','|---|---:|---:|']
    for kind,r in s['holdout_strata'].items():lines.append(f"| {kind} | {rate(r)} | {num(r['tracking']['root_xy_error_m']['rmse'],100)} |")
    lines+=['','| 训练集翻箱条件 | 完成率 |','|---|---:|']
    for condition in ['nominal','training_randomization','additional_reset_noise','friction_0.3','friction_1.2','goal_0.5','goal_1.5']:
        r=pc['train/'+condition+'/phase0']['box_climb'];lines.append(f"| {mode_names[condition]} | {rate(r)} |")
    lines+=['','| 起始相位 | 训练翻箱 60 条 × 3 次 | 完成且终点 ≤20 cm |','|---|---:|---:|']
    for phase in [0,20,50,80]:
        r=g[f'train_box/training_randomization/phase{phase}'];lines.append(f"| {phase}% | {rate(r)} | {r['completed_endpoint_20cm']}/{r['episodes']} |")
    lines+=['','中途重置使用该相位的参考姿态和速度，重新填充观测历史。它测试后缀初始化的鲁棒性，不等同于从第 0 帧自然运行到该阶段；不同相位的完成率不要求单调变化。单独翻箱批次为 60 个环境，全部类别批次为 260 个环境，两者种子对应的随机样本分配不同，分开统计。']
    if s['box_previous_comparison']:
        b=s['box_previous_comparison'];lines+=['',f"相同 60 环境、种子 0–2 的全程翻箱协议：第 {b['previous_iteration']} 次为 {b['previous_completed']}/{b['total']}，本次为 {b['current_completed']}/{b['total']}；{b['failure_to_success']} 次由失败转成功，{b['success_to_failure']} 次由成功转失败。摩擦、关节偏置、质心及目标前瞻的初始随机量逐项核对一致。"]
    if s.get('box_baseline_rerun'):
        b=s['box_baseline_rerun'];v=b['latest_legacy_verification']
        lines+=['',f"为复核回落，另用新增指标前的原脚本重测旧 {b['iteration']} 次权重，得到 {b['completed']}/{b['total']}（{100*b['completed']/b['total']:.1f}%），仍高于最新权重的 {b['current_completed']}/{b['total']}（{100*b['current_completed']/b['total']:.1f}%）。最新权重 seed 0 用原脚本得到 {v['legacy_completed']}/{v['total']}，新增统计脚本为 {v['instrumented_completed']}/{v['total']}；同种子 GPU 运行并非逐位确定，但复核结果不支持把回落解释为新增指标统计造成。",f"[旧权重复测]({link(out/'baseline_recheck/summary.json')})、[新旧评估脚本核对]({link(out/'legacy_box_verification.json')})。这些诊断复测不混入 4,220 个正式回合统计。"]
    lines+=['','## 接触与跟踪质量','',
        '以下为训练范围扰动的全部 1,300 个训练集回合。接触以当前机器人碰撞近似到地形的距离 ≤1 cm 判定，不是力传感器标注。位置误差是实际新接触相对最近 5 帧内有效参考目标的距离，目标取加噪前值；无匹配目标的接触不计入位置均值。落点距离会受大脚掌的前后接触位置切换影响，不能与脚 link 原点误差混为一谈。脚 link 速度包含滚脚，不能直接解释为纯滑移。','',
        '| 部位 | 参考接触帧 | Precision | Recall | F1 | 匹配/总新接触 | 落点均值 cm | ≤10 cm |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in main['contacts'].items():
        fmt=lambda x:'—' if x is None else pct(x)
        lines.append(f"| {name} | {r['label_frames']} | {fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} | {r['matched_onsets']}/{r['onsets']} | {num(r['placement_error_m']['mean'],100)} | {fmt(r['placement_within_0_1'])} |")
    lines+=['','没有正标签的部位不能据此判断对应接触技能已学会；手和膝的正标签主要来自翻箱。逐类别接触计数与误差见 JSON 总表。','','| 跟踪指标 | 有效帧合并 RMSE |','|---|---:|']
    metric_rows=[('根水平位置 cm','root_xy_error_m',100),('根高度 cm','root_z_abs_error_m',100),('根姿态 度','root_orientation_error_rad',180/math.pi),('根航向 度','root_yaw_error_rad',180/math.pi),('根线速度 m/s','root_linear_velocity_error_m_s',1),('关节角 度','joint_angle_rmse_rad',180/math.pi),('左脚 link 世界位置 cm','left_foot_link_world_error_m',100),('右脚 link 世界位置 cm','right_foot_link_world_error_m',100),('左脚 link 局部位置 cm','left_foot_link_local_error_m',100),('右脚 link 局部位置 cm','right_foot_link_local_error_m',100)]
    for title,key,scale in metric_rows:lines.append(f"| {title} | {num(main['tracking'][key]['rmse'],scale)} |")
    lines+=['',
        '## 失败分布','',
        f"默认训练扰动的有效仿真回合：仅高度阈值 {main['failure_height_only']} 次、仅倾斜阈值 {main['failure_tilt_only']} 次、两者同时 {main['failure_both']} 次。独立翻箱验证对应为 {hold['failure_height_only']}、{hold['failure_tilt_only']}、{hold['failure_both']} 次。",'',
        '| 训练扰动下重复失败的片段（仅有效仿真） | 失败次数 / 5 |','|---|---:|']
    failures=json.loads((out/'failures.json').read_text());counts=Counter(e['clip'] for e in failures if e['split']=='train' and e['condition']=='training_randomization' and e['valid_physics'])
    for name,count in sorted(counts.items(),key=lambda kv:(-kv[1],kv[0])):lines.append(f'| `{name}` | {count}/5 |')
    lines+=['',f"所有条件的逐回合失败、终止相位、误差与接触明细：[failures.json]({link(out/'failures.json')})。仿真无效回合单列于 [physics_invalid.json]({link(out/'physics_invalid.json')})；其中 overflow=32 表示 MuJoCo Warp 高度场碰撞缓存溢出，不等同于 NaN。",'',
        '## 视频与复现','',
        '视频为 seed 0 的训练范围扰动，全程原速。每类使用此前固定片段；纯平地补充按文件名排序的首条无箱体样本，验证集取排序首条。固定示例沿用以往选择；补充纯平地按几何条件及文件名确定，原固定的含箱体失败样本仍保留。另补充三段按结果选择的失败说明片：在 seed 0 的有效失败中选该地形族跨五次测试失败最多的片段，使用原始录制状态，不重跑筛选。白色实体为真实物理输出，蓝色半透明为参考动作，终止后两者一起定格；彩色点和线是实际输入策略的目标，包含指令噪声。','',
        '| 视频组 | 片段数 | 时长秒 |','|---|---:|---:|']
    for name,r in video['reels'].items():lines.append(f"| [{name}]({link(r['video'])}) | {len(r['chapters'])} | {r['duration_s']:.2f} |")
    lines+=['',f"[逐片段视频与章节索引]({link(Path(meta['video_dir'])/'index.json')})。所有单片与合集均通过完整解码，帧数已核对。",'',
        '```bash',f"env ACP_ROBOT=roban MUJOCO_GL=egl PYTHONPATH=. OMP_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 \\",f"  /home/liujunjie/InstinctMJ/.venv/bin/python tools/run_roban_comprehensive_suite.py {out.relative_to(root)}",'```','',
        '复现脚本读取已固定的 selection.json；已有结果经校验后跳过。要重新采样，应复制 selection.json 到新的空目录。指标与评测脚本的源快照、checkpoint 哈希、数据清单哈希均保存在 artifacts 中。',
        '这套测试覆盖已采集的地形参数，不是极限地形能力搜索，也没有测 stepping stone。参考与接触输入的格式沿用现有 tracker 接口。', '']
    box=pc[mainkey]['box_climb'];reset_box=pc['train/additional_reset_noise/phase0']['box_climb']
    perfect=sum(r['completion_rate']==1 for r in pc[mainkey].values())
    lines+=['## 结论与后续优先级','',
        f"- 在本次参考指令及默认初态设置下，{perfect}/21 类在五次扰动测试中全部完成。翻箱为 {pct(box['completion_rate'])}，独立翻箱验证为 {pct(hold['completion_rate'])}；应分别看待这些测试条件。",
        f"- 额外初态噪声下翻箱仅完成 {rate(reset_box)}，固定低摩擦 0.3 时为 {rate(pc['train/friction_0.3/phase0']['box_climb'])}。下一轮可优先对照测试逐步增大的初态扰动课程与低摩擦翻箱，并分别监测起步与上箱阶段。",
        '- 坡面应同时检查重复失败轨迹与高度场碰撞容量；物理无效回合需要先解决仿真有效性，不能据此直接归因于 policy 或世界模型。',
        '- 保留独立验证集用于后续权重选择。当前结果支持继续推进 tracker，但接入 planner 后仍需独立评估目标误差、闭环完成率和终点精度。','']
    report=root/'docs'/f"roban_tracker_evaluation_iter{s['iteration']:06d}_zh.md";report.write_text('\n'.join(lines));print(report)


if __name__=='__main__':main()
