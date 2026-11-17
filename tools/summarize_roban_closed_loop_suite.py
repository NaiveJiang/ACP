"""Audit and summarize same-checkpoint closed-loop reference/planner tests."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from summarize_roban_tracker_suite import aggregate,CATEGORIES

LABELS={'nominal_reference':'无噪声／参考目标','nominal_planner':'无噪声／纯 planner',
        'all_noise_reference':'三类噪声／参考目标','all_noise_planner':'三类噪声／纯 planner',
        'nominal_mixed':'无噪声／62.495% planner 混合'}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path);args=ap.parse_args()
    out=args.directory.resolve();root=Path(__file__).resolve().parents[1]
    m=json.loads((out/'manifest.json').read_text());status=json.loads((out/'status.json').read_text())
    assert status['phase']=='complete' and not status['failed']
    assert hashlib.sha256(Path(m['checkpoint']).read_bytes()).hexdigest()==m['checkpoint_sha256']
    assert hashlib.sha256((root/'data/roban_motion/manifest.json').read_bytes()).hexdigest()==m['dataset_manifest_sha256']
    for name,digest in m['source_sha256'].items():assert hashlib.sha256((out/'source'/name).read_bytes()).hexdigest()==digest
    dataset=json.loads((root/'data/roban_motion/manifest.json').read_text());names={r['clip'] for r in dataset}
    true_flat={r['clip'] for r in dataset if r['clip'].startswith('flat/') and r['box_count']==0}
    planner_validation={n for g in CATEGORIES for n in sorted(x for x in names if x.split('/')[0]==g)[::10]}
    assert len(planner_validation)==20
    grouped=defaultdict(list);trials={};results={};failure=[]
    for case in m['trials']:
        name=case['name'];d=json.loads((out/(name+'.json')).read_text());p=d['protocol'];ep=d['episodes']
        assert d['checkpoint_sha256']==p['planner_checkpoint_sha256']==m['checkpoint_sha256']
        assert p['eta_obs']==case['eta_obs'] and p['seed']==case['seed'] and p['noise']==case['noise']
        assert p['command_source']==case['mode']
        assert p['reference_commands']==(case['mode']=='reference')
        assert p['goal_s']==1 and not p['domain_randomization'] and p['friction_override'] is None
        assert p['finished_worlds_held_above_terrain'] and p['first_variant_only']
        assert d['original_evaluation']['term_window']==0
        assert len(ep)==200 and {r['clip'] for r in ep}==names
        assert Counter(r['category'] for r in ep)=={g:10 for g in CATEGORIES}
        assert sum(r['failed'] for r in ep)==d['original_evaluation']['n_failed']
        for r in ep:
            assert r['completed'] != r['failed'] and r['steps']<=r['reference_steps']
            for key,value in r['tracking'].items():
                if 'support' not in key:assert value['count']==r['steps']
            if r['failed']:failure.append({'trial':name,**{key:r[key] for key in ['clip','category','duration_s','reference_progress','failure_height','failure_tilt','valid_physics']}})
        stats=aggregate(ep)
        assert abs(stats['mean_normalized_return']-d['original_evaluation']['norm_return'])<.000051
        for foot in ['left_foot','right_foot']:
            c=stats['contacts'][foot];raw=d['original_evaluation']['contacts'][foot]
            for k in ['touch_frames','label_frames']:assert c[k]==raw[k]
            for k in ['precision','recall']:assert abs(c[k]-raw[k])<.000501
            assert abs(c['placement_error_m']['mean']-raw['placement_err'])<.000501
        key=case['condition']+'_'+case['mode'];grouped[key].extend(ep);trials[name]=stats;results[name]=ep
    assert sum(len(ep) for ep in grouped.values())==m['expected_episodes']
    summary={}
    for key,ep in grouped.items():
        cases=[c for c in m['trials'] if c['condition']+'_'+c['mode']==key]
        summary[key]={**aggregate(ep),'label':LABELS[key],'completed_per_seed':[trials[c['name']]['completed'] for c in cases],
                      'per_category':{g:aggregate([r for r in ep if r['category']==g]) for g in CATEGORIES},
                      'true_flat':aggregate([r for r in ep if r['clip'] in true_flat]),
                      'planner_internal_validation':aggregate([r for r in ep if r['clip'] in planner_validation]),
                      'planner_training_clips':aggregate([r for r in ep if r['clip'] not in planner_validation]),
                      'median_episode_root_xy_rmse_m':statistics.median(r['tracking']['root_xy_error_m']['rmse'] for r in ep),
                      'completed_and_endpoint_xy_within_m':{
                          str(threshold):sum(r['completed'] and r['root_xy_error_at_end_m']<=threshold for r in ep)
                          for threshold in [.1,.2,.3,.5]}}
    paired=[]
    for condition,seeds in [('nominal',[0,1]),('all_noise',[0,1,2])]:
        for seed in seeds:
            r={e['clip']:e for e in results[f'{condition}_reference_seed{seed}']};p={e['clip']:e for e in results[f'{condition}_planner_seed{seed}']}
            groups={'both_complete':[],'reference_only_complete':[],'planner_only_complete':[],'both_failed':[]}
            for name in sorted(names):
                key='both_complete' if r[name]['completed'] and p[name]['completed'] else 'reference_only_complete' if r[name]['completed'] else 'planner_only_complete' if p[name]['completed'] else 'both_failed'
                groups[key].append(name)
            common=groups['both_complete']
            paired.append({'condition':condition,'seed':seed,'counts':{k:len(v) for k,v in groups.items()},'clips':groups,
                           'both_completed_reference_metrics':aggregate([r[n] for n in common]),
                           'both_completed_planner_metrics':aggregate([p[n] for n in common])})
    import numpy as np
    with np.load(out/'nominal_reference_seed0.npz') as r,np.load(out/'nominal_planner_seed0.npz') as p:
        assert np.array_equal(r['names'],p['names']) and np.array_equal(r['boxes'],p['boxes'])
        for k in ['root_p','root_quat','q']:assert np.array_equal(r[k][0],p[k][0]),k
        for data in [r,p]:
            assert all(k in data for k in ['post_root_p','post_root_quat','post_q','post_f','heightfield_paths'])
    result={'checkpoint':m['checkpoint'],'checkpoint_sha256':m['checkpoint_sha256'],'groups':summary,'trials':trials,'paired':paired,
            'failures':failure,'overall':aggregate([r for ep in grouped.values() for r in ep]),'validation':'passed',
            'initial_states_and_geometry_equal_nominal_seed0':True,
            'scope':'200 training references, from first frame, reference root goal and exact terrain scan; planner supplies only future contact targets; no independent navigation test'}
    (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    table=['| 条件 | 完成 / 总数 | 各种子完成数（每次 200） | 根 XY RMSE cm | 左/右脚世界 RMSE cm | 归一化回报 |','|---|---:|---|---:|---:|---:|']
    for key,v in summary.items():
        t=v['tracking'];table.append(f"| {LABELS[key]} | {v['completed']}/{v['episodes']}（{v['completion_rate']:.2%}） | {', '.join(map(str,v['completed_per_seed']))} | {t['root_xy_error_m']['rmse']*100:.2f} | {t['left_foot_link_world_error_m']['rmse']*100:.2f}/{t['right_foot_link_world_error_m']['rmse']*100:.2f} | {v['mean_normalized_return']:.4f} |")
    table+=['','| 类别 | 无噪声参考 | 无噪声 planner | 带噪参考 | 带噪 planner | 混合 62.5% |','|---|---:|---:|---:|---:|---:|']
    order=list(LABELS)
    for g in CATEGORIES:
        cells=[summary[k]['per_category'][g] for k in order];table.append('| `'+g+'` | '+' | '.join(f"{v['completed']}/{v['episodes']}" for v in cells)+' |')
    table+=['','| 条件 | 左/右脚接触 precision | 左/右脚接触 recall | 对本组指令落点均值 cm | 对本组指令 ≤10 cm |','|---|---:|---:|---:|---:|']
    for key,v in summary.items():
        feet=[v['contacts'][b] for b in ['left_foot','right_foot']]
        cells=['/'.join(f"{r[k]:.1%}" for r in feet) for k in ['precision','recall']]
        cells+=['/'.join(f"{r['placement_error_m']['mean']*100:.2f}" for r in feet),'/'.join(f"{r['placement_within_0_1']:.1%}" for r in feet)]
        table.append('| '+LABELS[key]+' | '+' | '.join(cells)+' |')
    (out/'tables.md').write_text('\n'.join(table)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(15,9),layout='constrained',gridspec_kw={'width_ratios':[1.6,1]})
    heat=np.array([[summary[k]['per_category'][g]['completion_rate']*100 for k in order] for g in CATEGORIES])
    im=axes[0].imshow(heat,cmap='RdYlGn',vmin=50,vmax=100,aspect='auto')
    axes[0].set_yticks(range(len(CATEGORIES)),CATEGORIES);axes[0].set_xticks(range(len(order)),['Ref\nclean','Planner\nclean','Ref\nnoise','Planner\nnoise','Mixed\n62.5%'])
    for i,g in enumerate(CATEGORIES):
        for j,k in enumerate(order):
            v=summary[k]['per_category'][g];axes[0].text(j,i,f"{v['completed']}/{v['episodes']}",ha='center',va='center',fontsize=9,color='white' if heat[i,j]>=95 else '#222222')
    axes[0].set_title('Reference-clip completion (same tracker checkpoint)')
    fig.colorbar(im,ax=axes[0],label='Completion (%)',shrink=.5)
    y=np.arange(len(order));root_error=[summary[k]['tracking']['root_xy_error_m']['rmse']*100 for k in order]
    axes[1].barh(y,root_error,color=['#4c78a8','#f58518','#4c78a8','#f58518','#72b7b2'])
    axes[1].set_yticks(y,['Reference / clean','Planner / clean','Reference / noise','Planner / noise','Mixed / clean'])
    axes[1].invert_yaxis();axes[1].set_xlabel('Root XY RMSE (cm), alive frames');axes[1].set_title('Completion does not imply precise tracking')
    for i,x in enumerate(root_error):axes[1].text(x+.2,i,f'{x:.1f}',va='center')
    axes[1].set_xlim(0,max(root_error)*1.22);axes[1].grid(axis='x',alpha=.2)
    fig.suptitle('RoBan Stage 2 checkpoint 014500 — 200 references, 2200 episodes',fontsize=15)
    fig.savefig(out/'closed_loop_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps({'validation':'passed','episodes':result['overall']['episodes'],'invalid_physics':result['overall']['invalid_physics'],
                      'groups':{k:[v['completed'],v['episodes']] for k,v in summary.items()}}))


if __name__=='__main__':main()
