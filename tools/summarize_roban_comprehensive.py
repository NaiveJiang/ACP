"""Aggregate comprehensive tracker evaluation with frame-weighted error/contact metrics."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def pool(rows):
    rows=[r for r in rows if r['count']];n=sum(r['count'] for r in rows)
    if not n:return dict(count=0,mean=None,rmse=None,max=None)
    return dict(count=n,mean=sum(r['count']*r['mean'] for r in rows)/n,
        rmse=math.sqrt(sum(r['count']*r['rmse']**2 for r in rows)/n),max=max(r['max'] for r in rows))


def aggregate(es):
    valid_es=[e for e in es if e['valid_physics']]
    contacts={}
    for body in es[0]['contacts']:
        rows=[e['contacts'][body] for e in valid_es];tp=sum(r['true_positive_frames'] for r in rows)
        nt=sum(r['touch_frames'] for r in rows);nl=sum(r['label_frames'] for r in rows);nm=sum(r['matched_onsets'] for r in rows)
        contacts[body]=dict(true_positive_frames=tp,touch_frames=nt,label_frames=nl,
            precision=tp/nt if nt else None,recall=tp/nl if nl else None,f1=2*tp/(nt+nl) if nt+nl else None,
            onsets=sum(r['onsets'] for r in rows),matched_onsets=nm,
            placement_error_m=pool([r['placement_error_m'] for r in rows]),
            placement_within_0_1=sum((r['placement_within_0_1'] or 0)*r['matched_onsets'] for r in rows)/nm if nm else None)
    complete=[e for e in es if e['completed'] and e['valid_physics']]
    failures=[e for e in valid_es if e['failed']]
    return dict(episodes=len(es),completed=len(complete),completion_rate=len(complete)/len(es),
        completed_endpoint_20cm=sum(e['completed_endpoint_20cm'] for e in es),
        endpoint_20cm_rate=sum(e['completed_endpoint_20cm'] for e in es)/len(es),
        completed_endpoint_20cm_yaw15deg=sum(e['completed_endpoint_20cm'] and e['terminal_root_yaw_error_rad']<=math.radians(15) for e in es),
        invalid_physics=sum(not e['valid_physics'] for e in es),
        valid_episodes=len(valid_es),completion_rate_valid_only=len(complete)/len(valid_es) if valid_es else None,
        mean_terminal_phase=sum(e['terminal_phase'] for e in es)/len(es),
        simulated_seconds=sum(e['elapsed_s'] for e in es),
        failure_height_only=sum(e['terminal_root_z_error_m']>.25 and e['terminal_tilt_error']<=.8 for e in failures),
        failure_tilt_only=sum(e['terminal_root_z_error_m']<=.25 and e['terminal_tilt_error']>.8 for e in failures),
        failure_both=sum(e['terminal_root_z_error_m']>.25 and e['terminal_tilt_error']>.8 for e in failures),
        tracking={k:pool([e['tracking'][k] for e in valid_es]) for k in es[0]['tracking']},contacts=contacts)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path);args=ap.parse_args()
    out=args.directory.resolve();root=Path(__file__).resolve().parents[1]
    manifest=json.loads((out/'manifest.json').read_text());status=json.loads((out/'status.json').read_text())
    assert status['stage']=='complete' and not status['failed']
    assert hashlib.sha256(Path(manifest['checkpoint']).read_bytes()).hexdigest()==manifest['checkpoint_sha256']
    for info in manifest['datasets'].values():
        directory=Path(info['directory']);assert hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest()==info['manifest_sha256']
        for row in json.loads((directory/'manifest.json').read_text()):
            if 'heightfield_sha256' in row:
                assert hashlib.sha256((directory/(row['clip']+'_heightfield.npz')).read_bytes()).hexdigest()==row['heightfield_sha256']
    for name,h in manifest['source_sha256'].items():assert hashlib.sha256((out/'source'/name).read_bytes()).hexdigest()==h
    groups=defaultdict(list);all_es=[];trials={}
    for case in manifest['cases']:
        report=json.loads((out/case['name']/'results.json').read_text());es=report['episodes']
        assert report['checkpoint_sha256']==manifest['checkpoint_sha256'] and len(es)==case['expected_episodes']
        phase=es[0]['start_phase'];key=f"{case['split']}/{case['condition']}/phase{round(phase*100)}"
        expected=set(manifest['datasets']['holdout' if case['split']=='holdout' else 'train']['clip_names'])
        if case['split']=='train_box':expected={n for n in expected if n.startswith('box_climb/')}
        assert {e['clip'] for e in es}==expected
        for e in es:
            e.update(seed=case['seed'],trial=case['name'],split=case['split'],condition=case['condition'],category=e['clip'].split('/')[0])
            n=round(e['elapsed_s']/.02)
            assert 0<n<=e['remaining_steps'] and e['metric_frames']<=n
            if e['valid_physics']:assert e['metric_frames']==n
            if e['completed']:assert n==e['remaining_steps'] and not e['failed']
            for metric,stats in e['tracking'].items():
                if 'support' not in metric:
                    assert 0<=stats['count']<=e['metric_frames']
                    if e['finite_tracking_metrics']:assert stats['count']==e['metric_frames']
            if e['metric_frames'] and e['finite_tracking_metrics']:
                assert abs(e['root_xy_error_mean_m']-e['tracking']['root_xy_error_m']['mean'])<1e-4
            for c in e['contacts'].values():
                assert c['true_positive_frames']<=min(c['touch_frames'],c['label_frames'])
                assert c['matched_onsets']==c['placement_error_m']['count']<=c['onsets']
        groups[key].extend(es);all_es.extend(es);trials[case['name']]=aggregate(es)
    assert len(all_es)==manifest['expected_episodes']
    by_group={k:aggregate(es) for k,es in groups.items()}
    per_category={k:{cat:aggregate([e for e in es if e['category']==cat]) for cat in sorted({e['category'] for e in es})} for k,es in groups.items()}
    for key,cats in per_category.items():by_group[key]['category_macro_completion_rate']=sum(c['completion_rate'] for c in cats.values())/len(cats)
    main_group='train/training_randomization/phase0';failed=[e for e in all_es if e['failed'] or not e['valid_physics']]
    previous=root/'artifacts/roban/evaluation/stage1_v3_curriculum_no_reset_iter073500_box_phase0_training/summary.json'
    comparison=None
    if previous.exists():
        old=json.loads(previous.read_text());before={(e['seed'],e['clip']):e for e in old['episodes']};after={(e['seed'],e['clip']):e for e in groups['train_box/training_randomization/phase0']};assert before.keys()==after.keys()
        for k in before:
            for f in ['friction','joint_offsets_rad','torso_com_offset_m','goal_lookahead_s']:assert before[k][f]==after[k][f]
        ok=lambda e:e['completed'] and e['valid_physics']
        comparison=dict(previous_iteration=old['iteration'],current_iteration=manifest['iteration'],total=len(after),previous_completed=sum(ok(e) for e in before.values()),current_completed=sum(ok(e) for e in after.values()),failure_to_success=sum(not ok(before[k]) and ok(after[k]) for k in before),success_to_failure=sum(ok(before[k]) and not ok(after[k]) for k in before),matched_randomization=True)
    dataset=json.loads((root/'data/roban_motion/manifest.json').read_text());pure_flat={r['clip'] for r in dataset if r['clip'].startswith('flat/') and r['box_count']==0}
    flat_breakdown={condition:{kind:aggregate([e for e in groups['train/'+condition+'/phase0'] if e['category']=='flat' and ((e['clip'] in pure_flat)==(kind=='pure_flat'))]) for kind in ['pure_flat','mixed_terrain']} for condition in ['nominal','training_randomization','additional_reset_noise']}
    # Holdout geometry strata were assigned during collection, not after outcomes.
    hm=json.loads((root/'data/roban_box_climb_holdout/manifest.json').read_text());strata={r['clip']:r.get('stratum','unknown') for r in hm}
    holdout_strata={s:aggregate([e for e in groups['holdout/training_randomization/phase0'] if strata[e['clip']]==s]) for s in sorted(set(strata.values()))}
    result=dict(checkpoint=manifest['checkpoint'],checkpoint_sha256=manifest['checkpoint_sha256'],iteration=manifest['iteration'],episodes=len(all_es),invalid_physics=sum(not e['valid_physics'] for e in all_es),groups=by_group,per_category=per_category,trials=trials,box_previous_comparison=comparison,flat_breakdown=flat_breakdown,holdout_strata=holdout_strata)
    fresh_path=out/'baseline_recheck/summary.json'
    if fresh_path.exists():
        fresh=json.loads(fresh_path.read_text());now={(e['seed'],e['clip']):e for e in groups['train_box/training_randomization/phase0']}
        for e in fresh['episodes']:
            for f in ['friction','joint_offsets_rad','torso_com_offset_m','goal_lookahead_s']:
                assert e[f]==now[(e['seed'],e['clip'])][f]
        result['box_baseline_rerun']=dict(iteration=fresh['iteration'],completed=fresh['completed'],total=fresh['total'],current_completed=by_group['train_box/training_randomization/phase0']['completed'],matched_randomization=True,evaluator='pre-instrumentation script; latest seed0 independently rechecked with the same script',latest_legacy_verification=json.loads((out/'legacy_box_verification.json').read_text()))
    (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    (out/'failures.json').write_text(json.dumps(failed,indent=2,ensure_ascii=False)+'\n')
    invalid=[{k:e[k] for k in ['trial','seed','clip','condition','split','overflow_flags','finite_tracking_metrics','metric_frames','elapsed_s','terminal_phase']} for e in all_es if not e['valid_physics']]
    (out/'physics_invalid.json').write_text(json.dumps(dict(episodes=invalid,overflow_flag_32='mujoco_warp OverflowType.HFIELD (1 << 5): height-field collision buffer overflow',handling='Retain original attempts, count as non-completion conservatively, exclude entire episode from aggregated tracking/contact errors and from policy-failure threshold counts.'),indent=2)+'\n')
    # A compact matrix is also convenient for downstream plotting/reporting.
    matrix=[dict(category=cat,nominal=per_category['train/nominal/phase0'][cat]['completion_rate'],randomized=row['completion_rate'],endpoint=row['endpoint_20cm_rate'],randomized_completed=row['completed'],randomized_total=row['episodes'],xy_rmse_m=row['tracking']['root_xy_error_m']['rmse'],joint_rmse_rad=row['tracking']['joint_angle_rmse_rad']['rmse']) for cat,row in per_category[main_group].items()]
    (out/'category_matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')
    print(json.dumps(dict(episodes=len(all_es),invalid=result['invalid_physics'],groups={k:{f:v[f] for f in ['completed','episodes','completion_rate','endpoint_20cm_rate']} for k,v in by_group.items()},box_comparison=comparison),indent=2))


if __name__=='__main__':main()
