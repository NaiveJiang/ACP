"""Audit paired Stage 2 rollout results and aggregate without censoring failures."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from summarize_roban_comprehensive import aggregate


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('directory', type=Path)
    args = ap.parse_args(); out = args.directory.resolve()
    manifest = json.loads((out / 'manifest.json').read_text())
    status = json.loads((out / 'status.json').read_text())
    assert status['stage'] == 'complete' and not status['failed']
    for model in manifest['models'].values():
        assert hashlib.sha256(Path(model['checkpoint']).read_bytes()).hexdigest() == model['sha256']
    for name, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((out / 'source' / name).read_bytes()).hexdigest() == digest
    groups = defaultdict(list); reports = {}; all_es = []; trials = {}; initial = defaultdict(set)
    for case in manifest['cases']:
        report = json.loads((out / case['name'] / 'results.json').read_text()); es = report['episodes']
        model = manifest['models'][case['model']]
        assert report['checkpoint_sha256'] == report['planner_checkpoint_sha256'] == model['sha256']
        assert report['command_source'] == case['driver'] and report['eta_obs'] == case['eta_obs']
        assert report['seed'] == case['seed'] and len(es) == case['expected_episodes']
        assert {e['clip'] for e in es} == set(manifest['datasets'][case['split']]['clips'])
        initial[(case['split'], case['mode'], case['seed'])].add(report['initial_condition_sha256'][case['mode']])
        for e in es:
            e.update(model=case['model'], split=case['split'], condition=case['mode'], driver=case['driver'],
                     seed=case['seed'], trial=case['name'], category=e['clip'].split('/')[0])
            steps = round(e['elapsed_s'] / .02)
            assert 0 < steps <= e['remaining_steps'] and e['start_phase'] == 0
            assert e['reset_noise'] == (case['mode'] != 'nominal')
            if e['completed']:
                assert steps == e['remaining_steps'] and not e['failed']
            if e['valid_physics']:
                assert e['metric_frames'] == steps
            for k, v in e['tracking'].items():
                assert 0 <= v['count'] <= e['metric_frames']
                if e['finite_tracking_metrics'] and 'support' not in k:
                    assert v['count'] == e['metric_frames']
            for c in e['contacts'].values():
                assert c['matched_onsets'] == c['placement_error_m']['count'] <= c['onsets']
        reports[case['name']] = report; trials[case['name']] = aggregate(es)
        groups['/'.join(case[k] for k in ('model', 'split', 'mode', 'driver'))].extend(es); all_es.extend(es)
    assert all(len(v) == 1 for v in initial.values()), ('Initial states differ', initial)
    assert len(all_es) == manifest['expected_episodes']
    paired = defaultdict(lambda: defaultdict(list))
    ok = lambda e: e['completed'] and e['valid_physics']
    for case in manifest['cases']:
        if case['model'] != 'pre':
            continue
        before = {e['clip']: e for e in reports[case['name']]['episodes']}
        after = {e['clip']: e for e in reports[case['name'].replace('pre_', 'latest_', 1)]['episodes']}
        key = '/'.join(case[k] for k in ('split', 'mode', 'driver'))
        for name, e in before.items():
            for field in ('friction', 'joint_offsets_rad', 'torso_com_offset_m', 'goal_lookahead_s', 'initial_lift_plus_reset_noise_m'):
                assert e[field] == after[name][field], (name, field)
            a = after[name]
            kind = 'invalid_physics' if not e['valid_physics'] or not a['valid_physics'] else (
                'both_complete' if ok(e) and ok(a) else 'pre_only_complete' if ok(e) else 'latest_only_complete' if ok(a) else 'both_failed')
            paired[key][kind].append(dict(clip=name, seed=case['seed']))
    grouped = {k: aggregate(es) for k, es in groups.items()}
    per_category = {k: {cat: aggregate([e for e in es if e['category'] == cat])
                       for cat in sorted({e['category'] for e in es})} for k, es in groups.items()}
    invalid = [{k: e[k] for k in ('trial', 'clip', 'seed', 'overflow_flags', 'finite_tracking_metrics', 'elapsed_s')}
               for e in all_es if not e['valid_physics']]
    result = dict(models=manifest['models'], episodes=len(all_es), initial_states_and_randomization_matched=True,
                  groups=grouped, per_category=per_category, trials=trials, invalid_physics=invalid,
                  paired={k: dict(counts={a: len(b) for a, b in v.items()}, clips=v) for k, v in paired.items()},
                  scope='Reference-guided contact planning from frame zero; root goals and exact scans supplied. Independent box holdout is separate; training-reference scores are not unseen-terrain generalization.')
    (out / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    (out / 'failures.json').write_text(json.dumps([e for e in all_es if not ok(e)], indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(episodes=len(all_es), invalid=len(invalid), initial_states_matched=True,
                         groups={k: {f: v[f] for f in ('completed', 'episodes', 'endpoint_20cm_rate')} for k, v in grouped.items()}), indent=2))


if __name__ == '__main__':
    main()
