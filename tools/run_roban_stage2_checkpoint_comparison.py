"""Frozen, paired Stage 2 evaluation, with reference-command diagnostic controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path)
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.directory.resolve()
    prep = json.loads((out / 'preparation.json').read_text())
    assert not prep['latest_checkpoint_pending']
    models = {k: dict(checkpoint=prep[k + '_checkpoint'], sha256=prep[k + '_checkpoint_sha256'],
                     iteration=prep['pre_instability_iteration' if k == 'pre' else 'latest_iteration'])
              for k in ('pre', 'latest')}
    for m in models.values():
        assert sha(m['checkpoint']) == m['sha256']
    manifest_path = out / 'manifest.json'
    if not manifest_path.exists():
        datasets = {}
        for split, name in [('train', 'roban_motion'), ('holdout', 'roban_box_climb_holdout')]:
            directory = root / 'data' / name
            records = json.loads((directory / 'manifest.json').read_text())
            for row in records:
                for suffix, key, base in [('_motion.npz', 'motion_sha256', directory),
                                          ('_terrain.npy', 'terrain_sha256', directory),
                                          ('_heightfield.npz', 'heightfield_sha256', directory),
                                          ('.npz', 'contacts_sha256', root / 'data' / (name + '_contacts'))]:
                    if key in row:
                        assert sha(base / (row['clip'] + suffix)) == row[key], (row['clip'], key)
            datasets[split] = dict(directory=str(directory), contacts_dir=str(root / 'data' / (name + '_contacts')),
                                   manifest_sha256=sha(directory / 'manifest.json'),
                                   clips=sorted(r['clip'] for r in records),
                                   groups=sorted({r['split_group'] for r in records}))
        assert not set(datasets['train']['groups']) & set(datasets['holdout']['groups'])
        assert not set(datasets['train']['clips']) & set(datasets['holdout']['clips'])
        cases = []
        for condition, seeds in [('nominal', [0]), ('training_randomization', [0, 1, 2])]:
            for split in datasets:
                for driver in ('planner', 'reference'):
                    for seed in seeds:
                        for model in models:
                            cases.append(dict(name=f'{model}_{split}_{condition}_{driver}_seed{seed}',
                                              model=model, split=split, mode=condition, driver=driver, seed=seed,
                                              eta_obs=1. if driver == 'planner' else 0.,
                                              record=seed == 0, expected_episodes=len(datasets[split]['clips'])))
        files = list((root / 'supermimic').rglob('*.py')) + [root / 'tools' / n for n in
                 ('probe_roban_box_suffix.py', 'roban_tracker_metrics.py', Path(__file__).name)]
        sources = {}
        for src in files:
            name = str(src.relative_to(root)); dst = out / 'source' / name
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst); sources[name] = sha(src)
        save(manifest_path, dict(created_at=datetime.datetime.now().astimezone().isoformat(), models=models,
                                datasets=datasets, cases=cases, source_sha256=sources,
                                expected_episodes=sum(c['expected_episodes'] for c in cases), workers=args.workers,
                                protocol='All clips from phase zero, same explicit seed and reset parameters. Full planner targets versus reference controls; no optimizer updates. Reference root goals and exact terrain scan remain available. Training randomization includes reset/obs/command/action/physics randomization.'))
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest['source_sha256'].items():
        assert sha(root / name) == digest, name
    state = dict(pid=os.getpid(), stage='evaluation', started_at=datetime.datetime.now().astimezone().isoformat(),
                 running={}, completed={}, failed={})
    lock = threading.Lock(); stop = threading.Event()
    def persist():
        state['updated_at'] = datetime.datetime.now().astimezone().isoformat()
        save(out / 'status.json', state)
    persist()
    def trial(case):
        if stop.is_set():
            return case['name'], dict(cancelled=True), 1
        dest = out / case['name']; dest.mkdir(exist_ok=True)
        m = manifest['models'][case['model']]; data = manifest['datasets'][case['split']]
        cmd = [sys.executable, str(root / 'tools/probe_roban_box_suffix.py'), '--ckpt', m['checkpoint'],
               '--planner', m['checkpoint'], '--eta-obs', str(case['eta_obs']), '--out', str(dest / 'results.json'),
               '--category', 'all', '--phases', '0', '--modes', case['mode'], '--seed', str(case['seed']),
               '--motion-dir', data['directory'], '--contacts-dir', data['contacts_dir'], '--metrics', '--reset_noise']
        if case['record']:
            cmd += ['--record-dir', str(dest)]
        env = {**os.environ, 'ACP_ROBOT': 'roban', 'MUJOCO_GL': 'egl', 'OMP_NUM_THREADS': '2',
               'PYTHONPATH': str(root), 'TORCHDYNAMO_DISABLE': '1', 'PYTHONUNBUFFERED': '1'}
        if not (dest / 'results.json').exists():
            with (dest / 'evaluation.log').open('w') as log:
                p = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                with lock:
                    state['running'][case['name']] = dict(pid=p.pid, command=cmd); persist()
                code = p.wait()
            if code:
                stop.set(); return case['name'], dict(exit_code=code), code
        report = json.loads((dest / 'results.json').read_text()); es = report['episodes']
        assert report['checkpoint_sha256'] == report['planner_checkpoint_sha256'] == m['sha256']
        assert report['seed'] == case['seed'] and report['modes'] == [case['mode']]
        assert report['eta_obs'] == case['eta_obs'] and len(es) == case['expected_episodes']
        assert {e['clip'] for e in es} == set(data['clips'])
        assert all(e['start_phase'] == 0 and e['reset_noise'] == (case['mode'] != 'nominal') for e in es)
        return case['name'], dict(episodes=len(es), completed=sum(e['completed'] and e['valid_physics'] for e in es),
                                 endpoint_20cm=sum(e['completed_endpoint_20cm'] for e in es),
                                 invalid=sum(not e['valid_physics'] for e in es)), 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(trial, c): c for c in manifest['cases']}
        for future in as_completed(pending):
            case = pending[future]
            try:
                name, result, code = future.result()
            except Exception as error:
                stop.set(); name, result, code = case['name'], dict(error=repr(error)), 1
            with lock:
                state['running'].pop(name, None)
                state['failed' if code else 'completed'][name] = result; persist()
            print(name, json.dumps(result), flush=True)
    state['stage'] = 'failed' if state['failed'] else 'complete'; persist()
    if state['failed']:
        raise RuntimeError(state['failed'])


if __name__ == '__main__':
    main()
