"""Sequential, fixed-checkpoint evaluation of all tracker skills and held-out boxes."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def save(path, data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)


def cases():
    result=[]
    def add(name, split, condition, mode, seeds, count, options=None, record=False):
        for seed in seeds:
            result.append(dict(name=f'{name}_seed{seed}',split=split,condition=condition,mode=mode,seed=seed,
                               expected_episodes=count,options=options or [],record=record and seed==0))
    add('train_nominal','train','nominal','nominal',[0],260,record=True)
    add('train_randomized','train','training_randomization','training_randomization',range(5),260,record=True)
    add('holdout_nominal','holdout','nominal','nominal',[0],20,record=True)
    add('holdout_randomized','holdout','training_randomization','training_randomization',range(5),20,record=True)
    add('train_reset_noise','train','additional_reset_noise','training_randomization',range(3),260,['--reset_noise'])
    for value in [.3,1.2]:
        add(f'friction_{value:g}','train',f'friction_{value:g}','nominal',[0],260,['--friction',str(value)])
    for value in [.5,1.5]:
        add(f'goal_{value:g}','train',f'goal_{value:g}','nominal',[0],260,['--goal-s',str(value)])
    # Separate 60-world batches preserve the prior box-only experiment's RNG layout.
    for phase in [0.,.2,.5,.8]:
        add(f'box_phase{round(phase*100)}','train_box','training_randomization','training_randomization',range(3),60,
            ['--category','box_climb','--phases',str(phase)],record=phase==0)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];out=args.directory.resolve();meta=json.loads((out/'selection.json').read_text())
    assert hashlib.sha256(Path(meta['checkpoint']).read_bytes()).hexdigest()==meta['checkpoint_sha256']
    plan=cases();manifest=out/'manifest.json'
    if not manifest.exists():
        sources=[*sorted((root/'supermimic').rglob('*.py')),*[root/'tools'/n for n in
                 ['probe_roban_box_suffix.py','roban_tracker_metrics.py','render_roban_box_suffix.py','run_roban_comprehensive_suite.py']]]
        hashes={}
        for src in sources:
            rel=src.relative_to(root);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dest);hashes[str(rel)]=hashlib.sha256(src.read_bytes()).hexdigest()
        datasets={}
        for split,name in [('train','roban_motion'),('holdout','roban_box_climb_holdout')]:
            directory=root/'data'/name;records=json.loads((directory/'manifest.json').read_text())
            for row in records:
                for suffix,key,base in [('_motion.npz','motion_sha256',directory),('_terrain.npy','terrain_sha256',directory),('.npz','contacts_sha256',root/'data'/(name+'_contacts'))]:
                    if key in row:
                        assert hashlib.sha256((base/(row['clip']+suffix)).read_bytes()).hexdigest()==row[key],row['clip']
            datasets[split]=dict(directory=str(directory),clips=len(records),manifest_sha256=hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest(),
                clip_names=sorted(r['clip'] for r in records),groups=sorted({r['split_group'] for r in records}))
        assert not set(datasets['train']['groups']) & set(datasets['holdout']['groups'])
        assert not set(datasets['train']['clip_names']) & set(datasets['holdout']['clip_names'])
        save(manifest,dict(**meta,cases=plan,expected_episodes=sum(c['expected_episodes'] for c in plan),source_sha256=hashes,datasets=datasets))
    fixed=json.loads(manifest.read_text())
    for name,h in fixed['source_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
    status=dict(stage='evaluation',completed=[],running=None,failed=[])
    env={**os.environ,'ACP_ROBOT':'roban','MUJOCO_GL':'egl','OMP_NUM_THREADS':'4','PYTHONPATH':str(root),'TORCHDYNAMO_DISABLE':'1','PYTHONUNBUFFERED':'1'}
    for case in plan:
        dest=out/case['name'];dest.mkdir(exist_ok=True)
        cmd=[sys.executable,'tools/probe_roban_box_suffix.py','--ckpt',meta['checkpoint'],'--out',str(dest/'results.json'),
             '--category','all','--phases','0','--modes',case['mode'],'--seed',str(case['seed']),'--metrics']
        if case['split']=='holdout':
            cmd+=['--motion-dir','data/roban_box_climb_holdout','--contacts-dir','data/roban_box_climb_holdout_contacts']
        cmd+=case['options']
        if case['record']:cmd+=['--record-dir',str(dest)]
        status['running']=dict(name=case['name'],command=cmd);save(out/'status.json',status)
        if not (dest/'results.json').exists():
            print('RUN',case['name'],flush=True)
            with (dest/'evaluation.log').open('w') as log:
                process=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            if process.returncode:
                status['failed'].append(dict(name=case['name'],returncode=process.returncode));status['stage']='failed';save(out/'status.json',status)
                raise RuntimeError(f"Evaluation failed: {dest/'evaluation.log'}")
        report=json.loads((dest/'results.json').read_text());episodes=report['episodes']
        assert report['checkpoint_sha256']==meta['checkpoint_sha256'] and len(episodes)==case['expected_episodes']
        assert report['seed']==case['seed'] and report['modes']==[case['mode']]
        assert all(e['reset_noise']==(case['condition']=='additional_reset_noise') for e in episodes)
        complete=dict(name=case['name'],completed=sum(e['completed'] and e['valid_physics'] for e in episodes),total=len(episodes),invalid=sum(not e['valid_physics'] for e in episodes))
        status['completed'].append(complete);save(out/'status.json',status);print(json.dumps(complete),flush=True)
    status.update(stage='complete',running=None,finished_at=datetime.datetime.now().astimezone().isoformat());save(out/'status.json',status)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
