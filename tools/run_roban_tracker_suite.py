"""Run a fixed-checkpoint, 15-trial RoBan evaluation suite (200 clips/trial)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time


def cases():
    out = []
    def add(condition, seeds, options):
        for seed in seeds:
            out.append(dict(name=f"{condition}_seed{seed}", condition=condition, seed=seed, options=options))
    add("nominal", [0,1], [])
    add("obs_noise", [0], ["--noise","obs"])
    add("command_noise", [0], ["--noise","command"])
    add("action_noise", [0], ["--noise","action"])
    add("all_noise", [0,1,2], ["--noise","obs","command","action"])
    add("domain_all_noise", [0,1,2], ["--domain-randomization","--noise","obs","command","action"])
    add("friction_low", [0], ["--friction","0.3"])
    add("friction_high", [0], ["--friction","1.2"])
    add("goal_short", [0], ["--goal-s","0.5"])
    add("goal_long", [0], ["--goal-s","1.5"])
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--python",default="/home/liujunjie/InstinctMJ/.venv/bin/python")
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    directory=args.out.resolve(); directory.mkdir(parents=True,exist_ok=True)
    ckpt=args.ckpt.resolve(); digest=hashlib.sha256(ckpt.read_bytes()).hexdigest()
    plan=cases()
    manifest=dict(checkpoint=str(ckpt),checkpoint_sha256=digest,trials=plan,workers=args.workers,
                  expected_episodes=3000,started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                  training_manifest_sha256=hashlib.sha256((root/'data/roban_motion/manifest.json').read_bytes()).hexdigest(),
                  source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                                 ['tools/evaluate_roban_tracker_detailed.py','supermimic/acp/evaluate.py','supermimic/acp/env.py','supermimic/acp/task.py']})
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    snapshot=directory/'source';snapshot.mkdir(exist_ok=True)
    for name in manifest['source_sha256']:
        dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dest)
    status=dict(pid=os.getpid(),phase='running',completed={},running={},failed={})
    mutex=threading.Lock()
    def persist():
        status['updated_at']=time.strftime("%Y-%m-%d %H:%M:%S")
        temp=directory/'status.json.tmp';temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(directory/'status.json')
    persist()
    def run(case):
        name=case['name']; result=directory/(name+'.json')
        if result.exists():
            saved=json.loads(result.read_text())
            if saved.get('checkpoint_sha256')==digest and len(saved.get('episodes',[]))==200:
                return name,0,'existing'
            raise RuntimeError(f'Existing result does not match this checkpoint: {result}')
        command=[args.python,str(root/'tools/evaluate_roban_tracker_detailed.py'),'--ckpt',str(ckpt),'--out',str(result),
                 '--seed',str(case['seed']),*case['options']]
        if name=='nominal_seed0':
            command+=['--calibrate-contact','--record',str(directory/'nominal_seed0.npz')]
        environment={**os.environ,'PYTHONPATH':str(root),'ACP_ROBOT':'roban','TORCHDYNAMO_DISABLE':'1',
                     'OMP_NUM_THREADS':'2','PYTHONUNBUFFERED':'1','MUJOCO_GL':'egl'}
        with (directory/(name+'.log')).open('w') as log:
            process=subprocess.Popen(command,cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT)
            with mutex:
                status['running'][name]={'pid':process.pid,'command':command};persist()
            code=process.wait()
        return name,code,'executed'
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run,case):case['name'] for case in plan}
        for future in as_completed(futures):
            try:
                name,code,mode=future.result()
                if code==0:
                    data=json.loads((directory/(name+'.json')).read_text())
                    record={'mode':mode,'episodes':len(data['episodes']),
                            'completed':sum(r['completed'] for r in data['episodes']),
                            'invalid_physics':sum(not r['valid_physics'] for r in data['episodes'])}
                else:
                    record={'exit_code':code}
            except Exception as error:
                name=futures[future];code=1;record={'error':repr(error)}
            with mutex:
                status['running'].pop(name,None)
                status['completed' if code==0 else 'failed'][name]=record
                persist()
            print(name,json.dumps(record),flush=True)
    status['phase']='failed' if status['failed'] else 'complete';persist()


if __name__=='__main__':
    main()
