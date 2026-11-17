"""Execute a frozen-checkpoint reference/planner comparison manifest."""
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=args.directory.resolve()
    manifest=json.loads((out/'manifest.json').read_text());ckpt=Path(manifest['checkpoint'])
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest()==manifest['checkpoint_sha256']
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dest)
    status={'pid':os.getpid(),'phase':'running','running':{},'completed':{},'failed':{}}
    mutex=threading.Lock()
    def save():
        status['updated_at']=time.strftime('%Y-%m-%d %H:%M:%S')
        p=out/'status.json.tmp';p.write_text(json.dumps(status,indent=2)+'\n');p.replace(out/'status.json')
    save()
    def trial(case):
        name=case['name'];dest=out/(name+'.json')
        command=['/home/liujunjie/InstinctMJ/.venv/bin/python',str(root/'tools/evaluate_roban_tracker_detailed.py'),
                 '--ckpt',str(ckpt),'--planner',str(ckpt),'--eta-obs',str(case['eta_obs']),
                 '--seed',str(case['seed']),'--out',str(dest)]
        if case['noise']:command+=['--noise',*case['noise']]
        if case['record']:command+=['--record',str(out/(name+'.npz'))]
        if dest.exists():
            existing=json.loads(dest.read_text())
            assert existing['checkpoint_sha256']==manifest['checkpoint_sha256']
            assert existing['protocol']['eta_obs']==case['eta_obs'] and existing['protocol']['noise']==case['noise']
            assert existing['protocol']['seed']==case['seed'] and len(existing['episodes'])==200
            return name,0
        env={**os.environ,'ACP_ROBOT':'roban','PYTHONPATH':str(root),'OMP_NUM_THREADS':'2',
             'TORCHDYNAMO_DISABLE':'1','PYTHONUNBUFFERED':'1','MUJOCO_GL':'egl'}
        with (out/(name+'.log')).open('w') as log:
            p=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            with mutex:status['running'][name]={'pid':p.pid,'command':command};save()
            return name,p.wait()
    with ThreadPoolExecutor(max_workers=manifest['workers']) as pool:
        tasks={pool.submit(trial,c):c['name'] for c in manifest['trials']}
        for future in as_completed(tasks):
            name=tasks[future]
            try:
                name,code=future.result()
                if code:result={'exit_code':code}
                else:
                    d=json.loads((out/(name+'.json')).read_text());ep=d['episodes']
                    assert len(ep)==200
                    result={'episodes':len(ep),'completed':sum(r['completed'] for r in ep),
                            'invalid_physics':sum(not r['valid_physics'] for r in ep)}
            except Exception as error:code=1;result={'error':repr(error)}
            with mutex:
                status['running'].pop(name,None)
                status['failed' if code else 'completed'][name]=result;save()
            print(name,json.dumps(result),flush=True)
    status['phase']='failed' if status['failed'] else 'complete';save()


if __name__=='__main__':main()
