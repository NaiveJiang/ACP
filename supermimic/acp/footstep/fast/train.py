"""BlinkMimic-inspired contact training. Short validation only unless the user starts a run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',nargs='+',required=True)
    p.add_argument('--log-dir',required=True)
    p.add_argument('--cache-dir',default='data/cache/footstep_v2')
    p.add_argument('--controller',choices=['contact','stable','velocity'],default='contact')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--num-envs',type=int,default=4096)
    p.add_argument('--iterations',type=int,default=20000)
    p.add_argument('--hist',type=int,default=5)
    p.add_argument('--horizons',nargs='+',type=float,default=[.1,.2,.3,.5,.8,1.])
    p.add_argument('--horizon',type=int,default=16)
    p.add_argument('--wm-hidden',nargs='+',type=int,default=[512,512,512])
    p.add_argument('--policy-hidden',nargs='+',type=int,default=[512,256,128])
    p.add_argument('--buffer-steps',type=int,default=512)
    p.add_argument('--warmup-steps',type=int,default=64)
    p.add_argument('--wm-window',type=int,default=8)
    p.add_argument('--wm-batch',type=int,default=512)
    p.add_argument('--wm-batches',type=int,default=4)
    p.add_argument('--wm-only-iters',type=int,default=500)
    p.add_argument('--pi-batch',type=int,default=512)
    p.add_argument('--collect-steps',type=int,default=3)
    p.add_argument('--critic-epochs',type=int,default=2)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--pi-lr-mult',type=float,default=1.)
    p.add_argument('--gamma',type=float,default=.99)
    p.add_argument('--lam',type=float,default=.95)
    p.add_argument('--action-noise',type=float,default=.1)
    p.add_argument('--reset-noise',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--compile',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--cuda-graph',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--amp',choices=['bf16','none'],default='bf16')
    p.add_argument('--optimizer',choices=['adam','muon'],default='adam')
    p.add_argument('--structured-wm',action='store_true')
    p.add_argument('--ema',type=float,default=0.)
    p.add_argument('--log-every',type=int,default=25)
    p.add_argument('--ckpt-every',type=int,default=500)
    p.add_argument('--val-fraction',type=float,default=.2)
    p.add_argument('--max-clips',type=int,default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=4)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--resume')
    group.add_argument('--init',help='Initialize same-interface legacy weights; structured WM initializes policy only')
    args=p.parse_args()
    positive=(args.num_envs,args.iterations,args.hist,args.horizon,args.warmup_steps,args.wm_window,args.wm_batch,
              args.wm_batches,args.pi_batch,args.collect_steps,args.critic_epochs,args.log_every,args.ckpt_every,
              *args.wm_hidden,*args.policy_hidden)
    if min(positive)<1 or not args.buffer_steps>=args.warmup_steps>=args.wm_window or not 0<=args.ema<1:
        p.error('Invalid dimensions, replay warm-up, or EMA')
    if args.wm_only_iters<0 or args.lr<=0 or args.pi_lr_mult<=0 or args.max_clips<0:
        p.error('Invalid learning schedule')
    os.environ['ACP_ROBOT']='roban'
    os.environ.setdefault('MUJOCO_GL','egl')
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','2')
    import torch
    from .models import FastKinematics,FastWorldModel,FastContactTask,FastStableContactTask,FastVelocityTask,device_norms,flat_scan
    from .core import Updates,CudaUpdate
    from ..backend import backend_for_format,CONTACT_FORMAT,STABLE_CONTACT_FORMAT,VELOCITY_FORMAT
    from ..data import prepare,split_clips
    from ..reference import FootstepMotion
    from ..env import FootstepTrackingSim
    from ...buffer import ReplayBuffer
    from ...networks import Critic,WorldModel
    from ... import networks
    from ...mdp import INV_DIM,normalize_xi,unflatten_state
    from ...robot import metadata,default_joint_pos_list
    from ...sim_numerics import TransitionGuard
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    networks.AMP_DTYPE=torch.bfloat16 if args.amp=='bf16' else None
    dev=args.device
    out=Path(args.log_dir);out.mkdir(parents=True,exist_ok=False)
    (out/'args.json').write_text(json.dumps(vars(args),indent=2))
    load_start=time.monotonic()
    backend=backend_for_format(dict(contact=CONTACT_FORMAT,stable=STABLE_CONTACT_FORMAT,velocity=VELOCITY_FORMAT)[args.controller])
    clips=prepare(args.data,args.cache_dir)
    split=split_clips(clips,args.val_fraction,args.seed)
    config=dict(hist=args.hist,horizons=args.horizons,wm_hidden=args.wm_hidden,policy_hidden=args.policy_hidden,
                structured_wm=args.structured_wm)
    checkpoint=None
    if args.resume or args.init:
        checkpoint=torch.load(args.resume or args.init,map_location=dev,weights_only=True)
        if checkpoint['format']!=backend.format or checkpoint['robot']['fingerprint']!=metadata()['fingerprint']:
            raise ValueError('Checkpoint robot or controller interface differs')
        for key in ('hist','horizons','wm_hidden','policy_hidden'):
            if checkpoint['config'][key]!=config[key]:raise ValueError('Checkpoint '+key+' differs')
        if args.resume:
            if checkpoint.get('training_engine')!='blink_contact_v1' or checkpoint['config']!=config:
                raise ValueError('Fast --resume requires matching fast training config')
            old=checkpoint['args']
            if any(old[k]!=getattr(args,k) for k in ('optimizer','ema','controller','amp')):
                raise ValueError('Resume optimizer/EMA/controller/AMP differs')
        current={c['fingerprint']:c for c in clips}
        expected={c['fingerprint'] for v in checkpoint['split'].values() for c in v}
        if set(current)!=expected:raise ValueError('Checkpoint datasets changed')
        split={k:[dict(c,cache=current[c['fingerprint']]['cache']) for c in v] for k,v in checkpoint['split'].items()}
    kin=FastKinematics(dev)
    motion=FootstepMotion(split['train'],kin,dev,args.horizons,args.max_clips)
    if int(motion.clip_len.min())<=args.horizon+1:raise ValueError('Reference blocks too short')
    if bool(motion.clip_boxes.any()):raise ValueError('Fast flat-ground specialization only')
    geometry=max(r['max_m'] for r in motion.geometry_reports)
    if geometry>.05:raise ValueError('Capture/training FK differs by over 5cm')
    task_cls=dict(contact=FastContactTask,stable=FastStableContactTask,velocity=FastVelocityTask)[args.controller]
    reward_cfg=backend.RewardCfg(**checkpoint['reward']) if checkpoint else backend.RewardCfg()
    task=task_cls(motion,kin,reward_cfg)
    mdp=backend.MDP(kin,motion,torch.tensor(default_joint_pos_list(),device=dev),task)
    mdp.scan=lambda S,boxes:flat_scan(S,mdp.scan_dim)
    policy=device_norms(backend.Policy(args.hist,args.policy_hidden).to(dev))
    wm=FastWorldModel(mdp.scan_dim,args.wm_hidden,args.structured_wm,args.amp=='bf16').to(dev)
    critic=device_norms(Critic(getattr(mdp,'critic_input_dim',INV_DIM+mdp.critic_extra+backend.command_dim)).to(dev))
    def opt(model,lr,hidden=None,betas=(.9,.999)):
        if hidden and args.optimizer=='muon':
            from .vendor.optim import MuonAdam
            return MuonAdam(model,hidden,lr=lr,betas=betas,fused=True,capturable=True)
        return torch.optim.Adam(model.parameters(),lr=torch.tensor(lr,device=dev),betas=betas,fused=True,capturable=True)
    ow=opt(wm,args.lr,[wm.net]);op=opt(policy,args.lr*args.pi_lr_mult,[policy.enc,policy.dec],(.9,.95));ov=opt(critic,args.lr)
    updates=Updates(wm,policy,critic,mdp,task,ow,op,ov,args.horizon,args.gamma,args.lam,args.critic_epochs,args.compile,args.ema)
    if args.compile:
        options={'max_fusion_size':32}
        mdp.reward=torch.compile(mdp.reward,fullgraph=True,dynamic=False,options=options)
        mdp.termination=torch.compile(mdp.termination,fullgraph=True,dynamic=False,options=options)
        inference=torch.compile(policy,fullgraph=True,dynamic=False,options=options)
    else:inference=policy
    start_it=sim_steps=0
    if checkpoint:
        policy.load_state_dict(checkpoint.get('pi_live',checkpoint['pi']))
        same_wm_structure=bool(checkpoint['config'].get('structured_wm',False))==args.structured_wm
        if same_wm_structure:
            saved=dict(checkpoint['wm'])
            for key,value in list(saved.items()):
                if key.startswith('out_norm.'):
                    saved.setdefault('inc_norm.'+key[len('out_norm.'):],value)
            wm.load_state_dict(saved)
        if updates.average is not None:updates.average.load_state_dict(checkpoint['pi'])
        if args.resume:
            critic.load_state_dict(checkpoint['critic'])
            for optimizer,key in ((ow,'opt_wm'),(op,'opt_pi'),(ov,'opt_v')):
                rates=[g['lr'] for g in optimizer.param_groups]
                optimizer.load_state_dict(checkpoint[key])
                for group,rate in zip(optimizer.param_groups,rates):
                    rate.fill_(float(group['lr']));group['lr']=rate
            start_it,sim_steps=checkpoint['iteration'],checkpoint['sim_steps']
            if args.iterations<=start_it:raise ValueError('iterations must exceed checkpoint iteration')
    env=FootstepTrackingSim(args.num_envs,motion,mdp,dev,gamma=args.gamma,reset_noise=args.reset_noise)
    buffer=ReplayBuffer(args.buffer_steps,args.num_envs,dev,extra={'o':(backend.obs_dim,torch.float32)})
    history=torch.zeros(args.num_envs,args.hist,backend.obs_dim,device=dev)
    query=task.sample_queries(args.num_envs)
    sim_guard=TransitionGuard(str(out))
    def update_norms(S,S1,a,xi,valid):
        wm.in_norm.update(wm.inputs(S,a,normalize_xi(xi),flat_scan(S,mdp.scan_dim)),valid)
        wm.out_norm.update(wm.targets(S,S1),valid)
        wm.inc_norm.update(WorldModel.targets(S,S1),valid)
    def update_actor_norms(obs,cmd):
        policy.obs_norm.update(obs);policy.goal_norm.update(cmd)
    if args.compile:
        update_norms=torch.compile(update_norms,fullgraph=True,dynamic=False)
        update_actor_norms=torch.compile(update_actor_norms,fullgraph=True,dynamic=False)
    # Small GPU metrics; transfer to the host only at logging boundaries.
    sums=torch.zeros(4,device=dev)
    @torch.no_grad()
    def collect(n,iteration):
        nonlocal history,sim_steps
        for _ in range(n):
            obs=env.observe()
            history=torch.where((env.ep_step==0)[:,None,None],obs[:,None].expand_as(history),
                                torch.cat((history[:,1:],obs[:,None]),1))
            cmd=task.command(env.state(),env.f,query)
            update_actor_norms(obs,cmd)
            a=(inference(history,cmd)+args.action_noise*torch.randn_like(env.a_prev)).clamp(-5,5)
            if hasattr(policy,'clamp_action'):
                a=policy.clamp_action(a)
            tr=env.step(a);tr['goal_steps']=query.clone()
            sim_guard.observe(tr,motion,sim_steps,iteration)
            buffer.add(tr)
            update_norms(unflatten_state(tr['s']),unflatten_state(tr['s1']),a,tr['xi'],tr['valid'])
            sim_steps+=args.num_envs
            sums[0].add_(tr['r'].mean());sums[1].add_(1)
            sums[2].add_(tr['term'].float().sum());sums[3].add_(tr['done'].float().sum())
            # Fixed-size masked query updates avoid a host read of done.sum().
            query.copy_(torch.where(tr['done'],task.sample_queries(args.num_envs),query))
    def starts():
        s,h=buffer.sample_starts(2*args.pi_batch,args.hist)
        ok=s['valid'] & (s['f']+args.horizon<=motion.clip_end[motion.clip_id[s['f']]])
        order=torch.argsort((~ok).to(torch.int32),stable=True)
        count=ok.long().sum()
        take=order[torch.arange(args.pi_batch,device=dev)%count.clamp_min(1)]
        result={k:s[k][take] for k in ('s','a_prev','xi','f','goal_steps')}
        result['hist']=h['o'][take]
        result['usable']=ok[take]
        # Even an unusable placeholder may not gather outside the reference arrays.
        c=motion.clip_id[result['f']]
        result['f']=torch.minimum(result['f'],motion.clip_end[c]-args.horizon)
        return result
    modules=[wm,policy,critic]+([updates.average] if updates.average is not None else [])
    counters=[updates.errors,updates.runs,updates.worst]
    world=CudaUpdate(updates.world,modules,[ow],counters,args.cuda_graph)
    actor=CudaUpdate(updates.actor,modules,[op,ov],counters,args.cuda_graph)
    task_report=dict(format=backend.format,training_engine='blink_contact_v1',config=config,split=split,
                     command_dim=backend.command_dim,observation_dim=backend.obs_dim,reward=asdict(task.cfg),
                     train_frames=motion.num_frames,geometry_max_m=geometry,robot=metadata())
    (out/'task.json').write_text(json.dumps(task_report,indent=2))
    print(f"Fast {backend.command_dim}D: compile={args.compile}, graph={args.cuda_graph}, AMP={args.amp}, "
          f"structured={args.structured_wm}, optimizer={args.optimizer}, EMA={args.ema}",flush=True)
    collect(args.warmup_steps,start_it)
    pool=ThreadPoolExecutor(max_workers=1)
    pending=None
    def cpu(value):
        if torch.is_tensor(value):return value.detach().cpu().clone()
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,list):return [cpu(v) for v in value]
        if isinstance(value,tuple):return tuple(cpu(v) for v in value)
        return value
    def write(payload):
        torch.save(payload,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
    def save(iteration):
        nonlocal pending
        if pending is not None:pending.result()
        deployed=updates.average if updates.average is not None else policy
        payload=cpu(dict(**task_report,args=vars(args),iteration=iteration,sim_steps=sim_steps,
                         wm=wm.state_dict(),pi=deployed.state_dict(),pi_live=policy.state_dict(),critic=critic.state_dict(),
                         opt_wm=ow.state_dict(),opt_pi=op.state_dict(),opt_v=ov.state_dict()))
        pending=pool.submit(write,payload)
    initial_seconds=time.monotonic()-load_start
    start=time.monotonic();steady_start=None;steady_at=None
    wm_metrics=torch.zeros(3,device=dev);pi_metrics=torch.zeros(5,device=dev)
    try:
        with (out/'train.csv').open('w',newline='') as file:
            writer=csv.writer(file)
            writer.writerow(['iteration','wall_s','sim_steps','wm_loss','pi_loss','critic_loss','pi_gradient',
                             'imag_survival','sim_reward','term_fraction','bad_wm','bad_pi','bad_critic'])
            for iteration in range(start_it+1,args.iterations+1):
                collect(args.collect_steps,iteration)
                for _ in range(args.wm_batches):wm_metrics=world(buffer.sample_windows(args.wm_batch,args.wm_window))
                active=iteration>args.wm_only_iters
                if active:pi_metrics=actor(starts())
                if active and steady_start is None:
                    torch.cuda.synchronize();steady_start=time.monotonic();steady_at=iteration
                if iteration%args.log_every==0 or iteration==args.iterations:
                    values=torch.cat((wm_metrics[:1],pi_metrics[:4],sums,updates.errors.float(),updates.worst.float())).cpu().tolist()
                    row=[iteration,time.monotonic()-start,sim_steps,*values[:5],values[5]/max(values[6],1),
                         values[7]/max(values[8],1),*values[9:12]]
                    writer.writerow(row);file.flush();sums.zero_()
                    print(f"it {iteration} wm {values[0]:.4g} pi {values[1]:.4g} survival {values[4]:.4f} "
                          f"bad={values[9:12]}",flush=True)
                    if max(values[12:15])>=3:
                        save(iteration)
                        raise FloatingPointError('Repeated nonfinite updates were skipped on GPU; see checkpoint/log')
                if iteration%args.ckpt_every==0 or iteration==args.iterations:save(iteration)
    finally:
        if pending is not None:pending.result()
        pool.shutdown()
    torch.cuda.synchronize()
    steady_seconds=time.monotonic()-steady_start if steady_start is not None else None
    count=args.iterations-steady_at if steady_at is not None else 0
    performance=dict(startup_and_warmup_seconds=initial_seconds,total_update_seconds=time.monotonic()-start,
                     steady_iterations=count,steady_seconds=steady_seconds,
                     seconds_per_iteration=steady_seconds/count if count else None,
                     note='Steady window excludes first actor graph capture; total includes compilation/capture')
    (out/'performance.json').write_text(json.dumps(performance,indent=2))
    print(json.dumps(performance),flush=True)


if __name__=='__main__':main()
