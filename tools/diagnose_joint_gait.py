"""Read-only matched planner/controller swaps to diagnose short-stride gait."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--planner',required=True)
    p.add_argument('--out',required=True);p.add_argument('--seconds',type=float,default=4.)
    p.add_argument('--vx',type=float,default=.2);p.add_argument('--query-dt',type=float,default=.3)
    p.add_argument('--repair-costs',action='store_true',help='Evaluate v8 physical penalties without changing the old policies')
    a=p.parse_args()
    os.environ['ACP_ROBOT']='roban';os.environ.setdefault('MUJOCO_GL','egl')
    import numpy as np
    import torch
    from supermimic.acp.footstep.backend import backend_for_format
    from supermimic.acp.footstep.model import FootstepPlanner
    from supermimic.acp.footstep.joint import decode_plan
    from supermimic.acp.footstep.reference import FootstepMotion
    from supermimic.acp.footstep.fast.models import FastKinematics
    from supermimic.acp.robot import default_joint_pos_list
    from supermimic.acp.mdp import unflatten_state
    from supermimic.rotations import yaw_mat
    torch.set_num_threads(4);torch.manual_seed(123)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    pc=torch.load(a.planner,map_location='cpu',weights_only=True)
    backend=backend_for_format(ck['format']);config=ck['config'];dev='cuda:0'
    # Choose a held-out constant-speed window, not a random unmatched phase.
    selected=None
    for part in ck['split']['val']:
        vel=np.load(Path(part['cache'])/'velocity.npy',mmap_mode='r')[part['start']:part['end']]
        mask=np.abs(vel-np.array([a.vx,0.,0.])).max(1)<1e-5
        starts=np.flatnonzero(mask & np.r_[True,~mask[:-1]])
        ends=np.flatnonzero(mask & np.r_[~mask[1:],True])+1
        for s,e in zip(starts,ends):
            if e-s>=round(a.seconds*50)+20+round(a.query_dt*50):
                selected=part;offset=int(s)+20;break
        if selected is not None:break
    if selected is None:raise ValueError('No held-out constant command window long enough')
    kin=FastKinematics(dev);motion=FootstepMotion([selected],kin,dev,config['horizons'])
    task=backend.Task(motion,kin,backend.RewardCfg(**ck['reward']))
    mdp=backend.MDP(kin,motion,torch.tensor(default_joint_pos_list(),device=dev),task)
    mdp.velocity_override=torch.tensor([a.vx,0.,0.],device=dev)
    probe=None
    if a.repair_costs:
        from supermimic.acp.footstep.consistent_joint import ConsistentJointTask,ConsistentJointMDP,ConsistentJointCfg
        pt=ConsistentJointTask(motion,kin,ConsistentJointCfg(**ck['reward']))
        probe=ConsistentJointMDP(kin,motion,torch.tensor(default_joint_pos_list(),device=dev),pt)
    pi=backend.Policy(config['hist'],config['policy_hidden'],planner_config=config['planner_config']).to(dev).eval()
    pi.load_state_dict(ck['pi'])
    old=FootstepPlanner(**pc['config']).to(dev).eval();old.load_state_dict(pc['model'])
    names=['joint_plan_joint_control','offline_plan_joint_control','joint_plan_v6_control',
           'offline_plan_v6_control','oracle_plan_v6_control','oracle_plan_joint_control']
    n=len(names);env=backend.Env(n,motion,mdp,dev,eval_mode=True,reset_noise=False)
    frame=motion.clip_start[0]+offset;ids=torch.arange(n,device=dev)
    env.reset(ids,frames=frame.expand(n));env.refresh()
    h=config['horizons'].index(a.query_dt)
    command=torch.tensor([a.vx,0.,0.,a.query_dt],device=dev).expand(n,-1)
    history=None;dead=torch.zeros(n,dtype=torch.bool,device=dev)
    records={k:[] for k in ('root','velocity','feet','feet_base','gap','plan','world_goal','q','qd','dead','reference_feet','reference_gap')}
    if probe is not None:
        for key in ('phase_contact_cost','swing_clearance_cost','slip_cost'):records['repair_'+key]=[]
    with torch.no_grad():
        for _ in range(round(a.seconds*50)):
            obs=env.observe()
            history=(obs[:,None].expand(-1,config['hist'],-1).clone() if history is None
                     else torch.cat((history[:,1:],obs[:,None]),1))
            jp=pi.contact_plan(history,command)
            op=decode_plan(old(history[...,:48],command[:,:3],command[:,3:4]),command[:,3:4])
            oracle=task.oracle_command(env.state(),env.f,torch.full_like(env.f,h))
            plans=torch.stack((jp[0],op[1],jp[2],op[3],oracle[4],oracle[5]))
            motor=pi.controller_action(history,plans)
            teacher=pi.teacher(history,plans)
            motor[2:5]=teacher[2:5]
            S0=env.state();H=yaw_mat(S0[1])
            world=S0[0][:,None]+(H[:,None]@plans[:,3:9].reshape(n,2,3,1)).squeeze(-1)
            tr=env.step(motor);S=unflatten_state(tr['s1']);f=tr['f']+1
            feet,_,gap=task.foot_geometry(S)
            relative=(yaw_mat(S[1]).transpose(-1,-2)[:,None]@(feet-S[0][:,None])[...,None]).squeeze(-1)
            velocity=(yaw_mat(S[1]).transpose(-1,-2)@S[2][...,None]).squeeze(-1)
            dead|=tr['term']|~tr['valid']
            ref_feet,_,ref_gap=task.foot_geometry(motion.ref_state(f))
            if probe is not None:
                for key,value in probe.physical_gait_terms(S,f).items():records['repair_'+key].append(value.cpu().numpy().copy())
            for key,value in dict(root=S[0],velocity=velocity,feet=feet,feet_base=relative,gap=gap,plan=plans,
                                  world_goal=world,q=S[4],qd=S[5],dead=dead,reference_feet=ref_feet,reference_gap=ref_gap).items():
                records[key].append(value.cpu().numpy().copy())
    data={k:np.asarray(v) for k,v in records.items()}
    np.savez_compressed(out/'rollouts.npz',**data)

    def events(gap):
        # Geometric contact proxy with two-frame hysteresis, not force labels.
        contact=True;on=off=0;touch=[];lift=[]
        for i,value in enumerate(gap):
            on=on+1 if value<.01 else 0;off=off+1 if value>.025 else 0
            if contact and off>=2:contact=False;lift.append(i)
            if not contact and on>=2:contact=True;touch.append(i)
        return lift,touch

    result=[];cut=25
    for i,name in enumerate(names):
        end=int(np.flatnonzero(data['dead'][:,i])[0]) if data['dead'][:,i].any() else len(data['dead'])
        cut=min(25,max(0,end//4));stop=max(cut+1,end);footstats=[]
        for side in range(2):
            lift,touch=events(data['gap'][:stop,i,side])
            touch=[t for t in touch if t>=cut]
            step=[float(np.linalg.norm(data['feet'][b,i,side,:2]-data['feet'][aa,i,side,:2])) for aa,b in zip(touch[:-1],touch[1:])]
            swing=[]
            for l in lift:
                later=[t for t in touch if t>l]
                if later and l>=cut:swing.append((later[0]-l)*.02)
            fb=data['feet_base'][cut:stop,i,side,0]
            footstats.append(dict(touchdowns=len(touch),ipsilateral_stride_mean_m=float(np.mean(step)) if step else None,
                                  swing_duration_mean_s=float(np.mean(swing)) if swing else None,
                                  fore_aft_p95_range_m=float(np.quantile(fb,.95)-np.quantile(fb,.05)),
                                  clearance_p95_m=float(np.quantile(data['gap'][cut:stop,i,side],.95))))
        vel=data['velocity'][cut:stop,i]
        flags=data['plan'][cut:stop,i,1:3]>.5
        report=dict(variant=name,valid_seconds=end*.02,vx_mean=float(vel[:,0].mean()),vx_std=float(vel[:,0].std()),
                    horizontal_speed_error_mean=float(np.linalg.norm(vel[:,:2]-[a.vx,0.],axis=1).mean()),
                    planned_contact_switches=int((flags[1:]!=flags[:-1]).sum()),
                    legs_qd_rms=float(np.sqrt(np.mean(data['qd'][cut:stop,i,1:13]**2))),feet=footstats)
        delay=round(a.query_dt*50)-1
        last=stop-delay
        if last>cut:
            mask=data['plan'][cut:last,i,1:3]>.5
            error=np.linalg.norm(data['feet'][cut+delay:stop,i]-data['world_goal'][cut:last,i],axis=-1)
            heights=data['world_goal'][cut:last,i,:,2][mask]
            report['planned_future_support_position_error_mean_m']=float(error[mask].mean()) if mask.any() else None
            report['planned_support_world_height_mean_m']=float(heights.mean()) if len(heights) else None
        report['both_feet_near_ground_fraction']=float(np.all(data['gap'][cut:stop,i]<.015,axis=1).mean())
        report['velocity_reward_mean_before_dt']=float((4*np.exp(-((vel[:,:2]-[a.vx,0.])**2).sum(-1)/.35**2)).mean())
        if probe is not None:
            report['repair_costs_before_dt']={key:float(data['repair_'+key][cut:stop,i].mean())
                                            for key in ('phase_contact_cost','swing_clearance_cost','slip_cost')}
        result.append(report)
    report=dict(checkpoint=a.checkpoint,iteration=ck['iteration'],reference=selected,offset=offset,
                velocity=[a.vx,0.,0.],query_dt=a.query_dt,seconds=a.seconds,results=result,
                note='Matched initialization and physical scene; contact timing from geometry with hysteresis. Not a reward-weight ablation.')
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(3,1,figsize=(12,9),sharex=True);t=np.arange(len(data['root']))*.02
    for i,name in enumerate(names):
        ax[0].plot(t,data['velocity'][:,i,0],label=name)
        ax[1].plot(t,data['feet_base'][:,i,0,0],label=name)
        ax[2].plot(t,data['gap'][:,i,0],label=name)
    ax[0].axhline(a.vx,color='black',ls='--');ax[0].set_ylabel('vx (m/s)')
    ax[1].set_ylabel('Left foot x / base (m)');ax[2].set_ylabel('Left foot gap (m)');ax[2].set_xlabel('Time (s)')
    ax[0].legend(fontsize=8)
    for x in ax:x.grid(alpha=.3)
    fig.tight_layout();fig.savefig(out/'signals.png',dpi=150)


if __name__=='__main__':main()
