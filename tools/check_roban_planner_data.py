"""One real-reference planner batch across every category; no rollout-training claim."""
import os
os.environ['ACP_ROBOT']='roban'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
import torch
from supermimic.acp.robot import default_joint_pos_list,NUM_J
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.motion import MotionLib
from supermimic.acp.mdp import MDP,OBS_DIM,XI_DIM
from supermimic.acp.task import ContactTask,ContactCfg
from supermimic.acp.planner import PlannerCfg,PlannerScan,ContactPlanner,proprio,onset_index,hindsight_labels,planner_loss,planner_goal,height_target_supervision

ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--motion-dir',default='data/roban_motion');ap.add_argument('--report',required=True)
args=ap.parse_args();torch.set_num_threads(4);torch.manual_seed(20261120);dev='cuda:0'
kin=RobotKinematics(dev);m=MotionLib.from_dir(args.motion_dir,kin,dev)
mdp=MDP(kin,m,torch.tensor(default_joint_pos_list(),device=dev));ctc=ContactTask(ContactCfg(),mdp,args.motion_dir,dev)
cfg=PlannerCfg();scan=PlannerScan(cfg,dev);planner=ContactPlanner(OBS_DIM-6,cfg).to(dev)
seen=set();rows=[];hist=[];goals=[];points=[];targets=[];occ=[];masks=[]
for ci,name in enumerate(m.names):
    category=name.split('/')[0]
    if category in seen:continue
    seen.add(category);start=int(m.clip_start[ci]);length=int(m.clip_len[ci]);t=min(length-80,max(5,length//2))
    with np.load(Path(args.motion_dir+'_contacts')/(name+'.npz')) as z:
        touch=z['contact_info'][0,...,0]>.5;cp=z['contact_pos_w'][0]
    on=onset_index(touch,10,75);S=m.ref_state(torch.arange(start,start+length,device=dev))
    target,valid=hindsight_labels(torch.tensor(on[None],device=dev),torch.tensor(cp[None],device=dev),S[0][None],S[1][None],scan)
    mask=height_target_supervision(target[0],valid[0],S,m.clip_boxes[ci,0][None].expand(length,-1,-1))
    if category == 'box_climb':
        # Exercise actual hand targets, not a mid-clip sample with feet only.
        candidates = torch.where((valid[0, :, 4:6] & mask[:,4:6]).all(-1))[0]
        candidates = candidates[(candidates >= 4) & (candidates < length-1)]
        if not len(candidates):
            raise ValueError(f'{name}: no frame with two valid hand targets')
        t = int(candidates[len(candidates)//2])
    f=torch.tensor([start+t],device=dev);now=m.ref_state(f)
    hh=m.ref_state(torch.arange(start+t-4,start+t+1,device=dev))
    prop=proprio(mdp.obs(hh,torch.zeros(5,NUM_J,device=dev),torch.zeros(5,XI_DIM,device=dev)))
    hist.append(prop);goals.append(planner_goal(ctc,now,f,torch.full_like(f,50),cfg.goal_radius)[0])
    points.append(scan.points(scan.heights(now,m.boxes(f,torch.zeros_like(f))))[0])
    targets.append(target[0,t]);occ.append(valid[0,t]);masks.append(mask[t]);rows.append(dict(category=category,clip=name,reference_frame=t,positive_targets=int((valid[0,t]&mask[t]).sum()),positive_bodies=(valid[0,t]&mask[t]).tolist(),ignored_bodies=(~mask[t]).tolist()))
batch=[torch.stack(x) for x in (hist,goals,points,targets,occ)]
assert all(torch.isfinite(x).all() for x in batch)
alpha,logit=planner(*batch[:3]);losses=planner_loss(alpha,logit,*batch[2:],cfg.sigma_a,torch.stack(masks))
sum(losses).backward();assert all(torch.isfinite(p.grad).all() for p in planner.parameters() if p.grad is not None)
torch.optim.Adam(planner.parameters(),lr=cfg.lr).step()
report=dict(categories=len(rows),samples=rows,losses=[float(x.detach()) for x in losses],finite_gradients=True,
            input_shapes=[list(x.shape) for x in batch],scope='reference-data interface/gradient check; not trained ACP tracker rollouts')
Path(args.report).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
