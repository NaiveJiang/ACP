"""Prepare a weights-only Stage 3 initialization from a Stage 2 snapshot.

Preserve tracker/planner/WM exactly; initialize a new reference critic and fresh
optimizers. Stage 1 metadata supplies reward configuration and LR origin, never
newer model weights. Warm up both critics before enabling actor updates.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from supermimic.acp.checkpoint import load_checkpoint, save_checkpoint
from supermimic.acp.mdp import OBS_DIM, REF_ORI_DIMS
from supermimic.acp.networks import Critic, Policy, WorldModel
from supermimic.acp.planner import ContactPlanner, PlannerCfg
from supermimic.acp.robot import NB, ROBOT_ID
from supermimic.acp.terrain import CRITIC_SCAN, grid_offsets


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--stage1-metadata',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=20261112)
    args=ap.parse_args()
    if ROBOT_ID!='roban':ap.error('Set ACP_ROBOT=roban')
    if args.out.exists():raise FileExistsError(args.out)
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    ck=load_checkpoint(args.checkpoint,map_location='cpu',weights_only=False)
    meta=load_checkpoint(args.stage1_metadata,map_location='cpu',weights_only=False)
    if 'goal_cfg' in ck:raise ValueError('Expected Stage 2 source')
    for name in ('wm','pi','planner'):
        if not all(bool(torch.isfinite(v).all()) for v in ck[name].values()):raise ValueError(f'Nonfinite {name}')
    command_dim=ck['pi']['enc.0.weight'].shape[1]
    wm=WorldModel(len(grid_offsets(*CRITIC_SCAN)))
    pi=Policy(OBS_DIM,command_dim,hist=ck['init_args']['hist'])
    pcfg=PlannerCfg(**ck['planner_cfg']);planner=ContactPlanner(OBS_DIM-REF_ORI_DIMS,pcfg)
    critic=Critic(meta['critic']['net.0.weight'].shape[1]+NB*3+2)
    for name,model in (('wm',wm),('pi',pi),('planner',planner)):model.load_state_dict(ck[name])
    lr=ck['args']['lr']
    bootstrap=dict(mode='stage2_weights_only_to_stage3',source_checkpoint=str(args.checkpoint.resolve()),
        source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),source_iteration=ck['iteration'],
        reward_metadata_source=str(args.stage1_metadata.resolve()),critics_reinitialized=True,
        optimizers_reinitialized=True,preserved_modules=['pi','planner','wm'],seed=args.seed)
    ck.update(critic=critic.state_dict(),reward_cfg=meta['reward_cfg'],
        lr_it0=meta['iteration']+meta.get('lr_it0',0),bootstrap=bootstrap,
        opt_wm=torch.optim.Adam(wm.parameters(),lr=lr).state_dict(),
        opt_pi=torch.optim.Adam(pi.parameters(),lr=lr,betas=(.9,.95)).state_dict(),
        opt_v=torch.optim.Adam(critic.parameters(),lr=lr).state_dict(),
        opt_planner=torch.optim.Adam(planner.parameters(),lr=pcfg.lr,betas=(.9,.999),eps=1e-8,weight_decay=0.).state_dict())
    args.out.parent.mkdir(parents=True,exist_ok=True);save_checkpoint(ck,args.out)
    args.out.with_suffix('.json').write_text(json.dumps(bootstrap,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.out),**bootstrap),indent=2))


if __name__=='__main__':main()
