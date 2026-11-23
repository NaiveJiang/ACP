"""BlinkMimic-inspired contact training. Short validation only unless the user starts a run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',nargs='+',required=True)
    p.add_argument('--log-dir',required=True)
    p.add_argument('--cache-dir',default='data/cache/footstep_v2')
    p.add_argument('--profile',choices=['wbc'],help='Use the WBC RoBan Mimic training settings on the v6 contact task')
    p.add_argument('--controller',choices=['contact','stable','upper','gait','collision','joint','consistent','walking','velocity'],default='contact')
    p.add_argument('--body-goal',action='store_true',
                   help='Add body movement direction and relative heading to the v6 contact policy')
    p.add_argument('--event-stage2',action='store_true',
                   help='Train next-contact events and continue training a v6 controller')
    p.add_argument('--next-contact',action='store_true',help='Use the separate 12-D contact-event controller')
    p.add_argument('--next-contact-stage2',action='store_true',help='Continue training a next-contact controller with the ACP planner')
    p.add_argument('--event-obs-start',type=int,default=1000)
    p.add_argument('--event-reward-start',type=int,default=3000)
    p.add_argument('--event-ramp-iters',type=int,default=4000)
    p.add_argument('--event-rollout-every',type=int,default=50)
    p.add_argument('--event-rollout-envs',type=int,default=24)
    p.add_argument('--event-rollout-steps',type=int,default=200)
    p.add_argument('--event-batch',type=int,default=512)
    p.add_argument('--planner-init',help='Offline planner checkpoint, required for joint --init')
    p.add_argument('--planner-lr-mult',type=float,default=.25)
    p.add_argument('--planner-mix-start',type=float,default=.25)
    p.add_argument('--planner-mix-iters',type=int,default=2000)
    p.add_argument('--upper-supervision-weight',type=float,default=None,
                   help='v4 upper-head direct supervision coefficient (default 0.5)')
    p.add_argument('--leg-pose-weight',type=float,default=None,help='v5 soft leg pose cost (default 2.0)')
    p.add_argument('--leg-velocity-weight',type=float,default=None,help='v5 soft leg joint velocity cost (default 0.2)')
    p.add_argument('--stance-tilt-weight',type=float,default=None,help='v5 support-foot roll/pitch cost (default 0.5)')
    p.add_argument('--motion-style-weight',type=float,default=None,
                   help='Reference contact timing, swing trajectory and weight-transfer costs (default 0)')
    p.add_argument('--self-collision-weight',type=float,default=None,help='v6 opposite-leg proximity cost (default 2.0)')
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
    p.add_argument('--collect-steps',type=float,default=3,help='Mean simulator steps collected per update')
    p.add_argument('--critic-epochs',type=int,default=2)
    p.add_argument('--critic-minibatches',type=int,default=1)
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
    p.add_argument('--structured-wm',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--wm-mirror',type=float,default=0.,help='Fraction of flat world-model windows to mirror')
    p.add_argument('--ema',type=float,default=0.)
    p.add_argument('--failure-reset-share',type=float,default=0.,help='Mix in resets from the bin before a training fall')
    p.add_argument('--probe-share',type=float,default=0.,help='Mix in reset bins from EMA evaluation falls; zero disables probing')
    p.add_argument('--start-reset-fraction',type=float,default=0.,help='Fraction of resets that start at the first frame')
    p.add_argument('--probe-start',type=int,default=1000,help='First iteration eligible for EMA probing')
    p.add_argument('--probe-every',type=int,default=125,help='Iterations between EMA probes')
    p.add_argument('--probe-envs',type=int,default=16,help='Parallel environments used only by the EMA probe')
    p.add_argument('--log-every',type=int,default=25)
    p.add_argument('--ckpt-every',type=int,default=500)
    p.add_argument('--val-fraction',type=float,default=.2)
    p.add_argument('--max-clips',type=int,default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=4)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--resume')
    group.add_argument('--init',help='Initialize same-interface legacy weights; structured WM initializes policy only')
    p.add_argument('--init-new-reference',action='store_true',
                   help='Allow Stage 1 --init to start a fresh next-contact run on different reference data')
    profile, _ = p.parse_known_args()
    if profile.profile == 'wbc':
        p.set_defaults(controller='collision', num_envs=1024, hist=10, horizon=32,
                       buffer_steps=1000, wm_batch=1024, wm_batches=10, wm_only_iters=0,
                       collect_steps=2.5, critic_epochs=5, critic_minibatches=4,
                       lr=4e-4, pi_lr_mult=4., optimizer='muon', structured_wm=True,
                       wm_mirror=.5, ema=.995, seed=0, failure_reset_share=.30,
                       probe_share=.15, start_reset_fraction=.30)
    if profile.next_contact_stage2:
        p.set_defaults(event_batch=128,event_rollout_steps=250,event_rollout_envs=48)
    if profile.event_stage2 or profile.next_contact_stage2:
        p.set_defaults(lr=1e-4,pi_lr_mult=.5)
    args=p.parse_args()
    if args.profile == 'wbc' and args.controller != 'collision':
        p.error('--profile wbc uses the v6 collision controller')
    if args.body_goal and args.controller != 'collision':
        p.error('--body-goal requires --controller collision')
    next_contact = args.next_contact or args.next_contact_stage2
    event_stage = args.event_stage2 or args.next_contact_stage2
    if next_contact and (args.controller != 'collision' or args.body_goal or args.event_stage2):
        p.error('Next contact requires the collision controller without old event or body-goal mode')
    if args.init_new_reference and (not args.init or not next_contact or event_stage):
        p.error('--init-new-reference requires next-contact Stage 1 and --init')
    if event_stage:
        os.environ['ACP_ROBOT']='roban'
        from ..event_planner import MIN_SUPPORT, WINDOW
        if (args.controller!='collision' or args.body_goal or not (args.init or args.resume)
                or args.wm_only_iters or not args.ema or args.event_obs_start < 0
                or args.event_reward_start<=args.event_obs_start
                or args.event_rollout_steps < WINDOW+MIN_SUPPORT
                or min(args.event_ramp_iters,args.event_rollout_every,args.event_rollout_envs,args.event_batch)<1
                or (not next_contact and not all(any(abs(q-h)<1e-6 for h in args.horizons) for q in (.3,.5)))):
            p.error('Event Stage 2 requires initialization/resume, EMA, 0 WM-only iterations and a valid handover schedule')
    reset_shares = (args.failure_reset_share,args.probe_share,args.start_reset_fraction)
    if (not all(0 <= share <= 1 for share in reset_shares)
            or (any(reset_shares) and args.profile != 'wbc')
            or min(args.probe_start,args.probe_every,args.probe_envs) < 1
            or (args.probe_share and not args.ema)):
        p.error('Invalid reset/probe settings; focused resets use --profile wbc and probes require EMA')
    positive=(args.num_envs,args.iterations,args.hist,args.horizon,args.warmup_steps,args.wm_window,args.wm_batch,
              args.wm_batches,args.pi_batch,args.collect_steps,args.critic_epochs,args.log_every,args.ckpt_every,
              args.critic_minibatches,*args.wm_hidden,*args.policy_hidden)
    if min(positive)<1 or not args.buffer_steps>=args.warmup_steps>=args.wm_window or not 0<=args.ema<1:
        p.error('Invalid dimensions, replay warm-up, or EMA')
    if (args.wm_only_iters < 0 or args.max_clips < 0
            or not all(math.isfinite(rate) and rate > 0 for rate in (args.lr,args.pi_lr_mult))):
        p.error('Invalid learning schedule')
    if (not math.isfinite(args.collect_steps) or not 0 <= args.wm_mirror <= 1
            or (args.horizon * args.pi_batch) % args.critic_minibatches):
        p.error('Invalid collection rate, mirror fraction or critic minibatch size')
    if args.optimizer == 'muon' and (len(args.wm_hidden) < 2 or len(args.policy_hidden) < 2):
        p.error('Muon requires at least two hidden layers in the world model and policy')
    walking = args.controller == 'walking'
    joint=args.controller in ('joint','consistent','walking')
    consistent=args.controller=='consistent'
    if joint and (args.optimizer!='adam' or not (args.init or args.resume) or (args.init and not args.planner_init)):
        p.error('Joint fine tuning needs Adam and either --resume or --init <v6> --planner-init <offline planner>')
    if not math.isfinite(args.planner_lr_mult) or args.planner_lr_mult<=0 or not 0<=args.planner_mix_start<=1 or args.planner_mix_iters<1:
        p.error('Invalid planner LR multiplier or mixing schedule')
    if args.upper_supervision_weight is not None and (args.controller not in ('upper','gait','collision','joint','consistent') or not math.isfinite(args.upper_supervision_weight) or args.upper_supervision_weight<0):
        p.error('--upper-supervision-weight requires --controller upper/gait/collision and a finite nonnegative value')
    gait_options=('leg_pose_weight','leg_velocity_weight','stance_tilt_weight','motion_style_weight')
    for name in gait_options:
        value=getattr(args,name)
        if value is not None and (args.controller not in ('gait','collision','joint','consistent') or not math.isfinite(value) or value<0):
            p.error('--'+name.replace('_','-')+' requires --controller gait/collision and a finite nonnegative value')
    if args.self_collision_weight is not None and (args.controller not in ('collision','joint','consistent') or not math.isfinite(args.self_collision_weight) or args.self_collision_weight<0):
        p.error('--self-collision-weight requires --controller collision and a finite nonnegative value')
    os.environ['ACP_ROBOT']='roban'
    os.environ.setdefault('MUJOCO_GL','egl')
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','2')
    import torch
    from .models import FastKinematics,FastWorldModel,FastContactTask,FastStableContactTask,FastUpperSupervisedTask,FastGaitStyleTask,FastLegCollisionTask,FastJointTask,FastConsistentJointTask,FastWalkingPlannerTask,FastVelocityTask,device_norms,flat_scan
    from .models import FastBodyGoalTask,FastNextContactTask
    from .core import Updates,CudaUpdate
    from ..backend import backend_for_format,CONTACT_FORMAT,STABLE_CONTACT_FORMAT,UPPER_SUPERVISED_FORMAT,GAIT_STYLE_FORMAT,LEG_COLLISION_FORMAT,JOINT_FORMAT,CONSISTENT_FORMAT,WALKING_FORMAT,VELOCITY_FORMAT
    from ..backend import BODY_GOAL_FORMAT,NEXT_CONTACT_FORMAT
    from ..data import KINEMATIC_REFERENCE,prepare,split_clips
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
    formats = dict(contact=CONTACT_FORMAT,stable=STABLE_CONTACT_FORMAT,upper=UPPER_SUPERVISED_FORMAT,
                   gait=GAIT_STYLE_FORMAT,collision=LEG_COLLISION_FORMAT,joint=JOINT_FORMAT,
                   consistent=CONSISTENT_FORMAT,walking=WALKING_FORMAT,velocity=VELOCITY_FORMAT)
    backend=backend_for_format(NEXT_CONTACT_FORMAT if next_contact else BODY_GOAL_FORMAT if args.body_goal else formats[args.controller])
    clips=prepare(args.data,args.cache_dir)
    if any(c.get("reference_schema") == KINEMATIC_REFERENCE for c in clips) and (not next_contact or event_stage):
        p.error('Kinematic reference clips are supported by next-contact Stage 1')
    split=split_clips(clips,args.val_fraction,args.seed)
    config=dict(hist=args.hist,horizons=args.horizons,wm_hidden=args.wm_hidden,policy_hidden=args.policy_hidden,
                structured_wm=args.structured_wm)
    checkpoint=None
    init_v3=False
    init_v4=False
    init_v5=False
    init_joint=False
    init_next_contact=False
    planner_checkpoint=None
    if args.resume or args.init:
        checkpoint=torch.load(args.resume or args.init,map_location=dev,weights_only=True)
        init_v3=bool(args.init and not args.body_goal and not next_contact and args.controller in ('upper','gait','collision') and checkpoint['format']==STABLE_CONTACT_FORMAT)
        init_v4=bool(args.init and not args.body_goal and not next_contact and args.controller in ('gait','collision') and checkpoint['format']==UPPER_SUPERVISED_FORMAT)
        init_v5=bool(args.init and not args.body_goal and not next_contact and args.controller=='collision' and checkpoint['format']==GAIT_STYLE_FORMAT)
        init_joint=bool(args.init and joint and checkpoint['format']==LEG_COLLISION_FORMAT)
        init_next_contact=bool(args.init and next_contact and not event_stage and checkpoint['format']==LEG_COLLISION_FORMAT)
        if args.next_contact_stage2 and args.init and 'event_planner' in checkpoint:
            raise ValueError('Initialize from a recorded-target next-contact controller, or resume Stage 2')
        if (checkpoint['format']!=backend.format and not (init_v3 or init_v4 or init_v5 or init_joint or init_next_contact)) or checkpoint['robot']['fingerprint']!=metadata()['fingerprint']:
            raise ValueError('Checkpoint robot or controller interface differs')
        for key in ('hist','horizons','wm_hidden','policy_hidden'):
            if checkpoint['config'][key]!=config[key]:raise ValueError('Checkpoint '+key+' differs')
        if joint:
            if args.resume:planner_config=checkpoint['config']['planner_config']
            else:
                from ..data import JOINT_NAMES,STATE_FIELDS
                planner_checkpoint=torch.load(args.planner_init,map_location=dev,weights_only=True)
                if planner_checkpoint.get('format')!='two_foot_future_contact_v1' or planner_checkpoint['joint_names']!=JOINT_NAMES or planner_checkpoint['state_fields']!=STATE_FIELDS:
                    raise ValueError('Unsupported offline planner schema')
                if planner_checkpoint['horizons_s']!=config['horizons']:
                    raise ValueError('Planner/controller query windows differ')
                identity=lambda parts:{(x['fingerprint'],x['start'],x['end']) for x in parts}
                if any(identity(planner_checkpoint['split'][key])!=identity(checkpoint['split'][key]) for key in ('train','val')):
                    raise ValueError('Planner/controller train/validation split differs')
                planner_config=planner_checkpoint['config']
            config['planner_config']=planner_config
        if args.resume:
            if checkpoint.get('training_engine')!='blink_contact_v1' or checkpoint['config']!=config:
                raise ValueError('Fast --resume requires matching fast training config')
            old=checkpoint['args']
            if any(old[k]!=getattr(args,k) for k in ('optimizer','ema','controller','amp')):
                raise ValueError('Resume optimizer/EMA/controller/AMP differs')
            if args.event_stage2 != old.get('event_stage2',False):
                raise ValueError('Resume event Stage 2 setting differs')
            if args.next_contact_stage2 != old.get('next_contact_stage2', False):
                raise ValueError('Resume next-contact Stage 2 setting differs')
            if event_stage and any(old[k]!=getattr(args,k) for k in
                                        ('event_obs_start','event_reward_start','event_ramp_iters')):
                raise ValueError('Resume contact handover schedule differs')
            for key, default in (('profile', None), ('wm_mirror', 0.), ('critic_minibatches', 1)):
                if old.get(key, default) != getattr(args, key):
                    raise ValueError('Resume '+key+' differs')
            if args.profile == 'wbc' and any(old[k] != getattr(args, k) for k in ('lr','pi_lr_mult')):
                raise ValueError('Resume WBC learning-rate schedule differs')
            if joint and any(old[k]!=getattr(args,k) for k in ('planner_lr_mult','planner_mix_start','planner_mix_iters')):
                raise ValueError('Resume planner LR/mixing schedule differs')
            if consistent and old['horizon']!=args.horizon:raise ValueError('Resume realization horizon differs')
        current={c['fingerprint']:c for c in clips}
        expected={c['fingerprint'] for v in checkpoint['split'].values() for c in v}
        if not args.init_new_reference:
            if set(current)!=expected:raise ValueError('Checkpoint datasets changed')
            split={k:[dict(c,cache=current[c['fingerprint']]['cache']) for c in v] for k,v in checkpoint['split'].items()}
    if not split['train'] or not split['val']:
        raise ValueError('Reference requires nonempty training and validation clips')
    kin=FastKinematics(dev)
    motion=FootstepMotion(split['train'],kin,dev,args.horizons,args.max_clips,
                          reserve_steps=getattr(backend,'reserve_steps',0))
    if int(motion.clip_len.min())<=args.horizon+1:raise ValueError('Reference blocks too short')
    if bool(motion.clip_boxes.any()):raise ValueError('Fast flat-ground specialization only')
    geometry=max(r['max_m'] for r in motion.geometry_reports)
    if geometry>.05:raise ValueError('Capture/training FK differs by over 5cm')
    task_cls=dict(contact=FastContactTask,stable=FastStableContactTask,upper=FastUpperSupervisedTask,gait=FastGaitStyleTask,collision=FastLegCollisionTask,joint=FastJointTask,consistent=FastConsistentJointTask,walking=FastWalkingPlannerTask,velocity=FastVelocityTask)[args.controller]
    if args.body_goal:
        task_cls = FastBodyGoalTask
    if next_contact:
        task_cls = FastNextContactTask
    reward_cfg=backend.RewardCfg(**checkpoint['reward']) if checkpoint else backend.RewardCfg()
    if args.upper_supervision_weight is not None:
        if args.resume and args.upper_supervision_weight!=reward_cfg.upper_supervision_weight:
            raise ValueError('Resume upper supervision weight differs; use --init for a new run')
        reward_cfg.upper_supervision_weight=args.upper_supervision_weight
    for name in gait_options:
        value=getattr(args,name)
        if value is not None:
            if args.resume and value!=getattr(reward_cfg,name):
                raise ValueError('Resume '+name+' differs; use --init for a new run')
            setattr(reward_cfg,name,value)
    if args.self_collision_weight is not None:
        if args.resume and args.self_collision_weight!=reward_cfg.self_collision_weight:
            raise ValueError('Resume self collision weight differs; use --init for a new run')
        reward_cfg.self_collision_weight=args.self_collision_weight
    task=task_cls(motion,kin,reward_cfg)
    mdp=backend.MDP(kin,motion,torch.tensor(default_joint_pos_list(),device=dev),task)
    mdp.scan=lambda S,boxes:flat_scan(S,mdp.scan_dim)
    policy=device_norms(backend.Policy(args.hist,args.policy_hidden,**(dict(planner_config=planner_config) if joint else {})).to(dev))
    wm=FastWorldModel(mdp.scan_dim,args.wm_hidden,args.structured_wm,args.amp=='bf16').to(dev)
    critic=device_norms(Critic(getattr(mdp,'critic_input_dim',INV_DIM+mdp.critic_extra+backend.command_dim)).to(dev))
    event = None
    if event_stage:
        from ..event_planner import DuePlans, EventHandover, EventPlanner, FORMAT, handover
        from ..event_rollout import EventRollout
        if next_contact:
            from ..next_contact import NextContactHandover, NextContactPlanner, FORMAT
            event = NextContactHandover(NextContactPlanner(args.hist).to(dev),task)
        else:
            event = EventHandover(EventPlanner(args.hist).to(dev),task)
        event_opt = torch.optim.AdamW(event.planner.parameters(),lr=3e-4)
        event_replay = None
        event_collector = None
        event_loss = None
        event_successes = 0
        planner_rollout_ready = False
        eta_obs = eta_reward = 0.
        event_queries = torch.tensor([i for i,h in enumerate(args.horizons)
                                      if any(abs(h-q)<1e-6 for q in (.3,.5))],device=dev)
        if args.resume:
            if checkpoint['event_planner']['format'] != FORMAT:
                raise ValueError('Planner checkpoint format differs')
            event.planner.load_state_dict(checkpoint['event_planner']['weights'])
            event_opt.load_state_dict(checkpoint['opt_event'])
    def opt(model,lr,hidden=None,betas=(.9,.999)):
        if hidden and args.optimizer=='muon':
            from .vendor.optim import MuonAdam
            return MuonAdam(model,hidden,lr=lr,betas=betas,fused=True,capturable=True)
        return torch.optim.Adam(model.parameters(),lr=torch.tensor(lr,device=dev),betas=betas,fused=True,capturable=True)
    pi_hidden=[policy.enc,policy.dec]
    if args.controller in ('upper','gait','collision'):pi_hidden.extend([policy.upper_enc,policy.upper_dec])
    ow=opt(wm,args.lr,[wm.net]);ov=opt(critic,args.lr)
    if joint:
        groups=[dict(params=[v for k,v in policy.named_parameters() if v.requires_grad and not k.startswith('planner.')],
                     lr=torch.tensor(args.lr*args.pi_lr_mult,device=dev)),
                dict(params=list(policy.planner.parameters()),lr=torch.tensor(args.lr*args.pi_lr_mult*args.planner_lr_mult,device=dev))]
        op=torch.optim.Adam([g for g in groups if g["params"]],betas=(.9,.95),fused=True,capturable=True)
    else:op=opt(policy,args.lr*args.pi_lr_mult,pi_hidden,(.9,.95))
    updates=Updates(wm,policy,critic,mdp,task,ow,op,ov,args.horizon,args.gamma,args.lam,args.critic_epochs,
                    args.compile,args.ema,critic_minibatches=args.critic_minibatches,event=event)
    rates = mirror = None
    if args.profile == 'wbc' and event is None:
        from .wbc import WbcRates
        from ...motion import CONTROL_DT
        rates = WbcRates(args.lr, args.pi_lr_mult)
        reward_ceiling = CONTROL_DT * sum(getattr(reward_cfg, name) for name in
                                          ('contact_weight','position_weight','yaw_weight','upright_weight','height_weight'))
        if args.resume:
            rates = WbcRates(**checkpoint['wbc_rates'])
    if args.wm_mirror:
        from .wbc import FlatWindowMirror
        mirror = FlatWindowMirror(dev)
    if args.compile:
        options={'max_fusion_size':32}
        mdp.reward=torch.compile(mdp.reward,fullgraph=True,dynamic=False,options=options)
        mdp.termination=torch.compile(mdp.termination,fullgraph=True,dynamic=False,options=options)
        inference=torch.compile(policy.controller_action if joint else policy,fullgraph=True,dynamic=False,options=options)
        if joint:plan_inference=torch.compile(policy.contact_plan,fullgraph=True,dynamic=False,options=options)
    else:
        inference=policy.controller_action if joint else policy
        if joint:plan_inference=policy.contact_plan
    start_it=sim_steps=0
    if checkpoint:
        if init_joint:
            policy.init_controller(checkpoint['pi'])
            policy.planner.load_state_dict(planner_checkpoint['model'])
            print('Initialized joint agent from deployed v6 controller and offline planner',flush=True)
        elif init_next_contact:
            policy.init_from_query_controller(checkpoint['pi'])
            print('Initialized 12-D contact controller from v6 without its query feature',flush=True)
        elif init_v3:
            policy.init_from_v3(checkpoint.get('pi_live',checkpoint['pi']))
            print('Initialized legs from v3; the new waist/arm head starts fresh',flush=True)
        else:
            policy.load_state_dict(checkpoint['pi'] if event is not None and args.init
                                   else checkpoint.get('pi_live',checkpoint['pi']))
        if init_v4 or init_v5 or args.init_new_reference:
            norm_cap = 1e5 if args.init_new_reference else 1e6
            policy.obs_norm.count.clamp_max_(norm_cap);policy.goal_norm.count.clamp_max_(norm_cap)
            print('Initialized policy from previous task; the critic and optimizers start fresh',flush=True)
        same_wm_structure=bool(checkpoint['config'].get('structured_wm',False))==args.structured_wm
        if same_wm_structure:
            saved=dict(checkpoint['wm'])
            for key,value in list(saved.items()):
                if key.startswith('out_norm.'):
                    saved.setdefault('inc_norm.'+key[len('out_norm.'):],value)
            wm.load_state_dict(saved)
            if init_v3 or init_v4 or init_v5 or init_joint or args.init_new_reference:
                for norm in (wm.in_norm,wm.out_norm,wm.inc_norm):norm.count.clamp_max_(1e6)
        if updates.average is not None:
            updates.average.load_state_dict(policy.state_dict() if (init_v3 or init_v4 or init_v5 or init_joint or init_next_contact or args.init_new_reference) else checkpoint['pi'])
        if event is not None and args.init:
            critic.load_state_dict(checkpoint['critic'])
        if args.resume:
            critic.load_state_dict(checkpoint['critic'])
            for optimizer,key in ((ow,'opt_wm'),(op,'opt_pi'),(ov,'opt_v')):
                lr_tensors=[g['lr'] for g in optimizer.param_groups]
                optimizer.load_state_dict(checkpoint[key])
                for group,rate in zip(optimizer.param_groups,lr_tensors):
                    rate.fill_(float(group['lr']));group['lr']=rate
            start_it,sim_steps=checkpoint['iteration'],checkpoint['sim_steps']
            if args.iterations<=start_it:raise ValueError('iterations must exceed checkpoint iteration')
    env_cls = getattr(backend,'Env',FootstepTrackingSim)
    reset_cfg = None
    if any(reset_shares):
        from ...blink_reset_sampling import BlinkResetCfg
        reset_cfg = BlinkResetCfg(*reset_shares)
    env = env_cls(args.num_envs,motion,mdp,dev,gamma=args.gamma,reset_noise=args.reset_noise,blink_reset=reset_cfg)
    if event is not None and not next_contact:
        due_plans = DuePlans(event,args.num_envs,dev)
    sampler = env.blink_sampler
    probe_env = None
    if sampler is not None and args.resume and 'blink_reset' in checkpoint:
        sampler.load_state_dict(checkpoint['blink_reset'])
        env.reset(torch.arange(args.num_envs, device=dev))
    buffer=ReplayBuffer(args.buffer_steps,args.num_envs,dev,extra={'o':(backend.obs_dim,torch.float32)})
    history=torch.zeros(args.num_envs,args.hist,backend.obs_dim,device=dev)
    def sample_queries(n):
        if event is None or next_contact:
            return task.sample_queries(n)
        return event_queries[torch.randint(len(event_queries),(n,),device=dev)]
    query=sample_queries(args.num_envs)
    if event is not None:
        use_event = torch.zeros(args.num_envs,device=dev,dtype=torch.bool)
    if joint:
        from ..joint import PlannerSamples
        samples=PlannerSamples(motion,task,args.hist)
        if consistent:
            from ..consistent_joint import HindsightSamples
            hindsight=HindsightSamples(task,args.hist)
        initial_mix=min(1.,args.planner_mix_start+(1-args.planner_mix_start)*start_it/args.planner_mix_iters)
        use_planner=torch.rand(args.num_envs,device=dev)<initial_mix
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
    def collect(n,iteration,*,warmup=False):
        nonlocal history,sim_steps
        measure_return = rates is not None and rates.decay_start is None and not warmup
        reward_sum = sums[0].clone() if measure_return else None
        for _ in range(n):
            obs=env.observe()
            history=torch.where((env.ep_step==0)[:,None,None],obs[:,None].expand_as(history),
                                torch.cat((history[:,1:],obs[:,None]),1))
            if joint:query.copy_(task.pack_queries(env.f,query))
            state = env.state()
            cmd=task.command(state,env.f,query)
            contact_delta = None
            if event is not None:
                use_event.copy_(torch.where(env.ep_step==0,torch.rand(args.num_envs,device=dev)<eta_obs,use_event))
                if eta_obs:
                    planned = event.plan(history,state,env.f,query)
                    cmd = torch.where(use_event[:,None],planned,cmd)
                    if next_contact:
                        if eta_reward:
                            plan_in_world = event.world_plan(planned,state,env.f,query)
                            before = motion.requested_velocity[env.f]
                            def contact_delta(s1,f1):
                                same = (before == motion.requested_velocity[f1]).all(-1)
                                return eta_reward * event.reward_delta(s1,plan_in_world,use_event & same)
                    else:
                        due_plans.append(planned,state,env.f,use_event)
                    if eta_reward and not next_contact:
                        episode_step = env.ep_step.clone()
                        def contact_delta(s1,f1):
                            return due_plans.reward(s1,f1,query,episode_step,eta_reward)
            if joint:
                planned=plan_inference(history,cmd)
                oracle=task.oracle_command(env.state(),env.f,query)
                cmd=torch.where(use_planner[:,None],planned,oracle)
            if not walking:
                update_actor_norms(obs,cmd)
            a=(inference(history,cmd)+args.action_noise*torch.randn_like(env.a_prev)).clamp(-5,5)
            if hasattr(policy,'clamp_action'):
                a=policy.clamp_action(a)
            tr=env.step(a,contact_reward_delta=contact_delta);tr['goal_steps']=query.clone()
            sim_guard.observe(tr,motion,sim_steps,iteration)
            buffer.add(tr)
            update_norms(unflatten_state(tr['s']),unflatten_state(tr['s1']),a,tr['xi'],tr['valid'])
            sim_steps+=args.num_envs
            sums[0].add_(tr['r'].mean());sums[1].add_(1)
            sums[2].add_(tr['term'].float().sum());sums[3].add_(tr['done'].float().sum())
            # Fixed-size masked query updates avoid a host read of done.sum().
            query.copy_(torch.where(tr['done'],sample_queries(args.num_envs),query))
            if joint:
                fraction=min(1.,args.planner_mix_start+(1-args.planner_mix_start)*iteration/args.planner_mix_iters)
                use_planner.copy_(torch.where(tr['done'],torch.rand(args.num_envs,device=dev)<fraction,use_planner))
        if measure_return:
            rates.training_return = float((sums[0] - reward_sum) / (n * reward_ceiling))
    def starts():
        s,h=buffer.sample_starts(2*args.pi_batch,args.hist)
        ok=s['valid'] & (s['f']+args.horizon<=motion.clip_end[motion.clip_id[s['f']]])
        if joint:ok=ok & task.eligible(s['f'],s['goal_steps'],args.horizon)
        order=torch.argsort((~ok).to(torch.int32),stable=True)
        count=ok.long().sum()
        take=order[torch.arange(args.pi_batch,device=dev)%count.clamp_min(1)]
        result={k:s[k][take] for k in ('s','a_prev','xi','f','goal_steps')}
        result['hist']=h['o'][take]
        result['usable']=ok[take]
        # Even an unusable placeholder may not gather outside the reference arrays.
        c=motion.clip_id[result['f']]
        result['f']=torch.minimum(result['f'],motion.clip_end[c]-args.horizon)
        if event is not None:
            result['eta_obs'] = torch.tensor(eta_obs,device=dev)
            result['eta_reward'] = torch.tensor(eta_reward,device=dev)
        if joint:result.update(samples.sample(args.pi_batch))
        if consistent:result.update(hindsight.sample(buffer,args.pi_batch))
        return result
    modules=[wm,policy,critic]+([updates.average] if updates.average is not None else [])
    counters=[updates.errors,updates.runs,updates.worst]
    world=CudaUpdate(updates.world,modules,[ow],counters,args.cuda_graph)
    actor=CudaUpdate(updates.actor,modules,[op,ov],counters,args.cuda_graph)
    task_report=dict(format=backend.format,training_engine='blink_contact_v1',config=config,split=split,
                     command_dim=backend.command_dim,observation_dim=backend.obs_dim,reward=asdict(task.cfg),
                     train_frames=motion.num_frames,geometry_max_m=geometry,robot=metadata())
    if joint:task_report.update(controller_command_dim=13,planner_command_dim=4,joint_mode='fixed_controller' if walking else 'planner_and_controller')
    if consistent:task_report.update(future_realization_constraints=True,flat_ground_projection=True)
    if next_contact:
        task_report.update(stage='acp_next_contact_joint' if event is not None else 'acp_next_contact_reference',
                           controller_command_dim=backend.command_dim,event_window_s=1.5)
    elif event is not None:
        task_report.update(stage='contact_event_stage2',controller_command_dim=13,event_window_s=1.5)
    (out/'task.json').write_text(json.dumps(task_report,indent=2))
    print(f"Fast {backend.command_dim}D: compile={args.compile}, graph={args.cuda_graph}, AMP={args.amp}, "
          f"structured={args.structured_wm}, optimizer={args.optimizer}, EMA={args.ema}",flush=True)
    collect(args.warmup_steps,start_it,warmup=True)
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
        if walking or event is not None:
            os.link(out/'last.pt', out / f"iteration_{payload['iteration']:06d}.pt")
    def save(iteration):
        nonlocal pending
        if pending is not None:pending.result()
        deployed=updates.average if updates.average is not None else policy
        payload=dict(**task_report,args=vars(args),iteration=iteration,sim_steps=sim_steps,
                     wm=wm.state_dict(),pi=deployed.state_dict(),pi_live=policy.state_dict(),critic=critic.state_dict(),
                     opt_wm=ow.state_dict(),opt_pi=op.state_dict(),opt_v=ov.state_dict())
        if rates is not None:
            payload['wbc_rates'] = asdict(rates)
        if sampler is not None:
            payload['blink_reset'] = sampler.state_dict()
        if event is not None:
            payload['event_planner'] = dict(format=FORMAT,hist=args.hist,weights=event.planner.state_dict())
            payload['opt_event'] = event_opt.state_dict()
        pending=pool.submit(write,cpu(payload))
    initial_seconds=time.monotonic()-load_start
    start=time.monotonic();steady_start=None;steady_at=None
    upper_active=updates.upper_supervision is not None
    joint_active=updates.joint_supervision is not None
    trajectory_active=updates.trajectory_supervision is not None
    gait_columns=(['leg_pose_cost','leg_velocity_cost','pelvis_tilt_cost','stance_tilt_cost','knee_cost',
                   'contact_reward','position_reward','yaw_reward'] if args.controller in ('gait','collision') else [])
    if args.controller=='collision':gait_columns.extend(['self_collision_cost','min_leg_clearance_m','leg_proxy_overlap_fraction'])
    if joint:gait_columns=['leg_pose_cost','leg_velocity_cost','pelvis_tilt_cost','stance_tilt_cost','knee_cost',
                          'self_collision_cost','min_leg_clearance_m','leg_proxy_overlap_fraction',
                          'velocity_error_mps','yaw_rate_error_rps','velocity_reward','yaw_velocity_reward']
    if consistent:gait_columns.extend(['phase_contact_cost','swing_clearance_cost','slip_cost'])
    if getattr(reward_cfg,'motion_style_weight',0.):
        gait_columns.extend(['lift_timing_cost','swing_motion_cost','weight_shift_cost'])
    wm_metrics=torch.zeros(3,device=dev);pi_metrics=torch.zeros(5+int(upper_active)+int(joint_active)+int(trajectory_active),device=dev)
    try:
        with (out/'train.csv').open('w',newline='') as file:
            writer=csv.writer(file)
            writer.writerow(['iteration','wall_s','sim_steps','wm_loss','pi_loss','critic_loss','pi_gradient',
                             'imag_survival','sim_reward','term_fraction','bad_wm','bad_pi','bad_critic']
                            +(['lr','pi_lr','normalized_sim_reward'] if rates is not None else [])
                            +(['reset_failure_weight','reset_probe_weight'] if sampler is not None else [])
                            +(['event_loss','event_samples','event_successes','eta_obs','eta_reward'] if event is not None else [])
                            +(['upper_supervision_loss'] if upper_active else [])
                            +(['planner_supervision_loss','planner_collection_fraction','usable_actor_fraction'] if joint else [])+gait_columns
                            +(['trajectory_consistency_loss','hindsight_valid_fraction','in_model_query_fraction'] if consistent else []))
            for iteration in range(start_it+1,args.iterations+1):
                if event is not None:
                    if event_collector is None or iteration%args.event_rollout_every==0:
                        with torch.random.fork_rng(devices=[torch.device(dev)]):
                            if event_collector is None:
                                rollout_mdp = backend.MDP(kin,motion,mdp.default_q,task)
                                rollout_env = env_cls(args.event_rollout_envs,motion,rollout_mdp,dev,
                                                      eval_mode=True,reset_noise=False)
                                event_collector = EventRollout(rollout_env,task,next_contact=next_contact)
                            rollout_mix = min(.75,eta_obs) if next_contact else eta_obs
                            batch, successes = event_collector.collect(
                                updates.average,event.planner,args.event_rollout_steps,rollout_mix)
                        if next_contact:
                            attempts = event_collector.planned_moving_attempts
                            accepted = event_collector.planned_moving_successes
                            if attempts:
                                planner_rollout_ready = accepted >= max(2, math.ceil(attempts*.25))
                            print(f'Planned moving rollouts: {accepted}/{attempts}; handover ready: {planner_rollout_ready}',
                                  flush=True)
                        event_successes += successes
                        if batch['history'].numel():
                            if next_contact:
                                event.planner.affordance.prop_norm.update(event.planner.proprio(batch['history']))
                            event_replay = {name:(torch.cat((event_replay[name],data))[-32768:]
                                                  if event_replay is not None else data[-32768:])
                                            for name,data in batch.items()}
                        print(f'Event rollout at {iteration}: {successes} successful, '
                              f'{len(batch["history"])} examples',flush=True)
                    event_samples = 0 if event_replay is None else len(event_replay['history'])
                    if event_samples>=args.event_batch and event_replay['valid'].any():
                        ids = torch.randint(event_samples,(args.event_batch,),device=dev)
                        event_loss = event.planner.loss({k:x[ids] for k,x in event_replay.items()})
                        event_opt.zero_grad(set_to_none=True)
                        event_loss.backward()
                        grad = torch.nn.utils.clip_grad_norm_(event.planner.parameters(),1.)
                        if not (torch.isfinite(event_loss) & torch.isfinite(grad)):
                            raise FloatingPointError('Nonfinite contact event supervision')
                        event_opt.step()
                        event_loss = event_loss.detach()
                        eta_obs = handover(iteration-1,args.event_obs_start,args.event_ramp_iters)
                        eta_reward = handover(iteration-1,args.event_reward_start,args.event_ramp_iters)
                        if next_contact and not planner_rollout_ready:
                            eta_obs, eta_reward = min(eta_obs,.25), 0.
                if rates is not None:
                    lr, pi_lr = rates.rates(iteration-1)
                    for optimizer, rate in ((ow,lr),(op,pi_lr),(ov,lr)):
                        for group in optimizer.param_groups:
                            group['lr'].fill_(rate)
                collect(int(iteration*args.collect_steps)-int((iteration-1)*args.collect_steps),iteration)
                for _ in range(args.wm_batches):
                    windows = buffer.sample_windows(args.wm_batch,args.wm_window)
                    wm_metrics = world(mirror(windows,args.wm_mirror) if mirror is not None else windows)
                active=iteration>args.wm_only_iters
                if active:
                    start_batch=starts()
                    pi_metrics=actor(start_batch)
                if active and args.probe_share and iteration>=args.probe_start and iteration%args.probe_every==0:
                    from .wbc import probe_failures
                    with torch.random.fork_rng(devices=[torch.device(dev)]):
                        if probe_env is None:
                            probe_mdp = backend.MDP(kin,motion,mdp.default_q,task)
                            probe_env = env_cls(args.probe_envs,motion,probe_mdp,dev,eval_mode=True,reset_noise=False)
                            probe_env.max_episode_steps = env.max_episode_steps
                        failed = probe_failures(probe_env,updates.average,task)
                    sampler.set_focus(failed,preceding_bins=2)
                    print(f'Blink EMA probe at {iteration}: {len(failed)} failures',flush=True)
                if active and steady_start is None:
                    torch.cuda.synchronize();steady_start=time.monotonic();steady_at=iteration
                if iteration%args.log_every==0 or iteration==args.iterations:
                    values=torch.cat((wm_metrics[:1],pi_metrics[:4],sums,updates.errors.float(),updates.worst.float(),pi_metrics[5:])).cpu().tolist()
                    row=[iteration,time.monotonic()-start,sim_steps,*values[:5],values[5]/max(values[6],1),
                         values[7]/max(values[8],1),*values[9:12]]
                    if rates is not None:
                        row.extend((lr,pi_lr,row[8]/reward_ceiling))
                    if sampler is not None:
                        row.extend(torch.stack((sampler.failures.sum(),sampler.focus.sum())).tolist())
                    if event is not None:
                        row.extend((float(event_loss) if event_loss is not None else None,
                                    event_samples,event_successes,eta_obs,eta_reward))
                    if upper_active:row.append(values[15])
                    if joint:
                        row.extend([values[15 + int(upper_active)],float(use_planner.float().mean()),float(start_batch['usable'].float().mean()) if active else 0.])
                    if gait_columns:
                        with torch.no_grad():diag=mdp.training_diagnostics(env.state(),env.f)
                        row.extend(torch.stack([diag[key] for key in gait_columns]).cpu().tolist())
                    if consistent:
                        valid_ft=float(start_batch['ft_valid'].float().mean()) if active else 0.
                        coverage=float((motion.horizon_steps[start_batch['goal_steps'].remainder(task.num_queries)]<=args.horizon).float().mean()) if active else 0.
                        row.extend([values[17],valid_ft,coverage])
                    writer.writerow(row);file.flush();sums.zero_()
                    print(f"it {iteration} wm {values[0]:.4g} pi {values[1]:.4g} survival {values[4]:.4f} "
                          f"bad={values[9:12]}"+(f" upper_bc={values[15]:.4g}" if upper_active else '')
                          +(f" planner_bc={values[15 + int(upper_active)]:.4g}" if joint else ''),flush=True)
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
