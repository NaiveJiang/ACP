"""Adapt the v6 controller to fixed omni planner goals and speed changes."""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .omni_planner import OmniPlanner
from .phase_pose import predict
from .reference import FootstepMotion
from .backend import backend_for_format
from .fast.models import FastWorldModel, FastKinematics, device_norms, flat_scan
from ..mdp import normalize_xi, unflatten_state
from ..networks import WorldModel
from ..robot import default_joint_pos_list
from ...rotations import yaw_mat

DT = .02


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def local_plan(state, dt, flags, target, heading):
    basis = yaw_mat(state[1]).transpose(-1, -2)
    xyz = (basis[:,None] @ (target-state[0][:,None])[...,None]).squeeze(-1)
    body_yaw = torch.atan2(basis[:,0,1],basis[:,0,0])
    angle = heading-body_yaw[:,None]
    direction = torch.stack((angle.sin(),angle.cos()),-1)
    return torch.cat((dt,flags,(xyz*flags[...,None]).flatten(1),
                      (direction*flags[...,None]).flatten(1)),1)


def world_goal(state, plan):
    basis = yaw_mat(state[1])
    target = state[0][:,None] + (basis[:,None] @ plan[:,3:9].reshape(-1,2,3,1)).squeeze(-1)
    yaw = torch.atan2(basis[:,1,0],basis[:,0,0])[:,None]
    heading = yaw + torch.atan2(plan[:,9:13:2],plan[:,10:13:2])
    return target,heading


def learned_plan(planner, history, cmd, support, phase, initial, waiting):
    x = torch.cat((history[...,:48].flatten(1),cmd,support,phase),1)
    xyz,yaw = predict(planner.pose_model,planner.pose_weights,x)
    xyz = torch.where(waiting[:,None,None],initial[0],xyz)
    yaw = torch.where(waiting[:,None,None],initial[1],yaw)
    return torch.cat((cmd[:,3:4],support,(xyz*support[...,None]).flatten(1),
                      (yaw*support[...,None]).flatten(1)),1)


def precompute_timing(planner, start, end, switch, seconds, horizon, device):
    planner.reset(torch.zeros(1,2,3,device=device),torch.zeros(1,2,2,device=device))
    obs = torch.zeros(1,48,device=device)
    rows = []
    with torch.inference_mode():
        for i in range(seconds):
            vx = start if i<switch else end
            planner.step(obs,(vx,0.,0.),horizon)
            phase, code, waiting = planner.last_timing
            angle = 2*math.pi*phase
            rows.append((vx,float(bool(code&1)),float(bool(code&2)),math.sin(angle),math.cos(angle),waiting))
    return torch.tensor(rows,device=device)


@torch.no_grad()
def collect(env, policy, planner, task, motion, horizon, iteration, frames):
    n,dev = env.N,motion.device
    ids = torch.arange(n,device=dev)
    use_recorded = ids%4==0
    query_idx = int(torch.argmin((motion.horizons-horizon).abs()))
    query_steps = int(motion.horizon_steps[query_idx])
    q = torch.full((n,),query_idx,device=dev,dtype=torch.long)
    group = (ids//4)%2
    starts = motion.clip_start[group] + iteration%12
    env.reset(ids,frames=starts)
    env.refresh()
    start = env.state()
    zero_dt = start[0].new_full((n,1),horizon)
    fpos,frot,_,_ = task.ground.foot_state(start)
    eye = yaw_mat(start[1]).transpose(-1,-2)[:,None]
    local = (eye @ (fpos-start[0][:,None])[...,None]).squeeze(-1)
    frel = eye @ frot
    yaw = torch.atan2(frel[...,1,0],frel[...,0,0])
    initial = (local,torch.stack((yaw.sin(),yaw.cos()),-1))
    switch = min(frames-2*query_steps-3,100+iteration*17%65)
    table = torch.stack([precompute_timing(planner,a,b,switch,frames,horizon,dev)
                         for a,b in ((.2,.5),(.5,.2))],dim=1)
    history = None
    rows = []
    live = torch.ones(n,device=dev,dtype=torch.bool)
    for t in range(frames):
        state = env.state()
        obs = env.observe()
        history = (obs[:,None].expand(-1,planner.hist,-1).clone() if history is None else
                   torch.cat((history[:,1:],obs[:,None]),1))
        row = table[t,group]
        command = torch.cat((row[:,:1],torch.zeros_like(row[:,:2]),zero_dt),1)
        support, phase, waiting = row[:,1:3],row[:,3:5],row[:,5].bool()
        plan = learned_plan(planner,history,command,support,phase,initial,waiting)
        oracle = task.oracle_command(state,env.f,q)
        plan = torch.where(use_recorded[:,None],oracle,plan)
        target,heading = world_goal(state,plan)
        action = policy.controller_action(history,plan)
        tr = env.step(action)
        live = live & tr["valid"] & ~tr["done"]
        rows.append(dict(s=tr["s"],s1=tr["s1"],a=action,xi=tr["xi"],hist=history,
                         cmd=command,support=plan[:,1:3],phase=phase,waiting=waiting,
                         target=target,heading=heading,live=live))
        if not live.any():
            break
    return {k:torch.stack([row[k] for row in rows]) for k in rows[0]},use_recorded,initial,switch,query_steps


def fit_wm(wm, optimizer, mdp, data, indices, batch, count):
    losses = []
    for _ in range(count):
        t,e = indices[torch.randint(len(indices),(batch,),device=indices.device)].unbind(1)
        state = unflatten_state(data["s"][t,e])
        xi = normalize_xi(data["xi"][t,e])
        loss = state[0].new_zeros(())
        for k in range(3):
            state = wm.step(state,data["a"][t+k,e],xi,flat_scan(state,mdp.scan_dim))
            actual = unflatten_state(data["s1"][t+k,e])
            error = WorldModel.state_error(state,actual) / wm.inc_norm.std().clamp_min(.005)
            loss = loss+error.square().mean()/3
        update(loss,optimizer)
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def update(loss, optimizer):
    optimizer.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    loss.backward()
    params = [p for group in optimizer.param_groups for p in group["params"]]
    torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    optimizer.step()


def gait_cost(task, state, data, t, env_ids, query_steps, previous_feet=None):
    """Shape the swing path using support commands that are due now."""
    feet, _, gap, point_velocity = task.ground.foot_state(state)
    due = t - query_steps
    before = (due-2).clamp_min(0)
    after = (due+2).clamp_min(0)
    current = due.clamp_min(0)
    known = ((due>=2) & (data["cmd"][before,env_ids,0] == data["cmd"][t-1,env_ids,0])
             & (data["cmd"][after,env_ids,0] == data["cmd"][t-1,env_ids,0]))
    is_support = data["support"][current,env_ids].bool()
    mid_swing = (~is_support & ~data["support"][before,env_ids].bool()
                 & ~data["support"][after,env_ids].bool() & known[:,None])
    clearance = ((.025-gap).relu()/.025).square()
    swing = (clearance*mid_swing).sum()/mid_swing.sum().clamp_min(1)

    # Use the lowest point instead of sole-center motion so normal foot roll
    # is not penalized as slip. Detach proximity so this cost cannot buy a hop.
    proximity = ((.018-gap.detach())/.012).clamp(0,1)
    tangential_speed = point_velocity[...,:2].norm(dim=-1)
    slip = ((((tangential_speed-.10).relu()/.5).square())*proximity).mean()
    landing = (is_support | data["support"][after,env_ids].bool()) & known[:,None]
    near_landing = ((.035-gap.detach())/.025).clamp(0,1)
    descent = ((-point_velocity[...,2]-.35).relu()/.5).square()
    impact = (descent*near_landing*landing).mean()
    vertical = ((state[2][:,2]-.23).relu()/.25).square().mean()
    costs = torch.stack((swing,slip,impact,vertical))
    if previous_feet is None:
        return costs,feet,None
    # An early scuff can fall outside the planned landing interval. Slow
    # downward motion near the ground even in that case. Check sole center
    # as well as the lowest point to catch abrupt lowering during foot roll.
    sole_descent = (previous_feet[...,2]-feet[...,2])/DT
    downward_speed = torch.maximum(sole_descent,-point_velocity[...,2])
    near_floor = ((.04-gap.detach())/.03).clamp(0,1)
    excess = (downward_speed-.30).relu()/.45
    soft_touchdown = F.smooth_l1_loss(excess,torch.zeros_like(excess),
                                     reduction="none")*near_floor
    return costs,feet,soft_touchdown


def fit_controller(wm, policy, teacher, planner, optimizer, mdp, task, data, indices,
                   batch_size, query_steps, recorded, initial, gait_weights=None, yaw_rate_weight=0., pose_teacher=None,
                   speed_weight=0., smooth_weight=0., tune_position=False, soft_touchdown_weight=0.,
                   decel_indices=None, decel_mix=0.):
    decel_count = min(batch_size-1,round(decel_mix*batch_size)) if decel_indices is not None else 0
    chosen = indices[torch.randint(len(indices),(batch_size-decel_count,),device=indices.device)]
    if decel_count:
        extra = decel_indices[torch.randint(len(decel_indices),(decel_count,),device=indices.device)]
        chosen = torch.cat((chosen,extra))
    t,e = chosen.unbind(1)
    state = unflatten_state(data["s"][t,e])
    history = data["hist"][t,e]
    xi_raw = data["xi"][t,e]
    xi = normalize_xi(xi_raw)
    target = data["target"][t,e]
    heading = data["heading"][t,e]
    support = data["support"][t,e]
    distill = 0.
    gait = state[0].new_zeros(4)
    yaw_rate_error = state[0].new_zeros(batch_size)
    pose_anchor = state[0].new_zeros(())
    position_anchor = state[0].new_zeros(())
    speed_error = state[0].new_zeros(batch_size)
    speed_step_cost = state[0].new_zeros(batch_size)
    smooth = state[0].new_zeros(())
    previous_action = history[:,-1,-21:]
    previous_feet = task.ground.foot_state(state)[0] if soft_touchdown_weight else None
    touchdown_sum = state[0].new_zeros(batch_size,2)
    touchdown_peak = torch.zeros_like(touchdown_sum)
    for k in range(query_steps):
        cmd,support_k,phase,waiting = (data[name][t+k,e] for name in ("cmd","support","phase","waiting"))
        learned = learned_plan(planner,history,cmd,support_k,phase,
                               (initial[0][e],initial[1][e]),waiting)
        if pose_teacher is not None:
            with torch.no_grad():
                features = torch.cat((history[...,:48].detach().flatten(1),cmd,support_k,phase),1)
                reference_position, reference_heading = predict(pose_teacher,planner.pose_weights,features)
            active = (~recorded[e] & ~waiting)[:,None] * support_k
            angle_cost = 1-(learned[:,9:13].reshape(-1,2,2)*reference_heading).sum(-1)
            pose_anchor = pose_anchor + (angle_cost*active).sum()/active.sum().clamp_min(1)/query_steps
            if tune_position:
                scale = learned.new_tensor([.06,.04,.025])
                proposed = learned[:,3:9].reshape(-1,2,3)
                position_cost = F.smooth_l1_loss(proposed/scale,reference_position/scale,
                                                reduction="none").mean(-1)
                position_anchor = position_anchor + (position_cost*active).sum()/active.sum().clamp_min(1)/query_steps
        oracle = local_plan(state,cmd[:,3:4],support_k,data["target"][t+k,e],data["heading"][t+k,e])
        plan = torch.where(recorded[e,None],oracle,learned)
        action = policy.controller_action(history,plan)
        if smooth_weight:
            # Only penalize larger motor changes, preserving normal swing motion.
            change = ((action-previous_action).abs()-.15).relu()/.3
            smooth = smooth+change.square().mean()/query_steps
            previous_action = action
        with torch.no_grad():
            reference = teacher.controller_action(history.detach(),plan.detach())
        distill = distill + (action-reference).square().mean()/query_steps
        state = wm.step(state,action,xi,flat_scan(state,mdp.scan_dim))
        if speed_weight:
            direction = yaw_mat(state[1])[...,0]
            error = ((state[2]*direction).sum(-1)-cmd[:,0])/.20
            speed_error = speed_error+error/query_steps
            speed_step_cost = speed_step_cost+F.smooth_l1_loss(error,torch.zeros_like(error),
                                                               reduction="none")/query_steps
        if yaw_rate_weight:
            yaw_rate_error = yaw_rate_error+(state[3][:,2]-cmd[:,2])/query_steps
        if gait_weights is not None:
            costs,feet_k,near_cost = gait_cost(task,state,data,t+k+1,e,query_steps,previous_feet)
            gait = gait+costs/query_steps
            if soft_touchdown_weight:
                touchdown_sum = touchdown_sum+near_cost/query_steps
                touchdown_peak = torch.maximum(touchdown_peak,near_cost)
                previous_feet = feet_k
        obs = mdp.obs(state,action,xi_raw)
        history = torch.cat((history[:,1:],obs[:,None]),1)
    feet,rotation,gap,_ = task.ground.foot_state(state)
    actual_yaw = torch.atan2(rotation[...,1,0],rotation[...,0,0])
    pose_error = F.smooth_l1_loss(feet/.05,target/.05,reduction="none").sum(-1)
    facing_error = 1-(actual_yaw-heading).cos()
    contact = torch.sigmoid((task.cfg.contact_distance-gap)/task.cfg.contact_temperature)
    realized = ((pose_error+facing_error)*support).sum()/support.sum().clamp_min(1)
    measured = unflatten_state(data["s1"][t+query_steps-1,e])
    height_error = ((state[0][:,2]-measured[0][:,2])/.05).square().mean()
    tilt_error = (state[1][:,2,:]-measured[1][:,2,:]).square().sum(-1).mean()
    loss = realized+.5*(contact-support).square().mean()+2.*distill+.1*height_error+tilt_error
    if gait_weights is not None:
        loss = loss+(gait*gait_weights).sum()
    yaw_loss = loss.new_zeros(())
    if yaw_rate_weight:
        # Evaluate learned-plan rollouts only. Oracle rehearsals follow the
        # recorded targets independently of the synthetic speed schedule.
        valid = ~recorded[e] & ~data["waiting"][t,e]
        rate_error = F.smooth_l1_loss(yaw_rate_error/.35,torch.zeros_like(yaw_rate_error),
                                     reduction="none")
        yaw_loss = (rate_error*valid).sum()/valid.sum().clamp_min(1)
        loss = loss+yaw_rate_weight*yaw_loss
    speed_loss = loss.new_zeros(())
    if speed_weight:
        valid = ~recorded[e] & ~data["waiting"][t,e]
        error = F.smooth_l1_loss(speed_error,torch.zeros_like(speed_error),reduction="none")
        speed_loss = ((error+.2*speed_step_cost)*valid).sum()/valid.sum().clamp_min(1)
    loss = loss+.5*pose_anchor+position_anchor+speed_weight*speed_loss+smooth_weight*smooth
    soft_loss = (touchdown_sum+.25*touchdown_peak).mean()
    loss = loss+soft_touchdown_weight*soft_loss
    update(loss,optimizer)
    return (float(loss.detach()),gait.detach().tolist(),float(yaw_loss.detach()),float(pose_anchor.detach()),
            float(speed_loss.detach()),float(smooth.detach()),float(position_anchor.detach()),float(soft_loss.detach()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--planner",required=True)
    p.add_argument("--log-dir",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--num-envs",type=int,default=64)
    p.add_argument("--iterations",type=int,default=100)
    p.add_argument("--frames",type=int,default=300)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--updates",type=int,default=8)
    horizons = p.add_mutually_exclusive_group()
    horizons.add_argument("--query-dt",type=float,choices=(.3,.5),default=.3)
    horizons.add_argument("--query-dts",type=float,nargs="+",choices=(.3,.5),
                          help="Alternate among these query horizons in real collection and training")
    p.add_argument("--controller-lr",type=float,default=2e-5)
    p.add_argument("--wm-lr",type=float,default=2e-5)
    p.add_argument("--min-post-switch-fraction",type=float,default=0.,
                   help="Stop before updating if either direction loses too many post-switch deadlines")
    p.add_argument("--gait-guard", action="store_true",
                   help="Penalize premature ground contact, slip and hard touchdowns")
    p.add_argument("--slip-weight", type=float, default=.5)
    p.add_argument("--yaw-rate-weight",type=float,default=0.,
                   help="Train the controller to match commanded mean yaw rate on learned plans")
    p.add_argument("--joint-foot-yaw",action="store_true",
                   help="Also tune the planner heading outputs alongside the controller")
    p.add_argument("--joint-foot-pose",action="store_true",
                   help="Jointly tune all XYZ and yaw outputs of the planner with the controller")
    p.add_argument("--speed-weight",type=float,default=0.)
    p.add_argument("--smooth-weight",type=float,default=0.)
    p.add_argument("--landing-weight",type=float,default=.15)
    p.add_argument("--soft-touchdown-weight",type=float,default=0.,
                   help="Penalize downward speed close to the ground, even before planned touchdown")
    p.add_argument("--decel-mix",type=float,default=0.,
                   help="Reserve this fraction of actor samples for the first second after slowing down")
    p.add_argument("--pose-lr",type=float,default=1e-5)
    p.add_argument("--save-every",type=int,default=10)
    a = p.parse_args()
    if any(not math.isfinite(w) or w<0 for w in
           (a.slip_weight,a.yaw_rate_weight,a.speed_weight,a.smooth_weight,a.landing_weight,a.soft_touchdown_weight)):
        p.error("Loss weights must be finite and nonnegative")
    if not all(math.isfinite(v) and v>0 for v in (a.controller_lr,a.wm_lr)):
        p.error("Controller and world model learning rates must be finite and positive")
    if a.soft_touchdown_weight and not a.gait_guard:
        p.error("--soft-touchdown-weight requires --gait-guard")
    if not 0 <= a.decel_mix < 1:
        p.error("--decel-mix must be in [0,1)")
    if not 0 <= a.min_post_switch_fraction <= 1:
        p.error("--min-post-switch-fraction must be in [0,1]")
    query_dts = a.query_dts or [a.query_dt]
    joint_pose = a.joint_foot_yaw or a.joint_foot_pose
    if a.joint_foot_yaw and a.joint_foot_pose:
        p.error("Choose one of --joint-foot-yaw and --joint-foot-pose")
    if a.joint_foot_pose and a.speed_weight<=0:
        p.error("--joint-foot-pose requires positive --speed-weight")
    if joint_pose and (a.yaw_rate_weight<=0 or not math.isfinite(a.pose_lr) or a.pose_lr<=0):
        p.error("Joint fine tuning requires positive --yaw-rate-weight and --pose-lr")
    if a.num_envs<8 or a.frames<80 or not min(a.iterations,a.batch_size,a.updates,a.save_every)>0:
        p.error("Need at least 8 environments, 80 rollout frames and positive update counts")
    a.log_dir.mkdir(parents=True,exist_ok=False)
    (a.log_dir/"args.json").write_text(json.dumps(vars(a),default=str,indent=2)+"\n")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    ck = load(a.checkpoint)
    config = ck["config"]
    dev = a.device
    gait_weights = torch.tensor([.5,a.slip_weight,a.landing_weight,.15],device=dev) if a.gait_guard else None
    backend = backend_for_format(ck["format"])
    if ck["format"]!="two_foot_fixed_controller_planner_v9":
        raise ValueError("Stage 1 initializes from the v9 checkpoint containing the v6 controller")
    bundle = load(a.planner)
    if bundle["robot_fingerprint"]!=ck["robot"]["fingerprint"] or bundle["history_length"]!=config["hist"]:
        raise ValueError("Planner and tracker do not match")
    planner = OmniPlanner(bundle,dev)
    planner.pose_model.requires_grad_(False)
    # Use train-only stop/start clips for standing resets and oracle rehearsals.
    cases = {"0021_forward_r0","0084_forward_r0"}
    clips = [c for c in ck["split"]["train"] if c["case"] in cases]
    if len(clips)!=2 or any(c["end"]-c["start"]<a.frames+65 for c in clips):
        raise ValueError("Require the two forward training reference clips and shorter rollouts")
    kin = FastKinematics(dev)
    motion = FootstepMotion(clips,kin,dev,config["horizons"])
    task = backend.Task(motion,kin,backend.RewardCfg(**ck["reward"]))
    mdp = backend.MDP(kin,motion,torch.tensor(default_joint_pos_list(),device=dev),task)
    env = backend.Env(a.num_envs,motion,mdp,dev,eval_mode=True,reset_noise=False)
    policy = device_norms(backend.Policy(config["hist"],config["policy_hidden"],
                          planner_config=config["planner_config"]).to(dev))
    policy.load_state_dict(ck["pi"])
    teacher = copy.deepcopy(policy).requires_grad_(False)
    # The packaged path uses controller_action only. Train the existing low
    # level action heads and keep their input statistics and other modules fixed.
    for module in (policy.enc,policy.dec,policy.upper_enc,policy.upper_dec):
        module.requires_grad_(True)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    pose_teacher = None
    groups = [dict(params=trainable,lr=a.controller_lr)]
    if joint_pose:
        pose_teacher = copy.deepcopy(planner.pose_model).requires_grad_(False).eval()
        output = planner.pose_model[-1].requires_grad_(True)
        if not a.joint_foot_pose:
            mask = output.bias.new_tensor([0.,0.,0.,1.,1.]*2)
            output.weight.register_hook(lambda grad: grad*mask[:,None])
            output.bias.register_hook(lambda grad: grad*mask)
        # AdamW weight decay would otherwise also move masked position rows.
        groups.append(dict(params=list(output.parameters()),lr=a.pose_lr,weight_decay=0.))
    opt = torch.optim.AdamW(groups)
    wm = FastWorldModel(mdp.scan_dim,config["wm_hidden"],config["structured_wm"],amp=True).to(dev)
    wm.load_state_dict(ck["wm"])
    wmopt = torch.optim.AdamW(wm.parameters(),lr=a.wm_lr)
    def save(iteration, *, prefix=""):
        # Keep checkpoint compatible with the normal play entrypoint.
        # Drop stale optimizer internals from the original training engine.
        for key in ("opt_wm","opt_pi","opt_v"):
            ck.pop(key,None)
        ck.update(pi={k:v.detach().cpu() for k,v in policy.state_dict().items()},iteration=iteration,
                  wm={k:v.detach().cpu() for k,v in wm.state_dict().items()},
                  training_engine="omni_planned_tracker_stage1",gait_guard=a.gait_guard,
                  yaw_rate_weight=a.yaw_rate_weight,speed_weight=a.speed_weight,
                  slip_weight=a.slip_weight,smooth_weight=a.smooth_weight,landing_weight=a.landing_weight)
        ck["pi_live"] = ck["pi"]
        path = a.log_dir/f"{prefix}step_{iteration:04d}.pt"
        if joint_pose:
            planner_path = a.log_dir/f"{prefix}planner_{iteration:04d}.pt"
            pose_weights = {**bundle["pose"],
                            "weights":{k:v.detach().cpu() for k,v in planner.pose_model.state_dict().items()}}
            torch.save({**bundle,"pose":pose_weights},planner_path.with_suffix(".tmp"))
            planner_path.with_suffix(".tmp").replace(planner_path)
            ck["paired_planner"] = planner_path.name
            ck["training_engine"] = "omni_joint_pose" if a.joint_foot_pose else "omni_joint_heading"
        else:
            ck.pop("paired_planner",None)
        torch.save(ck,path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)

    for iteration in range(1,a.iterations+1):
        query_dt = query_dts[(iteration-1)%len(query_dts)]
        data, is_oracle, initial, switch, steps = collect(env,policy,planner,task,motion,query_dt,iteration,a.frames)
        live = data["live"]
        wm_candidates = live[2:].nonzero()
        frames = torch.arange(max(0,len(live)-steps+1),device=dev)
        usable = live[frames+steps-1] & ((frames+steps<=switch)|(frames>=switch))[:,None]
        actor_candidates = usable.nonzero()
        decel_indices = None
        if a.decel_mix:
            first_second = (frames>=switch) & (frames<switch+round(1./DT))
            speed_down = ~is_oracle & (((torch.arange(a.num_envs,device=dev)//4)%2)==1)
            selected = (usable & first_second[:,None] & speed_down[None]).nonzero()
            if len(selected):
                decel_indices = selected
        speed_groups = (torch.arange(a.num_envs,device=dev)//4)%2
        post_switch = frames>=switch
        possible_frames = a.frames-steps+1-switch
        fractions = []
        for g in (0,1):
            planned_envs = (~is_oracle) & (speed_groups==g)
            expected = possible_frames*int(planned_envs.sum())
            valid = int((usable & post_switch[:,None] & planned_envs[None]).sum())
            fractions.append(valid/max(1,expected))
        if min(fractions)<a.min_post_switch_fraction:
            stop = dict(iteration=iteration,query_dt=query_dt,post_switch_fractions=fractions,
                        reason="Insufficient post-switch deadlines; model rejected before update")
            (a.log_dir/"early_stop.json").write_text(json.dumps(stop,indent=2)+"\n")
            print(json.dumps(stop),flush=True)
            break
        if a.min_post_switch_fraction and (iteration==1 or (iteration-1)%a.save_every==0):
            save(iteration-1,prefix="checked_")
        if not len(wm_candidates) or not len(actor_candidates):
            raise RuntimeError("Not enough valid transitions for the requested deadline")
        wm.requires_grad_(True)
        wm_loss = fit_wm(wm,wmopt,mdp,data,wm_candidates,a.batch_size,a.updates)
        wm.requires_grad_(False)
        values = [fit_controller(wm,policy,teacher,planner,opt,mdp,task,data,actor_candidates,
                                 a.batch_size,steps,is_oracle,initial,gait_weights,a.yaw_rate_weight,pose_teacher,
                                 a.speed_weight,a.smooth_weight,a.joint_foot_pose,a.soft_touchdown_weight,
                                 decel_indices,a.decel_mix) for _ in range(a.updates)]
        result = dict(iteration=iteration,query_dt=query_dt,post_switch_fractions=fractions,wm_loss=wm_loss,controller_loss=float(np.mean([v[0] for v in values])),
                      valid_deadlines=len(actor_candidates),switch_frame=switch,
                      recorded_deadlines=int(is_oracle[actor_candidates[:,1]].sum()),
                      post_switch_planned_deadlines=int(((actor_candidates[:,0]>=switch) &
                                                       ~is_oracle[actor_candidates[:,1]]).sum()))
        if a.gait_guard:
            result.update(zip(("swing_clearance_loss","slip_loss","landing_speed_loss","root_up_speed_loss"),
                              np.mean([v[1] for v in values],axis=0).tolist()))
        if a.yaw_rate_weight:
            result["yaw_rate_loss"] = float(np.mean([v[2] for v in values]))
        if joint_pose:
            result["pose_heading_anchor_loss"] = float(np.mean([v[3] for v in values]))
        if a.joint_foot_pose:
            result["pose_position_anchor_loss"] = float(np.mean([v[6] for v in values]))
        if a.speed_weight:
            result["forward_speed_loss"] = float(np.mean([v[4] for v in values]))
        if a.smooth_weight:
            result["action_smoothness_loss"] = float(np.mean([v[5] for v in values]))
        if a.soft_touchdown_weight:
            result["soft_touchdown_loss"] = float(np.mean([v[7] for v in values]))
        if a.decel_mix:
            result["early_decel_deadlines"] = 0 if decel_indices is None else len(decel_indices)
        print(json.dumps(result),flush=True)
        with (a.log_dir/"train.jsonl").open("a") as f:
            f.write(json.dumps(result)+"\n")
        if iteration%a.save_every==0 or iteration==a.iterations:
            save(iteration)


if __name__=="__main__":
    main()
