"""Collect solver-labelled touchdown events in separate MuJoCo worlds."""
import mujoco_warp
import torch
import warp as wp

from .event_planner import hindsight
from ..mdp import unflatten_state
from ...rotations import yaw_mat


class FootContactForces:
    def __init__(self, env):
        self.sim = env.sim
        data = self.sim.wp_data
        capacity = data.contact.geom.shape[0]
        self.contact_ids = torch.arange(capacity,device=env.device,dtype=torch.int32)
        self.ids = wp.from_torch(self.contact_ids)
        self.force = wp.zeros(capacity,dtype=wp.spatial_vector,device=env.device)
        self.overflow = torch.zeros((),device=env.device,dtype=torch.bool)
        model = self.sim.mj_model
        self.foot = torch.full((model.ngeom,),-1,device=env.device,dtype=torch.long)
        for i, name in enumerate(("leg_l6_link","leg_r6_link")):
            body = model.body(f"robot/{name}").id
            for geom in range(int(model.body_geomadr[body]),int(model.body_geomadr[body]+model.body_geomnum[body])):
                self.foot[geom] = i
        self.floor = torch.zeros(model.ngeom,device=env.device,dtype=torch.bool)
        self.floor[env._terrain_geoms] = True
        self.count = env.N

    def contact(self):
        return self.normal_force() >= 5.

    def normal_force(self):
        data = self.sim.wp_data
        count = wp.to_torch(data.nacon)
        self.overflow.logical_or_((count>self.contact_ids.numel()).any())
        mujoco_warp.contact_force(self.sim.wp_model,data,self.ids,False,self.force)
        geom = wp.to_torch(data.contact.geom).long().clamp(0,self.foot.numel()-1)
        a,b = geom.unbind(-1)
        foot = torch.where(self.floor[b],self.foot[a],torch.where(self.floor[a],self.foot[b],-1))
        world = wp.to_torch(data.contact.worldid).long().clamp(0,self.count-1)
        normal = wp.to_torch(self.force)[:,0].clamp_min(0.)
        valid = (self.contact_ids < count) & (foot >= 0)
        summed = normal.new_zeros(2*self.count)
        summed.scatter_add_(0,2*world+foot.clamp_min(0),torch.where(valid,normal,0.))
        return summed.view(self.count,2)


class EventRollout:
    def __init__(self, env, task, *, next_contact=False):
        self.env, self.task = env,task
        self.sensor = FootContactForces(env)
        self.cursor = 0
        self.next_contact = next_contact
        # Round-robin across motion classes, with uniform starts inside a clip.
        groups = {}
        for i, part in enumerate(env.motion.sources):
            groups.setdefault(part["category"],[]).append(i)
        self.categories = [torch.tensor(v,device=env.device) for v in groups.values()]

    @torch.no_grad()
    def collect(self, policy, planner, steps, planner_fraction):
        env,task = self.env,self.task
        motion = env.motion
        ids = torch.arange(env.N,device=env.device)
        selected = []
        for i in range(env.N):
            group = self.categories[(self.cursor+i) % len(self.categories)]
            selected.append(group[torch.randint(len(group),(1,),device=env.device)])
        self.cursor += env.N
        clips = torch.cat(selected)
        lo, hi = motion.clip_start[clips],motion.clip_end[clips]
        span = (hi-lo-steps).clamp_min(0)
        start = lo + (torch.rand(env.N,device=env.device)*span).long()
        start = torch.where(ids % 4 == 0,lo,start)
        env.done_eval.zero_()
        env.failed_eval.zero_()
        self.sensor.overflow.zero_()
        env.reset(ids,frames=start)
        env.refresh()
        if self.next_contact:
            query = task.sample_queries(env.N)
        else:
            horizon = torch.tensor([i for i,h in enumerate(motion.horizons.tolist())
                                    if round(h,2) in (.3,.5)],device=env.device)
            query = horizon[ids % len(horizon)]
        live = (hi-start >= steps)
        use_planner = torch.rand(env.N,device=env.device) < planner_fraction
        history = None
        rows = []
        contact = self.sensor.contact()
        elapsed_error = torch.zeros(env.N,device=env.device)
        expected = torch.zeros((env.N,2),device=env.device)
        progress = torch.zeros_like(expected)
        for _ in range(steps):
            state = env.state()
            obs = env.observe()
            history = (obs[:,None].expand(-1,policy.hist,-1).clone() if history is None
                       else torch.cat((history[:,1:],obs[:,None]),dim=1))
            frame = env.f
            velocity = (motion.requested_velocity if self.next_contact else motion.velocity_command)[frame]
            feet,R,_ = task.foot_geometry(state)
            angle = torch.atan2(R[...,1,0],R[...,0,0])
            rows.append((feet,angle,state[0],state[1],contact,velocity,history[...,:48]))
            cmd = task.command(state,frame,query)
            if planner_fraction:
                plan = (planner.plan(history,velocity) if self.next_contact else
                        planner.plan(history,velocity,cmd[:,:1],state,task))
                cmd = torch.where(use_planner[:,None],plan,cmd)
            transition = env.step(policy(history,cmd))
            contact = self.sensor.contact()
            live &= ~transition["term"] & transition["valid"]
            S1 = unflatten_state(transition["s1"])
            ref_f = torch.minimum(frame + 1, hi)
            if self.next_contact:
                # Require progress along the reference motion so a stationary policy
                # cannot pass a short rolling-window rollout merely by staying upright.
                delta = motion.root_p[ref_f, :2] - motion.root_p[frame, :2]
                dist = delta.norm(dim=-1)
                advance = ((S1[0][:, :2] - state[0][:, :2])*delta).sum(-1)/dist.clamp_min(1e-6)
                def yaw_delta(before, after):
                    relative = yaw_mat(before).transpose(-1, -2) @ yaw_mat(after)
                    return torch.atan2(relative[:,1,0],relative[:,0,0])
                angle = yaw_delta(motion.root_R[frame],motion.root_R[ref_f])
                turn = yaw_delta(state[1],S1[1])*angle.sign()
                expected += torch.stack((dist,angle.abs()),-1)
                progress += torch.stack((advance,turn),-1)
            else:
                ref_f = env.f
            planar = S1[0][:,:2]-motion.root_p[ref_f,:2]
            h = yaw_mat(S1[1])
            r = yaw_mat(motion.root_R[ref_f])
            cos = (h[:,0,:2]*r[:,0,:2]).sum(-1)
            elapsed_error += (planar.norm(dim=-1)>.55) | (cos<.65)
            # Completed worlds share the simulator but cannot contribute labels.
            env.f = torch.minimum(env.f,hi-1)
        live &= elapsed_error < .20*steps
        if self.next_contact:
            moving = expected >= expected.new_tensor([.10,.20])
            live &= (~moving | (progress >= .45*expected)).all(-1)
            moving &= (hi-start >= steps)[:,None]
            planner_attempts = moving.any(-1) & use_planner
            self.planned_moving_attempts = int(planner_attempts.sum())
            self.planned_moving_successes = int((planner_attempts & live).sum())
        if bool(self.sensor.overflow):
            raise RuntimeError("MuJoCo contact buffer overflowed while collecting event labels")
        array = [torch.stack(column) for column in zip(*rows)]
        if self.next_contact:
            from .next_contact import hindsight as contact_hindsight
            return contact_hindsight(*array,live),int(live.sum())
        return hindsight(*array,live),int(live.sum())
