"""Joint velocity planner/contact controller fine tuning, isolated from oracle tasks."""
from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F

from .data import ContactDataset
from .model import FootstepPlanner,planner_loss
from .leg_collision import LegCollisionCfg,LegCollisionTask,LegCollisionMDP,LegCollisionPolicy
from ..mdp import INV_DIM,XI_DIM,invariant_features,unflatten_state
from ..motion import CONTROL_DT
from ...rotations import quat_to_mat,yaw_mat


def decode_plan(pred, dt):
    """Binary forward values, straight-through contact gradient in training."""
    probability=pred['contact_logits'].sigmoid()
    contact=(probability>.5).float()+(probability-probability.detach())
    heading=pred['heading']
    fallback=torch.stack((torch.zeros_like(heading[...,0]),torch.ones_like(heading[...,0])),-1)
    heading=torch.where(heading.norm(dim=-1,keepdim=True)>1e-6,F.normalize(heading,dim=-1,eps=1e-6),fallback)
    return torch.cat((dt,contact,(pred['position']*contact[...,None]).flatten(1),
                      (heading*contact[...,None]).flatten(1)),-1)


class JointPolicy(LegCollisionPolicy):
    def __init__(self,hist=5,hidden=(512,256,128),planner_config=None):
        super().__init__(hist,hidden)
        config=planner_config or dict(hist=hist,hidden=[256,256,128])
        if config['hist']!=hist:raise ValueError('Planner and controller history lengths must match')
        self.planner=FootstepPlanner(**config)
        self.teacher=LegCollisionPolicy(hist,hidden).requires_grad_(False).eval()

    def init_controller(self,state):
        merged=self.state_dict()
        expected={k for k in merged if not k.startswith(('planner.','teacher.'))}
        if set(state)!=expected:raise ValueError('Unexpected controller policy structure')
        for k,value in state.items():merged[k]=value;merged['teacher.'+k]=value
        self.load_state_dict(merged)
        self.obs_norm.count.clamp_max_(1e6);self.goal_norm.count.clamp_max_(1e6)

    def contact_plan(self,history,command):
        # First 48 controller observation fields are gravity, omega, measured
        # joint q and qd, already in the offline planner's units/order.
        pred=self.planner(history[...,:48],command[:,:3],command[:,3:4])
        return decode_plan(pred,command[:,3:4])

    def controller_action(self,history,contact):
        return super().forward(history,contact)

    def forward(self,history,command):
        return self.controller_action(history,self.contact_plan(history,command))


@dataclass
class JointCfg(LegCollisionCfg):
    velocity_weight: float=4.
    yaw_velocity_weight: float=1.
    linear_velocity_std: float=.35
    yaw_velocity_std: float=.5
    planner_supervision_weight: float=1.
    controller_rehearsal_weight: float=.1


class JointTask(LegCollisionTask):
    def __init__(self,motion,kin,cfg=None):
        super().__init__(motion,kin,cfg or JointCfg())
        c=self.cfg
        if not all(math.isfinite(x) and x>0 for x in (c.linear_velocity_std,c.yaw_velocity_std)) or not all(math.isfinite(x) and x>=0 for x in (c.velocity_weight,c.yaw_velocity_weight,c.planner_supervision_weight,c.controller_rehearsal_weight)):
            raise ValueError('Invalid joint training reward/supervision configuration')
        self.num_queries=len(motion.horizons)
        velocity=motion.velocity_command
        change=(velocity[1:]-velocity[:-1]).abs().amax(-1)>1e-6
        change=change | (motion.clip_id[1:]!=motion.clip_id[:-1])
        stop=torch.cat((change,torch.ones(1,device=velocity.device,dtype=torch.bool)))
        indices=torch.arange(len(velocity),device=velocity.device)+1
        markers=torch.where(stop,indices,len(velocity))
        self.command_until=markers.flip(0).cummin(0).values.flip(0)

    def pack_queries(self,frames,queries):
        return frames*self.num_queries+queries.remainder(self.num_queries)

    def command(self,S,frames,query):
        anchor=torch.div(query,self.num_queries,rounding_mode='floor')
        horizon=query.remainder(self.num_queries)
        return torch.cat((self.motion.velocity_command[anchor],self.motion.horizons[horizon,None]),-1)

    def oracle_command(self,S,frames,query):
        return super().command(S,frames,query.remainder(self.num_queries))

    def eligible(self,frames,query,horizon):
        # Neither imagined gait labels nor a queried future oracle target may
        # cross a future command change unknown to the planner at this start.
        steps=self.motion.horizon_steps[query.remainder(self.num_queries)]
        return frames+horizon+steps<self.command_until[frames]

    def supervision_loss(self,policy,batch):
        live=batch['usable']
        S=tuple(torch.where(live.reshape((-1,)+(1,)*(x.ndim-1)),x.detach(),0.)
                for x in unflatten_state(batch['s']))
        history=torch.where(live[:,None,None],batch['hist'].detach(),0.)
        command=self.command(S,batch['f'],batch['goal_steps'])
        # The existing arm supervision trains the controller, not a hidden
        # velocity encoding in the planner's contact targets.
        plan=policy.contact_plan(history,command).detach()
        raw=policy.upper_raw(history,plan)
        xi=torch.where(live[:,None],batch['xi'],0.)
        target=self.upper_target(policy,batch['f'],xi)
        ids=self.upper_indices
        radius=torch.where(target>=0,policy.action_hi[ids],-policy.action_lo[ids])
        raw_target=(radius*torch.atanh(target/radius)).detach()
        error=(F.smooth_l1_loss(raw,raw_target,reduction='none')*self.upper_weights).mean(-1)
        return torch.where(live,error,0.).sum()/live.float().sum().clamp_min(1)

    def joint_supervision_loss(self,policy,batch):
        pred=policy.planner(batch['bc_history'],batch['bc_velocity'],batch['bc_dt'])
        labels=dict(target_contact=batch['bc_contact'],support_mask=batch['bc_contact'],
                    target_position=batch['bc_position'],target_heading=batch['bc_heading'])
        bc=planner_loss(pred,labels)['total']
        live=batch['usable']
        history=torch.where(live[:,None,None],batch['hist'].detach(),0.)
        S=tuple(torch.where(live.reshape((-1,)+(1,)*(x.ndim-1)),x.detach(),0.)
                for x in unflatten_state(batch['s']))
        oracle=self.oracle_command(S,batch['f'],batch['goal_steps']).detach()
        with torch.no_grad():target=policy.teacher(history,oracle)
        error=(policy.controller_action(history,oracle)-target).square().mean(-1)
        rehearsal=torch.where(live,error,0.).sum()/live.float().sum().clamp_min(1)
        return self.cfg.planner_supervision_weight*bc+self.cfg.controller_rehearsal_weight*rehearsal


class JointMDP(LegCollisionMDP):
    def __init__(self,kin,motion,default_q,task):
        super().__init__(kin,motion,default_q,task)
        self.critic_input_dim=INV_DIM+XI_DIM+self.scan_dim+4+2+18+27+6

    def reward_with_velocity(self,S,f,a,prev,velocity):
        c=self.task.cfg;p,R,v,w,q,qd=S
        body_v=(yaw_mat(R).transpose(-1,-2)@v[...,None]).squeeze(-1)
        speed=torch.exp(-(body_v[:,:2]-velocity[:,:2]).square().sum(-1)/c.linear_velocity_std**2)
        yaw=torch.exp(-(w[:,2]-velocity[:,2]).square()/c.yaw_velocity_std**2)
        upper=self.regularization_terms(S,f,a,prev)
        limits=(torch.relu(q-self.soft_hi)+torch.relu(self.soft_lo-q)).sum(-1)
        return CONTROL_DT*(c.velocity_weight*speed+c.yaw_velocity_weight*yaw
            +c.upright_weight*torch.exp(-((1-R[:,2,2])/c.upright_std).square())
            +c.height_weight*torch.exp(-((p[:,2]-c.height_target)/c.height_std).square())
            -c.upper_pose_weight*upper['upper_pose']-c.upper_velocity_weight*upper['upper_velocity']
            +c.action_rate_weight*upper['action_rate']+c.joint_speed_weight*upper['joint_speed']
            +c.joint_limit_weight*limits-sum(self.weighted_gait_terms(S,f).values())
            -c.self_collision_weight*self.collision_cost(S))

    def reward(self,S,f,a,prev):
        # Physical transition f is driven by the command at f-1.
        velocity=(self.velocity_override.expand(len(f),-1) if hasattr(self,'velocity_override')
                  else self.motion.velocity_command[(f-1).clamp_min(0)])
        return self.reward_with_velocity(S,f,a,prev,velocity)

    def reward_with_query(self,S,f,a,prev,query):
        return self.reward_with_velocity(S,f,a,prev,self.task.command(S,f,query)[:,:3])

    def critic_input(self,S,f,query,xi_n):
        upper=self.upper_errors(S,f);legs=self.gait_errors(S,f)
        foot=quat_to_mat(self.motion.feet[f,:,3:7])[...,2,:].flatten(1)
        return torch.cat((invariant_features(S),xi_n,self.scan(S,self.motion.boxes(f,torch.zeros_like(f))),
                          self.task.command(S,f,query),self.motion.contact[f].float(),*upper,*legs,foot),-1)

    def training_diagnostics(self,S,f):
        result={key+'_cost':value.mean() for key,value in self.weighted_gait_terms(S,f).items()}
        gap=self.task.collision.clearance(S).flatten(1).min(-1).values
        result.update(self_collision_cost=(self.task.cfg.self_collision_weight*self.collision_cost(S)).mean(),
                      min_leg_clearance_m=gap.mean(),leg_proxy_overlap_fraction=(gap<0).float().mean())
        R=S[1];v=(yaw_mat(R).transpose(-1,-2)@S[2][...,None]).squeeze(-1);w=S[3]
        cmd=(self.velocity_override.expand(len(f),-1) if hasattr(self,'velocity_override') else self.motion.velocity_command[f]);c=self.task.cfg
        error=(v[:,:2]-cmd[:,:2]).square().sum(-1)
        yaw=(w[:,2]-cmd[:,2]).square()
        result.update(velocity_error_mps=error.sqrt().mean(),yaw_rate_error_rps=yaw.sqrt().mean(),
                      velocity_reward=(c.velocity_weight*torch.exp(-error/c.linear_velocity_std**2)).mean(),
                      yaw_velocity_reward=(c.yaw_velocity_weight*torch.exp(-yaw/c.yaw_velocity_std**2)).mean())
        return result


class PlannerSamples:
    """Balanced valid offline labels, gathered on GPU without Python sample IO."""
    def __init__(self,motion,task,hist):
        dataset=ContactDataset(motion.sources,hist)
        groups={};offset=0
        offsets={}
        for part in motion.sources:
            offsets[(part['cache'],part['start'],part['end'])]=offset
            offset+=part['end']-part['start']
        for part,pairs in dataset.entries:
            base=offsets[(part['cache'],part['start'],part['end'])]-part['start']
            for h in np.unique(pairs[:,1]):
                key=(part['category'],int(h))
                group=pairs[pairs[:,1]==h].astype(np.int64).copy();group[:,0]+=base
                groups.setdefault(key,[]).append(group)
        buckets=[np.concatenate(v) for v in groups.values()]
        lengths=np.array([len(v) for v in buckets]);starts=np.r_[0,lengths.cumsum()[:-1]]
        self.pairs=torch.as_tensor(np.concatenate(buckets),device=motion.device)
        self.starts=torch.as_tensor(starts,device=motion.device);self.lengths=torch.as_tensor(lengths,device=motion.device)
        self.motion,self.task,self.hist=motion,task,hist
        self.back=torch.arange(hist-1,-1,-1,device=motion.device)

    @torch.no_grad()
    def sample(self,n):
        bucket=torch.randint(len(self.lengths),(n,),device=self.motion.device)
        index=self.starts[bucket]+(torch.rand(n,device=self.motion.device)*self.lengths[bucket]).long()
        frames,query=self.pairs[index].unbind(-1)
        history=frames[:,None]-self.back
        m=self.motion
        omega=(m.root_R[history].transpose(-1,-2)@m.root_w[history][...,None]).squeeze(-1)
        state=torch.cat((-m.root_R[history][...,2,:],omega,m.q[history],m.qd[history]),-1)
        contact,pos,heading=self.task.target_components(m.ref_state(frames),frames+m.horizon_steps[query])
        return dict(bc_history=state,bc_velocity=m.velocity_command[frames],bc_dt=m.horizons[query,None],
                    bc_contact=contact,bc_position=pos,bc_heading=heading)
