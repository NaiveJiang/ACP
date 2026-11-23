"""Joint planning with physical future realization and non-sliding gait constraints."""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .joint import JointPolicy,JointTask,JointMDP,JointCfg,decode_plan
from .model import planner_loss
from ..task import BodySpheres,to_base
from ..mdp import unflatten_state
from ..motion import CONTROL_DT
from ...rotations import yaw_mat


class FootGroundGeometry(nn.Module):
    """Foot contact-point kinematics and flat-ground height from proprioception.

    Height inference assumes walking with at least one foot on the same plane.
    It uses measured q and gravity only, never privileged root height.
    """
    def __init__(self,kin=None):
        super().__init__()
        if kin is None:
            from .fast.models import FastKinematics
            kin=FastKinematics('cpu')
        self.kin=kin
        self.kin_fields=('axes','anchors','refs','off_p','off_R')
        for key in self.kin_fields:self.register_buffer(key,getattr(kin,key))
        dev=self.axes.device
        self.register_buffer('foot_ids',torch.tensor([kin.body_names.index(n) for n in ('leg_l6_link','leg_r6_link')],device=dev))
        spheres=BodySpheres();count=spheres.bounds[1][1]
        self.bounds=spheres.bounds[:2]
        self.register_buffer('points',spheres.pts[:count].to(dev))
        self.register_buffer('radii',spheres.rad[:count].to(dev))
        self.register_buffer('owners',spheres.owner[:count].to(dev))
        self.register_buffer('sole',torch.tensor([.0486,0.,-.04],device=dev))
        clearance=float(self.sole[2]-(self.points[:,2]-self.radii).min())
        self.register_buffer('flat_sole_clearance',torch.tensor(clearance,device=dev))

    def _apply(self,fn,recurse=True):
        result=super()._apply(fn,recurse=recurse)
        for key in self.kin_fields:setattr(self.kin,key,getattr(self,key))
        self.kin.device=str(self.axes.device)
        return result

    def relative_ground(self,observation):
        q=observation[:,6:27]
        p=torch.zeros_like(q[:,:3]);R=torch.eye(3,device=q.device).expand(len(q),-1,-1)
        if hasattr(self.kin,'pose'):pos,rot=self.kin.pose(p,R,q)
        else:pos,rot=self.kin.forward(p,R,p,p,q,torch.zeros_like(q),all_bodies=True)[:2]
        pos,rot=pos[:,self.foot_ids],rot[:,self.foot_ids]
        centers=pos[:,self.owners]+(rot[:,self.owners]@self.points[None,:,:,None]).squeeze(-1)
        up=F.normalize(-observation[:,:3],dim=-1,eps=1e-6)
        lowest=(centers*up[:,None]).sum(-1)-self.radii
        return lowest.min(-1).values+self.flat_sole_clearance

    def foot_state(self,S):
        p,R,v,w=self.kin.forward(*S,all_bodies=True)
        p,R,v,w=(x[:,self.foot_ids] for x in (p,R,v,w))
        centers=p[:,self.owners]+(R[:,self.owners]@self.points[None,:,:,None]).squeeze(-1)
        distances=centers[...,2]-self.radii
        floor_offsets=torch.stack((torch.zeros_like(self.radii),torch.zeros_like(self.radii),self.radii),-1)
        points=centers-floor_offsets
        batch=torch.arange(len(p),device=p.device)
        gaps=[];velocities=[]
        for foot,(start,end) in enumerate(self.bounds):
            index=distances[:,start:end].argmin(-1)+start
            contact=points[batch,index]
            gaps.append(distances[batch,index])
            velocities.append(v[:,foot]+torch.cross(w[:,foot],contact-p[:,foot],dim=-1))
        sole=p+(R@self.sole[None,None,:,None]).squeeze(-1)
        return sole,R,torch.stack(gaps,-1),torch.stack(velocities,1)


class ConsistentJointPolicy(JointPolicy):
    def __init__(self,hist=5,hidden=(512,256,128),planner_config=None):
        super().__init__(hist,hidden,planner_config)
        self.ground=FootGroundGeometry()

    def init_controller(self,state):
        merged=self.state_dict()
        expected={k for k in merged if not k.startswith(('planner.','teacher.','ground.'))}
        if set(state)!=expected:raise ValueError('Expected a clean v6 controller')
        for key,value in state.items():merged[key]=value;merged['teacher.'+key]=value
        self.load_state_dict(merged)
        self.obs_norm.count.clamp_max_(1e6);self.goal_norm.count.clamp_max_(1e6)

    def planning_prediction(self,history,command):
        pred=self.planner(history[...,:48],command[:,:3],command[:,3:4])
        z=self.ground.relative_ground(history[:,-1])[:,None,None].expand(-1,2,1)
        return dict(contact_logits=pred['contact_logits'],position=torch.cat((pred['position'][...,:2],z),-1),heading=pred['heading'])

    def planning_outputs(self,history,command):
        pred=self.planning_prediction(history,command)
        return torch.cat((decode_plan(pred,command[:,3:4]),pred['contact_logits'].sigmoid()),-1)

    def contact_plan(self,history,command):
        return self.planning_outputs(history,command)[:,:13]


@dataclass
class ConsistentJointCfg(JointCfg):
    controller_rehearsal_weight: float=1.
    planned_rehearsal_weight: float=.5
    hindsight_weight: float=1.
    realization_position_weight: float=1.
    realization_contact_weight: float=.5
    realization_heading_weight: float=.25
    plan_ground_weight: float=.5
    realization_position_std: float=.05
    plan_ground_std: float=.03
    phase_contact_weight: float=1.
    swing_clearance_weight: float=.5
    swing_clearance_std: float=.03
    slip_weight: float=.5
    slip_std: float=.15


class ConsistentJointTask(JointTask):
    def __init__(self,motion,kin,cfg=None):
        super().__init__(motion,kin,cfg or ConsistentJointCfg())
        self.ground=FootGroundGeometry(kin)
        c=self.cfg
        for key in ('realization_position_std','plan_ground_std','swing_clearance_std','slip_std'):
            if not math.isfinite(getattr(c,key)) or getattr(c,key)<=0:raise ValueError('Invalid '+key)
        for key in ('planned_rehearsal_weight','hindsight_weight','realization_position_weight','realization_contact_weight',
                    'realization_heading_weight','plan_ground_weight','phase_contact_weight','swing_clearance_weight','slip_weight'):
            if not math.isfinite(getattr(c,key)) or getattr(c,key)<0:raise ValueError('Invalid '+key)

    def hindsight_loss(self,policy,batch):
        live=batch['ft_valid'];weight=live.float()
        history=torch.where(live[:,None,None],batch['ft_history'],0.)
        pred=policy.planning_prediction(history,batch['ft_command'])
        contact=F.binary_cross_entropy_with_logits(pred['contact_logits'],batch['ft_contact'],reduction='none').mean(-1)
        mask=batch['ft_support']*weight[:,None]
        pos=F.smooth_l1_loss(pred['position']/self.cfg.realization_position_std,
                           batch['ft_position']/self.cfg.realization_position_std,reduction='none').sum(-1)
        heading=1-(pred['heading']*batch['ft_heading']).sum(-1)
        return ((contact*weight).sum()/weight.sum().clamp_min(1)
                +((pos+heading)*mask).sum()/mask.sum().clamp_min(1))

    def joint_supervision_loss(self,policy,batch):
        command=torch.cat((batch['bc_velocity'],batch['bc_dt']),-1)
        pred=policy.planning_prediction(batch['bc_history'],command)
        labels=dict(target_contact=batch['bc_contact'],support_mask=batch['bc_contact'],
                    target_position=batch['bc_position'],target_heading=batch['bc_heading'])
        bc=planner_loss(pred,labels)['total']
        live=batch['usable'];history=torch.where(live[:,None,None],batch['hist'].detach(),0.)
        S=tuple(torch.where(live.reshape((-1,)+(1,)*(x.ndim-1)),x.detach(),0.) for x in unflatten_state(batch['s']))
        oracle=self.oracle_command(S,batch['f'],batch['goal_steps']).detach()
        plan=policy.contact_plan(history,self.command(S,batch['f'],batch['goal_steps'])).detach()
        with torch.no_grad():
            teacher_oracle=policy.teacher(history,oracle);teacher_plan=policy.teacher(history,plan)
        mse=lambda x,y:torch.where(live,(x-y).square().mean(-1),0.).sum()/live.float().sum().clamp_min(1)
        c=self.cfg
        return (c.planner_supervision_weight*bc+c.controller_rehearsal_weight*mse(policy.controller_action(history,oracle),teacher_oracle)
                +c.planned_rehearsal_weight*mse(policy.controller_action(history,plan),teacher_plan)
                +c.hindsight_weight*self.hindsight_loss(policy,batch))

    def trajectory_supervision_loss(self,states,query,plans,usable,survival,alive):
        """Check goals at their actual query deadline, in the origin world frame."""
        H,B=plans.shape[:2];c=self.cfg
        flat=tuple(x.flatten(0,1) for x in states)
        feet,R,gap,_=self.ground.foot_state(flat)
        feet=feet.reshape(H+1,B,2,3);R=R.reshape(H+1,B,2,3,3);gap=gap.reshape(H+1,B,2)
        steps=self.motion.horizon_steps[query.remainder(self.num_queries)]
        time=torch.arange(H,device=query.device)[:,None]
        target=time+steps[None,:];index=target.clamp_max(H);batch=torch.arange(B,device=query.device)[None,:]
        valid=(target<=H).float()*usable[None].float()*survival.detach()
        live_states=torch.cat((survival,alive[None]),0)
        valid=valid*live_states[index,batch].detach()
        p0,R0=states[0][:-1],states[1][:-1]
        world=p0[:,:,None]+(yaw_mat(R0)[:,:,None]@plans[...,3:9].reshape(H,B,2,3,1)).squeeze(-1)
        support=plans[...,1:3].detach()
        mask=valid[...,None]*support
        error=F.smooth_l1_loss(feet[index,batch]/c.realization_position_std,world/c.realization_position_std,reduction='none').sum(-1)
        position=(error*mask).sum()/mask.sum().clamp_min(1)
        actual=torch.sigmoid((c.contact_distance-gap[index,batch])/c.contact_temperature)
        contact=((plans[...,13:15]-actual).square().mean(-1)*valid).sum()/valid.sum().clamp_min(1)
        future_R=R[index,batch]
        base=yaw_mat(R0)
        sine=base[...,1,0,None]*plans[...,10:13:2]+base[...,0,0,None]*plans[...,9:13:2]
        cosine=base[...,0,0,None]*plans[...,10:13:2]-base[...,1,0,None]*plans[...,9:13:2]
        norm=(future_R[...,1,0].square()+future_R[...,0,0].square()+1e-8).sqrt()
        dot=(sine*future_R[...,1,0]+cosine*future_R[...,0,0])/norm
        heading=((1-dot)*mask).sum()/mask.sum().clamp_min(1)
        ground_mask=survival.detach()[...,None]*usable[None,:,None].float()*support
        ground=((world[...,2]-self.ground.flat_sole_clearance).square()/c.plan_ground_std**2*ground_mask).sum()/ground_mask.sum().clamp_min(1)
        return c.realization_position_weight*position+c.realization_contact_weight*contact+c.realization_heading_weight*heading+c.plan_ground_weight*ground


class ConsistentJointMDP(JointMDP):
    def physical_gait_terms(self,S,f):
        c=self.task.cfg
        _,_,gap,velocity=self.task.ground.foot_state(S)
        with torch.no_grad():_,_,ref_gap,_=self.task.ground.foot_state(self.motion.ref_state(f))
        contact=self.motion.contact[f].float()
        probability=torch.sigmoid((c.contact_distance-gap)/c.contact_temperature)
        phase=(probability-contact).square().mean(-1)
        swing=1-contact
        clearance=(torch.relu(ref_gap.clamp_min(0)-gap)/c.swing_clearance_std).square()
        swing_cost=(clearance*swing).sum(-1)/swing.sum(-1).clamp_min(1)
        slip=(velocity[...,:2].square().sum(-1)/c.slip_std**2*probability.detach()).mean(-1)
        return dict(phase_contact_cost=c.phase_contact_weight*phase,
                    swing_clearance_cost=c.swing_clearance_weight*swing_cost,slip_cost=c.slip_weight*slip)

    def reward_with_velocity(self,S,f,a,prev,velocity):
        return super().reward_with_velocity(S,f,a,prev,velocity)-CONTROL_DT*sum(self.physical_gait_terms(S,f).values())

    def training_diagnostics(self,S,f):
        result=super().training_diagnostics(S,f)
        result.update({key:value.mean() for key,value in self.physical_gait_terms(S,f).items()})
        return result


class HindsightSamples:
    def __init__(self,task,hist):
        self.task,self.hist=task,hist
        self.window=torch.arange(int(task.motion.horizon_steps.max()),device=task.motion.device)
        self.back=torch.arange(hist-1,-1,-1,device=task.motion.device)

    @torch.no_grad()
    def sample(self,buffer,n):
        dev=buffer.device
        age=torch.randint(buffer.size,(n,),device=dev);env=torch.randint(buffer.N,(n,),device=dev)
        row=buffer._row(age);query=buffer.data['goal_steps'][row,env]
        steps=self.task.motion.horizon_steps[query.remainder(self.task.num_queries)]
        enough=age+steps<=buffer.size
        rows=buffer._row(age[:,None]+self.window[None]);active=self.window[None]<steps[:,None]
        valid=((buffer.data['valid'][rows,env[:,None]] & ~buffer.data['done'][rows,env[:,None]]) | ~active).all(-1)&enough
        frames=buffer.data['f'][row,env]
        valid=valid & (frames+steps<self.task.command_until[frames])
        ep=buffer.data['ep_step'][row,env].long()
        back=torch.minimum(self.back[None],torch.minimum(ep,age)[:,None])
        history=buffer.data['o'][buffer._row(age[:,None]-back),env[:,None]]
        S=unflatten_state(buffer.data['s'][row,env])
        target=unflatten_state(buffer.data['s1'][buffer._row(age+steps-1),env])
        sole,R,gap,_=self.task.ground.foot_state(target)
        position=to_base(sole,S[0],S[1])
        angle=torch.atan2(R[...,1,0],R[...,0,0])-torch.atan2(S[1][:,1,0],S[1][:,0,0])[:,None]
        heading=torch.stack((torch.sin(angle),torch.cos(angle)),-1)
        contact=torch.sigmoid((self.task.cfg.contact_distance-gap)/self.task.cfg.contact_temperature)
        safe=lambda x:torch.where(valid.reshape((-1,)+(1,)*(x.ndim-1)),x,0.)
        return dict(ft_history=safe(history),ft_command=self.task.command(S,frames,query),ft_valid=valid,
                    ft_position=safe(position),ft_contact=safe(contact),ft_support=safe((gap<self.task.cfg.contact_distance).float()),
                    ft_heading=safe(heading))
