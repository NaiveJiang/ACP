"""Cross-leg physical collisions and differentiable capsule proximity costs."""
from dataclasses import dataclass, replace
import math

import mujoco
import numpy as np
import torch

from .gait_style import GaitStyleCfg, GaitStyleTask, GaitStyleMDP, GaitStylePolicy
from .env import FootstepTrackingSim
from ..env import make_scene_cfg
from ..motion import CONTROL_DT
from ..robot import get_spec


def collision_spec():
    """Keep terrain interactions and enable only opposite-leg self contacts."""
    spec = get_spec()
    for body in spec.bodies:
        if body.name.startswith('leg_l'):
            masks = (2, 1|4)
        elif body.name.startswith('leg_r'):
            masks = (4, 1|2)
        else:
            continue
        for geom in body.geoms:
            if geom.contype or geom.conaffinity:
                geom.contype, geom.conaffinity = masks
    return spec


class LegCollisionSim(FootstepTrackingSim):
    def __init__(self, num_envs, motion, mdp, device, **kwargs):
        if 'scene_cfg' in kwargs:
            raise ValueError('Leg collision environment owns its collision scene configuration')
        scene = make_scene_cfg(num_envs, motion.clip_boxes.shape[2])
        scene.entities['robot'] = replace(scene.entities['robot'], spec_fn=collision_spec)
        super().__init__(num_envs,motion,mdp,device,scene_cfg=scene,**kwargs)


def segment_distance(a, b, c, d):
    """Exact segment distance with endpoint/parallel/point degeneracy handling."""
    u, v, w = b-a, d-c, a-c
    dot = lambda x,y:(x*y).sum(-1)
    aa,bb,cc,dd,ee = dot(u,u),dot(u,v),dot(v,v),dot(u,w),dot(v,w)
    # Four endpoint projections, plus the interior/interior stationary point.
    project = lambda point,start,delta,length: ((point-start)*delta).sum(-1)/length.clamp_min(1e-12)
    ds = []
    for point in (a,b):
        t = project(point,c,v,cc).clamp(0,1)
        ds.append((point-c-t[...,None]*v).square().sum(-1))
    for point in (c,d):
        s = project(point,a,u,aa).clamp(0,1)
        ds.append((point-a-s[...,None]*u).square().sum(-1))
    den = aa*cc-bb.square()
    nonparallel = den > 1e-6*(aa*cc).clamp_min(1e-24)
    safe_den = torch.where(nonparallel,den,torch.ones_like(den))
    s,t = (bb*ee-cc*dd)/safe_den,(aa*ee-bb*dd)/safe_den
    inside = nonparallel & (s>=0) & (s<=1) & (t>=0) & (t<=1)
    interior = (w+s[...,None]*u-t[...,None]*v).square().sum(-1)
    ds.append(torch.where(inside,interior,float('inf')))
    return torch.stack(ds,-1).min(-1).values.clamp_min(1e-12).sqrt()


class LegCollisionGeometry:
    """Exact capsule proxies; conservative covering spheres for thin foot solids.

    Proxies affect only the differentiable reward. Physics retains the original
    capsules, cylinders and boxes, with opposite-leg collision masks enabled.
    """
    def __init__(self, kin, device):
        model = get_spec().compile()
        proxies = {'l':[], 'r':[]}
        for g in range(model.ngeom):
            name = model.body(int(model.geom_bodyid[g])).name
            side = 'l' if name.startswith('leg_l') else 'r' if name.startswith('leg_r') else None
            if side is None or not (model.geom_contype[g] or model.geom_conaffinity[g]):continue
            R = np.zeros(9);mujoco.mju_quat2Mat(R,model.geom_quat[g]);R=R.reshape(3,3)
            center,size,kind = model.geom_pos[g],model.geom_size[g],model.geom_type[g]
            owner = kin.body_names.index(name)
            if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                extent=R[:,2]*size[1]
                proxies[side].append((owner,center-extent,center+extent,float(size[0])))
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                proxies[side].append((owner,center,center,float(size[0])))
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                proxies[side].append((owner,center,center,float(np.linalg.norm(size[:2]))))
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                axis=int(np.argmax(size));half=size[axis]/3
                radius=float(np.sqrt(np.sum(np.delete(size,axis)**2)+half**2))
                for x in np.linspace(-size[axis]+half,size[axis]-half,3):
                    p=center+R[:,axis]*x
                    proxies[side].append((owner,p,p,radius))
            else:raise ValueError('Unsupported leg collision primitive '+str(kind))
        self.kin=kin
        for side,items in proxies.items():
            if not items:raise ValueError('Missing '+side+' leg collision geometry')
            setattr(self,side+'_owners',torch.tensor([x[0] for x in items],device=device))
            for index,key in ((1,'a'),(2,'b'),(3,'radius')):
                setattr(self,side+'_'+key,torch.tensor(np.array([x[index] for x in items]),dtype=torch.float32,device=device))

    def clearance(self, S):
        if hasattr(self.kin,'pose'):p,R=self.kin.pose(S[0],S[1],S[4])
        else:p,R=self.kin.forward(*S,all_bodies=True)[:2]
        ends=[]
        for side in ('l','r'):
            ids=getattr(self,side+'_owners');pos,rot=p[:,ids],R[:,ids]
            ends.append(tuple(pos+(rot@getattr(self,side+'_'+key)[None,:,:,None]).squeeze(-1) for key in ('a','b')))
        la,lb=ends[0];ra,rb=ends[1]
        distance=segment_distance(la[:,:,None],lb[:,:,None],ra[:,None],rb[:,None])
        return distance-self.l_radius[None,:,None]-self.r_radius[None,None,:]


@dataclass
class LegCollisionCfg(GaitStyleCfg):
    self_collision_weight: float = 2.
    self_collision_margin: float = .01
    self_collision_std: float = .03


class LegCollisionPolicy(GaitStylePolicy):
    pass


class LegCollisionTask(GaitStyleTask):
    def __init__(self,motion,kin,cfg=None):
        super().__init__(motion,kin,cfg or LegCollisionCfg())
        c=self.cfg
        if not all(math.isfinite(x) for x in (c.self_collision_weight,c.self_collision_margin,c.self_collision_std)) or c.self_collision_weight<0 or c.self_collision_margin<0 or c.self_collision_std<=0:
            raise ValueError('Invalid leg collision configuration')
        self.collision=LegCollisionGeometry(kin,motion.device)


class LegCollisionMDP(GaitStyleMDP):
    def collision_cost(self,S):
        c=self.task.cfg
        gap=self.task.collision.clearance(S)
        return (torch.relu(c.self_collision_margin-gap)/c.self_collision_std).square().sum((-1,-2))

    def reward(self,S,f,a,a_prev):
        return super().reward(S,f,a,a_prev)-CONTROL_DT*self.task.cfg.self_collision_weight*self.collision_cost(S)

    def training_diagnostics(self,S,f):
        result=super().training_diagnostics(S,f)
        gap=self.task.collision.clearance(S)
        result['self_collision_cost']=(self.task.cfg.self_collision_weight*self.collision_cost(S)).mean()
        result['min_leg_clearance_m']=gap.flatten(1).min(-1).values.mean()
        result['leg_proxy_overlap_fraction']=(gap.flatten(1).min(-1).values<0).float().mean()
        return result
