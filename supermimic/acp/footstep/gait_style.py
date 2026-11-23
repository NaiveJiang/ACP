"""Phase-aligned motion rewards for contact control."""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .upper_supervised import UpperSupervisedCfg, UpperSupervisedTask, UpperSupervisedMDP, UpperSupervisedPolicy
from ..motion import CONTROL_DT
from ...rotations import quat_to_mat, yaw_mat


@dataclass
class GaitStyleCfg(UpperSupervisedCfg):
    leg_pose_weight: float = 2.
    leg_velocity_weight: float = .2
    pelvis_tilt_weight: float = .5
    stance_tilt_weight: float = .5
    knee_weight: float = .25
    motion_style_weight: float = 0.
    leg_pose_std: tuple = (.35, .25, .25, .45, .30, .20)
    leg_velocity_std: tuple = (3., 2., 2., 4., 3., 2.)
    tilt_std: float = .2
    knee_min: float = 0.
    knee_std: float = .1


class GaitStylePolicy(UpperSupervisedPolicy):
    pass


class GaitStyleTask(UpperSupervisedTask):
    def __init__(self, motion, kin, cfg=None):
        super().__init__(motion, kin, cfg or GaitStyleCfg())
        c = self.cfg
        scales = (*c.leg_pose_std, *c.leg_velocity_std, c.tilt_std, c.knee_std)
        weights = (c.leg_pose_weight,c.leg_velocity_weight,c.pelvis_tilt_weight,c.stance_tilt_weight,
                   c.knee_weight,c.motion_style_weight)
        if len(c.leg_pose_std)!=6 or len(c.leg_velocity_std)!=6 or not all(math.isfinite(x) and x>0 for x in scales):
            raise ValueError('Gait standard deviations must be finite positive, with six entries per leg')
        if not all(math.isfinite(x) and x>=0 for x in weights) or not math.isfinite(c.knee_min):
            raise ValueError('Invalid gait weights or knee floor')


class GaitStyleMDP(UpperSupervisedMDP):
    def __init__(self, kin, motion, default_q, task):
        super().__init__(kin, motion, default_q, task)
        dev = default_q.device
        self.leg_indices = torch.arange(1,13,device=dev)
        self.knee_indices = torch.tensor([4,10],device=dev)
        self.leg_pose_scale = torch.tensor(task.cfg.leg_pose_std,device=dev).repeat(2)
        self.leg_velocity_scale = torch.tensor(task.cfg.leg_velocity_std,device=dev).repeat(2)
        if task.cfg.motion_style_weight:
            self.swing_motion_scale = default_q.new_tensor([.10,.08,.055])
            self.shift_velocity_scale = default_q.new_tensor([.32,.32,.20])
        # Privileged style errors/targets are for the critic only.
        self.critic_input_dim += 24+3+6

    def gait_errors(self, S, f):
        ids = self.leg_indices
        # Gravity in the body frame is invariant to world translation and yaw.
        return (S[4][:,ids]-self.motion.q[f][:,ids],
                S[5][:,ids]-self.motion.qd[f][:,ids],
                S[1][:,2,:]-self.motion.root_R[f][:,2,:])

    def gait_terms(self, S, f):
        c = self.task.cfg
        pose, velocity, tilt = self.gait_errors(S,f)
        huber = lambda x: F.smooth_l1_loss(x,torch.zeros_like(x),reduction='none')
        # Huber keeps a restoring gradient at large errors without a rigid
        # full-body trajectory constraint or a vanishing exponential bonus.
        feet, foot_R, gap = self.task.foot_geometry(S)
        ref_R = quat_to_mat(self.motion.feet[f,:,3:7])
        support = self.motion.contact[f].float()
        foot_tilt = huber((foot_R[...,2,:]-ref_R[...,2,:])/c.tilt_std).sum(-1)
        terms = dict(leg_pose=huber(pose/self.leg_pose_scale).mean(-1),
                     leg_velocity=huber(velocity/self.leg_velocity_scale).mean(-1),
                     pelvis_tilt=huber(tilt/c.tilt_std).sum(-1),
                     stance_tilt=(foot_tilt*support).sum(-1)/support.sum(-1).clamp_min(1),
                     knee=huber(torch.relu(c.knee_min-S[4][:,self.knee_indices])/c.knee_std).mean(-1))
        if c.motion_style_weight:
            terms.update(self.motion_style_terms(S,f,feet,gap,support))
        return terms

    def motion_style_terms(self, S, f, feet, gap, support):
        m = self.motion
        p, R, v, w = S[:4]
        # Keep contact windows inside the same recorded motion.
        clips = m.clip_id[f]
        before = torch.maximum(f-2, m.clip_start[clips])
        after = torch.minimum(f+2, m.clip_end[clips])
        changing = (m.contact[before] != m.contact[after]).float()
        contact_logits = (self.task.cfg.contact_distance-gap.clamp_min(0.))/.025
        timing = F.binary_cross_entropy_with_logits(contact_logits,support,reduction='none')
        lift_timing = (timing*changing).mean(-1)

        basis = yaw_mat(R).transpose(-1,-2)
        ref_basis = yaw_mat(m.root_R[f]).transpose(-1,-2)
        local_feet = (basis[:,None] @ (feet-p[:,None])[...,None]).squeeze(-1)
        ref_feet = m.feet[f,:,:3]-m.root_p[f,None]
        ref_local_feet = (ref_basis[:,None] @ ref_feet[...,None]).squeeze(-1)
        delta = local_feet-ref_local_feet
        swing = 1-support
        swing_error = delta / self.swing_motion_scale
        swing_motion = (F.smooth_l1_loss(swing_error,torch.zeros_like(swing_error),reduction='none').mean(-1)
                        * swing).sum(-1)/swing.sum(-1).clamp_min(1)

        # Feet relative to the pelvis expose its displacement over the stance foot.
        # Compare in each pose's heading frame so there is no fixed global path.
        support_offset = (delta[...,:2]*support[...,None]).sum(-2)/support.sum(-1,keepdim=True).clamp_min(1)
        velocity = (basis @ v[...,None]).squeeze(-1)
        ref_velocity = (ref_basis @ m.root_v[f][...,None]).squeeze(-1)
        speed_error = (velocity-ref_velocity)/self.shift_velocity_scale
        yaw_error = (w[:,2]-m.root_w[f,2])/.65
        shift = torch.cat((support_offset/.09,speed_error,yaw_error[:,None]),dim=-1)
        weight_shift = F.smooth_l1_loss(shift,torch.zeros_like(shift),reduction='none').mean(-1)
        return dict(lift_timing=lift_timing,swing_motion=swing_motion,weight_shift=weight_shift)

    def weighted_gait_terms(self, S, f):
        terms = self.gait_terms(S,f)
        c = self.task.cfg
        weights = dict(lift_timing=.35*c.motion_style_weight,
                       swing_motion=.85*c.motion_style_weight,
                       weight_shift=.75*c.motion_style_weight)
        return {key:(weights[key] if key in weights else getattr(c,key+'_weight'))*value
                for key,value in terms.items()}

    def reward(self, S, f, a, a_prev):
        costs = self.weighted_gait_terms(S,f)
        return super().reward(S,f,a,a_prev)-CONTROL_DT*sum(costs.values())

    def critic_input(self, S, f, query, xi_n):
        pose, velocity, tilt = self.gait_errors(S,f)
        ref_foot_tilt = quat_to_mat(self.motion.feet[f,:,3:7])[...,2,:].flatten(1)
        return torch.cat((super().critic_input(S,f,query,xi_n),pose,velocity,tilt,ref_foot_tilt),-1)

    def training_diagnostics(self, S, f):
        result = {key+'_cost':value.mean() for key,value in self.weighted_gait_terms(S,f).items()}
        result.update({key+'_reward':(getattr(self.task.cfg,key+'_weight')*value).mean()
                       for key,value in self.task.reward_terms(S,f).items()})
        return result
