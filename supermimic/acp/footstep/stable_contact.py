"""13-D contact control with upper-body style and bounded physical PD targets.

Reference upper-body motion is training supervision only. The actor still receives
proprioception and contact plans, with no desired velocity or reference joint pose.
"""
from dataclasses import dataclass

import torch

from .contact_only.task import ContactOnlyPolicy, ContactOnlyMDP, ContactOnlyTask, ContactOnlyRewardCfg
from ..kinematics import RobotKinematics
from ..motion import CONTROL_DT
from ..networks import Policy
from ..robot import JOINT_NAMES, action_scale, default_joint_pos_list


class StableContactPolicy(ContactOnlyPolicy):
    def __init__(self, hist=5, hidden=(512, 256, 128)):
        super().__init__(hist, hidden)
        kin = RobotKinematics('cpu')
        default = torch.tensor(default_joint_pos_list())
        scale = torch.tensor([action_scale(n) for n in JOINT_NAMES])
        mid, half = (kin.q_lo+kin.q_hi)/2, (kin.q_hi-kin.q_lo)*.475
        # 95% soft range, reserving the environment's +/-0.01 rad encoder offset.
        lo = torch.maximum((mid-half+.01-default)/scale, torch.full_like(scale, -5.))
        hi = torch.minimum((mid+half-.01-default)/scale, torch.full_like(scale, 5.))
        if not bool(((lo < 0) & (hi > 0)).all()):
            raise ValueError('Default pose must lie strictly inside bounded PD target range')
        self.register_buffer('action_lo', lo)
        self.register_buffer('action_hi', hi)

    def clamp_action(self, action):
        """Apply the same joint bounds after physical exploration noise."""
        return torch.maximum(self.action_lo, torch.minimum(self.action_hi, action))

    def forward(self, obs_hist, goal):
        raw = Policy.forward(self, obs_hist, goal)
        # Zero raw output is the default pose; unit slope at zero on both sides.
        # Unlike a post-hoc hard clamp, this retains gradients inside each range.
        radius = torch.where(raw >= 0, self.action_hi, -self.action_lo)
        return radius*torch.tanh(raw/radius)


@dataclass
class StableContactRewardCfg(ContactOnlyRewardCfg):
    upper_pose_weight: float = 2.
    upper_velocity_weight: float = .03
    waist_pose_multiplier: float = 3.
    action_rate_weight: float = -.1
    joint_speed_weight: float = -.0005
    upright_weight: float = 1.
    height_weight: float = .5


class StableContactTask(ContactOnlyTask):
    def __init__(self, motion, kin, cfg=None):
        super().__init__(motion, kin, cfg or StableContactRewardCfg())


class StableContactMDP(ContactOnlyMDP):
    def __init__(self, kin, motion, default_q, task):
        super().__init__(kin, motion, default_q, task)
        self.critic_input_dim += 2*len(self.upper_indices)
        self.upper_weights = torch.ones(len(self.upper_indices), device=default_q.device)
        self.upper_weights[0] = task.cfg.waist_pose_multiplier

    def upper_errors(self, S, f):
        ids = self.upper_indices
        return S[4][:, ids]-self.motion.q[f][:, ids], S[5][:, ids]-self.motion.qd[f][:, ids]

    def regularization_terms(self, S, f, a, a_prev):
        pose, velocity = self.upper_errors(S, f)
        # Quadratic costs keep exerting a corrective gradient at large errors;
        # the former tiny exponential bonus became nearly flat at bad postures.
        return dict(upper_pose=(pose.square()*self.upper_weights).mean(-1),
                    upper_velocity=velocity.square().mean(-1),
                    action_rate=(a-a_prev).square().sum(-1),
                    joint_speed=S[5].square().sum(-1))

    def reward(self, S, f, a, a_prev):
        c = self.task.cfg
        p, R, _, _, q, _ = S
        terms = self.task.reward_terms(S, f)
        reg = self.regularization_terms(S, f, a, a_prev)
        upright = torch.exp(-((1-R[:, 2, 2])/c.upright_std).square())
        height = torch.exp(-((p[:, 2]-c.height_target)/c.height_std).square())
        limits = (torch.relu(q-self.soft_hi)+torch.relu(self.soft_lo-q)).sum(-1)
        return CONTROL_DT*(c.contact_weight*terms['contact']+c.position_weight*terms['position']
            +c.yaw_weight*terms['yaw']+c.upright_weight*upright+c.height_weight*height
            -c.upper_pose_weight*reg['upper_pose']-c.upper_velocity_weight*reg['upper_velocity']
            +c.action_rate_weight*reg['action_rate']+c.joint_speed_weight*reg['joint_speed']
            +c.joint_limit_weight*limits)

    def critic_input(self, S, f, query, xi_n):
        pose, velocity = self.upper_errors(S, f)
        # Only the privileged critic sees reference style errors, never the actor.
        return torch.cat((super().critic_input(S, f, query, xi_n), pose, velocity), -1)
