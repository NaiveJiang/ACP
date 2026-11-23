"""Contact-first actor and rewards: no commanded or reference velocity objective."""
from dataclasses import dataclass

import torch

from ..task import FootstepTask, FootstepMDP, TrackerPolicy, OBS_DIM
from ...mdp import INV_DIM, XI_DIM, invariant_features
from ...motion import CONTROL_DT
from ...networks import RunningNorm, mlp
from ...task import to_base
from ....rotations import quat_to_mat

COMMAND_DIM = 13


class ContactOnlyPolicy(TrackerPolicy):
    command_dim = COMMAND_DIM

    def __init__(self, hist=5, hidden=(512, 256, 128)):
        super().__init__(hist, hidden)
        self.enc = mlp(self.command_dim, [128, 128], 32)
        self.goal_norm = RunningNorm(self.command_dim)


@dataclass
class ContactOnlyRewardCfg:
    contact_weight: float = 2.
    position_weight: float = 4.
    yaw_weight: float = 1.5
    upright_weight: float = .5
    height_weight: float = .25
    upper_pose_weight: float = .05
    action_rate_weight: float = -.01
    joint_limit_weight: float = -.5
    joint_speed_weight: float = -.0001
    position_std: float = .08
    yaw_std: float = .4
    height_target: float = .77
    height_std: float = .15
    upright_std: float = .3
    upper_pose_std: float = .5
    contact_distance: float = .015
    contact_temperature: float = .005
    minimum_height: float = .45
    minimum_upright: float = .5


class ContactOnlyTask(FootstepTask):
    def __init__(self, motion, kin, cfg=None):
        super().__init__(motion, kin, cfg or ContactOnlyRewardCfg())

    def target_components(self, S, target):
        m = self.motion
        contact = m.contact[target]
        position = to_base(m.feet[target, :, :3], S[0], S[1])
        R = quat_to_mat(m.feet[target, :, 3:7])
        angle = torch.atan2(R[..., 1, 0], R[..., 0, 0])-torch.atan2(S[1][:, 1, 0], S[1][:, 0, 0])[:, None]
        heading = torch.stack((torch.sin(angle), torch.cos(angle)), -1)
        return contact.float(), torch.where(contact[..., None], position, 0.), torch.where(contact[..., None], heading, 0.)

    def command(self, S, frames, query):
        target = frames+self.motion.horizon_steps[query]
        contact, position, heading = self.target_components(S, target)
        return torch.cat((self.motion.horizons[query, None], contact, position.flatten(1), heading.flatten(1)), -1)

    def reward_terms(self, S, frames):
        # Same current-reference phase supervision as the oracle baseline; no future
        # foot is pulled to its landing target early, and no velocity command is read.
        c, m = self.cfg, self.motion
        sole, R, gap = self.foot_geometry(S)
        contact = m.contact[frames].float()
        probability = torch.sigmoid((c.contact_distance-gap)/c.contact_temperature)
        label = (contact*probability+(1-contact)*(1-probability)).mean(-1)
        count = contact.sum(-1).clamp_min(1)
        position = (torch.exp(-((sole-m.feet[frames, :, :3])**2).sum(-1)/c.position_std**2)*contact).sum(-1)/count
        ref_R = quat_to_mat(m.feet[frames, :, 3:7])
        delta = torch.atan2(R[..., 1, 0], R[..., 0, 0])-torch.atan2(ref_R[..., 1, 0], ref_R[..., 0, 0])
        delta = torch.atan2(torch.sin(delta), torch.cos(delta))
        heading = (torch.exp(-delta.square()/c.yaw_std**2)*contact).sum(-1)/count
        return dict(contact=label, position=position, yaw=heading)


class ContactOnlyMDP(FootstepMDP):
    def __init__(self, kin, motion, default_q, task):
        super().__init__(kin, motion, default_q, task)
        self.critic_input_dim = INV_DIM+XI_DIM+self.scan_dim+COMMAND_DIM+12
        # Only upper-body pose is a weak style prior; no leg-pose imitation.
        self.upper_indices = torch.tensor([0]+list(range(13, 21)), device=default_q.device)

    def reward(self, S, f, a, a_prev):
        c = self.task.cfg
        p, R, _, _, q, qd = S
        terms = self.task.reward_terms(S, f)
        upright = torch.exp(-((1-R[:, 2, 2])/c.upright_std).square())
        height = torch.exp(-((p[:, 2]-c.height_target)/c.height_std).square())
        upper = (q[:, self.upper_indices]-self.motion.q[f][:, self.upper_indices]).square().mean(-1)
        pose = torch.exp(-upper/c.upper_pose_std**2)
        limits = (torch.relu(q-self.soft_hi)+torch.relu(self.soft_lo-q)).sum(-1)
        return CONTROL_DT*(c.contact_weight*terms["contact"]+c.position_weight*terms["position"]
            +c.yaw_weight*terms["yaw"]+c.upright_weight*upright+c.height_weight*height
            +c.upper_pose_weight*pose+c.action_rate_weight*(a-a_prev).square().sum(-1)
            +c.joint_limit_weight*limits+c.joint_speed_weight*qd.square().sum(-1))

    def termination(self, S, f):
        c = self.task.cfg
        return (S[0][:, 2] < c.minimum_height) | (S[1][:, 2, 2] < c.minimum_upright)

    def soft_termination(self, S, f, tz=.02, tt=.04):
        c = self.task.cfg
        upright = 1-torch.sigmoid((c.minimum_upright-S[1][:, 2, 2])/tt)
        height = 1-torch.sigmoid((c.minimum_height-S[0][:, 2])/tz)
        return 1-upright*height

    def critic_input(self, S, f, query, xi_n):
        # Privileged *current* contact targets help evaluate phase-aligned reward.
        # No reference root trajectory, body velocity or leg-pose errors are used.
        contact, pos, yaw = self.task.target_components(S, f)
        current = torch.cat((contact, pos.flatten(1), yaw.flatten(1)), -1)
        return torch.cat((invariant_features(S), xi_n,
                          self.scan(S, self.motion.boxes(f, torch.zeros_like(f))),
                          self.task.command(S, f, query), current), -1)
