"""Oracle 16-D contact command and phase-aligned differentiable tracking rewards."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..mdp import MDP
from ..motion import CONTROL_DT
from ..networks import Policy, RunningNorm, mlp
from ..robot import NUM_J
from ..task import BodySpheres, to_base
from ...rotations import quat_to_mat, yaw_mat

OBS_DIM = 48+NUM_J  # no measured contact flags, no reference pose in actor observations
COMMAND_DIM = 16


class TrackerPolicy(Policy):
    def __init__(self, hist=5, hidden=(512, 256, 128)):
        # Independent network size; keep the established SuperMimic interface.
        torch.nn.Module.__init__(self)
        self.hist = hist
        self.enc = mlp(COMMAND_DIM, [128, 128], 32)
        self.dec = mlp(32+hist*OBS_DIM, list(hidden), NUM_J)
        self.obs_norm, self.goal_norm = RunningNorm(OBS_DIM), RunningNorm(COMMAND_DIM)
        with torch.no_grad():
            self.dec[-1].weight.mul_(.01)
            self.dec[-1].bias.zero_()

    def forward(self, obs_hist, goal):
        raw = super().forward(obs_hist, goal)
        return 5*torch.tanh(raw/5)  # same bounds in imagination and physical execution


@dataclass
class ContactRewardCfg:
    contact_weight: float = 1.
    position_weight: float = 1.
    yaw_weight: float = .5
    velocity_weight: float = .5
    position_std: float = .08
    yaw_std: float = .4
    velocity_std: float = .5
    contact_distance: float = .015
    contact_temperature: float = .005


class FootstepTask:
    def __init__(self, motion, kin, cfg=None):
        self.motion, self.kin = motion, kin
        self.cfg = cfg or ContactRewardCfg()
        self.foot_ids = motion.foot_ids
        # Reuse only collision geometry extraction, not legacy contact labels/commands.
        spheres = BodySpheres()
        count = spheres.bounds[1][1]
        self.sphere_points = spheres.pts[:count].to(motion.device)
        self.sphere_radii = spheres.rad[:count].to(motion.device)
        self.sphere_owner = spheres.owner[:count].to(motion.device)
        self.sphere_bounds = spheres.bounds[:2]

    def sample_queries(self, n):
        return torch.randint(0, len(self.motion.horizons), (n,), device=self.motion.device)

    def command(self, S, frames, query):
        m = self.motion
        target = frames+m.horizon_steps[query]
        contact = m.contact[target]
        pos = to_base(m.feet[target, :, :3], S[0], S[1])
        foot_R = quat_to_mat(m.feet[target, :, 3:7])
        angle = torch.atan2(foot_R[..., 1, 0], foot_R[..., 0, 0])-torch.atan2(S[1][:, 1, 0], S[1][:, 0, 0])[:, None]
        heading = torch.stack((torch.sin(angle), torch.cos(angle)), -1)
        pos = torch.where(contact[..., None], pos, 0.)
        heading = torch.where(contact[..., None], heading, 0.)
        return torch.cat((m.velocity_command[frames], m.horizons[query, None], contact.float(), pos.flatten(1), heading.flatten(1)), -1)

    def foot_geometry(self, S):
        p, R = self.kin.forward(*S, all_bodies=True)[:2]
        p, R = p[:, self.foot_ids], R[:, self.foot_ids]
        sole = p+(R@self.motion.sole_offset.view(1, 1, 3, 1)).squeeze(-1)
        points = p[:, self.sphere_owner]+(R[:, self.sphere_owner]@self.sphere_points[None, :, :, None]).squeeze(-1)
        gap = points[..., 2]-self.sphere_radii
        distance = torch.stack([gap[:, s:e].min(-1).values for s, e in self.sphere_bounds], -1)
        return sole, R, distance

    def reward_terms(self, S, frames):
        """Score *current* reference contact, never pull a swing foot to its future goal early."""
        c, m = self.cfg, self.motion
        sole, R, gap = self.foot_geometry(S)
        contact = m.contact[frames].float()
        probability = torch.sigmoid((c.contact_distance-gap)/c.contact_temperature)
        label = (contact*probability+(1-contact)*(1-probability)).mean(-1)
        count = contact.sum(-1).clamp_min(1)
        position_error = ((sole-m.feet[frames, :, :3])**2).sum(-1)
        position = (torch.exp(-position_error/c.position_std**2)*contact).sum(-1)/count
        ref_R = quat_to_mat(m.feet[frames, :, 3:7])
        delta = torch.atan2(R[..., 1, 0], R[..., 0, 0])-torch.atan2(ref_R[..., 1, 0], ref_R[..., 0, 0])
        wrapped = torch.atan2(torch.sin(delta), torch.cos(delta))
        heading = (torch.exp(-wrapped.square()/c.yaw_std**2)*contact).sum(-1)/count
        Ht = yaw_mat(S[1]).transpose(-1, -2)
        velocity = (Ht@S[2].unsqueeze(-1)).squeeze(-1)[:, :2]
        yaw_rate = S[3][:, 2:3]  # world vertical axis, consistent with yaw-only command frame
        measured = torch.cat((velocity, yaw_rate), -1)
        speed = torch.exp(-((measured-m.velocity_command[frames])**2).sum(-1)/c.velocity_std**2)
        return dict(contact=label, position=position, yaw=heading, velocity=speed)


class FootstepMDP(MDP):
    def __init__(self, kin, motion, default_q, task):
        super().__init__(kin, motion, default_q)
        self.task = task
        self.obs_noise = torch.cat((torch.full((3,), .02), torch.full((3,), .1),
                                    torch.full((NUM_J,), .01), torch.full((NUM_J,), .2),
                                    torch.zeros(NUM_J))).to(default_q.device)

    def obs(self, S, a_prev, xi):
        _, R, _, w, q, qd = S
        gravity = -R[:, 2, :]
        omega = (R.transpose(-1, -2)@w.unsqueeze(-1)).squeeze(-1)
        # Same first 48 features as the planner; previous action belongs only to tracker.
        measured_q = q-xi[:, 1:1+NUM_J]
        return torch.cat((gravity, omega, measured_q, qd, a_prev), -1)

    def reward(self, S, f, a, a_prev):
        base = super().reward(S, f, a, a_prev)
        r = self.task.reward_terms(S, f)
        c = self.task.cfg
        return base+CONTROL_DT*(c.contact_weight*r["contact"]+c.position_weight*r["position"]
                                +c.yaw_weight*r["yaw"]+c.velocity_weight*r["velocity"])

    def critic_input(self, S, f, query, xi_n):
        v = torch.zeros_like(f)
        return torch.cat((self.critic_features(S, f, xi_n, self.motion.boxes(f, v)),
                          self.task.command(S, f, query)), -1)
