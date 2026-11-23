"""Contact-first legs and a separately supervised waist/arm action head.

Reference joints are training labels only. Deployment still needs 69-D
proprioception history and a 13-D contact plan, without desired velocity.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .stable_contact import StableContactPolicy, StableContactMDP, StableContactTask, StableContactRewardCfg
from ..mdp import unflatten_state
from ..networks import Policy, mlp, run_mlp
from ..robot import JOINT_NAMES, action_scale, default_joint_pos_list, pd_params

UPPER_INDICES = (0, 13, 14, 15, 16, 17, 18, 19, 20)


class UpperSupervisedPolicy(StableContactPolicy):
    def __init__(self, hist=5, hidden=(512, 256, 128)):
        super().__init__(hist, hidden)
        # The original encoder/decoder now emits only the twelve leg actions.
        self.dec[-1] = torch.nn.Linear(self.dec[-1].in_features, 12)
        self.upper_enc = mlp(self.command_dim, [128, 128], 32)
        self.upper_dec = mlp(32+hist*69, [256, 128], len(UPPER_INDICES))
        self.register_buffer('upper_indices', torch.tensor(UPPER_INDICES))
        with torch.no_grad():
            for head in (self.dec, self.upper_dec):
                head[-1].weight.mul_(.01)
                head[-1].bias.zero_()

    def upper_raw(self, history, goal):
        latent = torch.tanh(run_mlp(self.upper_enc, self.goal_norm(goal)))
        obs = self.obs_norm(history).flatten(-2)
        return run_mlp(self.upper_dec, torch.cat((latent, obs), -1))

    def init_from_v3(self, state):
        """Preserve the trained leg mapping; initialize the new upper head fresh."""
        merged = self.state_dict()
        expected = {k for k in merged if not k.startswith(('upper_enc.', 'upper_dec.')) and k != 'upper_indices'}
        if set(state) != expected:
            raise ValueError('Unexpected v3 policy keys; refusing partial initialization')
        last = f'dec.{len(self.dec)-1}.'
        for key, value in state.items():
            merged[key] = value[1:13] if key in (last+'weight', last+'bias') else value
        self.load_state_dict(merged)
        # Old long runs may have frozen statistics at max_count. Keep the
        # current transform but let it adapt to the newly moving arms.
        self.obs_norm.count.clamp_max_(1e6)
        self.goal_norm.count.clamp_max_(1e6)

    def forward(self, history, goal):
        legs = Policy.forward(self, history, goal)
        upper = self.upper_raw(history, goal)
        raw = torch.cat((upper[:, :1], legs, upper[:, 1:]), -1)
        radius = torch.where(raw >= 0, self.action_hi, -self.action_lo)
        return radius*torch.tanh(raw/radius)


@dataclass
class UpperSupervisedCfg(StableContactRewardCfg):
    # Actor auxiliary loss coefficient; physical reward remains the v3 reward.
    upper_supervision_weight: float = .5
    upper_target_margin: float = .02


class UpperSupervisedTask(StableContactTask):
    def __init__(self, motion, kin, cfg=None):
        super().__init__(motion, kin, cfg or UpperSupervisedCfg())
        dev = motion.device
        self.upper_indices = torch.tensor(UPPER_INDICES, device=dev)
        self.default_q = torch.tensor(default_joint_pos_list(), device=dev)
        self.alpha = torch.tensor([action_scale(n) for n in JOINT_NAMES], device=dev)
        kp, kd, _, _ = pd_params()
        self.damping_lead = torch.tensor(kd, device=dev)/torch.tensor(kp, device=dev)
        self.upper_weights = torch.ones(len(UPPER_INDICES), device=dev)
        self.upper_weights[0] = self.cfg.waist_pose_multiplier
        if not 0 < self.cfg.upper_target_margin < 1 or self.cfg.upper_supervision_weight < 0:
            raise ValueError('Invalid upper supervision weight or target margin')

    def upper_target(self, policy, frames, xi):
        """Nominal next-frame PD targets, not recorded expert motor commands.

        kd/kp * qd_ref compensates the controller's velocity damping. Gravity,
        inertia and model mismatch remain for the physical tracking objective.
        Encoder offsets follow the exact environment action-to-target mapping.
        """
        ids = self.upper_indices
        target = frames+1
        qdes = self.motion.q[target][:, ids]+self.damping_lead[ids]*self.motion.qd[target][:, ids]
        action = (qdes-self.default_q[ids]-xi[:, 1:1+len(JOINT_NAMES)][:, ids])/self.alpha[ids]
        margin = 1-self.cfg.upper_target_margin
        return torch.maximum(policy.action_lo[ids]*margin,
                             torch.minimum(policy.action_hi[ids]*margin, action)).detach()

    def supervision_loss(self, policy, batch):
        """Real replay samples only; no gradient through imagined states or WM.

        Supervise pre-tanh outputs so a wrong saturated joint still receives a
        restoring gradient. The twelve leg outputs are not part of this loss.
        """
        live = batch['usable']
        S = tuple(torch.where(live.reshape((-1,)+(1,)*(x.ndim-1)), x.detach(), 0.)
                  for x in unflatten_state(batch['s']))
        history = torch.where(live[:, None, None], batch['hist'].detach(), 0.)
        xi = torch.where(live[:, None], batch['xi'].detach(), 0.)
        goal = self.command(S, batch['f'], batch['goal_steps']).detach()
        raw = policy.upper_raw(history, goal)
        target = self.upper_target(policy, batch['f'], xi)
        ids = self.upper_indices
        radius = torch.where(target >= 0, policy.action_hi[ids], -policy.action_lo[ids])
        raw_target = (radius*torch.atanh(target/radius)).detach()
        error = (F.smooth_l1_loss(raw, raw_target, reduction='none')*self.upper_weights).mean(-1)
        return torch.where(live, error, 0.).sum()/live.float().sum().clamp_min(1)


class UpperSupervisedMDP(StableContactMDP):
    pass
