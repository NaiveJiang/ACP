"""BlinkMimic structured dynamics and Muon for the terrain contact tracker.

Uses the attributed optimizer already bundled in footstep.fast.vendor. Unlike
the flat-footstep engine, this adapter retains ACP's real terrain scans and
nine-body contact command. Reference-action skips are intentionally absent.
"""

import torch

from ..rotations import so3_exp, yaw_mat
from .footstep.fast.vendor.optim import MuonAdam
from .mdp import XI_DIM
from .motion import CONTROL_DT
from .networks import WM_OUT, RunningNorm, WorldModel, run_mlp
from .robot import NUM_J


class BlinkContactWorldModel(WorldModel):
    drift_weight = 0.625

    def __init__(self, scan_dim, hidden=(1024, 1024, 1024), *, context_dim=XI_DIM):
        super().__init__(scan_dim, hidden, context_dim=context_dim)
        self.inc_norm = RunningNorm(WM_OUT, eps=1e-4)

    def drift(self, state, velocity, angular, joint_velocity):
        _, rotation, old_velocity, old_angular, _, old_joint_velocity = state
        old = CONTROL_DT * (1 - self.drift_weight)
        new = CONTROL_DT * self.drift_weight
        return (
            (
                yaw_mat(rotation).transpose(-1, -2)
                @ (old * old_velocity + new * velocity).unsqueeze(-1)
            ).squeeze(-1),
            (
                rotation.transpose(-1, -2)
                @ (old * old_angular + new * angular).unsqueeze(-1)
            ).squeeze(-1),
            old * old_joint_velocity + new * joint_velocity,
        )

    def targets(self, state, next_state):
        increments = WorldModel.targets(state, next_state)
        position, orientation, velocity, angular, joints, joint_velocity = (
            increments.split((3, 3, 3, 3, NUM_J, NUM_J), -1)
        )
        drift_position, drift_orientation, drift_joints = self.drift(
            state, next_state[2], next_state[3], next_state[5]
        )
        return torch.cat(
            (
                position - drift_position,
                orientation - drift_orientation,
                velocity,
                angular,
                joints - drift_joints,
                joint_velocity,
            ),
            -1,
        )

    def integrate(self, state, increment):
        position, rotation, velocity, angular, joints, joint_velocity = state
        dp, dr, dv, dw, dq, dqd = increment.split((3, 3, 3, 3, NUM_J, NUM_J), -1)
        next_velocity = velocity + (rotation @ dv.unsqueeze(-1)).squeeze(-1)
        next_angular = angular + (rotation @ dw.unsqueeze(-1)).squeeze(-1)
        next_joint_velocity = joint_velocity + dqd
        drift_position, drift_orientation, drift_joints = self.drift(
            state, next_velocity, next_angular, next_joint_velocity
        )
        return (
            position
            + (yaw_mat(rotation) @ (dp + drift_position).unsqueeze(-1)).squeeze(-1),
            rotation @ so3_exp(dr + drift_orientation),
            next_velocity,
            next_angular,
            joints + dq + drift_joints,
            next_joint_velocity,
        )

    def step(self, state, action, context, scan):
        prediction = run_mlp(
            self.net, self.in_norm(self.inputs(state, action, context, scan))
        )
        return self.integrate(
            state, prediction * self.out_norm.std() + self.out_norm.mean
        )


def blink_optimizer(module, mlps, lr):
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError(
            "Blink contact training requires PyTorch with torch.optim.Muon"
        )
    return MuonAdam(module, mlps, lr, fused=next(module.parameters()).is_cuda)


@torch.no_grad()
def update_ema(average, live, decay):
    for target, source in zip(average.parameters(), live.parameters(), strict=True):
        target.lerp_(source, 1 - decay)
    for target, source in zip(average.buffers(), live.buffers(), strict=True):
        target.copy_(source)
