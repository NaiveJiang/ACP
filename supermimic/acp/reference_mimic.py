"""Reference-conditioned BlinkMimic baseline, independent of contact labels."""

from dataclasses import dataclass

import torch

from ..rotations import mat_to_6d, yaw_mat
from .mdp import OBS_DIM
from .networks import Policy
from .robot import JOINT_NAMES, NUM_J, action_scale
from .s17_actuation import ACTION_SCALE as S17_ACTION_SCALE
from .terrain import grid_offsets, height_scan


@dataclass
class ReferenceCfg:
    offsets: tuple[int, ...] = (1, 5, 10, 20, 30, 50)
    reference_action_skip: bool = True
    root_reference: str = "world"

    def __post_init__(self):
        if self.root_reference not in ("world", "local"):
            raise ValueError("root_reference must be world or local")


class ReferenceTask:
    def __init__(self, cfg, mdp):
        self.cfg, self.mdp, self.motion = cfg, mdp, mdp.motion
        self.scan_offsets = grid_offsets(
            (-0.4, 0.4, 9), (-0.3, 0.3, 7), device=mdp.default_q.device
        )
        self.frame_dim = 15 + 2 * NUM_J
        self.cmd_dim = len(cfg.offsets) * self.frame_dim + len(self.scan_offsets)
        self.offsets = torch.tensor(cfg.offsets, device=mdp.default_q.device)

    def sample_goal_steps(self, count):
        return torch.zeros(count, dtype=torch.long, device=self.mdp.default_q.device)

    def command(self, state, frames, variants, goal_steps=None, noise=False):
        del goal_steps, noise
        motion = self.motion
        future = torch.minimum(
            frames[:, None] + self.offsets,
            motion.clip_end[motion.clip_id[frames]][:, None],
        )
        window = (motion.root_p[future], motion.root_R[future], motion.root_v[future],
                  motion.root_w[future], motion.q[future], motion.qd[future])
        anchor = (motion.root_p[frames], motion.root_R[frames]) if self.cfg.root_reference == "local" else None
        return self.command_from_reference(state, window, motion.boxes(frames, variants), anchor=anchor)

    def command_from_reference(self, state, window, boxes, *, anchor=None):
        """Encode an externally planned window in the same order as training."""
        root_position, root_rotation, linear_velocity, angular_velocity, joint_position, joint_velocity = window
        if self.cfg.root_reference == "local":
            if anchor is None:
                raise ValueError("Local reference conditioning requires the current reference pose")
            origin = torch.cat((anchor[0][..., :2], state[0][..., 2:3]), dim=-1)
            yaw = anchor[1]
        else:
            origin, yaw = state[:2]
        inverse = yaw_mat(yaw).transpose(-1, -2)[:, None]
        position = (
            inverse @ (root_position - origin[:, None])[..., None]
        ).squeeze(-1)
        orientation = mat_to_6d(inverse @ root_rotation)
        velocity = (inverse @ linear_velocity[..., None]).squeeze(-1)
        angular = (inverse @ angular_velocity[..., None]).squeeze(-1)
        reference = torch.cat(
            (
                position,
                orientation,
                velocity,
                angular,
                joint_position,
                joint_velocity,
            ),
            -1,
        ).flatten(1)
        scan = height_scan(
            state[0], state[1], boxes, self.scan_offsets
        )
        return torch.cat((reference, scan), -1)

    def critic_input(self, state, frames, variants, goal_steps, context):
        return torch.cat(
            (
                self.mdp.critic_features(
                    state, frames, context, self.motion.boxes(frames, variants)
                ),
                self.command(state, frames, variants, goal_steps),
            ),
            -1,
        )


class ReferencePolicy(Policy):
    def __init__(
        self,
        obs_dim,
        goal_dim,
        default_q,
        lower,
        upper,
        hist=10,
        reference_action_skip=True,
        s17_actuators=False,
    ):
        super().__init__(obs_dim, goal_dim, hist)
        self.reference_action_skip = reference_action_skip
        scale = (
            torch.full_like(default_q, S17_ACTION_SCALE)
            if s17_actuators
            else torch.tensor(
                [action_scale(name) for name in JOINT_NAMES], device=default_q.device
            )
        )
        self.register_buffer("reference_default", default_q.detach().clone())
        self.register_buffer("reference_scale", scale)
        middle, half = (lower + upper) / 2, (upper - lower) * 0.475
        self.register_buffer("action_lo", (middle - half + 0.01 - default_q) / scale)
        self.register_buffer("action_hi", (middle + half - 0.01 - default_q) / scale)

    def project_action(self, action):
        return torch.maximum(self.action_lo, torch.minimum(self.action_hi, action))

    def forward(self, history, goal):
        action = super().forward(history, goal)
        if self.reference_action_skip:
            action = (
                action
                + (goal[:, 15 : 15 + NUM_J] - self.reference_default)
                / self.reference_scale
            )
        return self.project_action(action)


def build_policy(mdp, task, hist):
    return ReferencePolicy(
        OBS_DIM,
        task.cmd_dim,
        mdp.default_q,
        mdp.kin.q_lo,
        mdp.kin.q_hi,
        hist,
        task.cfg.reference_action_skip,
        s17_actuators=bool(mdp.delay_steps),
    )


def check_interface(checkpoint):
    if checkpoint.get("task_interface") != "reference_mimic_v1":
        raise ValueError(
            "Expected a reference Mimic checkpoint, not a contact-controller checkpoint"
        )
