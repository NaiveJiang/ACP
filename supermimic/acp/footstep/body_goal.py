"""Contact tracking with an ACP-style body direction and heading goal."""
import torch

from .contact_only.task import COMMAND_DIM as CONTACT_DIM
from .leg_collision import LegCollisionMDP, LegCollisionPolicy, LegCollisionTask
from ..motion import CONTROL_DT
from ..task import goal_command

COMMAND_DIM = CONTACT_DIM + 3
BODY_GOAL_SECONDS = .8


class BodyGoalPolicy(LegCollisionPolicy):
    command_dim = COMMAND_DIM


class BodyGoalTask(LegCollisionTask):
    def command(self, S, frames, query):
        contacts = super().command(S, frames, query)
        motion = self.motion
        # Use the measured root pose at the lookahead, within the same clip.
        future = torch.minimum(frames + round(BODY_GOAL_SECONDS / CONTROL_DT),
                               motion.data_clip_end[motion.clip_id[frames]])
        ref_R = motion.root_R[future]
        ref_yaw = torch.atan2(ref_R[:, 1, 0], ref_R[:, 0, 0])
        direction = goal_command(S, motion.root_p[future], ref_yaw, radius=.3)
        return torch.cat((contacts, direction), dim=-1)


class BodyGoalMDP(LegCollisionMDP):
    def __init__(self, kin, motion, default_q, task):
        super().__init__(kin, motion, default_q, task)
        self.critic_input_dim += COMMAND_DIM - CONTACT_DIM
