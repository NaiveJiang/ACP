"""Flat-ground walking planner trained through a fixed contact controller."""
import torch
import torch.nn.functional as F

from .consistent_joint import ConsistentJointPolicy, ConsistentJointTask


def phase_logits(prediction):
    # Independent Bernoulli logits conditioned on at least one supporting foot.
    left, right = prediction["contact_logits"].unbind(-1)
    return torch.stack((left, right, left + right), dim=-1)


class WalkingPlannerPolicy(ConsistentJointPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requires_grad_(False)
        self.planner.requires_grad_(True)

    def init_controller(self, state):
        super().init_controller(state)
        self.obs_norm.count.copy_(state["obs_norm.count"])
        self.goal_norm.count.copy_(state["goal_norm.count"])

    def contact_plan(self, history, command):
        prediction = self.planning_prediction(history, command)
        logits = phase_logits(prediction)
        probability = logits.softmax(-1)
        hard = F.one_hot(logits.argmax(-1), num_classes=3).to(probability)
        phase = hard + (probability - probability.detach())
        support = phase[:, :2] + phase[:, 2:3]
        position = prediction["position"] * support.unsqueeze(-1)
        heading = prediction["heading"] * support.unsqueeze(-1)
        return torch.cat((command[:, 3:4], support, position.flatten(1), heading.flatten(1)), -1)


class WalkingPlannerTask(ConsistentJointTask):
    supervision_loss = None
    trajectory_supervision_loss = None

    def joint_supervision_loss(self, policy, batch):
        command = torch.cat((batch["bc_velocity"], batch["bc_dt"]), -1)
        prediction = policy.planning_prediction(batch["bc_history"], command)
        support = batch["bc_contact"].float()
        valid = support.any(-1)
        # [1,0], [0,1], [1,1] correspond to classes 0, 1, 2.
        label = (support[:, 0] + 2 * support[:, 1] - 1).long().clamp_min(0)
        phase = F.cross_entropy(phase_logits(prediction), label, reduction="none")
        phase = (phase * valid).sum() / valid.sum().clamp_min(1)
        position = F.smooth_l1_loss(prediction["position"] / .05, batch["bc_position"] / .05,
                                    reduction="none").mean(-1)
        heading = 1 - (prediction["heading"] * batch["bc_heading"]).sum(-1)
        geometry = ((position + heading) * support).sum() / support.sum().clamp_min(1)
        return self.cfg.planner_supervision_weight * (phase + geometry)
