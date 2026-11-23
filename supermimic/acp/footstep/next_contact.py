"""Two-foot contact-event controller and ACP affordance planner for flat ground."""
import torch
from torch import nn
from torch.nn import functional as F

from ..motion import CONTROL_DT
from ..planner import ContactPlanner, PlannerCfg, planner_loss
from ..task import from_base, to_base
from ...rotations import quat_to_mat
from .event_planner import MIN_SUPPORT, WINDOW, _next
from .leg_collision import LegCollisionMDP, LegCollisionPolicy, LegCollisionTask

FORMAT = "two_foot_acp_affordance_v1"
COMMAND_DIM = 12
REFERENCE_TAIL = WINDOW + MIN_SUPPORT
PROPRIO_DIM = 48


def pack(valid, position, heading):
    valid = valid.bool()
    return torch.cat((valid.to(position.dtype), torch.where(valid[..., None], position, 0.).flatten(1),
                      torch.where(valid[..., None], heading, 0.).flatten(1)), -1)


def world_plan(plan, state):
    valid = plan[:, :2]
    position = from_base(plan[:, 2:8].reshape(-1, 2, 3), state[0], state[1])
    yaw = torch.atan2(state[1][:, 1, 0], state[1][:, 0, 0])[:, None]
    heading = yaw + torch.atan2(plan[:, 8::2], plan[:, 9::2])
    return valid, position, heading


def touchdown_indices(contact, min_support=MIN_SUPPORT):
    middle = (contact[:-2].int() + contact[1:-1].int() + contact[2:].int()) >= 2
    contact = torch.cat((contact[:1], middle, contact[-1:]))
    t = torch.arange(len(contact), device=contact.device)[:, None, None]
    release = _next(~contact)
    starts = contact & ~torch.cat((torch.ones_like(contact[:1]), contact[:-1]))
    sustained = (release - t >= min_support) & (t + min_support <= len(contact))
    return _next(starts & sustained)


class NextContactTask(LegCollisionTask):
    def __init__(self, motion, kin, cfg=None):
        super().__init__(motion, kin, cfg)
        local = []
        for start, end, source in zip(motion.clip_start.tolist(), motion.data_clip_end.tolist(), motion.sources):
            contact = motion.contact[start:end+1, None]
            index = touchdown_indices(contact, source.get("min_support_steps", MIN_SUPPORT))[:, 0]
            now = torch.arange(len(contact), device=motion.device)[:, None]
            moving = motion.requested_velocity[start:end+1].abs().amax(-1) > 1e-5
            next_command = _next(moving[:, None, None])[:, 0]
            visible = moving[:, None] | (index < next_command)
            local.append(torch.where((index < len(contact)) & (index <= now + WINDOW) & visible,
                                     index + start, -1))
        self.next_index = torch.cat(local)
        rotation = quat_to_mat(motion.feet[..., 3:7])
        self.recorded_yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])

    def sample_queries(self, n):
        return torch.zeros(n, device=self.motion.device, dtype=torch.long)

    def command(self, state, frames, query):
        index = self.next_index[frames]
        valid = index >= 0
        index = index.clamp_min(0)
        feet = torch.arange(2, device=frames.device)
        position = to_base(self.motion.feet[index, feet, :3], state[0], state[1])
        yaw = self.recorded_yaw[index, feet] - torch.atan2(state[1][:, 1, 0], state[1][:, 0, 0])[:, None]
        return pack(valid, position, torch.stack((yaw.sin(), yaw.cos()), -1))


class NextContactPolicy(LegCollisionPolicy):
    command_dim = COMMAND_DIM

    def init_from_query_controller(self, saved):
        weights = dict(saved)
        for key in ("enc.0.weight", "upper_enc.0.weight"):
            weights[key] = weights[key][:, 1:]
        for key in ("goal_norm.mean", "goal_norm.var"):
            weights[key] = weights[key][1:]
        self.load_state_dict(weights)
        self.goal_norm.count.clamp_max_(1e5)


class NextContactMDP(LegCollisionMDP):
    def __init__(self, *args):
        super().__init__(*args)
        self.critic_input_dim -= 1


def hindsight(positions, angles, roots, rotations, contact, velocity, history, accepted):
    """Keep positive events observed before a switch and fully observed negatives."""
    t = torch.arange(len(contact), device=contact.device)[:, None, None]
    index = touchdown_indices(contact)
    n = len(contact) - REFERENCE_TAIL + 1
    if n <= 0:
        raise ValueError("Rollout too short for contact-event supervision")
    changed = (velocity[1:] - velocity[:-1]).abs().amax(-1) > 1e-5
    change = _next(F.pad(changed[:, :, None], (0, 0, 0, 0, 1, 0)))[:n]
    observed = index[:n] <= t[:n] + WINDOW
    valid = observed & (index[:n] + MIN_SUPPORT <= change)
    known = valid | (~observed & (change > t[:n] + WINDOW + MIN_SUPPORT - 1))
    clean = known.any(-1) & accepted[None]

    lookup = index[:n].clamp_max(len(contact)-1)
    pos = positions.gather(0, lookup[..., None].expand(-1, -1, -1, 3))[:n]
    pos = to_base(pos.flatten(0, 1), roots[:n].flatten(0, 1), rotations[:n].flatten(0, 1)).unflatten(0, (n, -1))
    yaw = angles.gather(0, lookup) - torch.atan2(rotations[:n, ..., 1, 0], rotations[:n, ..., 0, 0])[..., None]
    return dict(history=history[:n][clean], velocity=velocity[:n][clean], valid=valid[clean],
                known=known[clean], position=pos[clean], heading=torch.stack((yaw.sin(), yaw.cos()), -1)[clean])


class NextContactPlanner(nn.Module):
    def __init__(self, hist):
        super().__init__()
        self.cfg = PlannerCfg(hist=hist, scan_x=(-1.2, 1.2, 25), scan_y=(-.7, .7, 15))
        self.affordance = ContactPlanner(PROPRIO_DIM, self.cfg, bodies=2)
        self.pose = nn.Linear(96 + 64, 3)
        x, y = (torch.linspace(*grid) for grid in (self.cfg.scan_x, self.cfg.scan_y))
        xy = torch.stack(torch.meshgrid(x, y, indexing="ij"), -1).reshape(-1, 2)
        # The occupancy network sees a flat floor. Sole height relative to the root is learned separately.
        self.register_buffer("grid", F.pad(xy, (0, 1), value=-.77), persistent=False)

    @staticmethod
    def proprio(history):
        x = history[..., :PROPRIO_DIM]
        return torch.cat((x[..., :3], x[..., 3:6]/3., x[..., 6:27], x[..., 27:48]/6.), -1)

    def forward(self, history, velocity):
        log_alpha, logits, context = self.affordance.features(self.proprio(history), velocity, self.grid[None])
        pose = self.pose(context)
        # Decode around the most likely point to avoid averaging separated modes.
        grid_xy = self.grid[:, :2]
        center = grid_xy[log_alpha.argmax(-1)]
        neighborhood = (grid_xy - center[..., None, :]).square().sum(-1) <= .15**2
        weight = log_alpha.masked_fill(~neighborhood, -torch.inf).softmax(-1)
        xy = weight @ grid_xy
        position = torch.cat((xy, -.77 + .30*pose[..., :1].tanh()), -1)
        return log_alpha, logits, position, F.normalize(pose[..., 1:], dim=-1, eps=1e-6)

    def loss(self, batch):
        log_alpha, logits, position, heading = self(batch["history"], batch["velocity"])
        label = batch["position"]
        grid = self.grid[None].expand(len(label), -1, -1)
        on_grid = (label[..., :2].abs() <= self.grid[:, :2].amax(0)).all(-1)
        known = batch["known"] & (~batch["valid"] | on_grid)
        planar_label = torch.cat((label[..., :2], torch.full_like(label[..., :1], -.77)), -1)
        affordance, occupancy = planner_loss(log_alpha, logits, grid, planar_label, batch["valid"], .08, known)
        positive = batch["valid"] & known
        height_error = F.smooth_l1_loss(position[..., 2]/.05, label[..., 2]/.05, reduction="none")
        yaw_error = 1 - (heading * batch["heading"]).sum(-1)
        return affordance + occupancy + ((height_error+yaw_error)*positive).sum()/positive.sum().clamp_min(1)

    @torch.no_grad()
    def plan(self, history, velocity):
        _, logits, position, heading = self(history, velocity)
        return pack(logits >= 0, position, heading)


class NextContactHandover:
    def __init__(self, planner, task):
        self.planner, self.task = planner, task

    @torch.no_grad()
    def plan(self, history, state, frame, query):
        return self.planner.plan(history, self.task.motion.requested_velocity[frame])

    @torch.no_grad()
    def world_plan(self, command, state, frame, query):
        valid, position, heading = world_plan(command, state)
        gap = self.task.foot_geometry(state)[2]
        reference = world_plan(self.task.command(state, frame, query), state)
        return valid, position, heading, gap, *reference

    def reward_delta(self, state, plan, eligible):
        valid, position, heading, previous_gap, ref_valid, reference, reference_heading = plan
        cfg = self.task.cfg
        feet, rotation, gap = self.task.foot_geometry(state)
        yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
        p0 = torch.sigmoid((cfg.contact_distance-previous_gap)/cfg.contact_temperature)
        p1 = torch.sigmoid((cfg.contact_distance-gap.detach())/cfg.contact_temperature)
        touchdown = (p1-p0).clamp(0., 1.)
        def score(target, target_yaw):
            error = yaw - target_yaw
            angle = torch.atan2(error.sin(), error.cos())
            return (cfg.position_weight * torch.exp(-(feet-target).square().sum(-1)/cfg.position_std**2)
                    + cfg.yaw_weight * torch.exp(-angle.square()/cfg.yaw_std**2))
        return CONTROL_DT * .5 * (valid*touchdown*(score(position, heading)-ref_valid*score(reference, reference_heading))).sum(-1)*eligible

    def imagined_delta(self, predicted, frames, query, plans, use):
        plan = tuple(torch.stack(series).flatten(0, 1) for series in zip(*plans))
        velocity = self.task.motion.requested_velocity
        f = torch.stack(frames)
        same = (velocity[f[1:]] == velocity[f[:-1]]).all(-1)
        return self.reward_delta(predicted, plan, (same & use).flatten())
