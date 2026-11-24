"""Track timed footholds with gait supervision from the same physical rollout."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .consistent_joint import FootGroundGeometry
from .leg_collision import LegCollisionCfg, LegCollisionMDP, LegCollisionPolicy, LegCollisionTask
from ..frax_rollout import FEET, SCHEMA
from ..motion import CONTROL_DT, MotionLib
from ..robot import JOINT_NAMES, NUM_J, metadata
from ..task import to_base
from ...rotations import quat_to_mat, yaw_mat
from .event_planner import WINDOW
from .motion_matching import FOOT_OFFSET, heading


EVENT_FIELDS = ("valid", "left", "right", "time_s", "x", "y", "z", "sin_yaw", "cos_yaw")


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rollout_clips(datasets, min_frames):
    """Discover successful physical episodes through their collection manifests."""
    clips = []
    seen = set()
    fingerprint_expected = metadata()["fingerprint"]
    for dataset in datasets:
        root = Path(dataset).resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        if (manifest.get("sample_schema") != SCHEMA or not manifest.get("completed")
                or manifest.get("control_dt") != CONTROL_DT):
            raise ValueError(f"{root} requires a completed Frax rollout export at 50 Hz")
        if manifest["robot"].get("fingerprint") != fingerprint_expected:
            raise ValueError(f"{root} was captured with a different robot or actuation configuration")
        for entry in manifest["attempts"]:
            if not entry["successful"] or entry["valid_samples"] < min_frames:
                continue
            path = root / entry["samples"]
            with np.load(path, allow_pickle=False) as data:
                bounds = data["future_event_bounds"][:entry["valid_samples"]]
                slots = int(np.diff(bounds, axis=1).max())
            motion = root / entry["motion"]
            sample_hash, motion_hash = digest(path), digest(motion)
            fingerprint = f"{sample_hash}:{motion_hash}"
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            clips.append(dict(source=str(root), case=entry["source_clip"],
                              category=entry["upstream"]["category"], take=entry["upstream"]["raw_motion"],
                              sample=str(path), motion=str(motion), fingerprint=fingerprint,
                              frames=entry["steps"], usable=entry["valid_samples"], event_slots=slots))
    if not clips:
        raise ValueError("No successful episodes with enough confirmed future frames")
    return clips


def split_rollouts(clips, val_fraction, seed):
    """Keep all segments, attempts, and mirrors of a source motion in one partition."""
    if not 0 < val_fraction < 1:
        raise ValueError("Validation fraction must be in (0, 1)")
    categories = {}
    for clip in clips:
        categories.setdefault(clip["take"], set()).add(clip["category"])
    takes = sorted(categories)
    if len(takes) < 2:
        raise ValueError("A held-out split requires at least two distinct upstream source motions")
    order = np.random.default_rng(seed).permutation(takes)
    held = set()
    full = set.union(*categories.values())
    wanted = min(len(takes) - 1, max(1, round(val_fraction * len(takes))))
    for take in order:
        remaining = set(takes) - held - {take}
        if set.union(*(categories[key] for key in remaining)) == full:
            held.add(take)
        if len(held) == wanted:
            break
    if not held:
        held.add(order[0])
    return {name: [c for c in clips if (c["take"] in held) == is_val]
            for name, is_val in (("train", False), ("val", True))}


def encode_event_plan(state, valid, foot, time_s, position_w, yaw_w):
    """Pack ordered, fixed-width world events for the actor at its current base pose.

    The caller provides one row per slot, sorted by remaining time. Use nonnegative
    times no greater than 1.5 s and foot indices 0 (left) or 1 (right).
    """
    valid = valid.bool()
    position_w = torch.where(valid[..., None], position_w, state[0][:, None])
    position = to_base(position_w, state[0], state[1])
    root_yaw = torch.atan2(state[1][:, 1, 0], state[1][:, 0, 0])[:, None]
    angle = torch.where(valid, yaw_w, root_yaw) - root_yaw
    fields = torch.cat((valid[..., None], (foot == 0)[..., None], (foot == 1)[..., None],
                        time_s[..., None], position, angle.sin()[..., None], angle.cos()[..., None]), -1)
    return torch.where(valid[..., None], fields, 0.).flatten(1)


def read_episode(part, slots):
    """Expand CSR windows and compute per-foot deadline bands inside one episode."""
    path = part["sample"]
    with np.load(path, allow_pickle=False) as source:
        if (source["schema"].item() != SCHEMA or source["fps"].item() != round(1 / CONTROL_DT)
                or source["horizon_steps"].item() != WINDOW
                or source["joint_names"].tolist() != JOINT_NAMES
                or tuple(source["foot_names"]) != FEET or not np.allclose(source["sole_offset"], FOOT_OFFSET)):
            raise ValueError(f"Incompatible rollout shard {path}")
        n = part["frames"]
        t = np.arange(n)
        valid = source["sample_valid"]
        if valid.dtype != np.bool_ or not np.array_equal(valid, t < part["usable"]):
            raise ValueError(f"Invalid confirmed prefix in {path}")
        if not np.allclose(source["time_s"], (t + 1) * CONTROL_DT, atol=1e-6, rtol=0):
            raise ValueError(f"Invalid rollout timeline in {path}")
        feet = np.concatenate((source["sole_pos_w"], source["sole_quat_w"]), axis=-1)
        support = source["foot_support"]
        qpos, qvel = source["qpos"], source["qvel"]
        frame, foot = source["event_frame"], source["event_foot"]
        position, yaw = source["event_pos_w"], source["event_yaw_w"]
        bounds = source["future_event_bounds"]
    if (qpos.shape != (n, 7 + NUM_J) or qvel.shape != (n, 6 + NUM_J) or feet.shape != (n, 2, 7)
            or support.shape != (n, 2) or support.dtype != np.bool_
            or not all(np.isfinite(x).all() for x in (feet, qpos, qvel, position, yaw))
            or not np.allclose(np.linalg.norm(feet[..., 3:7], axis=-1), 1., atol=1e-3)
            or not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=-1), 1., atol=1e-3)):
        raise ValueError(f"Invalid aligned states in {path}")
    if (frame.ndim != 1 or frame.dtype.kind not in "iu" or foot.shape != frame.shape
            or foot.dtype.kind not in "iu" or position.shape != (len(frame), 3) or yaw.shape != frame.shape
            or np.any((foot < 0) | (foot > 1)) or np.any((frame < 1) | (frame >= n))
            or np.any(np.diff(frame) < 0) or not support[frame, foot].all()
            or not np.allclose(position, feet[frame, foot, :3])
            or not np.allclose(np.cos(yaw - heading(feet[frame, foot, 3:7])), 1., atol=1e-5)):
        raise ValueError(f"Invalid touchdown events in {path}")
    expected = np.stack((np.searchsorted(frame, t, side="right"),
                         np.searchsorted(frame, t + WINDOW, side="right")), axis=-1)
    if bounds.dtype.kind not in "iu" or not np.array_equal(bounds, expected):
        raise ValueError(f"Invalid future-event index in {path}")
    if np.diff(bounds[:part["usable"]], axis=1).max() > slots:
        raise ValueError(f"Event capacity {slots} is too small for {path}")
    index = bounds[:, :1] + np.arange(slots)
    active = (index < bounds[:, 1:]) & valid[:, None]
    # Pad with a sentinel for episodes without any takeoff or touchdown.
    index = np.where(active, index, len(frame))
    event_foot = np.r_[foot, 0][index]
    dt = np.where(active, (np.r_[frame, 0][index] - t[:, None]) * CONTROL_DT, 0.)
    event_pos = np.concatenate((position, np.zeros((1, 3))))[index]
    event_yaw = np.r_[yaw, 0.][index]

    # Score XY and yaw in a short band after touchdown while the foot is loaded.
    # Touchdown targets are from the event table, never a moving swing reference.
    anchor = np.zeros((n, 2, 3))
    anchor_yaw = np.zeros((n, 2))
    near = np.zeros((n, 2), dtype=bool)
    for offset in range(3):
        target_frame = np.minimum(frame + offset, n - 1)
        select = (frame + offset < n) & support[target_frame, foot]
        dest = (target_frame[select], foot[select])
        anchor[dest], anchor_yaw[dest], near[dest] = position[select], yaw[select], True
    return (qpos, qvel), dict(feet=feet, contact=support, event_active=active, event_foot=event_foot,
                             event_time=dt, event_pos=event_pos, event_yaw=event_yaw,
                             landing_pos=anchor, landing_yaw=anchor_yaw, landing_active=near)


class RolloutMotion(MotionLib):
    def __init__(self, parts, kin, device, event_slots, max_clips=0):
        self.sources = parts[:max_clips] if max_clips else parts
        if not self.sources:
            raise ValueError("No rollout episodes in this split")
        files = [part["motion"] for part in self.sources]
        super().__init__(files, kin, device, root=str(Path(files[0]).parent))
        self.names = [part["source"] + "/" + part["case"] + "/" + Path(part["sample"]).stem
                      for part in self.sources]
        self.event_slots = event_slots
        self.foot_ids = [kin.body_names.index(name) for name in ("leg_l6_link", "leg_r6_link")]
        self.sole_offset = torch.as_tensor(FOOT_OFFSET, dtype=torch.float32, device=device)
        blocks = []
        self.geometry_reports = []
        for clip, part in enumerate(self.sources):
            if f"{digest(part['sample'])}:{digest(part['motion'])}" != part["fingerprint"]:
                raise ValueError(f"Rollout files changed since the split was created: {part['sample']}")
            (qpos, qvel), block = read_episode(part, event_slots)
            start, end = int(self.clip_start[clip]), int(self.clip_end[clip]) + 1
            if end - start != part["frames"]:
                raise ValueError(f"Rollout motion and event shard lengths differ: {part['sample']}")
            ref = self.ref_state(torch.arange(start, end, device=device))
            expected_qpos = torch.as_tensor(qpos, device=device)
            expected_qvel = torch.as_tensor(qvel, device=device)
            expected_r = quat_to_mat(expected_qpos[:, 3:7])
            expected_w = (expected_r @ expected_qvel[:, 3:6, None]).squeeze(-1)
            expected = (expected_qpos[:, :3], expected_r, expected_qvel[:, :3], expected_w,
                        expected_qpos[:, 7:], expected_qvel[:, 6:])
            if not all(torch.allclose(a, b, atol=1e-5, rtol=1e-5) for a, b in zip(ref, expected)):
                raise ValueError(f"Rollout style state and event source are misaligned: {part['sample']}")
            pick = torch.arange(0, end - start, max(1, (end - start) // 32), device=device)
            body, rot = kin.forward(*(x[pick] for x in ref), all_bodies=True)[:2]
            sole = body[:, self.foot_ids] + rot[:, self.foot_ids] @ self.sole_offset
            err = (sole - torch.as_tensor(block["feet"], device=device)[pick, :, :3]).norm(dim=-1)
            self.geometry_reports.append(dict(source=part["sample"], max_m=float(err.max())))
            blocks.append(block)
        for key in blocks[0]:
            values = torch.as_tensor(np.concatenate([block[key] for block in blocks]), device=device)
            setattr(self, key, values.float() if values.is_floating_point() else values)
        self.data_clip_end = self.clip_end
        self.clip_len = torch.tensor([part["usable"] for part in self.sources], device=device)
        self.clip_end = self.clip_start + self.clip_len - 1
        bins = self.bin_start <= self.clip_end[self.bin_clip]
        self.frame_bin = (bins.long().cumsum(0) - 1)[self.frame_bin]
        self.bin_start, self.bin_clip = self.bin_start[bins], self.bin_clip[bins]
        self.num_bins = len(self.bin_start)
        velocity = (yaw_mat(self.root_R).transpose(-1, -2) @ self.root_v[..., None]).squeeze(-1)
        # Measured rollout twist for diagnostics, never a joystick target.
        self.velocity_command = torch.cat((velocity[:, :2], self.root_w[:, 2:3]), dim=-1)


@dataclass
class RolloutRewardCfg(LegCollisionCfg):
    contact_weight: float = 2.
    position_weight: float = 2.
    yaw_weight: float = .75
    motion_style_weight: float = .75
    upper_supervision_weight: float = 0.
    slip_weight: float = .4
    impact_weight: float = .2


class RolloutPolicy(LegCollisionPolicy):
    def __init__(self, hist=10, hidden=(512, 256, 128), *, event_slots):
        self.command_dim = event_slots * len(EVENT_FIELDS)
        super().__init__(hist, hidden)


class RolloutTask(LegCollisionTask):
    def __init__(self, motion, kin, cfg=None):
        super().__init__(motion, kin, cfg or RolloutRewardCfg())
        self.ground = FootGroundGeometry(kin)
        if not self.cfg.upper_supervision_weight:
            self.supervision_loss = None

    def sample_queries(self, n):
        return torch.zeros(n, device=self.motion.device, dtype=torch.long)

    def command(self, S, frames, query):
        m = self.motion
        return encode_event_plan(S, m.event_active[frames], m.event_foot[frames], m.event_time[frames],
                                 m.event_pos[frames], m.event_yaw[frames])

    def contact_terms(self, S, f):
        c, m = self.cfg, self.motion
        sole, rotation, gap, velocity = self.ground.foot_state(S)
        support = m.contact[f].float()
        probability = torch.sigmoid((c.contact_distance - gap) / c.contact_temperature)
        contact = (support * probability + (1 - support) * (1 - probability)).mean(-1)
        due = m.landing_active[f].float()
        xy2 = (sole[..., :2] - m.landing_pos[f, :, :2]).square().sum(-1)
        yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0]) - m.landing_yaw[f]
        yaw = torch.atan2(yaw.sin(), yaw.cos())
        # A close but airborne foot does not earn full landing credit.
        position = (torch.exp(-xy2 / c.position_std**2) * due * probability).mean(-1)
        heading = (torch.exp(-yaw.square() / c.yaw_std**2) * due * probability).mean(-1)
        near_floor = ((.025 - gap.detach()) / .02).clamp(0, 1)
        slip = (((velocity[..., :2].norm(dim=-1) - .10).relu() / .5).square() * support * near_floor).mean(-1)
        # Allow descending speed seen in the demonstrated touchdown; only penalize excess.
        ref_descent = m.sole_descent[f]
        impact = (((-velocity[..., 2] - ref_descent - .30).relu() / .5).square() * due * near_floor).mean(-1)
        return dict(contact=contact, position=position, yaw=heading), slip, impact

    def reward_terms(self, S, frames):
        return self.contact_terms(S, frames)[0]


class RolloutMDP(LegCollisionMDP):
    def __init__(self, kin, motion, default_q, task):
        super().__init__(kin, motion, default_q, task)
        self.critic_input_dim += motion.event_slots * len(EVENT_FIELDS) - 13
        # Use measured sole-center descent within each episode as the impact allowance.
        sole_z = motion.feet[..., 2]
        previous = torch.maximum(torch.arange(len(sole_z), device=motion.device) - 1,
                                 motion.clip_start[motion.clip_id])
        motion.sole_descent = ((sole_z[previous] - sole_z) / CONTROL_DT).clamp_min(0)

    def reward(self, S, f, a, a_prev, *, return_self_overlap=False):
        # The inherited local gait and upper-body terms read the same rollout poses.
        c = self.task.cfg
        terms, slip, impact = self.task.contact_terms(S, f)
        p, R, _, _, q, _ = S
        reg = self.regularization_terms(S, f, a, a_prev)
        gait = self.weighted_gait_terms(S, f)
        limits = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        height = torch.exp(-((p[:, 2] - self.motion.root_p[f, 2]) / c.height_std).square())
        upright = torch.exp(-((R[:, 2, 2] - self.motion.root_R[f, 2, 2]) / c.upright_std).square())
        reward = CONTROL_DT * (c.contact_weight * terms["contact"] + c.position_weight * terms["position"]
                               + c.yaw_weight * terms["yaw"] + c.height_weight * height + c.upright_weight * upright
                               - c.upper_pose_weight * reg["upper_pose"] - c.upper_velocity_weight * reg["upper_velocity"]
                               + c.action_rate_weight * reg["action_rate"] + c.joint_speed_weight * reg["joint_speed"]
                               + c.joint_limit_weight * limits - sum(gait.values())
                               - c.self_collision_weight * self.collision_cost(S)
                               - c.slip_weight * slip - c.impact_weight * impact)
        return (reward, None) if return_self_overlap else reward

    def training_diagnostics(self, S, f):
        result = super().training_diagnostics(S, f)
        _, slip, impact = self.task.contact_terms(S, f)
        result["slip_cost"] = (self.task.cfg.slip_weight * slip).mean()
        result["impact_cost"] = (self.task.cfg.impact_weight * impact).mean()
        return result
