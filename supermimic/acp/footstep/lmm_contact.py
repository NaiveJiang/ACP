"""Frozen-LMM forecasts and solver-supervised timed footholds."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .event_planner import WINDOW
from .fast.models import FastKinematics
from .learned_motion import FORMAT as LMM_FORMAT
from .motion_codec import Q, load_codec
from .motion_matching import Database, FOOT_OFFSET, Robot, upstream_clips
from .motion_stepper import load_latents, load_stepper
from .rollout_controller import digest, encode_event_plan
from ..frax_rollout import DEBOUNCE_STEPS
from ..motion import CONTROL_DT
from ..task import to_base
from ...rotations import mat_to_quat, quat_slerp, quat_to_mat, yaw_mat


FORMAT = "roban_lmm_timed_footholds_v2"
EVENT_FRAME = "robot_root_yaw_at_query"
# Supply extra context to confirm a touchdown at the final query frame.
SUPPORT_STEPS = 5
LOOKAHEAD = WINDOW + SUPPORT_STEPS + DEBOUNCE_STEPS - 1
HISTORY = 10
OBS = 69
POSE_FEATURES = 21 + 3 + 3 + 6 + 6 + 4


def lmm_identity(directory):
    return {name: digest(Path(directory) / name) for name in ("model.pt", "stepper.pt", "projector.pt")}


def check_seed_codec(model_dir, codec_dir):
    identity = lmm_identity(model_dir)
    if digest(codec_dir / "model.pt") != identity["model.pt"]:
        raise ValueError("--codec must contain the exact encoder used by this LMM")
    projector = torch.load(model_dir / "projector.pt", map_location="cpu", weights_only=True)
    if (projector.get("format") != LMM_FORMAT
            or projector.get("codec_sha256") != identity["model.pt"]
            or projector.get("stepper_sha256") != identity["stepper.pt"]):
        raise ValueError("Projector, codec and Stepper must come from the same LMM run")
    return identity


def angle_of(rotation):
    return torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])


def relative_yaw(yaw, anchor):
    angle = yaw - anchor
    return torch.stack((angle.sin(), angle.cos()), dim=-1)


@dataclass
class Preview:
    feature: torch.Tensor
    sole: torch.Tensor
    yaw: torch.Tensor

    def rows(self):
        return dict(feature=self.feature, sole=self.sole, yaw=self.yaw)


class FootstepPreview:
    def __init__(self, kin):
        self.kin = kin
        self.ids = torch.tensor([kin.body_names.index(f"leg_{side}6_link") for side in "lr"], device=kin.device)
        self.offset = torch.as_tensor(FOOT_OFFSET, device=kin.device, dtype=torch.float32)

    def __call__(self, qpos):
        p, R, q = qpos[..., :3], quat_to_mat(qpos[..., 3:7]), qpos[..., 7:]
        links, rotations = self.kin.pose(p, R, q)
        rotations = rotations[..., self.ids, :, :]
        sole = links[..., self.ids, :] + rotations @ self.offset
        root, anchor = p[:, 0], R[:, 0]
        foot = to_base(sole, root, anchor)
        ground = torch.cat((foot[..., :2], sole[..., 2:3]), dim=-1)
        speed = torch.diff(ground, dim=1, prepend=ground[:, :1]) / CONTROL_DT
        yaw = angle_of(rotations) - angle_of(anchor)[:, None, None]
        path = to_base(p, root, anchor)
        path = torch.cat((path[..., :2], p[..., 2:3]), dim=-1)
        feature = torch.cat((q, path, R[..., 2, :], ground.flatten(2),
                             (speed / 3).flatten(2), relative_yaw(yaw, 0).flatten(2)), dim=-1)
        return Preview(feature, foot, yaw)


def recorded_history(data, frames, hist=HISTORY):
    index = np.maximum(frames[:, None] + np.arange(1 - hist, 1), 0)
    qpos, qvel = data["qpos"][index], data["qvel"][index]
    rotation = quat_to_mat(torch.from_numpy(qpos[..., 3:7]))
    return torch.cat((-rotation[..., 2, :], torch.from_numpy(qvel[..., 3:6]),
                      torch.from_numpy(qpos[..., 7:]), torch.from_numpy(qvel[..., 6:]),
                      torch.from_numpy(data["previous_action"][index]).float()), dim=-1)


def label_rows(preview, query_root_pose, contact, feet, foot_yaw, event, known):
    """Supervise future feet relative to the measured robot root at the query."""
    root = query_root_pose[:, :3]
    rotation = quat_to_mat(query_root_pose[:, 3:7])
    position = to_base(feet, root, rotation)
    yaw = foot_yaw - angle_of(rotation)[:, None, None]
    return dict(**preview.rows(), contact=contact, foot_position=position,
                foot_heading=relative_yaw(yaw, 0), event=event, known=known)


class ContactHead(nn.Module):
    def __init__(self, hidden=96):
        super().__init__()
        self.hidden = hidden
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(HISTORY * OBS, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.start = nn.Conv1d(POSE_FEATURES, hidden, 1)
        self.temporal = nn.Sequential(nn.Conv1d(hidden, hidden, 5, padding=2), nn.GroupNorm(1, hidden), nn.SiLU(),
                                      nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2), nn.SiLU())
        self.output = nn.Conv1d(hidden, 10, 1)
        with torch.no_grad():
            self.output.weight.mul_(.01)
            self.output.bias.zero_()

    def forward(self, feature, sole, yaw, history):
        gravity, omega, q, qd, previous = history.split((3, 3, 21, 21, 21), dim=-1)
        body = self.body(torch.cat((gravity, omega / 3, q, qd / 6, previous / 4), dim=-1))
        code = self.start(feature.transpose(1, 2)) + body[..., None]
        raw = self.output(code + self.temporal(code)).transpose(1, 2)
        raw = raw.reshape(*raw.shape[:-1], 2, 5)
        position = sole + .35 * raw[..., 1:4].tanh()
        angle = yaw + torch.pi * raw[..., 4].tanh()
        return raw[..., 0], position, angle

    def predict(self, batch):
        return self(batch["feature"], batch["sole"], batch["yaw"], batch["history"])

    def loss(self, batch):
        logits, position, angle = self.predict(batch)
        known = batch["known"][..., None]
        contact = batch["contact"].float()
        event = batch["event"] & known
        boundary = torch.zeros_like(contact)
        boundary[:, 1:] = (contact[:, 1:] != contact[:, :-1]).float()
        boundary[:, 0] = 0
        weight = (1 + 2 * F.max_pool1d(boundary.transpose(1, 2), 5, stride=1, padding=2).transpose(1, 2)) * known
        support = (F.binary_cross_entropy_with_logits(logits, contact, reduction="none") * weight).sum() / weight.sum().clamp_min(1)
        error = F.smooth_l1_loss(position / .06, batch["foot_position"] / .06, reduction="none").mean(-1)
        rotation = 1 - (relative_yaw(angle, 0) * batch["foot_heading"]).sum(-1)
        touchdown = ((error + rotation) * event).sum() / event.sum().clamp_min(1)
        return support + touchdown


def decode_events(logits, position, yaw, slots):
    """Confirm sustained support and return a sorted fixed-width event window."""
    support = logits >= 0
    # Match the rollout recorder's two-frame gap tolerance.
    single = support[:, :-2] & support[:, 2:]
    double = support[:, :-3] & support[:, 3:]
    support = (support | F.pad(single, (0, 0, 1, 1))
               | F.pad(double, (0, 0, 1, 2)) | F.pad(double, (0, 0, 2, 1)))
    runs = support.unfold(1, SUPPORT_STEPS, 1).all(-1)
    rises = runs[:, 1:WINDOW + 1] & ~support[:, :WINDOW]
    batch, device = len(logits), logits.device
    times = torch.arange(1, WINDOW + 1, device=device)[None, :, None].expand_as(rises)
    foot = torch.arange(2, device=device)[None, None, :].expand_as(rises)
    none = 2 * (WINDOW + 1)
    ordered = torch.where(rises, 2 * times + foot, none).flatten(1).sort().values
    ordered = F.pad(ordered, (0, max(0, slots - ordered.shape[1])), value=none)[:, :slots]
    valid = ordered < none
    frame = (ordered // 2).clamp_max(WINDOW)
    side = ordered % 2
    rows = torch.arange(batch, device=device)[:, None]
    return valid, side, frame * CONTROL_DT, position[rows, frame, side], yaw[rows, frame, side]


def world_events(state, events):
    """Anchor new local proposals to the controlled robot's current root and yaw."""
    valid, foot, time, position, yaw = events
    basis = yaw_mat(state[1])
    position = state[0][:, None] + (basis[:, None] @ position[..., None]).squeeze(-1)
    return valid, foot, time, position, yaw + angle_of(basis)[:, None]


def event_command(head, preview, history, state, slots):
    logits, position, yaw = head(**preview.rows(), history=history)
    return encode_event_plan(state, *world_events(state, decode_events(logits, position, yaw, slots)))


class SeededForecast:
    """Offline only: sample the LMM catalogue at an observed upstream frame."""
    def __init__(self, model_dir, codec_dir, upstream, device):
        self.identity = check_seed_codec(model_dir, codec_dir)
        robot = Robot()
        db = Database.from_upstream_preview(robot, upstream, 60.)
        self.codec, meta = load_codec(model_dir / "model.pt", device)
        self.stepper, step = load_stepper(model_dir / "stepper.pt", device)
        if (meta["database"] != db.projection_signature() or step["database"] != meta["database"]
                or step["codec_sha256"] != self.identity["model.pt"] or step["fps"] != db.fps):
            raise ValueError("Stage 1 needs the exact Frax upstream library and codec/Stepper used by the LMM")
        state, frames, _, _, _ = load_latents(codec_dir / "latent.npz", meta, len(self.codec.feature_mean), device)
        self.states = (state - self.stepper.mean) / self.stepper.std
        self.rows = np.full(len(db.pose), -1, dtype=int)
        self.rows[frames] = np.arange(len(frames))
        self.root = torch.as_tensor(db.pose[:, robot.root:robot.root + 7], device=device, dtype=torch.float32)
        self.cases = {clip["case"]: (int(np.flatnonzero(db.clip == i)[0]), clip["frames"])
                      for i, clip in enumerate(upstream_clips(upstream))}
        self.window = int(step["window"])
        self.device = device
        self.preview = FootstepPreview(FastKinematics(device))

    @torch.no_grad()
    def __call__(self, case, local_frame):
        start, count = self.cases[case]
        base = np.minimum(np.floor(local_frame).astype(int), count - 1)
        ids = start + base
        row = self.rows[ids]
        if np.any(row < 0):
            raise ValueError(f"Missing playable LMM latent for {case}; inspect upstream eligibility")
        ticks = np.arange(LOOKAHEAD + 1) * (60. * CONTROL_DT) + (local_frame - base)[:, None]
        last = int(np.ceil(ticks.max()))
        trajectory = self.stepper.rollout(self.states[row], last) * self.stepper.std + self.stepper.mean
        d = len(self.codec.feature_mean)
        decoded, _ = self.codec.decode(trajectory[..., :d], trajectory[..., d:])
        q = integrate_forecast(self.root[ids], decoded)
        ticks = torch.as_tensor(ticks, device=self.device, dtype=q.dtype)
        before = ticks.floor().long()
        after = (before + 1).clamp_max(last)
        batch = torch.arange(len(ids), device=self.device)[:, None]
        left, right = q[batch, before], q[batch, after]
        fraction = ticks - before
        interp = left.lerp(right, fraction[..., None])
        interp[..., 3:7] = quat_slerp(left[..., 3:7], right[..., 3:7], fraction)
        return self.preview(interp)


def integrate_forecast(root, decoded):
    """Use the decoded velocities, including tilt-induced heading rate, as LMM playback does."""
    tilt = quat_to_mat(F.normalize(decoded[..., 1:5], dim=-1))
    forward = tilt[..., :, 0]
    angular = (tilt @ decoded[..., 8:11, None]).squeeze(-1)
    derivative = torch.cross(angular, forward, dim=-1)
    rate = (forward[..., 0] * derivative[..., 1] - forward[..., 1] * derivative[..., 0])
    rate = rate / forward[..., :2].square().sum(-1).clamp_min(1e-6)
    yaw0 = angle_of(quat_to_mat(root[:, 3:7]))
    delta = F.pad((rate[:, :-1] / 60.).cumsum(1), (1, 0))
    angle = yaw0[:, None] + delta
    middle = angle + rate / 120.
    vx, vy = decoded[..., 5:7].unbind(-1)
    c, s = middle.cos(), middle.sin()
    displacement = torch.stack((c * vx - s * vy, s * vx + c * vy), dim=-1)
    xy = root[:, None, :2] + F.pad((displacement[:, :-1] / 60.).cumsum(1), (0, 0, 1, 0))
    heading = torch.stack((angle.cos(), -angle.sin(), angle.sin(), angle.cos()), dim=-1).unflatten(-1, (2, 2))
    R = F.pad(heading, (0, 1, 0, 1))
    R[..., 2, 2] = 1.
    tilt = yaw_mat(tilt).transpose(-1, -2) @ tilt
    return torch.cat((xy, decoded[..., :1], mat_to_quat(R @ tilt), decoded[..., Q]), dim=-1)


def load_head(path, lmm_dir, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved.get("format") != FORMAT or saved.get("event_frame") != EVENT_FRAME:
        raise ValueError("Foothold head must use robot-root labels; rerun stage1 with this version")
    if saved.get("lmm") != lmm_identity(lmm_dir):
        raise ValueError("Foothold checkpoint and LMM weights do not match")
    head = ContactHead(saved["hidden"]).to(device)
    head.load_state_dict(saved["head"])
    return head, saved
