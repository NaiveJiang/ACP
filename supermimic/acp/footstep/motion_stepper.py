"""Train a multi-step Learned Motion Matching Stepper from RoBan codec outputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...rotations import quat_to_mat
from .motion_codec import LATENT_DIM, Q, apply_decoded_pose, load_codec
from .motion_matching import heading, rotate

FORMAT = "roban_motion_stepper_v1"


class MotionStepper(nn.Module):
    def __init__(self, mean, std, delta_scale):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("delta_scale", delta_scale)
        dim = len(mean)
        self.net = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), nn.Linear(512, 512),
                                 nn.ReLU(), nn.Linear(512, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state):
        return state + self.net(state)*self.delta_scale

    def rollout(self, start, steps):
        states = [start]
        for _ in range(steps):
            states.append(self(states[-1]))
        return torch.stack(states, dim=1)


def windows(frame, clip, train, width):
    """Starts indexing consecutive exported frame IDs in one clip and one split."""
    if len(frame) < width:
        return np.empty(0, dtype=np.int64)
    broken = ((np.diff(frame) != 1) | (clip[1:] != clip[:-1]) | (train[1:] != train[:-1]))
    edges = np.r_[0, np.cumsum(broken)]
    return np.flatnonzero(edges[width-1:] == edges[:len(frame)-width+1])


def load_latents(path, stored, feature_dim, device):
    with np.load(path, allow_pickle=False) as archive:
        frame = archive["frame"]
        clip = archive["clip"]
        train = archive["training_clip"]
        feature = archive["features"]
        latent = archive["latent"]
        fps = float(archive["fps"])
        compatible = str(archive["database"]) == stored["database"] and fps == stored["fps"]
    n = len(frame)
    if (not compatible or not math.isfinite(fps) or fps <= 0
            or frame.shape != (n,) or clip.shape != (n,) or train.shape != (n,) or train.dtype != np.bool_
            or feature.shape != (n, feature_dim) or latent.shape != (n, LATENT_DIM)
            or np.any(np.diff(frame) <= 0)
            or not (np.isfinite(feature).all() and np.isfinite(latent).all())):
        raise ValueError("latent.npz does not match the RoBan codec or contains invalid frames")
    state = torch.as_tensor(np.column_stack((feature, latent)), dtype=torch.float32, device=device)
    return state, frame, clip, train, fps


def load_stepper(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved.get("format") != FORMAT:
        raise ValueError(f"{path} is not a RoBan motion Stepper checkpoint")
    weights = saved["stepper"]
    model = MotionStepper(weights["mean"], weights["std"], weights["delta_scale"]).to(device)
    model.load_state_dict(weights)
    return model.eval(), saved


@torch.inference_mode()
def integrate_root(reference, decoded, root_index, dt):
    """Integrate heading-frame linear velocity and body-frame angular velocity."""
    path = reference.copy()
    root = path[:, root_index:root_index+7]
    tilt = quat_to_mat(decoded[:, 1:5])
    forward = tilt[..., :, 0]
    angular = (tilt @ decoded[:, 8:11, None]).squeeze(-1)
    derivative = torch.cross(angular, forward, dim=-1)
    rate = ((forward[:, 0]*derivative[:, 1] - forward[:, 1]*derivative[:, 0]) /
            forward[:, :2].square().sum(-1).clamp_min(1e-6)).numpy()
    yaw = heading(root[0, 3:7]) + np.r_[0., np.cumsum(.5*dt*(rate[1:]+rate[:-1]))]
    velocity = rotate(decoded[:, 5:7].numpy(), yaw)
    root[:, :2] = root[0, :2] + np.vstack((np.zeros((1, 2)), np.cumsum(.5*dt*(velocity[1:]+velocity[:-1]), axis=0)))
    root[:, 3:7] = np.column_stack((np.cos(.5*yaw), np.zeros((len(yaw), 2)), np.sin(.5*yaw)))
    return path


class StepperPlayback:
    """Advance a recorded seed, optionally integrating the decoded planar motion."""

    def __init__(self, checkpoint, directory, robot, db, *, decoded_root=False):
        codec_path = directory/"model.pt"
        self.codec, codec_saved = load_codec(codec_path)
        self.model, saved = load_stepper(checkpoint)
        codec_hash = hashlib.sha256(codec_path.read_bytes()).hexdigest()
        db_hash = db.projection_signature()
        self.feature_dim = len(self.codec.feature_mean)
        if (saved["codec_sha256"] != codec_hash or saved["database"] != db_hash
                or saved["fps"] != db.fps or saved["feature_dim"] != self.feature_dim
                or codec_saved["joints"] != robot.joints):
            raise ValueError("Stepper requires the exact codec, database and frame rate used for training")
        self.states, frames, _, _, _ = load_latents(directory/"latent.npz", codec_saved, self.feature_dim, "cpu")
        eligible = np.flatnonzero(db.end > np.arange(len(db.pose)))
        if not np.array_equal(frames, eligible):
            raise ValueError("Latent catalogue is missing playable motion frames")
        self.rows = np.full(len(db.pose), -1, dtype=np.int64)
        self.rows[frames] = np.arange(len(frames))
        self.robot, self.db = robot, db
        self.window = saved["window"]
        self.decoded_root = decoded_root

    @torch.inference_mode()
    def seed(self, frame):
        row = self.rows[frame]
        if row < 0:
            raise ValueError(f"Stepper seed frame {frame} is outside the playable motion")
        self.start = frame
        self.last = min(frame+self.window-1, self.db.end[frame]-1)
        normalized = (self.states[row]-self.model.mean)/self.model.std
        trajectory = self.model.rollout(normalized[None], self.last-frame)[0]
        states = trajectory*self.model.std+self.model.mean
        pose, _ = self.codec.decode(states[:, :self.feature_dim], states[:, self.feature_dim:])
        if not torch.isfinite(pose).all():
            raise ValueError("Stepper produced nonfinite reconstructed poses")
        reference = self.db.pose[self.start:self.last+1]
        if self.decoded_root:
            reference = integrate_root(reference, pose, self.robot.root, 1./self.db.fps)
        self.poses = apply_decoded_pose(self.robot, reference, pose)

    def pose(self, frame):
        return self.poses[frame-self.start]


def rollout_loss(model, source, feature_dim):
    predicted = model.rollout(source[:, 0], source.shape[1]-1)
    error = predicted[:, 1:]-source[:, 1:]
    feature = F.smooth_l1_loss(error[..., :feature_dim], torch.zeros_like(error[..., :feature_dim]))
    latent = F.smooth_l1_loss(error[..., feature_dim:], torch.zeros_like(error[..., feature_dim:]))
    change = (predicted.diff(dim=1)-source.diff(dim=1))/model.delta_scale
    velocity = F.smooth_l1_loss(change, torch.zeros_like(change))
    return feature+latent+.25*velocity, predicted


@torch.no_grad()
def validate(model, codec, state, starts, offsets, feature_dim, batch_size):
    names = ("loss", "joint_mae_deg", "last_joint_mae_deg", "root_planar_velocity_mae_mps",
             "root_yaw_rate_mae_rad_s", "contact_agreement", "contact_changes_per_window",
             "reference_contact_changes_per_window")
    totals = torch.zeros(len(names), device=state.device)
    for batch in starts.split(batch_size):
        target = state[batch[:, None]+offsets]
        loss, prediction = rollout_loss(model, target, feature_dim)
        prediction = (prediction[:, 1:]*model.std+model.mean)
        target = (target[:, 1:]*model.std+model.mean)
        gp, gc = codec.decode(target[..., :feature_dim], target[..., feature_dim:])
        pp, pc = codec.decode(prediction[..., :feature_dim], prediction[..., feature_dim:])
        q_error = (gp[..., Q]-pp[..., Q]).abs()*180./math.pi

        def yaw_rate(pose):
            vertical = quat_to_mat(pose[..., 1:5])[..., 2, :]
            return (vertical*pose[..., 8:11]).sum(-1)

        def contacts(logits):
            touching = logits >= 0
            return touching, (touching[:, 1:] != touching[:, :-1]).sum((1, 2)).float().mean()

        reference, reference_changes = contacts(gc)
        planned, planned_changes = contacts(pc)
        totals += len(batch)*torch.stack((
            loss, q_error.mean(), q_error[:, -1].mean(),
            (gp[..., 5:7]-pp[..., 5:7]).abs().mean(),
            (yaw_rate(gp)-yaw_rate(pp)).abs().mean(),
            (reference == planned).float().mean(), planned_changes, reference_changes,
        ))
    return dict(zip(names, (totals/len(starts)).tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec", type=Path, required=True, help="Directory with model.pt and latent.npz")
    parser.add_argument("--out", type=Path, required=True, help="New directory for the trained Stepper")
    parser.add_argument("--window", type=int, default=16, help="Frames per window, including the initial frame")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--eval-frames", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.window < 3 or min(args.steps, args.batch, args.eval_frames, args.eval_every) < 1:
        parser.error("Use a window of at least three frames and positive training counts")
    if args.out.exists():
        parser.error(f"{args.out} already exists; use a new output directory")

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    codec_file = args.codec/"model.pt"
    codec, stored = load_codec(codec_file, args.device)
    feature_dim = len(codec.feature_mean)
    state, frame, clip, train_mask, fps = load_latents(args.codec/"latent.npz", stored, feature_dim, args.device)
    starts = windows(frame, clip, train_mask, args.window)
    train_start = torch.as_tensor(starts[train_mask[starts]], device=args.device)
    val = starts[~train_mask[starts]]
    if not len(val) or not len(train_start):
        parser.error("Need consecutive training and held-out windows at this window length")
    validation = torch.as_tensor(rng.choice(val, size=min(args.eval_frames, len(val)), replace=False),
                                 device=args.device)
    training_states = state[torch.as_tensor(np.flatnonzero(train_mask), device=args.device)]
    mean, std = training_states.mean(0), training_states.std(0, unbiased=False).clamp_min(.03)
    state = (state-mean)/std
    del training_states
    delta = state[train_start+1]-state[train_start]
    delta_scale = delta.square().mean(0).sqrt().clamp_min(.025)
    model = MotionStepper(mean, std, delta_scale).to(args.device)
    codec.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=6e-5)
    offsets = torch.arange(args.window, device=args.device)
    identity = hashlib.sha256(codec_file.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=False)
    print(f"{len(train_start)} training windows, {len(val)} held-out windows; "
          f"{args.window-1} predicted frames ({(args.window-1)/fps:.2f} s)", flush=True)
    best = math.inf
    for step in range(1, args.steps+1):
        batch = train_start[torch.randint(len(train_start), (args.batch,), device=args.device)]
        target = state[batch[:, None]+offsets]
        loss, _ = rollout_loss(model, target, feature_dim)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        if step % args.eval_every == 0 or step == args.steps:
            scores = validate(model, codec, state, validation, offsets, feature_dim, args.batch)
            scores.update(step=step, training_loss=float(loss.detach()))
            print(json.dumps(scores), flush=True)
            if scores["loss"] < best:
                best = scores["loss"]
                torch.save(dict(format=FORMAT, stepper=model.state_dict(), feature_dim=feature_dim,
                                fps=fps, window=args.window, codec_sha256=identity, database=stored["database"],
                                validation_clips=stored["validation_clips"], metrics=scores),
                           args.out/"model.tmp")
                (args.out/"model.tmp").replace(args.out/"model.pt")
                (args.out/"metrics.json").write_text(json.dumps(scores, indent=2)+"\n")
    print(f"Saved best Stepper to {args.out/'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
