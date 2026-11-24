"""Train the RoBan Learned Motion Matching compressor and decompressor."""
import argparse
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...rotations import mat_to_quat, quat_to_mat, yaw_mat
from .motion_matching import Database, FOOT_OFFSET, Robot, clip_sampling_weights, heading, local

FORMAT = "roban_motion_codec_v1"
LATENT_DIM = 32
Q = slice(11, 32)
QD = slice(32, 53)
PHYSICAL_DIM = 53


class MotionCodec(nn.Module):
    def __init__(self, feature_mean, feature_std, pose_mean, pose_std):
        super().__init__()
        for name, value in (("feature_mean", feature_mean), ("feature_std", feature_std),
                            ("pose_mean", pose_mean), ("pose_std", pose_std)):
            self.register_buffer(name, value)
        self.compressor = nn.Sequential(nn.Linear(PHYSICAL_DIM + 2, 512), nn.ELU(),
                                        nn.Linear(512, 512), nn.ELU(), nn.Linear(512, 512), nn.ELU(),
                                        nn.Linear(512, LATENT_DIM))
        self.decompressor = nn.Sequential(nn.Linear(len(feature_mean) + LATENT_DIM, 512),
                                          nn.ReLU(), nn.Linear(512, PHYSICAL_DIM + 2))

    def encode(self, pose, contact):
        return self.compressor(torch.cat(((pose-self.pose_mean)/self.pose_std, contact), -1))

    def decode(self, features, latent):
        value = self.decompressor(torch.cat(((features-self.feature_mean)/self.feature_std, latent), -1))
        pose, logits = value.split((PHYSICAL_DIM, 2), dim=-1)
        return pose*self.pose_std+self.pose_mean, logits


def load_codec(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved.get("format") != FORMAT:
        raise ValueError(f"{path} is not a RoBan motion codec checkpoint")
    state = saved["codec"]
    codec = MotionCodec(*(state[name] for name in ("feature_mean", "feature_std", "pose_mean", "pose_std")))
    codec.to(device).load_state_dict(state)
    return codec.eval(), saved


@torch.inference_mode()
def decode_catalogue(directory, robot, db):
    """Reconstruct the pose catalogue while retaining its original planar root path."""
    codec, saved = load_codec(directory/"model.pt")
    if saved["database"] != db.projection_signature() or saved["joints"] != robot.joints:
        raise ValueError("Codec requires the same motions, joints, clip filters and frame rate used for training")
    with np.load(directory/"latent.npz", allow_pickle=False) as archive:
        frames = archive["frame"]
        expected = np.flatnonzero(db.end > np.arange(len(db.pose)))
        if str(archive["database"]) != saved["database"] or not np.array_equal(frames, expected):
            raise ValueError("Latent catalogue does not cover this codec's playable motion frames")
        features = torch.from_numpy(archive["features"])
        latent = torch.from_numpy(archive["latent"])
    if features.shape != (len(frames), len(codec.feature_mean)) or latent.shape != (len(frames), LATENT_DIM):
        raise ValueError("Latent catalogue has unexpected feature or latent dimensions")
    decoded = torch.cat([codec.decode(x, z)[0] for x, z in zip(features.split(4096), latent.split(4096))])
    if not torch.isfinite(decoded).all():
        raise ValueError("Codec produced nonfinite reconstructed poses")
    poses = db.pose.copy()
    poses[frames] = apply_decoded_pose(robot, poses[frames], decoded)
    return poses


@torch.inference_mode()
def apply_decoded_pose(robot, reference_pose, decoded):
    """Use predicted joints, height and tilt with the supplied planar root path."""
    poses = reference_pose.copy()
    root = poses[:, robot.root:robot.root+7]
    tilt = quat_to_mat(decoded[:, 1:5])
    tilt = yaw_mat(tilt).transpose(-1, -2) @ tilt
    reference = quat_to_mat(torch.from_numpy(root[:, 3:7]).float())
    root[:, 3:7] = mat_to_quat(yaw_mat(reference) @ tilt).numpy()
    root[:, 2] = decoded[:, 0].numpy()
    poses[:, robot.joint_qpos] = decoded[:, Q].numpy()
    return poses


def pose_data(robot, db, device):
    pose = db.pose
    n = len(pose)
    root = pose[:, robot.root:robot.root+7]
    yaw = heading(root[:, 3:7])
    current = np.arange(n)
    prior = np.maximum(current-1, 0)
    following = np.minimum(current+1, n-1)
    prior = np.where(db.clip[prior] == db.clip, prior, current)
    following = np.where(db.clip[following] == db.clip, following, current)
    velocity = np.zeros((n, robot.model.nv))
    for i in range(n):
        elapsed = max(1, following[i]-prior[i])/db.fps
        mujoco.mj_differentiatePos(robot.model, velocity[i], elapsed, pose[prior[i]], pose[following[i]])
    root_dof = int(robot.model.jnt_dofadr[robot.free])
    joint_dofs = [robot.model.jnt_dofadr[robot.model.joint(name).id] for name in robot.joints]
    to_device = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    rotation = quat_to_mat(to_device(root[:, 3:7]))
    tilt = mat_to_quat(yaw_mat(rotation).transpose(-1, -2) @ rotation)
    physical = torch.cat((to_device(root[:, 2:3]), tilt,
                          to_device(local(velocity[:, root_dof:root_dof+3], yaw)),
                          to_device(velocity[:, root_dof+3:root_dof+6]),
                          to_device(pose[:, robot.joint_qpos]), to_device(velocity[:, joint_dofs])), -1)
    features = to_device(db.features*db.scale)
    # LAFAN1 has no solver contact forces. These are kinematic support labels.
    contact = to_device(db.stance >= .5)
    feet = to_device(db.features[:, :6].reshape(n, 2, 3)).clone()
    feet[..., 2] += physical[:, :1]
    next_yaw = yaw[following]-yaw
    c, s = np.cos(next_yaw), np.sin(next_yaw)
    turn = to_device(np.stack((c, -s, s, c), -1).reshape(n, 2, 2))
    translation = to_device(local(root[following, :3]-root[:, :3], yaw)[:, :2])
    return physical, features, contact, feet, turn, translation


class Reconstruction:
    def __init__(self, data, fps, device):
        from .fast.models import FastKinematics
        self.pose, self.features, self.contact, self.feet, self.turn, self.translation = data
        self.fps = fps
        self.kin = FastKinematics(device)
        self.foot_idx = [self.kin.body_names.index(f"leg_{side}6_link") for side in "lr"]
        self.sole = torch.as_tensor(FOOT_OFFSET, dtype=torch.float32, device=device)
        self.offsets = torch.arange(2, device=device)

    def __call__(self, codec, starts):
        rows = starts[:, None]+self.offsets
        target = self.pose[rows]
        contact = self.contact[rows]
        latent = codec.encode(target, contact)
        predicted, logits = codec.decode(self.features[rows], latent)
        root = F.pad(predicted[..., :1], (2, 0))
        feet, rotations = self.kin.pose(root, quat_to_mat(predicted[..., 1:5]), predicted[..., Q])
        sole = feet[..., self.foot_idx, :] + (rotations[..., self.foot_idx, :, :] @ self.sole)
        target_sole = self.feet[rows]

        def sole_velocity(position):
            xy = (self.turn[starts, None] @ position[:, 1, :, :2, None]).squeeze(-1)
            return (xy + self.translation[starts, None] - position[:, 0, :, :2])*self.fps

        foot_speed = sole_velocity(sole)
        reference_speed = sole_velocity(target_sole)
        support = contact[:, 0]*contact[:, 1]
        speed_error = F.smooth_l1_loss(foot_speed/.25, reference_speed/.25, reduction="none")
        positions = F.smooth_l1_loss(sole/.03, target_sole/.03)
        velocity = (speed_error*(1.+support[..., None])).mean()
        pose_error = F.smooth_l1_loss((predicted-target)/codec.pose_std, torch.zeros_like(predicted))
        contacts = F.binary_cross_entropy_with_logits(logits, contact)
        angles = predicted[..., Q]
        delta_error = ((angles[:, 1]-angles[:, 0]) - (target[:, 1, Q]-target[:, 0, Q]))*self.fps
        temporal = F.smooth_l1_loss(delta_error/2., torch.zeros_like(delta_error))
        regularizer = .002*latent.square().mean() + .01*(latent[:, 1]-latent[:, 0]).abs().mean()
        loss = pose_error + positions + .25*velocity + .25*contacts + .15*temporal + regularizer
        measured = torch.stack((
            loss.detach(),
            (angles-target[..., Q]).abs().mean().detach()*180./math.pi,
            (sole-target_sole).norm(dim=-1).mean().detach()*1000.,
            ((logits >= 0) == contact.bool()).float().mean(),
            (predicted[..., QD]-target[..., QD]).abs().mean().detach(),
            (foot_speed.norm(dim=-1)*support).sum().detach(),
            (reference_speed.norm(dim=-1)*support).sum().detach(),
            support.sum(),
        ))
        return loss, measured


def evaluate(reconstruction, codec, ids, batch_size):
    names = ("loss", "joint_mae_deg", "sole_mean_error_mm", "contact_accuracy", "joint_velocity_mae_rad_s",
             "predicted_support_foot_speed_mps", "reference_support_foot_speed_mps")
    ordinary, support = 0., 0.
    with torch.no_grad():
        for chunk in ids.split(batch_size):
            _, measured = reconstruction(codec, chunk)
            ordinary = ordinary + measured[:5]*len(chunk)
            support = support + measured[5:]
    scores = torch.cat((ordinary/len(ids), support[:2]/support[2].clamp_min(1)))
    return dict(zip(names, scores.tolist()))


def upstream_holdout(db):
    clips = np.unique(db.clip[db.candidates])
    # The original clip and its mirror share a split.
    names = [db.names[i].removesuffix(" mirrored") for i in clips]
    groups, group_ids = np.unique(names, return_inverse=True)
    if len(groups) < 2:
        raise ValueError("Need at least two independent clips in the mapped library")
    counts = np.bincount(db.clip[db.candidates], minlength=len(db.names))
    sizes = np.bincount(group_ids, weights=counts[clips])
    high_step = np.unique(group_ids[db.high_step_clips[clips]])
    if db.preserve_ground and len(high_step) > 1:
        # Keep the longer high-step segment for training and hold out the shorter one.
        chosen = high_step[np.argmin(sizes[high_step])]
    else:
        small = np.flatnonzero(sizes <= .25*sizes.sum())
        choices = small if len(small) else np.arange(len(groups))
        chosen = choices[np.argmax(sizes[choices])]
    return clips[group_ids == chosen]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path.home()/"roban数据/lafan1")
    parser.add_argument("--clips", nargs="+", default=["walk*.csv", "run*.csv"], help="Quoted CSV glob patterns")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--upstream-roban", type=Path, help="Mapped 60 Hz standing, walking and running library")
    parser.add_argument("--out", type=Path, required=True, help="New directory for model.pt and latent.npz")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--eval-frames", type=int, default=2048)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.fps is None:
        args.fps = 60. if args.upstream_roban else 30.
    if args.upstream_roban and args.fps != 60.:
        parser.error("The mapped upstream RoBan library uses 60 Hz")
    if (min(args.steps, args.batch, args.eval_frames, args.eval_every) < 1
            or not math.isfinite(args.fps) or args.fps < 10.):
        parser.error("Training counts must be positive and frame rate must be finite and at least 10 Hz")
    if args.out.exists():
        parser.error(f"{args.out} already exists; pass a new output directory")

    os.environ["ACP_ROBOT"] = "roban"
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    robot = Robot()
    db = (Database.from_upstream_preview(robot, args.upstream_roban, args.fps) if args.upstream_roban
          else Database(robot, args.data, args.fps, args.clips))
    clips = np.unique(db.clip[db.candidates])
    if len(clips) < 2:
        parser.error("Use at least two clips with upright motion to hold out a validation clip")
    if args.upstream_roban:
        try:
            holdout = upstream_holdout(db)
        except ValueError as error:
            parser.error(str(error))
    else:
        holdout = rng.permutation(clips)[:max(1, len(clips)//5)]
    keep = ~np.isin(db.clip[db.candidates], holdout)
    train = torch.as_tensor(db.candidates[keep], device=args.device)
    sampling = None
    if db.preserve_ground:
        sampling = torch.as_tensor(clip_sampling_weights(db.clip[db.candidates[keep]]),
                                   dtype=torch.float32, device=args.device)
    validation = db.candidates[~keep]
    validation = torch.as_tensor(rng.choice(validation, size=min(args.eval_frames, len(validation)), replace=False),
                                 device=args.device)
    data = pose_data(robot, db, args.device)
    measured, features = (array[train] for array in data[:2])
    codec = MotionCodec(features.mean(0), features.std(0, unbiased=False).clamp_min(.025),
                        measured.mean(0), measured.std(0, unbiased=False).clamp_min(.015)).to(args.device)
    reconstruction = Reconstruction(data, db.fps, args.device)
    optimizer = torch.optim.AdamW(codec.parameters(), lr=8e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=8e-5)
    args.out.mkdir(parents=True, exist_ok=False)
    print(f"{len(train)} training windows; {len(validation)} validation windows in "
          f"{[db.names[i] for i in holdout]}", flush=True)

    best = math.inf
    for step in range(1, args.steps+1):
        batch = (torch.randint(len(train), (args.batch,), device=args.device) if sampling is None
                 else torch.multinomial(sampling, args.batch, replacement=True))
        ids = train[batch]
        loss, _ = reconstruction(codec, ids)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = nn.utils.clip_grad_norm_(codec.parameters(), 3., error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        if step % args.eval_every == 0 or step == args.steps:
            scores = evaluate(reconstruction, codec, validation, args.batch)
            scores.update(step=step, training_loss=float(loss.detach()), gradient_norm=float(grad))
            print(json.dumps(scores), flush=True)
            if scores["loss"] < best:
                best = scores["loss"]
                torch.save(dict(format=FORMAT, codec=codec.state_dict(), fps=db.fps,
                                database=db.projection_signature(), clips=db.names, joints=robot.joints,
                                validation_clips=[db.names[i] for i in holdout],
                                balance_clips=db.preserve_ground, metrics=scores),
                           args.out/"model.tmp")
                (args.out/"model.tmp").replace(args.out/"model.pt")

    chosen = torch.load(args.out/"model.pt", map_location=args.device, weights_only=True)
    codec.load_state_dict(chosen["codec"])
    physical, features, contact = data[:3]
    frames = np.flatnonzero(db.end > np.arange(len(db.pose)))
    with torch.no_grad():
        indices = torch.as_tensor(frames, device=args.device)
        latent = torch.cat([codec.encode(physical[i], contact[i]).cpu() for i in indices.split(4096)])
    np.savez_compressed(args.out/"latent.npz", latent=latent.numpy(), frame=frames, clip=db.clip[frames],
                        features=features[indices].cpu().numpy(), database=chosen["database"], fps=db.fps,
                        training_clip=~np.isin(db.clip[frames], holdout))
    (args.out/"metrics.json").write_text(json.dumps(chosen["metrics"], indent=2)+"\n")
    print(f"Saved best step {chosen['metrics']['step']} and {len(frames)} frame latents to {args.out}", flush=True)


if __name__ == "__main__":
    main()
