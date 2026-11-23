"""Train and run the zero-lookup Projector, Stepper and decoder for RoBan."""

import argparse
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.nn import functional as F

from .motion_codec import Q, load_codec
from .motion_matching import (Database, POSTURE_WEIGHT, Robot, STANCE_WEIGHT,
                              command_trajectory, intent_mode, rotate)
from .motion_stepper import load_latents, load_stepper


FORMAT = "roban_learned_motion_v1"


class LatentProjector(nn.Sequential):
    def __init__(self, input_dim, state_dim):
        super().__init__(nn.Linear(input_dim, 512), nn.ELU(), nn.Linear(512, 512),
                         nn.ELU(), nn.Linear(512, state_dim))


@dataclass
class PlaybackState:
    motion: torch.Tensor
    xy: np.ndarray
    yaw: float
    previous_command: np.ndarray
    ticks_to_project: int = 0
    outgoing: torch.Tensor | None = None
    blend_tick: int = 0


class LearnedPlayback:
    """The three networks and a standing seed are sufficient at runtime."""

    def __init__(self, directory, robot):
        self.robot = robot
        self.codec, codec_data = load_codec(directory/"model.pt")
        self.stepper, stepper_data = load_stepper(directory/"stepper.pt")
        saved = torch.load(directory/"projector.pt", map_location="cpu", weights_only=True)
        if (saved.get("format") != FORMAT or saved["joints"] != list(robot.joints)
                or saved["codec_sha256"] != checksum(directory/"model.pt")
                or saved["stepper_sha256"] != checksum(directory/"stepper.pt")
                or codec_data["fps"] != stepper_data["fps"] or saved["fps"] != codec_data["fps"]):
            raise ValueError("The RoBan LMM weights are incompatible")
        self.fps = int(saved["fps"])
        self.feature_dim = len(self.codec.feature_mean)
        self.feature_scale = saved["feature_scale"]
        self.input_mean, self.input_std = saved["input_mean"], saved["input_std"]
        self.projector = LatentProjector(len(self.input_mean), len(self.stepper.mean)).eval()
        self.projector.load_state_dict(saved["projector"])
        seed = (saved["standing_seed"]-self.stepper.mean)/self.stepper.std
        self.state = PlaybackState(seed, np.zeros(2), 0., np.zeros(3))
        self.blend_ticks = round(.22*self.fps)
        self.project_interval = round(.25*self.fps)

    def decode(self, motion):
        state = motion*self.stepper.std+self.stepper.mean
        return self.codec.decode(state[:self.feature_dim], state[self.feature_dim:])

    def pose_and_velocity(self, state):
        pose, _ = self.decode(state.motion)
        if state.outgoing is not None:
            old, _ = self.decode(state.outgoing)
            phase = state.blend_tick/self.blend_ticks
            weight = phase*phase*(3.-2.*phase)
            if torch.dot(old[1:5], pose[1:5]) < 0:
                pose[1:5] *= -1
            pose = old.lerp(pose, weight)
        pose = pose.numpy()
        tilt = Rotation.from_quat(pose[1:5], scalar_first=True)
        forward = tilt.as_matrix()[:, 0]
        angular = tilt.apply(pose[8:11])
        derivative = np.cross(angular, forward)
        yaw_rate = (forward[0]*derivative[1]-forward[1]*derivative[0])/max(np.dot(forward[:2], forward[:2]), 1e-6)
        tilt = Rotation.from_rotvec([0., 0., -np.arctan2(forward[1], forward[0])])*tilt
        rotation = Rotation.from_rotvec([0., 0., state.yaw])*tilt
        q = self.robot.model.qpos0.copy()
        q[self.robot.root:self.robot.root+7] = np.r_[state.xy, pose[0], rotation.as_quat(scalar_first=True)]
        q[self.robot.joint_qpos] = pose[Q]
        return q, pose[5:7], yaw_rate

    def projection(self, motion, command):
        state = motion*self.stepper.std+self.stepper.mean
        current = state[:self.feature_dim]
        displacement, direction = command_trajectory(command)
        desired = torch.as_tensor(np.r_[displacement.ravel(), direction.ravel()], dtype=current.dtype)
        desired *= self.feature_scale[-len(desired):]
        pose, contact = self.decode(motion)
        query = torch.cat((current[:-len(desired)], desired, torch.as_tensor(command, dtype=current.dtype),
                           pose[Q]*POSTURE_WEIGHT, contact.sigmoid()*STANCE_WEIGHT))
        projected = self.projector((query-self.input_mean)/self.input_std)
        predicted = projected*self.stepper.std+self.stepper.mean
        error = (current[-len(desired):]-desired).square().sum()
        improved = (predicted[self.feature_dim-len(desired):self.feature_dim]-desired).square().sum()
        return projected, improved < error

    def step(self, state, command, root_velocity=None):
        changed = np.any(np.abs(command-state.previous_command) > [.075, .075, .12])
        if state.outgoing is None and (state.ticks_to_project <= 0 or changed):
            target, improves = self.projection(state.motion, command)
            if changed or improves:
                state = replace(state, motion=target, outgoing=state.motion, blend_tick=0)
            state = replace(state, previous_command=command.copy(), ticks_to_project=self.project_interval)

        if root_velocity is None:
            _, local_velocity, rate = self.pose_and_velocity(state)
        else:
            local_velocity, rate = root_velocity
        next_yaw = state.yaw + rate/self.fps
        xy = state.xy + rotate(local_velocity, .5*(state.yaw+next_yaw))/self.fps
        outgoing = state.outgoing
        blend_tick = state.blend_tick+1 if outgoing is not None else 0
        outgoing = self.stepper(outgoing) if outgoing is not None and blend_tick < self.blend_ticks else None
        return PlaybackState(self.stepper(state.motion), xy, next_yaw, state.previous_command,
                             state.ticks_to_project-1, outgoing, blend_tick)

    @torch.inference_mode()
    def advance_reference(self, command, steps):
        for _ in range(steps):
            self.state = self.step(self.state, command)

    @torch.inference_mode()
    def forecast_reference(self, command, steps):
        state = self.state
        pose, velocity, rate = self.pose_and_velocity(state)
        poses = [pose]
        for _ in range(steps):
            state = self.step(state, command, (velocity, rate))
            pose, velocity, rate = self.pose_and_velocity(state)
            poses.append(pose)
        return np.stack(poses)


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_pairs(db, state, rows, candidate_frames, bank_mask, rng, count, device):
    from .motion_projector import teach

    frames = rng.choice(candidate_frames, size=count, replace=True)
    command = db.intent[frames].copy()
    command += rng.normal(size=command.shape)*[.18, .13, .25]
    change = rng.random(count)
    command[change < .1] = 0.
    jump = (change >= .1) & (change < .5)
    bank_frames = db.candidates[bank_mask]
    bounds = np.quantile(db.intent[bank_frames], [.005, .995], axis=0)
    command[jump] = rng.uniform(*bounds, size=(jump.sum(), 3))
    query = db.features[frames].copy()
    displacement, direction = command_trajectory(command)
    query[:, 15:21] = displacement.reshape(-1, 6)
    query[:, 21:27] = direction.reshape(-1, 6)
    inputs = db.projection_inputs(query, command, db.pose_features[frames])
    keys = torch.as_tensor(db.projection_keys()[bank_mask], device=device)
    labels = teach(db, keys, (inputs, frames, intent_mode(command)), device, bank_frames)
    target = bank_frames[labels]
    # Keep explicit startup, stationary and unmodified continuations in the training set.
    identity = frames[:max(1, count//7)]
    idle = (change < .1) & db.idle[frames]
    target[idle] = frames[idle]
    current = db.projection_inputs(db.features[identity], db.intent[identity], db.pose_features[identity])
    return (torch.as_tensor(np.concatenate((inputs, current)), device=device),
            state[rows[np.concatenate((target, identity))]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-roban", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--stepper", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--samples", type=int, default=60000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.steps, args.samples, args.batch) < 1 or args.out.exists():
        parser.error("Use positive training counts and a new output directory")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    codec_path = args.codec/"model.pt"
    codec, saved = load_codec(codec_path, args.device)
    stepper, stepper_data = load_stepper(args.stepper, args.device)
    robot = Robot()
    db = Database.from_upstream_preview(robot, args.upstream_roban, 60.)
    if (db.projection_signature() != saved["database"] or saved["joints"] != robot.joints
            or stepper_data["codec_sha256"] != checksum(codec_path) or stepper_data["fps"] != db.fps):
        parser.error("Train the codec and Stepper on this mapped 60 Hz library first")
    state, frames, _, train, _ = load_latents(args.codec/"latent.npz", saved,
                                             len(codec.feature_mean), args.device)
    rows = np.full(len(db.pose), -1)
    rows[frames] = np.arange(len(frames))
    if np.any(rows[db.candidates] < 0) or rows[db.start] < 0:
        parser.error("The codec archive is missing playable motion frames")
    candidate_train = train[rows[db.candidates]]
    normalized = (state-stepper.mean)/stepper.std
    inputs, targets = training_pairs(db, normalized, rows, db.candidates[candidate_train], candidate_train,
                                    rng, args.samples, args.device)
    held, held_targets = training_pairs(db, normalized, rows, db.candidates[~candidate_train], candidate_train,
                                       rng, min(args.samples, 2048), args.device)
    mean = inputs.mean(0)
    std = inputs.std(0, unbiased=False).clamp_min(.03)
    inputs, held = (inputs-mean)/std, (held-mean)/std
    model = LatentProjector(inputs.shape[1], targets.shape[1]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    for _ in range(args.steps):
        batch = torch.randint(len(inputs), (args.batch,), device=args.device)
        loss = F.smooth_l1_loss(model(inputs[batch]), targets[batch])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        validation = float(F.smooth_l1_loss(model(held), held_targets))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/"model.pt").write_bytes(codec_path.read_bytes())
    (args.out/"stepper.pt").write_bytes(args.stepper.read_bytes())
    torch.save(dict(format=FORMAT, projector=model.state_dict(), fps=db.fps, joints=list(robot.joints),
                    standing_seed=state[rows[db.start]].cpu(), feature_scale=torch.tensor(db.scale).float(),
                    input_mean=mean.cpu(), input_std=std.cpu(),
                    codec_sha256=checksum(codec_path), stepper_sha256=checksum(args.stepper)),
               args.out/"projector.pt")
    print(f"Projector held-out loss {validation:.4f}; runtime weights: {args.out}")


if __name__ == "__main__":
    main()
