"""Classic Motion Matching over retargeted RoBan CSVs, with MuJoCo playback."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

import mujoco
import numpy as np


LOOKAHEAD = np.array([1/3, 2/3, 1.])
FOOT_OFFSET = np.array([.0486, 0., -.04])
INTENT_SCALE = np.array([7., 7., 14.], dtype=np.float32)
TURN_INTENT_SCALE = np.array([9., 9., 22.], dtype=np.float32)
PIVOT_INTENT_SCALE = np.array([12., 12., 22.], dtype=np.float32)
POSTURE_WEIGHT = 4.
STANCE_WEIGHT = 4.5
TURN_THRESHOLD = .12
FEATURE_WEIGHT = np.repeat([.8, .45, .65, 6., 3.], [6, 6, 3, 6, 6])


def rotate(xy, yaw):
    x, y = np.moveaxis(xy, -1, 0)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack((c*x-s*y, s*x+c*y), -1)


def local(vectors, yaw):
    return np.concatenate((rotate(vectors[..., :2], -yaw), vectors[..., 2:]), -1)


def heading(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def planar_heading(angle):
    return rotate(np.broadcast_to([1., 0.], (*np.shape(angle), 2)), angle)


def command_trajectory(command):
    vx, vy, wz = np.moveaxis(np.asarray(command)[..., None, :], -1, 0)
    angles = wz * LOOKAHEAD
    # Exact integration of a constant velocity in the turning base frame.
    along = LOOKAHEAD * np.sinc(angles / np.pi)
    across = .5 * LOOKAHEAD * angles * np.sinc(angles / (2*np.pi))**2
    return np.stack((along*vx-across*vy, across*vx+along*vy), -1), planar_heading(angles)


def planted(feet, velocity):
    height = np.clip((.075-feet[..., 2])/.050, 0., 1.)
    speed = np.linalg.norm(velocity, axis=-1)
    return height * np.clip((.65-speed)/.45, 0., 1.)


def smooth_pose(pose, root, joints):
    # The retargeted CSVs are sampled at 30 Hz. A short symmetric filter
    # damps frame-to-frame IK noise without delaying live joystick response.
    columns = np.r_[np.arange(root, root+7), joints]
    values = pose[:, columns]
    rotation = values[:, 3:7]
    signs = np.where((rotation[1:]*rotation[:-1]).sum(-1) < 0., -1., 1.)
    rotation[1:] *= np.cumprod(signs)[:, None]
    padded = np.pad(values, ((2, 2), (0, 0)), mode="edge")
    values = sum(weight*padded[i:i+len(pose)] for i, weight in enumerate((1, 4, 6, 4, 1)))/16.
    values[:, 3:7] /= np.linalg.norm(values[:, 3:7], axis=-1, keepdims=True)
    pose[:, columns] = values


def stick_vector(axes, deadzone):
    axes = np.asarray(axes, dtype=float)
    norm = np.linalg.norm(axes)
    magnitude = np.clip((norm-deadzone)/(1-deadzone), 0., 1.)
    return axes * (magnitude/max(norm, 1e-8))


def gamepad_command(axes, limits, deadzone):
    turn = np.sign(axes[2]) * max(0., (abs(axes[2])-deadzone)/(1-deadzone))
    return np.r_[stick_vector(axes[:2], deadzone), turn] * limits


def orbit_camera(axes, azimuth, elevation, deadzone, dt):
    up, left = stick_vector((axes[3], axes[2]), deadzone)
    if left:
        azimuth = (azimuth + left*150.*dt + 180.) % 360. - 180.
    if up:
        elevation = float(np.clip(elevation + up*90.*dt, -75., -5.))
    return azimuth, elevation


class HeadingStick:
    """Face the left stick and walk along the robot's resulting forward axis."""

    def __init__(self, limits, deadzone):
        self.limits = limits
        self.deadzone = deadzone
        self.reset()

    def reset(self):
        self.turn_sign = 0.
        self.target_yaw = None

    def update(self, axes, camera_yaw, body_yaw, *, run_speed=None):
        forward, left = stick_vector(axes[:2], self.deadzone)
        strength = np.hypot(forward, left)
        if strength < 1e-8:
            self.reset()
            return np.zeros(3)
        self.target_yaw = camera_yaw + np.arctan2(left, forward)
        delta = self.target_yaw - body_yaw
        error = np.arctan2(np.sin(delta), np.cos(delta))
        # Hold the selected side near a half-turn so stick noise cannot flip it.
        if abs(error) > np.pi-.10:
            self.turn_sign = self.turn_sign or (1. if abs(error) > np.pi-1e-6 else np.sign(error))
            error = abs(error)*self.turn_sign
        else:
            self.turn_sign = np.sign(error)
        turn_first = np.cos(np.deg2rad(35.))
        aligned = np.clip((np.cos(error)-turn_first)/(1.-turn_first), 0., 1.)
        # Walking begins only near the desired heading, using positive body-frame vx.
        forward_fraction = strength*aligned
        max_yaw = self.limits[2] - max(0., self.limits[2]-2.)*forward_fraction
        turn = 0. if abs(error) < .12 else np.clip(2.8*error, -max_yaw, max_yaw)
        maximum = self.limits[0] if run_speed is None else max(self.limits[0], run_speed)
        return np.array([maximum*forward_fraction, 0., turn])


class Robot:
    def __init__(self):
        from ..roban import JOINT_NAMES, get_spec
        self.joints = JOINT_NAMES
        spec = get_spec()
        spec.add_texture(name="motion_grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=64, height=64,
                         rgb1=[.27, .31, .35], rgb2=[.20, .24, .28])
        spec.add_material(name="motion_floor", textures=["motion_grid"], texrepeat=[1., 1.], texuniform=True)
        spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[0, 0, .25], material="motion_floor")
        spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1])
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.free = int(np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE).item())
        self.root = int(self.model.jnt_qposadr[self.free])
        joint_ids = np.array([self.model.joint(name).id for name in self.joints])
        self.joint_qpos = self.model.jnt_qposadr[joint_ids]
        limited = joint_ids[self.model.jnt_limited[joint_ids].astype(bool)]
        self.limited_qpos = self.model.jnt_qposadr[limited]
        self.joint_ranges = self.model.jnt_range[limited].T
        self.feet = [self.model.body(f"leg_{side}6_link").id for side in "lr"]

    def clamp_joint_pose(self, pose):
        pose[self.limited_qpos] = np.clip(pose[self.limited_qpos], *self.joint_ranges)

    def foot_positions(self):
        matrices = self.data.xmat[self.feet].reshape(2, 3, 3)
        return self.data.xpos[self.feet] + matrices @ FOOT_OFFSET

    def load(self, path, fps):
        fields = ["Frame", *("root_translate"+a for a in "XYZ"), *("root_rotate"+a for a in "WXYZ"), *self.joints]
        with path.open(newline="", encoding="utf-8-sig") as stream:
            header = next(csv.reader(stream))
            if missing := set(fields)-set(header):
                raise ValueError(f"{path.name} missing columns {sorted(missing)}")
            values = np.loadtxt(stream, delimiter=",", usecols=[header.index(n) for n in fields], ndmin=2)
        if len(values) < round(fps)+3 or not np.isfinite(values).all():
            raise ValueError(f"{path.name} is too short or contains a nonfinite pose")
        if not np.all(np.diff(values[:, 0]) == 1):
            raise ValueError(f"{path.name} has nonconsecutive source frames")
        pose = np.broadcast_to(self.model.qpos0, (len(values), self.model.nq)).copy()
        root = pose[:, self.root:self.root+7]
        root[:] = values[:, 1:8]
        length = np.linalg.norm(root[:, 3:7], axis=-1)
        if np.any(length < 1e-8):
            raise ValueError(f"{path.name} has a zero root quaternion")
        root[:, 3:7] /= length[:, None]
        pose[:, self.joint_qpos] = values[:, 8:]
        return pose

    def align(self, pose, target):
        q = pose.copy()
        root = q[self.root:self.root+7]
        goal = target[self.root:self.root+7]
        yaw = heading(goal[3:7]) - heading(root[3:7])
        shift = goal[:2] - rotate(root[:2], yaw)
        return yaw, shift

    def apply_alignment(self, pose, yaw, shift):
        q = pose.copy()
        root = q[self.root:self.root+7]
        root[:2] = rotate(root[:2], yaw) + shift
        rotation = np.array([np.cos(.5*yaw), 0., 0., np.sin(.5*yaw)])
        mujoco.mju_mulQuat(root[3:7], rotation, root[3:7].copy())
        return q


def intent_mode(command):
    turning = np.abs(command[..., 2]) >= TURN_THRESHOLD
    pivot = np.linalg.norm(command[..., :2], axis=-1) < .10
    return np.where(turning, np.where(pivot, 2, 1), 0)


INTENT_MODES = np.stack((INTENT_SCALE, TURN_INTENT_SCALE, PIVOT_INTENT_SCALE))


def upstream_clips(directory):
    reference = directory/"reference"
    manifest = json.loads((reference/"manifest.json").read_text())
    if manifest.get("schema") != "roban_motion_matching_preview":
        raise ValueError(f"Expected a mapped Motion Matching library at {directory}")
    return manifest["clips"]


def read_upstream_poses(robot, directory):
    return read_upstream_tracks(robot, directory/"reference", upstream_clips(directory))


def upstream_clip_name(clip):
    name = clip["raw_motion"]
    if clip.get("purpose") == "high_step_run":
        name += f"_{clip['source_row_start']}-{clip['source_row_stop']}"
    return name + (" mirrored" if clip["mirrored"] else "")


def read_upstream_tracks(robot, reference, clips):
    for clip in clips:
        name = upstream_clip_name(clip)
        with np.load(reference/(clip["case"]+"_labels.npz")) as data:
            pose = data["qpos"]
        if pose.shape != (clip["frames"], robot.model.nq) or not np.isfinite(pose).all():
            raise ValueError(f"Invalid RoBan pose archive for {name}")
        yield name, pose


class Database:
    def __init__(self, robot, directory, fps, masks):
        paths = sorted({p for mask in masks for p in directory.glob(mask) if p.is_file()})
        if not paths:
            raise ValueError(f"No matching CSV clips in {directory}")
        self._build(robot, fps, ((p.stem, robot.load(p, fps)) for p in paths), smooth=True)

    @classmethod
    def from_upstream_preview(cls, robot, directory, fps, *, high_step_running=True):
        clips = upstream_clips(directory)
        db = cls.__new__(cls)
        db._build(robot, fps, read_upstream_tracks(robot, directory/"reference", clips), smooth=False)
        if high_step_running:
            high_step = np.array([clip.get("purpose") == "high_step_run" for clip in clips])
            db.high_step_clips = high_step
            db.running_frames = db.candidates[high_step[db.clip[db.candidates]]]
            if len(db.running_frames):
                weights = (.75, .75, 1., 1., 1., 1., 1.5)
                stops = (0, 3, 6, 9, 12, 15, 21, 27)
                scale = [np.full(b-a, weight/max(db.features[:, a:b].std(axis=0).mean(), .01))
                         for a, b, weight in zip(stops[:-1], stops[1:], weights)]
                db.running_scale = np.concatenate(scale)
                db.running_features = db.features[db.running_frames]*db.running_scale
        return db

    def _build(self, robot, fps, tracks, *, smooth):
        steps = np.rint(LOOKAHEAD * fps).astype(int)
        self.names = []
        self.fps = float(fps)
        self.projector = None
        self.running_frames = np.empty(0, dtype=int)
        self.ignore_neighbor = max(1, round(.40*fps))
        poses, features, intents, contacts, candidates, ends, clip_ids, idle = [], [], [], [], [], [], [], []
        look = max(1, round(.20*fps))
        cursor = 0
        for clip, (name, pose) in enumerate(tracks):
            self.names.append(name)
            if smooth:
                smooth_pose(pose, robot.root, robot.joint_qpos)
            feet = np.empty((len(pose), 2, 3))
            for i, frame in enumerate(pose):
                robot.data.qpos[:] = frame
                mujoco.mj_kinematics(robot.model, robot.data)
                feet[i] = robot.foot_positions()
            root = pose[:, robot.root:robot.root+7]
            lowest_foot = feet[..., 2].min(axis=1)
            upright = 1. - 2.*np.square(root[:, 4:6]).sum(-1)
            clearance = root[:, 2] - lowest_foot
            # Some walk-tagged sequences include low or severely tilted retargeted
            # poses. Calibrate ground and match clips using upright locomotion only.
            upright_frames = (upright > .82) & (clearance > .59) & (clearance < .87)
            ground = np.median(lowest_foot[upright_frames]) if upright_frames.any() else 0.
            root[:, 2] -= ground
            feet[..., 2] -= ground
            good = upright_frames & (lowest_foot-ground > -.045) & (lowest_foot-ground < .085)
            yaw = heading(root[:, 3:7])
            ids = np.arange(len(pose))
            next_bad = np.minimum.accumulate(np.where(good, len(pose), ids)[::-1])[::-1]
            end = np.maximum(ids, next_bad-1)
            future = np.minimum(ids[:, None] + steps, end[:, None])
            root_velocity = local(np.gradient(root[:, :3], axis=0)*fps, yaw)
            foot_velocity = local(np.gradient(feet, axis=0)*fps, yaw[:, None])
            foot_position = local(feet-root[:, None, :3], yaw[:, None])
            contacts.append(planted(feet, foot_velocity))
            trajectory = rotate(root[future, :2]-root[:, None, :2], -yaw[:, None])
            direction = planar_heading(yaw[future]-yaw[:, None])
            feature = np.column_stack((foot_position.reshape(-1, 6), foot_velocity.reshape(-1, 6),
                                       root_velocity, trajectory.reshape(-1, 6), direction.reshape(-1, 6)))
            usable = (ids > 0) & good & (ids + steps[-1] <= end)
            still = (usable & (np.linalg.norm(trajectory, axis=-1).max(axis=-1) < .045)
                     & (np.abs(direction[..., 1]).max(axis=-1) < .15)
                     & (np.linalg.norm(foot_velocity, axis=-1).max(axis=-1) < .20)
                     & (feet[..., 2].max(axis=-1) < .065) & (direction[..., 0].min(axis=-1) > 0.))
            poses.append(pose)
            features.append(feature)
            soon = np.minimum(ids + look, len(pose)-1)
            elapsed = np.maximum(soon-ids, 1)/fps
            imminent = rotate(root[soon, :2]-root[:, :2], -yaw)
            turn = np.arctan2(np.sin(yaw[soon]-yaw), np.cos(yaw[soon]-yaw))
            intents.append(np.column_stack((imminent, turn))/elapsed[:, None])
            candidates.append(usable)
            ends.append(cursor+end)
            clip_ids.append(np.full(len(pose), clip))
            idle.append(still)
            cursor += len(pose)
        self.pose = np.concatenate(poses)
        self.features = np.concatenate(features)
        self.intent = np.concatenate(intents)
        self.stance = np.concatenate(contacts)
        self.candidates = np.flatnonzero(np.concatenate(candidates))
        if not len(self.candidates):
            raise ValueError("No continuous upright locomotion windows found in the selected clips")
        self.end = np.concatenate(ends)
        self.clip = np.concatenate(clip_ids)
        self.idle = np.concatenate(idle)
        self.scale = FEATURE_WEIGHT / np.maximum(self.features[self.candidates].std(axis=0), .075)
        pose_features = self.pose[:, robot.joint_qpos]
        self.pose_features = np.column_stack((pose_features*POSTURE_WEIGHT, self.stance*STANCE_WEIGHT)).astype(np.float32)
        self.candidate_pose = self.pose_features[self.candidates]
        self.candidate_intent = self.intent[self.candidates].astype(np.float32)
        self.normalized = (self.features[self.candidates]*self.scale).astype(np.float32)
        self.still_candidates = self.candidates[self.idle[self.candidates]]
        neutral = self.still_candidates if len(self.still_candidates) else self.candidates
        rest_cost = np.square(self.features[neutral, 6:21]).sum(axis=1)
        self.start = int(neutral[rest_cost.argmin()])

    def projection_inputs(self, query, command, pose_query):
        return np.concatenate(((query*self.scale).astype(np.float32),
                               command.astype(np.float32), pose_query.astype(np.float32)), axis=-1)

    def projection_keys(self):
        return np.column_stack((self.normalized, self.candidate_intent, self.candidate_pose))

    def projection_signature(self):
        digest = hashlib.sha256()
        for array in (self.projection_keys(), self.candidates, self.clip[self.candidates], INTENT_MODES):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(repr((self.names, self.fps, self.ignore_neighbor)).encode())
        return digest.hexdigest()

    def shortlist(self, query, command, pose_query, mode):
        return self.projector.propose(self.projection_inputs(query, command, pose_query), mode)

    def eligibility(self, command, current):
        return None, True

    def is_straight_run(self, command):
        vx, vy, wz = command
        return len(self.running_frames) > 0 and vx >= 1.85 and abs(vy) <= .12 and abs(wz) <= .18

    def search_running(self, query, current, force):
        # Continue the visible gait phase using the source frame's pose features.
        # Only the trajectory in the query comes from the live command.
        target = np.r_[self.features[current, :15], query[15:]]*self.running_scale
        error = self.running_features-target
        cost = np.einsum("ij,ij->i", error, error)
        frames = self.running_frames
        if not force:
            nearby = (self.clip[frames] == self.clip[current]) & (np.abs(frames-current) < round(self.fps/3.))
            cost[nearby] = np.inf
        best = int(cost.argmin())
        if not np.isfinite(cost[best]):
            return None
        current_cost = np.square(self.features[current]*self.running_scale-target).sum()
        if not force and self.high_step_clips[self.clip[current]] and cost[best]+3. >= current_cost:
            return current
        return int(frames[best])

    def search(self, query, command, pose_query, current, force, command_changed):
        if self.is_straight_run(command):
            match = self.search_running(query, current, force)
            if match is not None:
                return match
        mode = int(intent_mode(command))
        intent_scale = INTENT_MODES[mode]
        current_cost = (np.square((self.features[current]-query)*self.scale).sum()
                        + np.square((self.intent[current]-command)*intent_scale).sum()
                        + np.square(self.pose_features[current]-pose_query).sum())
        allowed, keep_current = self.eligibility(command, current)
        if self.projector is None:
            return self.pick(query, command, pose_query, current, force, command_changed, current_cost, intent_scale,
                             slice(None), allowed, keep_current)
        shortlist = self.shortlist(query, command, pose_query, mode)
        picked = self.pick(query, command, pose_query, current, force, command_changed, current_cost, intent_scale,
                           shortlist, allowed, keep_current)
        if picked is not None:
            return picked
        return self.pick(query, command, pose_query, current, force, command_changed, current_cost, intent_scale,
                         slice(None), allowed, keep_current)

    def pick(self, query, command, pose_query, current, force, command_changed, current_cost, intent_scale,
             pool, allowed, keep_current):
        frames = self.candidates[pool]
        difference = self.normalized[pool] - (query*self.scale).astype(np.float32)
        cost = np.einsum("ij,ij->i", difference, difference)
        # A long-horizon match can turn out and back. Prefer candidates moving
        # as requested immediately as well, especially for straight walking.
        intent_error = (self.candidate_intent[pool] - command.astype(np.float32))*intent_scale
        cost += np.einsum("ij,ij->i", intent_error, intent_error)
        pose_error = self.candidate_pose[pool] - pose_query
        cost += np.einsum("ij,ij->i", pose_error, pose_error)
        if not force:
            nearby = ((self.clip[frames] == self.clip[current])
                      & (np.abs(frames-current) <= self.ignore_neighbor))
            cost[nearby] = np.inf
        if allowed is not None:
            cost[~allowed[pool]] = np.inf
        picked = int(cost.argmin())
        if not np.isfinite(cost[picked]):
            return None if not isinstance(pool, slice) else current
        margin = 1.2 if command_changed else max(6., .32*current_cost)
        if not force and keep_current and cost[picked] + margin >= current_cost:
            return current
        return int(frames[picked])


class Player:
    def __init__(self, robot, db, fps, *, source_pose=None, stepper=None, simulation_root=None,
                 stabilize_transitions=False):
        self.robot, self.db = robot, db
        self.source_pose = db.pose if source_pose is None else source_pose
        self.stepper = stepper
        self.simulation_root = simulation_root
        self.dt = 1/fps
        self.cadence = max(1, round(.10*fps))
        transition, turn_transition = (.47, .31) if stabilize_transitions else (.40, .27)
        self.minimum_transition = max(1, round(transition*fps))
        self.minimum_turn_transition = max(1, round(turn_transition*fps))
        self.minimum_run_transition = max(1, round(.30*fps))
        self.command_interrupt_after = .20 if stabilize_transitions else 0.
        self.reset()

    def reset(self):
        self.frame = self.db.start
        if self.simulation_root is not None:
            self.simulation_root.reset()
        self.yaw, self.shift = 0., np.zeros(2)
        origin = self.robot.model.qpos0.copy()
        origin[self.robot.root:self.robot.root+7] = [0., 0., 0., 1., 0., 0., 0.]
        if self.stepper is not None:
            self.stepper.seed(self.frame)
        self.yaw, self.shift = self.robot.align(self.raw_pose(self.frame), origin)
        self.pose = self.aligned(self.frame)
        if self.simulation_root is not None:
            self.robot.clamp_joint_pose(self.pose)
        self.offset = np.zeros(self.robot.model.nv)
        self.offset_rate = np.zeros_like(self.offset)
        self.blend_rate = np.full_like(self.offset, 10.)
        self.velocity = np.zeros_like(self.offset)
        self.command_at_search = np.zeros(3)
        self.until_search = 0
        self.age = self.minimum_transition*self.dt
        self.switches = 0
        self.stepper_reanchors = 0
        self.robot.data.qpos[:] = self.pose
        mujoco.mj_kinematics(self.robot.model, self.robot.data)
        self.feet = self.robot.foot_positions()
        self.foot_velocity = np.zeros_like(self.feet)

    def raw_pose(self, index):
        return self.source_pose[index] if self.stepper is None else self.stepper.pose(index)

    @property
    def clip_name(self):
        return self.db.names[self.db.clip[self.frame]]

    def aligned(self, index):
        return self.robot.apply_alignment(self.raw_pose(index), self.yaw, self.shift)

    def blend_from(self, frame, turning, turn_changed):
        model = self.robot.model
        target = self.aligned(frame)
        following = self.aligned(frame+1)
        mujoco.mj_differentiatePos(model, self.offset, 1., target, self.pose)
        target_rate = np.empty_like(self.velocity)
        mujoco.mj_differentiatePos(model, target_rate, self.dt, target, following)
        self.offset_rate = self.velocity - target_rate
        # During a turn change, release the old yaw velocity sooner.
        self.blend_rate[5] = 19. if turn_changed else 13. if turning else 10.
        self.frame = frame
        self.age = 0.

    def trajectories(self, command, heading_target):
        root = self.pose[self.robot.root:self.robot.root+7]
        yaw = heading(root[3:7])
        desired, desired_facing = (self.simulation_root.query(command, root[:2], yaw, heading_target)
                                   if self.simulation_root is not None else command_trajectory(command))
        matched = self.db.features[self.frame, 15:21].reshape(3, 2)
        matched_facing = self.db.features[self.frame, 21:27].reshape(3, 2)
        return [(root[:2]+rotate(path, yaw), rotate(facing, yaw))
                for path, facing in ((desired, desired_facing), (matched, matched_facing))]

    def step(self, command, heading_target=None):
        model, data = self.robot.model, self.robot.data
        root = self.pose[self.robot.root:self.robot.root+7]
        yaw = heading(root[3:7])
        turning = abs(command[2]) >= TURN_THRESHOLD
        turn_direction = np.sign(command[2]) if turning else 0
        old_direction = (np.sign(self.command_at_search[2])
                         if abs(self.command_at_search[2]) >= TURN_THRESHOLD else 0)
        turn_changed = turn_direction != old_direction
        best = self.frame
        tired = self.frame + 1 >= self.db.end[self.frame]
        changed = turn_changed or np.linalg.norm(command-self.command_at_search) >= .20
        if self.until_search <= 0 or (changed and self.age >= self.command_interrupt_after) or tired:
            if self.simulation_root is None:
                displacement, facing = command_trajectory(command)
            else:
                displacement, facing = self.simulation_root.query(command, root[:2], yaw, heading_target)
            query = np.r_[local(self.feet-root[:3], yaw).ravel(), local(self.foot_velocity, yaw).ravel(),
                          local(self.velocity[:3], yaw), displacement.ravel(), facing.ravel()]
            stance = planted(self.feet, self.foot_velocity)
            pose_query = np.r_[self.pose[self.robot.joint_qpos]*POSTURE_WEIGHT, stance*STANCE_WEIGHT].astype(np.float32)
            best = self.db.search(query, command, pose_query, self.frame, tired, changed)
            hold = self.minimum_turn_transition if turning else self.minimum_transition
            if self.db.is_straight_run(command):
                hold = self.minimum_run_transition
            self.until_search = hold if best != self.frame else self.cadence
            self.command_at_search = command.copy()
        if best != self.frame:
            if self.stepper is not None:
                self.stepper.seed(best)
            self.yaw, self.shift = self.robot.align(self.raw_pose(best), self.pose)
            stance = planted(self.feet, self.foot_velocity)
            shared = stance * self.db.stance[best]
            if shared.sum() > .25:
                # Align a foot likely planted in both poses before blending.
                candidate_feet = root[:2] + rotate(self.db.features[best, :6].reshape(2, 3)[:, :2], yaw)
                error = ((self.feet[:, :2]-candidate_feet)*shared[:, None]).sum(0)/shared.sum()
                self.shift += error * min(1., .045/max(np.linalg.norm(error), 1e-8))
            self.blend_from(best, turning, turn_changed)
            self.switches += 1
        if self.stepper is not None and self.frame == self.stepper.last:
            self.stepper.seed(self.frame)
            if self.stepper.decoded_root:
                self.yaw, self.shift = self.robot.align(self.raw_pose(self.frame), self.pose)
            self.blend_from(self.frame, turning, False)
            self.stepper_reanchors += 1
        self.until_search -= 1
        self.frame += 1
        self.age += self.dt
        # Critically damped inertialization preserves pose and approximate velocity at a search jump.
        rate = self.blend_rate
        correction = (self.offset + self.age*(self.offset_rate + rate*self.offset))*np.exp(-rate*self.age)
        pose = self.aligned(self.frame)
        mujoco.mj_integratePos(model, pose, correction, 1.)
        if self.simulation_root is not None:
            self.simulation_root.correct(self, pose, command, heading_target)
            self.robot.clamp_joint_pose(pose)
        mujoco.mj_differentiatePos(model, self.velocity, self.dt, self.pose, pose)
        self.pose = pose
        data.qpos[:] = pose
        data.qvel[:] = self.velocity
        data.time += self.dt
        mujoco.mj_forward(model, data)
        feet = self.robot.foot_positions()
        self.foot_velocity = (feet-self.feet)/self.dt
        self.feet = feet


def draw_trajectories(scene, origin, paths):
    scene.ngeom = 0
    colors = ([1., .52, .13, 1.], [.08, .78, 1., 1.])
    identity = np.eye(3).ravel()
    zeros = np.zeros(3)
    for height, (points, facing), color in zip((.025, .035), paths, colors):
        prev = np.r_[origin, height]
        for xy, direction in zip(points, facing):
            if scene.ngeom+3 > scene.maxgeom:
                return
            point = np.r_[xy, height]
            if np.linalg.norm(point-prev) > 1e-4:
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, zeros, zeros, identity, color)
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .009, prev, point)
                scene.ngeom += 1
            marker = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(marker, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3,.018), point, identity, color)
            arrow = scene.geoms[scene.ngeom+1]
            mujoco.mjv_initGeom(arrow, mujoco.mjtGeom.mjGEOM_ARROW, zeros, zeros, identity, color)
            mujoco.mjv_connector(arrow, mujoco.mjtGeom.mjGEOM_ARROW, .009, point, point+np.r_[direction*.13, 0.])
            scene.ngeom += 2
            prev = point


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Use play_motion_matching.sh --s17-search [--upstream-roban PATH] for C++ motion playback in this viewer.",
    )
    parser.add_argument("--data", type=Path, default=Path.home()/"roban数据/lafan1")
    parser.add_argument("--clips", nargs="+", default=["walk*.csv", "run*.csv"], help="Quoted CSV glob patterns")
    parser.add_argument("--fps", type=float, help="Source frame rate for CSVs (default: 30)")
    parser.add_argument("--upstream-roban", type=Path, nargs="?",
                        const=Path("logs/motion_matching_upstream_roban_001"),
                        help="Run classic search on the mapped original idle, walk, and run clips")
    parser.add_argument("--upstream-baseline", action="store_true",
                        help="Replay the mapped corpus with the original transition and root correction settings")
    parser.add_argument("--s17-search-library", type=Path, nargs=3, help=argparse.SUPPRESS)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--gamepad", action="store_true")
    parser.add_argument("--strafe", action="store_true", help="Use body-relative movement and independent turning")
    parser.add_argument("--hold-heading", action="store_true", help="Keep the current heading in C++ playback")
    source.add_argument("--velocity", type=float, nargs=3, metavar=("VX", "VY", "WZ"))
    parser.add_argument("--projector", type=Path, help="Use a trained network to shortlist motion matches")
    parser.add_argument("--s17-aligned", action="store_true", help="Use S17 selections, mirroring and simulation-root correction")
    parser.add_argument("--s17-selection", type=Path,
                        default=Path(os.environ.get("S17_REFERENCE_SOURCE", str(Path.home()/"Motion-Matching-main")))
                        / "resources/s17_lafan1_selection.txt")
    parser.add_argument("--codec", type=Path, help="Play poses reconstructed by the codec in this run directory")
    parser.add_argument("--stepper", type=Path, help="Advance the selected motion with a trained Stepper")
    parser.add_argument("--decoded-root", action="store_true", help="Integrate decoded planar root motion")
    parser.add_argument("--gamepad-library")
    parser.add_argument("--gamepad-wait", type=float, default=10., help="Seconds to wait for a gamepad at startup")
    parser.add_argument("--limits", type=float, nargs=3, metavar=("VX", "VY", "WZ"),
                        help="Gamepad limits; defaults to .9 1.15 2 for upstream, 1 .65 1.5 for strafe, or 1.6 .65 3")
    parser.add_argument("--run-speed", type=float, default=2.2, help="Forward speed limit in the run gear")
    parser.add_argument("--deadzone", type=float)
    parser.add_argument("--switch", action="append", type=float, nargs=4, default=[], metavar=("TIME", "VX", "VY", "WZ"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=0., help="Zero runs until the viewer is closed")
    args = parser.parse_args()
    if args.upstream_baseline and args.upstream_roban is None:
        parser.error("--upstream-baseline requires --upstream-roban")
    if args.hold_heading and not args.s17_search_library:
        parser.error("--hold-heading requires --s17-search")
    if args.upstream_roban is not None:
        if args.fps is not None or args.s17_aligned or args.projector or args.codec or args.stepper:
            parser.error("--upstream-roban uses the mapped source rate and classic search")
        try:
            args.fps = float(json.loads((args.upstream_roban/"metadata.json").read_text())["sample_hz"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(f"Cannot load mapped Motion Matching rate: {error}")
    elif args.fps is None:
        args.fps = 30.
    stabilize_transitions = args.upstream_roban is not None and not args.upstream_baseline
    show_trajectories = args.s17_aligned or args.upstream_roban is not None
    use_s17_stick = args.upstream_roban is not None and args.gamepad and not args.strafe
    if use_s17_stick:
        args.limits = args.limits or [.9, 1.15, 2.]
    else:
        args.limits = args.limits or ([1., .65, 1.5] if args.strafe else [1.6, .65, 3.])
    if args.deadzone is None:
        args.deadzone = .20 if use_s17_stick else .18
    if args.s17_search_library:
        if args.upstream_roban is None or args.upstream_baseline or args.strafe:
            parser.error("C++ search requires mapped RoBan clips and the default gamepad layout")
        if args.gamepad and (args.limits != [.9, 1.15, 2.] or args.deadzone != .20 or args.run_speed != 2.2):
            parser.error("C++ gamepad search uses main's built-in limits and deadzone")
    if (not np.isfinite([args.fps, args.seconds, args.gamepad_wait, args.run_speed, *args.limits, *(args.velocity or [])]).all()
            or args.fps < 10 or args.seconds < 0 or args.gamepad_wait < 0 or args.run_speed <= 0
            or not 0 <= args.deadzone < 1 or min(args.limits) <= 0):
        parser.error("Invalid frame rate, duration, velocity, gamepad limits or deadzone")
    if args.decoded_root and not args.stepper:
        parser.error("--decoded-root requires --stepper")
    if args.stepper and not args.codec:
        parser.error("--stepper requires its training --codec directory")
    if args.headless and args.seconds <= 0:
        parser.error("--headless requires a positive --seconds")
    if (args.gamepad and args.switch) or ((args.gamepad_library or args.strafe) and not args.gamepad):
        parser.error("Use --switch for scripted playback; --strafe and --gamepad-library require --gamepad")
    switches = [(round(t*args.fps), np.asarray(v)) for t, *v in args.switch
                if np.isfinite([t, *v]).all() and t > 0]
    frames = [t for t, _ in switches]
    if len(switches) != len(args.switch) or frames != sorted(set(frames)) or 0 in frames:
        parser.error("Switches must be finite and in strictly increasing playback frames")
    os.environ["ACP_ROBOT"] = "roban"
    gamepad = viewer = native_playback = None
    viewer_threads = []
    paths_visible = threading.Event()
    if show_trajectories:
        paths_visible.set()
    try:
        if args.gamepad:
            from .gamepad import Gamepad
            if args.gamepad_wait:
                print(f"正在查找手柄，最多等待 {args.gamepad_wait:g} 秒。", flush=True)
            gamepad = Gamepad(args.gamepad_library, wait=args.gamepad_wait)
            if args.s17_search_library:
                mapping = ("左摇杆给速度，沿机身朝向移动，右摇杆指定航向，LT 锁定朝向并允许后退或横移，"
                           "RT + 右摇杆转镜头，A 慢走，LB/RB 缩放")
            elif use_s17_stick:
                mapping = "左摇杆前后/侧移，右摇杆转向，A 慢走，RT + 右摇杆转镜头，LB/RB 缩放"
            else:
                mapping = ("左摇杆前后/侧移，右摇杆左右转向" if args.strafe else
                           "左摇杆选方向，机器人转过去向前走，右摇杆转镜头")
                mapping += ("，RB 跑步" if show_trajectories else "") + "，A 重置"
            print(f"已连接 {gamepad.name}。{mapping}。", flush=True)
        robot = Robot()
        if args.s17_search_library:
            from .native_search import NativePlayback
            clips = [(upstream_clip_name(clip), clip["frames"]) for clip in upstream_clips(args.upstream_roban)]
            native_playback = NativePlayback(*args.s17_search_library, robot=robot, clips=clips)
            base = None
            print("C++ S17 motion playback; Python input and viewer", flush=True)
        elif args.upstream_roban is not None:
            base = Database.from_upstream_preview(robot, args.upstream_roban, args.fps,
                                                  high_step_running=not args.upstream_baseline)
            print("Classic search on the mapped original Motion Matching ranges", flush=True)
        else:
            base = Database(robot, args.data, args.fps, args.clips)
        db = base
        simulation_root = None
        s17_stick = None
        if show_trajectories:
            from .s17_alignment import S17Stick, SimulationRoot, SupportAwareSimulationRoot
            root_class = SupportAwareSimulationRoot if stabilize_transitions else SimulationRoot
            if use_s17_stick:
                s17_stick = S17Stick(args.limits, args.run_speed, args.deadzone, 1./args.fps)
            if native_playback is None:
                simulation_root = root_class(robot, 1./args.fps,
                                             velocity_source=s17_stick.world_velocity if s17_stick else None)
        if args.s17_aligned:
            from .s17_alignment import AlignedDatabase, MirroredStepper
            db = AlignedDatabase(base, robot, args.data, args.s17_selection)
            print(f"S17-aligned playback, {db.sections} selected sections with left/right mirrors", flush=True)
        if args.projector:
            import torch
            from .motion_projector import MotionProjector
            torch.set_num_threads(2)
            db.projector = MotionProjector(args.projector, base, pool=db.curated_columns if args.s17_aligned else None)
            print(f"Learned Projector, {db.projector.k} candidates per query", flush=True)
        reconstructed = stepper = None
        if args.codec:
            import torch
            torch.set_num_threads(2)
            if args.stepper:
                from .motion_stepper import StepperPlayback
                stepper = StepperPlayback(args.stepper, args.codec, robot, base, decoded_root=args.decoded_root)
                if args.s17_aligned:
                    stepper = MirroredStepper(stepper, db)
                root_source = "decoded root velocities" if args.decoded_root else "recorded root path"
                print(f"Stepper playback, up to {stepper.window-1} predicted frames per seed, {root_source}", flush=True)
            else:
                from .motion_codec import decode_catalogue
                reconstructed = decode_catalogue(args.codec, robot, base)
                if args.s17_aligned:
                    reconstructed = np.concatenate((reconstructed, db.mirror.poses(reconstructed)))
                print(f"Playing reconstructed poses from {args.codec}", flush=True)
        if native_playback is not None:
            player = native_playback
            library_info = f"Loaded {len(clips)} clips, {player.ends[-1]} frames at {args.fps:g} Hz"
        else:
            player = Player(robot, db, args.fps, source_pose=reconstructed, stepper=stepper,
                            simulation_root=simulation_root, stabilize_transitions=stabilize_transitions)
            library_info = (f"Loaded {len(db.names)} clips, {len(db.pose)} frames at {args.fps:g} Hz, "
                            f"{len(db.candidates)} search candidates, {len(db.still_candidates)} standing frames")
        print(library_info, flush=True)
        render_data = None
        if not args.headless:
            import glfw
            from mujoco import viewer as mj_viewer

            def key_callback(key):
                if key == glfw.KEY_F1 and show_trajectories:
                    if paths_visible.is_set():
                        paths_visible.clear()
                    else:
                        paths_visible.set()

            before_launch = set(threading.enumerate())
            render_data = mujoco.MjData(robot.model) if args.s17_aligned else robot.data
            viewer = mj_viewer.launch_passive(robot.model, render_data, key_callback=key_callback)
            viewer_threads = [thread for thread in threading.enumerate() if thread not in before_launch]
            with viewer.lock():
                viewer.cam.distance = 2.5
                if native_playback is not None:
                    viewer.cam.azimuth = -25.
                    viewer.cam.distance = 3.2
                elif args.gamepad and not args.strafe:
                    viewer.cam.azimuth = 0.
                viewer.cam.elevation = -19
                viewer.opt.geomgroup[3] = 0
        elapsed = count = 0
        segment_velocity = np.zeros(3)
        segment_start = player.switches
        segment_reanchors = player.stepper_reanchors
        command = np.asarray(args.velocity or [0., 0., 0.])
        schedule = dict(switches)
        limit = math.ceil(args.seconds*args.fps)
        limits = np.asarray(args.limits)
        heading_stick = HeadingStick(limits, args.deadzone)
        camera_azimuth, camera_elevation = (-25. if native_playback is not None else 0.), -19.

        def report():
            if count:
                shown_command = native_playback.command if native_playback is not None else command
                info = dict(command=shown_command.tolist(), mean_velocity=(segment_velocity/count).tolist(),
                            steps=count, matches=player.switches-segment_start,
                            clip=player.clip_name)
                if stepper is not None:
                    info["stepper_reanchors"] = player.stepper_reanchors-segment_reanchors
                print(json.dumps(info), flush=True)

        while (not limit or elapsed < limit) and (viewer is None or viewer.is_running()):
            start = time.monotonic()
            heading_target = None
            native_input = None
            hold_heading = args.hold_heading
            running = False
            if elapsed in schedule:
                report()
                command = schedule[elapsed]
                segment_velocity[:] = 0.
                count, segment_start = 0, player.switches
                segment_reanchors = player.stepper_reanchors
            if gamepad:
                axes, reset = gamepad.read(full_right=not args.strafe)
                if s17_stick is not None:
                    walk, orbit, zoom = gamepad.motion_matching_buttons()
                    running = not walk
                    if native_playback is not None:
                        hold_heading = hold_heading or gamepad.hold_heading_pressed()
                    else:
                        body_yaw = heading(player.pose[robot.root+3:robot.root+7])
                        command, heading_target = s17_stick.update(axes, walk, orbit, body_yaw, simulation_root)
                    if viewer is not None and (orbit or zoom or native_playback is not None):
                        with viewer.lock():
                            if orbit:
                                viewer.cam.azimuth, viewer.cam.elevation = s17_stick.orbit(
                                    axes, viewer.cam.azimuth, viewer.cam.elevation)
                            viewer.cam.distance = float(np.clip(viewer.cam.distance+10.*zoom*player.dt, .1, 100.))
                            camera_azimuth = viewer.cam.azimuth
                    elif viewer is None and orbit:
                        camera_azimuth, camera_elevation = s17_stick.orbit(axes, camera_azimuth, camera_elevation)
                    if native_playback is not None:
                        native_input = (axes, walk, orbit, np.deg2rad(camera_azimuth))
                else:
                    running = show_trajectories and gamepad.run_pressed()
                if reset and s17_stick is None:
                    player.reset()
                    heading_stick.reset()
                    segment_velocity[:] = 0.
                    count, segment_start, segment_reanchors = 0, 0, 0
                if args.strafe:
                    active_limits = [max(limits[0], args.run_speed), limits[1], limits[2]] if running else limits
                    command = gamepad_command(axes, active_limits, args.deadzone)
                elif s17_stick is None:
                    if viewer is None:
                        camera_azimuth, camera_elevation = orbit_camera(
                            axes, camera_azimuth, camera_elevation, args.deadzone, player.dt)
                    else:
                        with viewer.lock():
                            camera_azimuth, camera_elevation = orbit_camera(
                                axes, viewer.cam.azimuth, viewer.cam.elevation, args.deadzone, player.dt)
                            viewer.cam.azimuth = camera_azimuth
                            viewer.cam.elevation = camera_elevation
                    body_yaw = heading(player.pose[robot.root+3:robot.root+7])
                    command = heading_stick.update(axes, np.deg2rad(camera_azimuth), body_yaw,
                                                   run_speed=args.run_speed if running else None)
                    heading_target = body_yaw if heading_stick.target_yaw is None else heading_stick.target_yaw
            step_args = ({"gamepad_input": native_input, "hold_heading": hold_heading}
                         if native_playback is not None else {"heading_target": heading_target})
            if viewer is None:
                player.step(command, **step_args)
            else:
                previous = player.pose
                with viewer.lock():
                    player.step(command, **step_args)
                if elapsed % player.cadence == 0:
                    shown_command = native_playback.command if native_playback is not None else command
                    cmd = "  ".join(f"{value:+.2f}" for value in shown_command)
                    velocity_label = "World vx / vy / yaw rate" if native_playback is not None else "vx / vy / yaw rate"
                    info_keys = f"{velocity_label}\nMatched clip"
                    info_values = f"{cmd}\n{player.clip_name}"
                    if native_playback is not None and gamepad is not None:
                        info_keys = "World vx / vy\nTarget heading\nMatched clip"
                        vx, vy = shown_command[:2]
                        info_values = f"{vx:+.2f}  {vy:+.2f}\n{np.rad2deg(native_playback.target_heading):+.0f} deg\n{player.clip_name}"
                    if show_trajectories:
                        info_keys += "\nGait\nF1 Trajectories"
                        is_running = running if s17_stick is not None else running or np.linalg.norm(shown_command[:2]) >= 1.85
                        info_values += f"\n{'Run' if is_running else 'Walk'}"
                        info_values += "\nOrange: predicted   Cyan: matched" if paths_visible.is_set() else "\nHidden"
                    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, None, info_keys, info_values))
                if args.s17_aligned:
                    # Render the genuine half-step pose without changing the 30 Hz learned model cadence.
                    with viewer.lock():
                        render_data.qpos[:] = previous
                        mujoco.mj_integratePos(robot.model, render_data.qpos, player.velocity, .5*player.dt)
                        render_data.time = robot.data.time-.5*player.dt
                        mujoco.mj_forward(robot.model, render_data)
                        viewer.cam.lookat[:] = render_data.qpos[robot.root:robot.root+3]
                if show_trajectories:
                    with viewer.lock():
                        if paths_visible.is_set():
                            origin = (native_playback.trajectory_origin if native_playback is not None
                                      else player.pose[robot.root:robot.root+2])
                            draw_trajectories(viewer.user_scn, origin, player.trajectories(command, heading_target))
                        else:
                            viewer.user_scn.ngeom = 0
                if args.s17_aligned:
                    viewer.sync()
                    time.sleep(max(0., .5*player.dt-(time.monotonic()-start)))
                    with viewer.lock():
                        render_data.qpos[:] = player.pose
                        render_data.time = robot.data.time
                        mujoco.mj_forward(robot.model, render_data)
                with viewer.lock():
                    viewer.cam.lookat[:] = player.pose[robot.root:robot.root+3]
                    if native_playback is not None:
                        viewer.cam.lookat[2] = .76
                viewer.sync()
            yaw = heading(player.pose[robot.root+3:robot.root+7])
            world_rotation_rate = robot.data.xmat[robot.model.jnt_bodyid[robot.free]].reshape(3, 3) @ player.velocity[3:6]
            segment_velocity += np.r_[rotate(player.velocity[:2], -yaw), world_rotation_rate[2]]
            count += 1
            elapsed += 1
            if viewer is not None or gamepad is not None:
                time.sleep(max(0., player.dt-(time.monotonic()-start)))
        report()
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    finally:
        if viewer is not None:
            viewer.close()
            # Let the render thread release GL before GLFW's atexit handler.
            for thread in viewer_threads:
                thread.join(timeout=5.)
        if gamepad is not None:
            gamepad.close()
        if native_playback is not None:
            native_playback.close()


if __name__ == "__main__":
    main()
