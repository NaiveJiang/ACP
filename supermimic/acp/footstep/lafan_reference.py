"""Convert retargeted RoBan LAFAN1 CSVs to 50 Hz kinematic contact references."""
import argparse
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter, minimum_filter1d, uniform_filter1d
from scipy.spatial.transform import Rotation, Slerp

from .data import KINEMATIC_REFERENCE
from .motion_matching import Database, FOOT_OFFSET, Robot, heading, rotate

FPS = 50
MIN_SUPPORT = 5
STRAIGHT_WALK = dict(forward_min_mps=.18, forward_max_mps=1.35, lateral_base_mps=.065,
                     lateral_ratio=.30, yaw_rate_max_rad_s=.42, yaw_range_2s_max_rad=.60,
                     segment_heading_range_max_rad=.42, speed_change_05s_max_mps=.35,
                     flight_fraction_max=.10, touchdown_alternation_min=.75)


def intervals(mask):
    return np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)


def straight_walk_intervals(pose, root, fps, playable):
    """Find steady forward walking while retaining normal stride-scale oscillations."""
    rule = STRAIGHT_WALK
    samples = lambda duration: max(3, 2*round(duration*fps/2)+1)
    yaw = uniform_filter1d(np.unwrap(heading(pose[:, root+3:root+7])), samples(.13), mode="nearest")
    velocity = np.gradient(pose[:, root:root+2], 1./fps, axis=0)
    world_velocity = uniform_filter1d(velocity, samples(1.07), axis=0, mode="nearest")
    vx, vy = rotate(world_velocity, -yaw).T
    yaw_rate = uniform_filter1d(np.gradient(yaw, 1./fps), samples(1.07), mode="nearest")
    yaw_range = (maximum_filter1d(yaw, samples(2.)) - minimum_filter1d(yaw, samples(2.)))
    speed_range = (maximum_filter1d(vx, samples(.46)) - minimum_filter1d(vx, samples(.46)))
    steady = (playable & (vx >= rule["forward_min_mps"]) & (vx <= rule["forward_max_mps"])
              & (np.abs(vy) <= rule["lateral_base_mps"]+rule["lateral_ratio"]*vx)
              & (np.abs(yaw_rate) <= rule["yaw_rate_max_rad_s"])
              & (yaw_range <= rule["yaw_range_2s_max_rad"])
              & (speed_range <= rule["speed_change_05s_max_mps"]))

    route_heading = uniform_filter1d(yaw, samples(.65), mode="nearest")
    for start, stop in intervals(steady):
        begin = start
        low = high = route_heading[start]
        for i in range(start+1, stop):
            y = route_heading[i]
            low, high = min(low, y), max(high, y)
            if high-low > rule["segment_heading_range_max_rad"]:
                yield begin, i
                begin = i
                low = high = y
        yield begin, stop


def walking_contacts(contact):
    onset = contact[1:] & ~contact[:-1]
    foot = onset[onset.sum(axis=1) == 1].argmax(axis=1)
    alternation = float(np.mean(foot[1:] != foot[:-1])) if len(foot) > 1 else 0.
    valid = (onset.sum(axis=0).min() >= 2 and alternation >= STRAIGHT_WALK["touchdown_alternation_min"]
             and (~contact.any(-1)).mean() <= STRAIGHT_WALK["flight_fraction_max"])
    return bool(valid), alternation


def resample(pose, root, fps):
    source_time = np.arange(len(pose))/fps
    time = np.arange(math.floor(source_time[-1]*FPS + 1e-7)+1)/FPS
    out = np.stack([np.interp(time, source_time, x) for x in pose.T], -1)
    rotation = Rotation.from_quat(pose[:, root+3:root+7], scalar_first=True)
    out[:, root+3:root+7] = Slerp(source_time, rotation)(time).as_quat(scalar_first=True)
    return time, out


def support_from_gap(gap, feet):
    speed = np.linalg.norm(np.gradient(feet[..., :2], 1./FPS, axis=0), axis=-1)
    touch = np.zeros(gap.shape, dtype=bool)
    state = np.zeros(2, dtype=bool)
    for i in range(len(gap)):
        state = np.where(state, (gap[i] < .034) & (speed[i] < 1.05),
                         (gap[i] < .018) & (speed[i] < .65))
        touch[i] = state
    touch = median_filter(touch, size=(3, 1), mode="nearest")
    for foot in range(2):
        for start, end in intervals(touch[:, foot]):
            if end-start < MIN_SUPPORT:
                touch[start:end, foot] = False
    return touch


class ReferenceGeometry:
    def __init__(self, robot):
        from ..roban import action_scale, default_joint_pos_list

        self.robot = robot
        model = robot.model
        arms = [i for i, name in enumerate(robot.joints) if name.startswith("zarm_")]
        self.arm_columns = robot.joint_qpos[arms]
        joints = [model.joint(robot.joints[i]).id for i in arms]
        self.arm_home = np.asarray(default_joint_pos_list())[arms]
        alpha = np.array([action_scale(robot.joints[i]) for i in arms])
        limits = model.jnt_range[joints]
        mid = limits.mean(axis=1)
        radius = np.diff(limits, axis=1)[:, 0]*.475
        self.arm_lo = np.maximum(mid-radius+.02, self.arm_home-4.7*alpha)
        self.arm_hi = np.minimum(mid+radius-.02, self.arm_home+4.7*alpha)
        self.geoms = []
        for body in robot.feet:
            active = np.flatnonzero((model.geom_bodyid == body) & (model.geom_conaffinity != 0))
            if not len(active):
                raise ValueError("Missing RoBan foot collision geometry")
            kinds = model.geom_type[active]
            box = kinds == mujoco.mjtGeom.mjGEOM_BOX
            cylinder = kinds == mujoco.mjtGeom.mjGEOM_CYLINDER
            if not np.all(box | cylinder):
                raise ValueError("Unsupported RoBan foot collision shape")
            self.geoms.append((active, box, model.geom_size[active]))

    def measure(self, pose):
        robot = self.robot
        data, model = robot.data, robot.model
        feet = np.empty((len(pose), 2, 7))
        gap = np.empty((len(pose), 2))
        for i, q in enumerate(pose):
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            feet[i, :, :3] = robot.foot_positions()
            feet[i, :, 3:] = data.xquat[robot.feet]
            for j, (geom, box, size) in enumerate(self.geoms):
                world_z = np.abs(data.geom_xmat[geom].reshape(-1, 3, 3)[:, 2])
                extension = np.where(box, (world_z*size).sum(-1),
                                     world_z[:, 2]*size[:, 1]+np.sqrt(np.maximum(0., 1.-world_z[:, 2]**2))*size[:, 0])
                gap[i, j] = (data.geom_xpos[geom, 2]-extension).min()
        return feet, gap

    def clip(self, pose, fps):
        robot = self.robot
        time, pose = resample(pose, robot.root, fps)
        # Keep small arm motions intact and smoothly compress the larger excursions
        # into the actual PD target range of the existing contact controller.
        delta = pose[:, self.arm_columns]-self.arm_home
        distance = np.where(delta >= 0., self.arm_hi-self.arm_home, self.arm_home-self.arm_lo)
        fraction = np.abs(delta)/distance
        compressed = np.where(fraction <= .7, fraction, .7+.3*np.tanh((fraction-.7)/.3))
        pose[:, self.arm_columns] = self.arm_home+np.sign(delta)*distance*compressed
        feet, gap = self.measure(pose)
        # Offset the whole clip so stance geometry is near the flat floor.
        floor = np.quantile(gap.min(1), .30)
        pose[:, robot.root+2] -= floor
        feet[..., 2] -= floor
        gap -= floor
        contact = support_from_gap(gap, feet)
        velocity = np.empty((len(pose), robot.model.nv))
        for t in range(len(pose)):
            a, b = max(0, t-1), min(len(pose)-1, t+1)
            mujoco.mj_differentiatePos(robot.model, velocity[t], (b-a)/FPS, pose[a], pose[b])
        yaw = heading(pose[:, robot.root+3:robot.root+7])
        angular = Rotation.from_quat(pose[:, robot.root+3:robot.root+7], scalar_first=True).apply(velocity[:, 3:6])
        speed = uniform_filter1d(np.column_stack((rotate(velocity[:, :2], -yaw), angular[:, 2])),
                                 size=9, axis=0, mode="nearest")
        stationary = (np.linalg.norm(speed[:, :2], axis=1) < .06) & (abs(speed[:, 2]) < .10)
        speed[stationary] = 0.
        return dict(sim_time=time, qpos=pose.astype(np.float32), qvel=velocity.astype(np.float32),
                    foot_state=feet.astype(np.float32), contact=contact,
                    velocity_command=speed.astype(np.float32), requested=speed.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path.home()/"roban数据/lafan1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clips", nargs="+")
    parser.add_argument("--straight-walk", action="store_true",
                        help="Keep only steady forward walking with alternating foot contacts")
    parser.add_argument("--source-fps", type=float, default=30.)
    parser.add_argument("--min-seconds", type=float, default=3.)
    parser.add_argument("--validation-subject", type=int)
    args = parser.parse_args()
    if args.validation_subject is None:
        args.validation_subject = 3 if args.straight_walk else 5
    args.clips = args.clips or (["walk*.csv"] if args.straight_walk else ["walk*.csv", "run*.csv"])
    if (not all(math.isfinite(value) for value in (args.source_fps, args.min_seconds))
            or args.source_fps < 10. or args.min_seconds < 3. or args.validation_subject < 1):
        parser.error("Require at least 3 s per clip, a valid source rate, and a positive validation subject")
    if args.out.exists():
        parser.error(f"{args.out} already exists")
    os.environ["ACP_ROBOT"] = "roban"
    robot = Robot()
    model = robot.model
    if robot.root != 0 or model.jnt_dofadr[robot.free] != 0:
        raise ValueError("Expected the RoBan free root at the first MuJoCo address")
    db = Database(robot, args.data, args.source_fps, args.clips)
    geometry = ReferenceGeometry(robot)
    metadata = dict(sample_hz=FPS, source_hz=args.source_fps, sole_offset=FOOT_OFFSET.tolist(),
                    contact_source="kinematic", velocity_source="measured_pose_derivative",
                    arm_reference="smoothly_bounded_pd_target",
                    motion_selection="straight_forward_walk" if args.straight_walk else "upright_walk_run",
                    joints=[dict(name=name, qpos_adr=int(model.jnt_qposadr[model.joint(name).id]),
                                 dof_adr=int(model.jnt_dofadr[model.joint(name).id])) for name in robot.joints])
    spans = []
    frames = np.arange(len(db.pose))
    for i, source in enumerate(db.names):
        if args.straight_walk and not source.startswith("walk"):
            continue
        rows = frames[db.clip == i]
        playable = db.end[rows] > rows
        partition = "val" if source.endswith(f"_subject{args.validation_subject}") else "train"
        sections = (straight_walk_intervals(db.pose[rows], robot.root, args.source_fps, playable)
                    if args.straight_walk else intervals(playable))
        for start, end in sections:
            if (end-start-1)/args.source_fps >= args.min_seconds:
                spans.append((source, rows[0]+start, rows[0]+end, partition, start, end))
    if {item[3] for item in spans} != {"train", "val"}:
        parser.error("Selected clips must include eligible train and validation subjects")

    source_dir = args.out/"reference"
    args.out.mkdir(parents=True, exist_ok=False)
    source_dir.mkdir()
    (args.out/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    reports = []
    for source, start, end, partition, row_start, row_stop in spans:
        data = geometry.clip(db.pose[start:end], args.source_fps)
        case = f"{source}_{start:06d}"
        contact = data["contact"]
        if args.straight_walk:
            keep, alternation = walking_contacts(contact)
            if not keep:
                continue
        starts = contact[1:] & ~contact[:-1]
        report = dict(case=case, category="run" if source.startswith("run") else "walk",
                      partition=partition, raw_motion=source, frames=len(contact),
                      source_row_start=int(row_start), source_row_stop=int(row_stop),
                      touchdowns=starts.sum(0).tolist(), support_fraction=contact.mean(0).tolist(),
                      double_air_fraction=float((~contact.any(-1)).mean()))
        if args.straight_walk:
            report["touchdown_alternation"] = alternation
            report["mean_forward_mps"] = float(data["velocity_command"][:, 0].mean())
        np.savez_compressed(source_dir/(case+"_labels.npz"), **data)
        reports.append(report)
    if {r["partition"] for r in reports} != {"train", "val"}:
        raise ValueError("No walking contacts survived in one split; choose another held-out subject")
    manifest = dict(schema=KINEMATIC_REFERENCE, status="complete", min_support_steps=MIN_SUPPORT,
                    database=db.projection_signature(), validation_subject=args.validation_subject, clips=reports)
    if args.straight_walk:
        manifest["straight_walk_filter"] = STRAIGHT_WALK
        manifest["windows_before_contact_filter"] = len(spans)
    (source_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    for partition in ("train", "val"):
        clips = [c for c in reports if c["partition"] == partition]
        print(f"{partition}: {len(clips)} clips, {sum(c['frames'] for c in clips)} frames, "
              f"{np.sum([c['touchdowns'] for c in clips], axis=0).tolist()} left/right touchdown labels")
    print(source_dir)


if __name__ == "__main__":
    main()
