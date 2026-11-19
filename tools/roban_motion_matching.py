"""Offline RoBan experiment using PHP feature search and inertialization."""

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from roban_sole_alignment import align_ascent_soles
from scipy.spatial.transform import Rotation, Slerp

JOINT_NAMES = (
    ["waist_yaw_joint"]
    + [f"leg_{side}{index}_joint" for side in "lr" for index in range(1, 7)]
    + [f"zarm_{side}{index}_joint" for side in "lr" for index in range(1, 5)]
)
FPS = 50
DT = 1 / FPS


def rotations(quaternion):
    return Rotation.from_quat(np.asarray(quaternion)[..., [1, 2, 3, 0]])


def quaternions(rotation):
    return rotation.as_quat()[..., [3, 0, 1, 2]]


def angular_velocity(quaternion):
    rotation = rotations(quaternion)
    velocity = np.empty((len(quaternion), 3))
    velocity[1:-1] = (rotation[2:] * rotation[:-2].inv()).as_rotvec() / (2 * DT)
    velocity[0] = (rotation[1] * rotation[0].inv()).as_rotvec() / DT
    velocity[-1] = (rotation[-1] * rotation[-2].inv()).as_rotvec() / DT
    return velocity


def load_model(path):
    path = Path(path).resolve()
    root = ET.parse(path).getroot()
    root.find("compiler").set("meshdir", str(path.parent.parent / "meshes"))
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "body" and child.get("name") in {
                "head_camera_base",
                "head_radar",
            }:
                parent.remove(child)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def load_clip(path):
    with np.load(path, allow_pickle=False) as archive:
        legacy = "dof_positions" in archive
        position = archive["body_positions" if legacy else "body_pos_w"][:, 0]
        quaternion = archive["body_rotations" if legacy else "body_quat_w"][:, 0]
        joints = archive["dof_positions" if legacy else "joint_pos"]
        fps = float(np.asarray(archive["fps"]).item())
        if legacy:
            names = archive["dof_names"].tolist()
            joints = joints[:, [names.index(name) for name in JOINT_NAMES]]
        if joints.shape[1] != 21 or fps != FPS:
            raise ValueError(f"Expected 21 joints at 50 Hz: {path}")
        result = np.c_[position, quaternion, joints].astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite input: {path}")
    result[:, 3:7] = quaternions(rotations(result[:, 3:7]))
    return result


def mirror_pose(pose, center_y):
    result = pose.copy()
    result[:, 1] = 2 * center_y - result[:, 1]
    polar = np.array([1, -1, 1])
    matrices = rotations(result[:, 3:7]).as_matrix()
    result[:, 3:7] = quaternions(
        Rotation.from_matrix(matrices * polar[None, :, None] * polar[None, None, :])
    )
    order = [0, *range(7, 13), *range(1, 7), *range(17, 21), *range(13, 17)]
    signs = [-1, *([-1, -1, -1, 1, 1, -1] * 2), *([1, -1, -1, 1] * 2)]
    result[:, 7:] = pose[:, 7:][:, order] * signs
    return result


def forward(model, pose):
    data = mujoco.MjData(model)
    addresses = [model.jnt_qposadr[model.joint(name).id] for name in JOINT_NAMES]
    positions = np.empty((len(pose), model.nbody - 1, 3))
    quaternion = np.empty((len(pose), model.nbody - 1, 4))
    for frame, state in enumerate(pose):
        data.qpos[:7] = state[:7]
        data.qpos[addresses] = state[7:]
        mujoco.mj_kinematics(model, data)
        positions[frame] = data.xpos[1:]
        quaternion[frame] = data.xquat[1:]
    return positions, quaternion


def features(model, pose):
    positions, _ = forward(model, pose)
    heading = rotations(pose[:, 3:7]).as_euler("xyz")[:, 2]
    inverse = Rotation.from_euler("z", -heading)
    feet = [model.body(f"leg_{side}6_link").id - 1 for side in "lr"]
    root_velocity = np.gradient(pose[:, :3], DT, axis=0)
    parts = [inverse.apply(positions[:, foot] - pose[:, :3]) for foot in feet]
    parts += [
        inverse.apply(np.gradient(positions[:, foot], DT, axis=0)) for foot in feet
    ]
    parts.append(inverse.apply(root_velocity))
    future_position, future_heading = [], []
    for offset in (round(FPS / 3), round(2 * FPS / 3), FPS):
        indices = np.minimum(np.arange(len(pose)) + offset, len(pose) - 1)
        future_position.append(inverse.apply(pose[indices, :3] - pose[:, :3])[:, :2])
        angles = heading[indices] - heading
        future_heading.append(np.c_[np.cos(angles), np.sin(angles)])
    return np.concatenate(parts + future_position + future_heading, axis=1)


def match_window(database, query, frames, upstream, reverse=False):
    candidates, labels = [], []
    for path, pose, feature in database:
        indices = (
            np.arange(frames - 1, len(pose) - 1)
            if reverse
            else np.arange(0, len(pose) - frames)
        )
        for index in indices:
            labels.append((path, pose, int(index)))
        candidates.extend(feature[indices])
    if not labels:
        raise ValueError("Walking clips are too short for the requested duration")
    raw = np.asarray(candidates)
    offset = raw.mean(axis=0)
    scale = np.empty(27)
    for start, size, weight in (
        (0, 3, 0.75),
        (3, 3, 0.75),
        (6, 3, 1),
        (9, 3, 1),
        (12, 3, 1),
        (15, 6, 1),
        (21, 6, 1.5),
    ):
        scale[start : start + size] = (
            max(raw[:, start : start + size].std(0).mean(), 0.05) / weight
        )
    normalized = (raw - offset) / scale
    index, cost = upstream.motion_matching_search_fixed(
        -1,
        float("inf"),
        np.array([0]),
        np.array([len(raw)]),
        normalized,
        offset,
        scale,
        (query - offset) / scale,
        ignore_range_end=0,
        ignore_surrounding=0,
        mirror_tie_epsilon=None,
    )
    path, pose, source_frame = labels[index]
    start = source_frame - frames + 1 if reverse else source_frame
    return pose[start : start + frames].copy(), {
        "source": str(path),
        "source_range_exclusive": [start, start + frames],
        "php_feature_cost": float(cost),
    }


def place_walk(walk, target, at_end=False, elevation=0):
    anchor = -1 if at_end else 0
    source_yaw = rotations(walk[anchor, 3:7]).as_euler("xyz")[2]
    target_yaw = rotations(target[3:7]).as_euler("xyz")[2]
    rotation = Rotation.from_euler("z", target_yaw - source_yaw)
    result = walk.copy()
    result[:, :3] = rotation.apply(walk[:, :3] - walk[anchor, :3])
    result[:, :2] += target[:2]
    result[:, 2] += walk[anchor, 2] + elevation
    result[:, 3:7] = quaternions(rotation * rotations(walk[:, 3:7]))
    return result


def inertialize(previous, following, upstream, halflife=0.10):
    result = following.copy()
    source_velocity = (previous[-1] - previous[-2]) / DT
    destination_velocity = (following[1] - following[0]) / DT
    channels = np.r_[0:3, 7:28]
    position_offset = previous[-1, channels] - following[0, channels]
    velocity_offset = source_velocity[channels] - destination_velocity[channels]
    rotation_offset = quaternions(
        rotations(previous[-1, 3:7]) * rotations(following[0, 3:7]).inv()
    )
    angular_offset = (
        (rotations(previous[-1, 3:7]) * rotations(previous[-2, 3:7]).inv()).as_rotvec()
        - (
            rotations(following[1, 3:7]) * rotations(following[0, 3:7]).inv()
        ).as_rotvec()
    ) / DT
    for frame in range(len(result)):
        value = np.zeros(len(channels))
        velocity = np.zeros(len(channels))
        quaternion = np.zeros(4)
        angular = np.zeros(3)
        elapsed = 0 if frame == 0 else DT
        upstream.inertialize_update_vec3(
            value,
            velocity,
            position_offset,
            velocity_offset,
            following[frame, channels],
            destination_velocity[channels],
            halflife,
            elapsed,
        )
        upstream.inertialize_update_quat(
            quaternion,
            angular,
            rotation_offset,
            angular_offset,
            following[frame, 3:7],
            np.zeros(3),
            halflife,
            elapsed,
        )
        result[frame, channels] = value
        result[frame, 3:7] = quaternion / np.linalg.norm(quaternion)
    return result


def join(blocks, upstream):
    result = blocks[0][1].copy()
    stages = [blocks[0][0]] * len(result)
    seams = []
    for name, block in blocks[1:]:
        seams.append(len(result) - 1)
        block = inertialize(result, block, upstream)
        result = np.concatenate((result[:-1], block))
        stages = stages[:-1] + [name] * len(block)
    return result, np.asarray(stages), seams


def turn_approach(walk, angle):
    result = walk.copy()
    fraction = np.linspace(0, 1, len(walk))
    blend = fraction**3 * (10 - 15 * fraction + 6 * fraction**2)
    correction = -angle * (1 - blend)
    yaw = Rotation.from_euler("z", correction)
    steps = np.diff(walk[:, :3], axis=0)
    steps = yaw[1:].apply(steps)
    result[-1, :3] = walk[-1, :3]
    result[:-1, :3] = walk[-1, :3] - np.cumsum(steps[::-1], axis=0)[::-1]
    result[:, 3:7] = quaternions(yaw * rotations(walk[:, 3:7]))
    return result


def audit(model, pose, box, seams):
    positions, _ = forward(model, pose)
    feet = positions[:, [model.body(f"leg_{side}6_link").id - 1 for side in "lr"]]
    rotation = rotations(box[3:7]).inv()
    local = rotation.apply((feet - box[:3]).reshape(-1, 3)).reshape(feet.shape)
    inside = (np.abs(local[..., :2]) <= box[7:9] / 2).all(-1)
    support = np.where(inside, box[2] + box[9] / 2, 0)
    sole = feet[..., 2] - 0.035
    root_steps = np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=-1)
    joint_steps = np.abs(np.diff(pose[:, 7:], axis=0)).max(-1)
    return {
        "finite": bool(np.isfinite(pose).all()),
        "root_max_step_m": float(root_steps.max()),
        "joint_max_step_rad": float(joint_steps.max()),
        "joint_max_speed_rad_s": float(
            np.abs(np.gradient(pose[:, 7:], DT, axis=0)).max()
        ),
        "seam_root_steps_m": [float(root_steps[index]) for index in seams],
        "seam_joint_steps_rad": [float(joint_steps[index]) for index in seams],
        "approx_foot_below_support_frames": int(
            np.any(sole < support - 0.025, axis=1).sum()
        ),
        "foot_check": "ankle origin minus 35 mm; not collision/contact certification",
        "dynamics_validated": False,
    }


def ease_from_rest(pose):
    source_frames = np.arange(len(pose))
    ramp_frames = min(round(0.4 * FPS), len(pose) - 1)
    ramp = np.arange(2 * ramp_frames, dtype=float) ** 2 / (4 * ramp_frames)
    positions = np.concatenate((ramp, source_frames[ramp_frames:]))
    result = np.empty((len(positions), pose.shape[1]), dtype=pose.dtype)
    for column in (*range(3), *range(7, pose.shape[1])):
        result[:, column] = np.interp(positions, source_frames, pose[:, column])
    result[:, 3:7] = quaternions(
        Slerp(source_frames, rotations(pose[:, 3:7]))(positions)
    )
    return result


def generate(args):
    php_path = args.php.resolve() / "motion_matching"
    sys.path.insert(0, str(php_path))
    search = importlib.import_module("motion_matching.core.database")
    animation = importlib.import_module("motion_matching.utils.animation_state")
    model = load_model(args.robot)
    walking = []
    paths = sorted((args.motions / "new_walking_cleaned/平地直线").glob("*.npz"))
    for path in paths:
        if "后退" in path.name:
            continue
        pose = load_clip(path)
        walking.append((path, pose, features(model, pose)))
    if not walking:
        raise ValueError("No forward walking clips found")
    reference = args.motions / "supermimic_teacher_hand_clearance"
    ascent, descent = (
        load_clip(reference / f"{name}.npz") for name in ("ascent", "descent")
    )
    with np.load(reference / "ascent.npz") as archive:
        center, half = archive["box_center"], archive["box_half"]
    original_far = -1.27541032 + 0.215517241
    front = center[0] - half[0]
    height = center[2] + half[2]
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "kinematic_preview_not_training_validated",
        "terrain_position_format": "center_xyz_wxyz_full_extents",
        "php_revision": subprocess.check_output(
            ["git", "-C", str(args.php), "rev-parse", "HEAD"], text=True
        ).strip(),
        "method": "PHP 27D feature search + PHP inertialization; RoBan FK adapter",
        "approach_turning": "explicit smooth yaw/path warping, not learned turning",
        "align_ascent_soles": args.align_ascent_soles,
        "turn_top_approach": args.turn_top_approach,
        "include_stand_up": args.include_stand_up,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths + [reference / "ascent.npz", reference / "descent.npz"]
        },
        "robot_xml": str(args.robot.resolve()),
        "clips": [],
    }
    for mirrored in (False, True):
        up, down = ascent.copy(), descent.copy()
        if mirrored:
            up, down = (mirror_pose(pose, center[1]) for pose in (up, down))
        up_features = features(model, up)
        if args.align_ascent_soles:
            up = align_ascent_soles(model, up, center, half)
        approach, approach_match = match_window(
            walking,
            up_features[0],
            round(args.walk_seconds * FPS),
            search,
            True,
        )
        top, top_match = match_window(
            walking, up_features[-1], round(args.top_seconds * FPS), search
        )
        top = place_walk(top, up[-1], elevation=height)
        down_after = down.copy()
        down_after[:, 0] += top[-1, 0] - down[0, 0]
        platform_far = original_far + top[-1, 0] - down[0, 0]
        if platform_far - front < 0.7 or np.ptp(top[:, 1]) > 0.5:
            raise ValueError("Matched top walking does not fit the platform")
        top_before, down_match = match_window(
            walking,
            features(model, down)[0],
            round(args.top_seconds * FPS),
            search,
            True,
        )
        top_before = place_walk(top_before, down[0], at_end=True, elevation=height)
        cases = [
            (
                "ground_up",
                [("ground_walk", place_walk(approach, up[0], True)), ("ascent", up)],
                front,
                center[0] + half[0],
                [approach_match],
            ),
            (
                "top_down",
                [("top_walk", top_before), ("descent", down)],
                min(top_before[0, 0] - 0.5, front),
                original_far,
                [down_match],
            ),
            (
                "stand_down",
                [
                    ("stand", np.repeat(down[:1], FPS, axis=0)),
                    ("descent", ease_from_rest(down)),
                ],
                original_far - 2 * half[0],
                original_far,
                [],
            ),
            (
                "full",
                [
                    ("ground_walk", place_walk(approach, up[0], True)),
                    ("ascent", up),
                    ("top_walk", top),
                    ("descent", down_after),
                ],
                front,
                platform_far,
                [approach_match, top_match],
            ),
        ]
        if args.include_stand_up:
            cases.insert(
                1,
                (
                    "stand_up",
                    [
                        ("stand", np.repeat(up[:1], FPS, axis=0)),
                        ("ascent", ease_from_rest(up)),
                    ],
                    front,
                    center[0] + half[0],
                    [],
                ),
            )
        for scenario, blocks, box_front, box_far, matches in cases:
            standing = blocks[0][0] == "stand"
            placements = ((0, 0),) if standing else ((0, 0), (20, 0.6), (-20, -0.6))
            for degrees, lateral in placements:
                angle = np.deg2rad(degrees)
                transformed = [(name, pose.copy()) for name, pose in blocks]
                if transformed[0][0] == "ground_walk":
                    transformed[0] = (
                        "ground_walk",
                        turn_approach(transformed[0][1], angle),
                    )
                if args.turn_top_approach and scenario == "top_down" and degrees:
                    walk = transformed[0][1]
                    initial_yaw = rotations(walk[0, 3:7]).as_euler("xyz")[2]
                    transformed[0] = (
                        "top_walk",
                        turn_approach(walk, angle + initial_yaw),
                    )
                if standing:
                    pose = np.concatenate([block for _, block in transformed])
                    stages = np.concatenate(
                        [np.repeat(stage, len(block)) for stage, block in transformed]
                    )
                    seams = [len(transformed[0][1]) - 1]
                else:
                    pose, stages, seams = join(transformed, animation)
                box = np.array(
                    [
                        (box_front + box_far) / 2,
                        center[1],
                        height / 2,
                        1,
                        0,
                        0,
                        0,
                        box_far - box_front,
                        2.0,
                        height,
                    ]
                )
                rotation = Rotation.from_euler("z", angle)
                pivot = box[:3].copy()
                pivot[2] = 0
                shift = np.array([3.0, lateral, 0.0])
                pose[:, :3] = rotation.apply(pose[:, :3] - pivot) + shift
                pose[:, 3:7] = quaternions(rotation * rotations(pose[:, 3:7]))
                box[:3] = shift + np.array([0.0, 0.0, height / 2])
                box[3:7] = quaternions(rotation)
                name = f"{scenario}_{'mirror' if mirrored else 'original'}_yaw{degrees:+03d}"
                positions, quaternion = forward(model, pose)
                np.savez_compressed(
                    args.output / f"{name}_motion.npz",
                    fps=FPS,
                    joint_pos=pose,
                    joint_vel=np.c_[
                        np.gradient(pose[:, :3], DT, axis=0),
                        angular_velocity(pose[:, 3:7]),
                        np.gradient(pose[:, 7:], DT, axis=0),
                    ],
                    joint_names=np.asarray(JOINT_NAMES),
                    robot_id="roban",
                    body_pos_w=positions,
                    body_quat_w=quaternion,
                    body_names=np.asarray(
                        [model.body(index).name for index in range(1, model.nbody)]
                    ),
                    stage=stages,
                    seam_frames=np.asarray(seams),
                )
                np.save(args.output / f"{name}_terrain.npy", box[None, None])
                manifest["clips"].append(
                    {
                        "name": name,
                        "frames": len(pose),
                        "duration_s": (len(pose) - 1) / FPS,
                        "matches": matches,
                        "audit": audit(model, pose, box, seams),
                    }
                )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Generated {len(manifest['clips'])} previews in {args.output}")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--php", type=Path, default=root.parent / "php_parkour")
    parser.add_argument(
        "--motions",
        type=Path,
        default=root.parent / "kuavo_amp_mjlab/src/mjlab/motions",
    )
    parser.add_argument(
        "--robot",
        type=Path,
        default=root.parent
        / "kuavo_amp_mjlab/src/mjlab/asset_zoo/robots/biped_s17_ver/xml/biped_s17_verified_handball_fixedhead.xml",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "data/motion_matching_preview"
    )
    parser.add_argument("--walk-seconds", type=float, default=3.0)
    parser.add_argument("--top-seconds", type=float, default=5.0)
    parser.add_argument("--align-ascent-soles", action="store_true")
    parser.add_argument(
        "--turn-top-approach",
        action="store_true",
        help="Give angled top_down variants a fixed +X heading at their first frame",
    )
    parser.add_argument(
        "--include-stand-up",
        action="store_true",
        help="Add original and mirrored stationary ground-start ascent previews",
    )
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
