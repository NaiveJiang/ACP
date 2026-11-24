"""Pair the original Motion Matching database with retargeted RoBan poses."""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from supermimic.acp.footstep.lafan_reference import ReferenceGeometry
from supermimic.acp.footstep.motion_matching import Robot, read_upstream_poses
from supermimic.acp.footstep.s17_alignment import Mirror


# Original resources/generate_database.py writes these in this order.
TAKES = (
    ("idle", "pushAndStumble1_subject5", 194, 351),
    ("run", "run1_subject5", 90, 7086),
    ("walk", "walk1_subject5", 80, 7791),
)
DATABASE_FPS = 60
CSV_FPS = 30
# These two ranges were exercised by the native RoBan matcher at 2.2 m/s.
HIGH_STEP_SOURCE = "run2_subject4"
HIGH_STEP_RANGES = ((322, 786), (5731, 5912))


def read_database_index(path):
    with path.open("rb") as stream:
        for components in (3, 3, 4, 3):
            frames, bones = struct.unpack("<II", stream.read(8))
            stream.seek(frames*bones*components*4, 1)
        parent_count, = struct.unpack("<I", stream.read(4))
        stream.seek(parent_count*4, 1)

        def indices():
            count, = struct.unpack("<I", stream.read(4))
            return np.fromfile(stream, dtype="<i4", count=count)

        starts, stops = indices(), indices()
        expected_lengths = np.repeat([int(.9*(2*(stop-start)-1)) for _, _, start, stop in TAKES], 2)
        ranges = np.r_[0, np.cumsum(expected_lengths)]
        if not (np.array_equal(starts, ranges[:-1]) and np.array_equal(stops, ranges[1:])):
            raise ValueError("Expected the original database with its six LAFAN1 ranges")
        frames, feet = struct.unpack("<II", stream.read(8))
        data = np.fromfile(stream, dtype=np.uint8)
        if frames != ranges[-1] or feet != 2 or len(data) != frames*feet or (data > 1).any():
            raise ValueError("Unexpected contact table in database.bin")
    return starts, stops, data.reshape(frames, feet).astype(bool)


def interpolate(poses, source_rows, root):
    rows = np.arange(len(poses))
    output = np.stack([np.interp(source_rows, rows, channel) for channel in poses.T], axis=-1)
    rotation = slice(root+3, root+7)
    output[:, rotation] = Slerp(rows, Rotation.from_quat(poses[:, rotation], scalar_first=True))(
        source_rows).as_quat(scalar_first=True)
    return output


def build_reference(database, csv_dir, out, *, add_high_step_run=False, frax_dir=None):
    starts, stops, human_contacts = read_database_index(database)
    robot = Robot()
    mirror = Mirror(robot)
    geometry = ReferenceGeometry(robot)
    if frax_dir is not None:
        source_manifest = json.loads((frax_dir/"manifest.json").read_text())
        if (source_manifest.get("schema") != "roban_lafan1_frax_reference"
                or source_manifest.get("robot") != "biped_s17_ver"
                or source_manifest.get("fps") != CSV_FPS
                or source_manifest.get("source_start_frame", 0) != 0):
            raise ValueError("Expected full-length Frax S17-ver LaFAN1 references at 30 Hz")
        catalogue = {Path(clip["clip"]).name.removesuffix("_motion.npz"): frax_dir/clip["clip"]
                     for clip in source_manifest["clips"]}
        columns = np.r_[np.arange(robot.root, robot.root+7), robot.joint_qpos]

    def load_take(source, stop):
        if frax_dir is None:
            path = csv_dir/(source+".csv")
            poses = robot.load(path, CSV_FPS)
        else:
            if source not in catalogue:
                raise ValueError(f"Missing Frax take {source} in {frax_dir}")
            path = catalogue[source]
            with np.load(path, allow_pickle=False) as data:
                values = data["joint_pos"]
                if (data["robot_id"].item() != "roban" or float(data["fps"]) != CSV_FPS
                        or list(data["joint_names"]) != list(robot.joints)
                        or values.ndim != 2 or values.shape[1] != len(columns) or not np.isfinite(values).all()
                        or not np.allclose(np.linalg.norm(values[:, 3:7], axis=1), 1., atol=1e-3)):
                    raise ValueError(f"Invalid Frax S17 pose archive: {path}")
            poses = np.broadcast_to(robot.model.qpos0, (len(values), robot.model.nq)).copy()
            poses[:, columns] = values
        if stop > len(poses):
            raise ValueError(f"{path.name} does not cover required source row {stop-1}")
        return poses

    clips = []
    for take in (0, 2, 1):  # Preview order: idle, walk, run.
        gait, source, start, stop = TAKES[take]
        raw = load_take(source, stop)
        for mirrored in (False, True):
            index = 2*take+mirrored
            db_start, db_stop = int(starts[index]), int(stops[index])
            source_rows = np.linspace(start, stop-1, db_stop-db_start)
            poses = interpolate(raw, source_rows, robot.root)
            if mirrored:
                poses = mirror.poses(poses)
            human_stance = human_contacts[db_start:db_stop]
            _, gap = geometry.measure(poses)
            stance_gap = gap[human_stance]
            joints = poses[:, robot.limited_qpos]
            low, high = robot.joint_ranges
            audit = dict(
                source_stance_sole_gap_cm_p05_p50_p95=(
                    np.percentile(stance_gap, [5, 50, 95])*100).tolist(),
                pose_outside_joint_limits_pct=float(((joints < low-1e-4) | (joints > high+1e-4)).any(1).mean()*100),
            )
            case = gait + ("_mirror" if mirrored else "")
            clips.append((dict(case=case, category=gait, raw_motion=source, mirrored=mirrored,
                               source_row_start=start, source_row_stop=stop, database_start=db_start,
                               database_stop=db_stop, frames=len(poses), diagnostics=audit), poses, source_rows))

    if add_high_step_run:
        source = HIGH_STEP_SOURCE
        raw = load_take(source, max(stop for _, stop in HIGH_STEP_RANGES))
        for start, stop in HIGH_STEP_RANGES:
            # Supplementary clips follow main's timing; the original six retain the demo's retiming.
            frames = (stop-start-1)*DATABASE_FPS//CSV_FPS + 1
            source_rows = np.linspace(start, stop-1, frames)
            original = interpolate(raw, source_rows, robot.root)
            for mirrored in (False, True):
                poses = mirror.poses(original) if mirrored else original
                _, gap = geometry.measure(poses)
                joints = poses[:, robot.limited_qpos]
                low, high = robot.joint_ranges
                audit = dict(sole_gap_p95_cm=float(np.quantile(gap, .95)*100),
                             pose_outside_joint_limits_pct=float(
                                 ((joints < low-1e-4) | (joints > high+1e-4)).any(1).mean()*100))
                case = f"run_high_step_{start}" + ("_mirror" if mirrored else "")
                clips.append((dict(case=case, category="run", purpose="high_step_run",
                                   raw_motion=source, mirrored=mirrored, source_row_start=start,
                                   source_row_stop=stop, frames=frames, diagnostics=audit), poses, source_rows))

    reference = out/"reference"
    out.mkdir(parents=True, exist_ok=False)
    reference.mkdir()
    (out/"metadata.json").write_text(json.dumps(dict(sample_hz=DATABASE_FPS, source_hz=CSV_FPS), indent=2)+"\n")
    source = (dict(retargeter="frax", robot="biped_s17_ver", motion_dir=str(frax_dir.resolve()))
              if frax_dir is not None else dict(csv_dir=str(csv_dir.resolve())))
    manifest = dict(schema="roban_motion_matching_preview", status="preview",
                    database=str(database.resolve()), **source, clips=[clip for clip, _, _ in clips])
    (reference/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    for clip, poses, source_rows in clips:
        np.savez_compressed(reference/(clip["case"]+"_labels.npz"), qpos=poses, source_row=source_rows)
        print(clip["case"], f"{clip['frames']} frames", json.dumps(clip["diagnostics"]))


def write_native_catalogue(preview, out):
    metadata = json.loads((preview/"metadata.json").read_text())
    if metadata["sample_hz"] != DATABASE_FPS:
        raise ValueError("The native matcher requires a prepared 60 Hz library")
    robot = Robot()
    columns = np.r_[np.arange(robot.root, robot.root+7), robot.joint_qpos]
    fields = ["Frame", *("root_translate"+axis for axis in "XYZ"),
              *("root_rotate"+axis for axis in "WXYZ"), *robot.joints]
    out.mkdir()
    files = []
    for index, (name, pose) in enumerate(read_upstream_poses(robot, preview)):
        filename = f"{index:02d}_{name.replace(' ', '_')}.csv"
        rows = np.column_stack((np.arange(len(pose)), pose[:, columns]))
        np.savetxt(out/filename, rows, delimiter=",", fmt=["%d"]+["%.17g"]*len(columns),
                   header=",".join(fields), comments="")
        files.append(filename)
    if not files:
        raise ValueError("No mapped clips to export")
    (out/"clips.txt").write_text("\n".join(files)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path.home()/"Motion-Matching-main/resources/database.bin")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv-dir", type=Path, default=Path.home()/"roban数据/lafan1")
    source.add_argument("--frax-data", type=Path, help="ACP dataset exported by Frax for biped_s17_ver")
    parser.add_argument("--add-high-step-run", action="store_true",
                        help="Include the two validated run2_subject4 ranges and their mirrors")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"{args.out} already exists")
    build_reference(args.database, args.csv_dir, args.out, add_high_step_run=args.add_high_step_run,
                    frax_dir=args.frax_data)


if __name__ == "__main__":
    main()
