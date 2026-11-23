"""Export all retargeted RoBan LAFAN1 takes for WBC reference tracking."""
import argparse
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np

from supermimic.acp.footstep.motion_matching import Robot


TAKE_NAME = re.compile(r"([A-Za-z]+)\d+_subject(\d+)")
SOURCE_FPS = 30.


def export_motions(source, output):
    takes = [(path, match) for path in sorted(source.glob("*.csv"))
             if (match := TAKE_NAME.fullmatch(path.stem))]
    if not takes:
        raise ValueError(f"No LAFAN1 takes in {source}")
    if output.exists():
        raise FileExistsError(output)
    robot = Robot()
    columns = np.r_[np.arange(robot.root, robot.root+7), robot.joint_qpos]
    low, high = robot.joint_ranges
    records = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".lafan-mimic-", dir=output.parent) as temporary:
        dataset = Path(temporary)/"dataset"
        dataset.mkdir()
        for path, match in takes:
            category, subject = match.groups()
            pose = robot.load(path, SOURCE_FPS)
            origin = pose[0, robot.root:robot.root+2].copy()
            pose[:, robot.root:robot.root+2] -= origin
            joint_pos = pose[:, columns].astype(np.float32)
            joints = pose[:, robot.limited_qpos]
            violations = int(np.count_nonzero(((joints < low-1e-4) | (joints > high+1e-4)).any(1)))
            folder = dataset/category
            folder.mkdir(exist_ok=True)
            motion = folder/(path.stem+"_motion.npz")
            np.savez_compressed(motion, joint_pos=joint_pos, joint_names=np.array(robot.joints),
                                fps=np.float32(SOURCE_FPS), robot_id=np.array("roban"))
            np.save(folder/(path.stem+"_terrain.npy"), np.empty((0, 1, 10), dtype=np.float32))
            records.append(dict(clip=str(motion.relative_to(dataset)), source=path.name,
                                category=category, subject=int(subject), frames=len(pose),
                                original_xy=origin.tolist(), joint_limit_violation_frames=violations))
        manifest = dict(schema="roban_lafan1_wbc_reference", source=str(source.resolve()),
                        source_fps=SOURCE_FPS, terrain="flat", split="all_takes",
                        total_frames=sum(record["frames"] for record in records), clips=records)
        (dataset/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        dataset.rename(output)
    print(f"Exported {len(records)} takes, {manifest['total_frames']} source frames at {SOURCE_FPS:g} Hz")
    print(f"Manifest: {output/'manifest.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.home()/"roban数据/lafan1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export_motions(args.source, args.out)
