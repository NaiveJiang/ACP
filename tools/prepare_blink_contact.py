"""Package approved previews and the matching RoBan asset for contact training."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def prepare(preview, output, robot_directory, asset_output):
    manifest = json.loads((preview / "manifest.json").read_text())
    if manifest.get("terrain_position_format") != "center_xyz_wxyz_full_extents":
        raise ValueError("Preview terrain must use box centers, not bottom heights")
    if output.exists():
        raise FileExistsError(output)
    if asset_output.exists():
        raise FileExistsError(
            f"Refusing to replace existing robot asset: {asset_output}"
        )
    names = [clip["name"] for clip in manifest["clips"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("Expected a nonempty preview set with unique clip names")
    sources = []
    for clip in manifest["clips"]:
        if not clip["audit"]["finite"]:
            raise ValueError(f"Nonfinite clip: {clip['name']}")
        if (
            clip["audit"]["root_max_step_m"] > 0.2
            or clip["audit"]["joint_max_step_rad"] > 0.8
        ):
            raise ValueError(f"Discontinuous clip: {clip['name']}")
        for suffix in ("_motion.npz", "_terrain.npy"):
            path = preview / (clip["name"] + suffix)
            if not path.is_file():
                raise FileNotFoundError(path)
            sources.append(path)
    model_path = robot_directory / "xml/biped_s17_verified_handball_fixedhead.xml"
    if not model_path.is_file() or not (robot_directory / "meshes").is_dir():
        raise FileNotFoundError(model_path)
    output.mkdir(parents=True)
    asset_output.mkdir(parents=True)
    shutil.copytree(robot_directory / "meshes", asset_output / "meshes")
    (asset_output / "xml").mkdir()
    shutil.copy2(model_path, asset_output / "xml" / model_path.name)
    records = []
    split_counts = {"train": 0, "validation": 0}
    for clip in manifest["clips"]:
        name = clip["name"]
        split = "validation" if "yaw-20" in name else "train"
        split_counts[split] += 1
        group = name.split("_original")[0].split("_mirror")[0]
        destination = output / split / group
        destination.mkdir(parents=True, exist_ok=True)
        for suffix in ("_motion.npz", "_terrain.npy"):
            source = preview / (name + suffix)
            target = destination / source.name
            shutil.copy2(source, target)
            records.append(
                {
                    "source": str(source.resolve()),
                    "output": str(target.relative_to(output)),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            )
    report = {
        "source_preview": str(preview.resolve()),
        "files": records,
        "train_clips": split_counts["train"],
        "validation_clips": split_counts["validation"],
        "split_note": "Held-out box pose only; both splits reuse underlying source motions",
        "physical_execution_validated": False,
        "contact_label_method": "pending mujoco_geomDistance, not contact forces",
        "robot_source": str(model_path.resolve()),
        "robot_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    (output / "dataset.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Prepared {split_counts['train']} training and "
        f"{split_counts['validation']} box-pose validation clips: {output}"
    )


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview", type=Path, default=root / "data/motion_matching_preview"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "data/roban_blink_contact"
    )
    parser.add_argument(
        "--robot-directory",
        type=Path,
        default=root.parent
        / "kuavo_amp_mjlab/src/mjlab/asset_zoo/robots/biped_s17_ver",
    )
    parser.add_argument(
        "--asset-output", type=Path, default=root / "data/roban_description"
    )
    args = parser.parse_args()
    prepare(args.preview, args.output, args.robot_directory, args.asset_output)


if __name__ == "__main__":
    main()
