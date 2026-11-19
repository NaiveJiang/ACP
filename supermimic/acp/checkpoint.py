"""Robot-aware ACP checkpoint I/O. Legacy metadata-free G1 files remain readable."""
import torch
from pathlib import Path
from .robot import ROBOT_ID, metadata


def load_checkpoint(path, **kwargs):
    checkpoint = torch.load(path, **kwargs)
    # Distribution-only relocation: keep the checkpoint bytes/weights intact.
    # Original machine paths in configuration resolve inside this unpacked copy.
    old_root = "/home/liujunjie/Acp_supermimic_vibed/"
    package_root = Path(__file__).resolve().parents[2]
    def relocate(value):
        if isinstance(value, str) and value.startswith(old_root):
            return str(package_root / value[len(old_root):])
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, tuple):
            return tuple(relocate(item) for item in value)
        return value
    for key in ("args", "init_args", "ctc_cfg", "planner_cfg", "terrain_args", "goal_cfg"):
        if key in checkpoint:
            checkpoint[key] = relocate(checkpoint[key])
    saved = checkpoint.get("robot")
    args = checkpoint.get("args", {})
    expected = metadata(s17_actuators=args.get("s17_actuators", False),
                        primitive_soles=args.get("primitive_soles", False),
                        legacy_primitive_soles=(
                            saved is not None and "primitive_sole" in saved
                            and "primitive_mjcf_sha256" not in saved
                        ))
    if saved is None:
        if ROBOT_ID != "g1":
            raise ValueError(f"{path}: checkpoint has no robot metadata; cannot load as {ROBOT_ID}")
    elif saved.get("fingerprint") != expected["fingerprint"]:
        raise ValueError(f"{path}: incompatible robot/control/contact definition ({saved.get('id')} vs {ROBOT_ID})")
    return checkpoint


def save_checkpoint(checkpoint, path, **kwargs):
    args = checkpoint.get("args", {})
    robot = metadata(s17_actuators=args.get("s17_actuators", False),
                     primitive_soles=args.get("primitive_soles", False))
    torch.save({**checkpoint, "robot": robot}, path, **kwargs)
