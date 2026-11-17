"""Convert root+joint CSV data to ACP at 50 Hz, with explicit frame/time/terrain conventions."""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch

from .robot import JOINT_NAMES, NUM_J, ROBOT_ID, ROOT_BODY, get_spec
from .motion import _resample, load_php_npz
from ..rotations import mat_to_quat, quat_to_mat


def convert(csv, out, *, fps, quat_order, root_frame, terrain, translation=(0., 0., 0.)):
    out, terrain = Path(out), Path(terrain)
    if not str(out).endswith("_motion.npz"):
        raise ValueError("Output must end in _motion.npz")
    raw = np.loadtxt(csv, delimiter=",", ndmin=2)
    if raw.shape[1] != 7 + NUM_J or len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError(f"Expected finite CSV with (T >= 2, {7 + NUM_J}) columns")
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("Source fps must be positive")
    if quat_order not in ("xyzw", "wxyz"):
        raise ValueError("quat_order must be xyzw or wxyz")
    quat = raw[:, 3:7].copy()
    if quat_order == "xyzw":
        quat = quat[:, [3, 0, 1, 2]]
    if not np.allclose(np.linalg.norm(quat, axis=-1), 1., atol=.01):
        raise ValueError("Invalid source quaternion norm")
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    p, quat, q = _resample(*(torch.tensor(a, dtype=torch.float32) for a in (raw[:, :3], quat, raw[:, 7:])), fps)
    # If the CSV root is another model body (e.g. waist_yaw_link), recover base_link
    # using that frame's joint-dependent transform at every resampled frame.
    model = get_spec().compile()
    body_id = model.body(root_frame).id
    if root_frame != ROOT_BODY:
        data = mujoco.MjData(model)
        rotations = quat_to_mat(quat).numpy()
        for t in range(len(q)):
            data.qpos[:] = np.r_[np.zeros(3), 1., 0., 0., 0., q[t].numpy()]
            mujoco.mj_kinematics(model, data)
            rotation = rotations[t] @ data.xmat[body_id].reshape(3, 3).T
            p[t] -= torch.from_numpy(rotation @ data.xpos[body_id]).float()
            quat[t] = mat_to_quat(torch.tensor(rotation, dtype=torch.float32))
    boxes = np.load(terrain, allow_pickle=False)
    if boxes.ndim != 3 or boxes.shape[-1] != 10 or boxes.shape[1] < 1 or not np.isfinite(boxes).all():
        raise ValueError("Terrain must be a finite (K, V >= 1, 10) ACP box array")
    if boxes.size and (np.any(boxes[..., 7:] <= 0) or not np.allclose(boxes[..., 4:6], 0., atol=1e-6)
                       or not np.allclose(np.linalg.norm(boxes[..., 3:7], axis=-1), 1., atol=1e-3)):
        raise ValueError("Terrain requires positive extents and unit yaw-only wxyz quaternions")
    p += torch.tensor(translation)
    boxes = boxes.astype(np.float64, copy=True)
    boxes[..., :3] += np.asarray(translation)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, joint_pos=torch.cat((p, quat, q), -1).numpy(), joint_names=np.array(JOINT_NAMES),
        fps=np.array([50.]), robot_id=np.array(ROBOT_ID), source_csv=np.array(str(csv)),
        source_fps=np.array(fps), source_quat_order=np.array(quat_order), source_root_frame=np.array(root_frame))
    np.save(str(out).removesuffix("_motion.npz") + "_terrain.npy", boxes)
    load_php_npz(str(out))
    return len(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="No header: xyz, quaternion, joints in the selected robot's canonical order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--quat-order", choices=("xyzw", "wxyz"), required=True)
    ap.add_argument("--root-frame", required=True, help="Model body whose world pose is recorded by the CSV")
    ap.add_argument("--terrain", required=True, help="Matched source-coordinate (K,V,10) terrain NPY; empty K for flat")
    ap.add_argument("--translation", nargs=3, type=float, default=(0., 0., 0.), help="Shared xyz translation of motion and boxes")
    args = ap.parse_args()
    count = convert(args.csv, args.out, fps=args.fps, quat_order=args.quat_order, root_frame=args.root_frame,
                    terrain=args.terrain, translation=args.translation)
    print(f"Wrote {args.out}: {count} frames, 50 Hz, {NUM_J} joints ({ROBOT_ID})")


if __name__ == "__main__":
    main()
