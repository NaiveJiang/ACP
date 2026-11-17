"""Export successful source episodes to a quarantined ACP motion directory.

Does not invent terrain or publish unvalidated clips. Run make_labels and the
dataset validator on the output before using it for training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def terrain_boxes(snapshot):
    """Compiled colliders -> exact yaw boxes; constant hfields are flat slabs."""
    boxes, geom_ids = [], []
    for i, kind in enumerate(snapshot["types"]):
        p, q, size = snapshot["pos"][i].copy(), snapshot["quat"][i].copy(), snapshot["size"][i].copy()
        if not np.allclose(q[1:3], 0., atol=1e-6):
            raise ValueError(f"non-yaw collider {snapshot['names'][i]}")
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            full = 2 * size
        elif kind == mujoco.mjtGeom.mjGEOM_HFIELD:
            h = snapshot["dataid"][i]
            start = snapshot["hfield_adr"][h]
            count = snapshot["hfield_nrow"][h] * snapshot["hfield_ncol"][h]
            values = snapshot["hfield_data"][start:start + count]
            if np.ptp(values) > 1e-7:
                raise ValueError(f"nonconstant hfield {snapshot['names'][i]}")
            sx, sy, scale, depth = snapshot["hfield_size"][h]
            top = float(values[0]) * scale
            p[2] += (top - depth) / 2
            full = np.array([2 * sx, 2 * sy, top + depth])
        elif kind == mujoco.mjtGeom.mjGEOM_PLANE:
            # Snapshot planes are checked separately; finite route-box crops
            # use their top as the implicit support plane.
            boxes.append(np.r_[p, q, 1e6, 1e6, 1e-3])
            boxes[-1][2] -= 5e-4
            geom_ids.append(snapshot["geom_id"][i])
            continue
        else:
            raise ValueError(f"unsupported terrain collider {snapshot['names'][i]} type={kind}")
        boxes.append(np.r_[p, q, full])
        geom_ids.append(snapshot["geom_id"][i])
    return np.asarray(boxes).reshape(-1, 10), np.asarray(geom_ids)


def heights(points, boxes, floor=-np.inf):
    out = np.full(points.shape[:-1], floor, dtype=np.float64)
    for b in boxes:
        yaw = 2 * np.arctan2(b[6], b[3])
        c, s = np.cos(yaw), np.sin(yaw)
        d = points - b[:2]
        inside = (np.abs(c*d[..., 0] + s*d[..., 1]) <= b[7]/2 + 1e-8) & (np.abs(-s*d[..., 0] + c*d[..., 1]) <= b[8]/2 + 1e-8)
        out = np.maximum(out, np.where(inside, b[2] + b[9]/2, -np.inf))
    return out


def export_episode(run, folder, output, all_boxes, gids, *, min_frames=301):
    meta = json.loads((folder / "episode.json").read_text())
    if not meta["source_success"]:
        return {"episode": str(folder), "accepted": False, "reason": "source_not_successful"}
    with np.load(folder / "states.npz", allow_pickle=False) as raw:
        qp, names, times = raw["qpos"].astype(np.float64), raw["joint_names"], raw["time"]
    if not min_frames <= len(qp) <= 1001 or not np.isfinite(qp).all():
        raise ValueError("invalid frame count or nonfinite state")
    if qp.shape[1] != 28 or not np.allclose(np.diff(times), .02, atol=1e-7):
        raise ValueError("requires actual 50 Hz, 28-column states")
    if not np.allclose(np.linalg.norm(qp[:, 3:7], axis=-1), 1., atol=1e-3):
        raise ValueError("invalid source quaternion")
    qp[:, 3:7] /= np.linalg.norm(qp[:, 3:7], axis=-1, keepdims=True)
    for t in range(1, len(qp)):
        if qp[t, 3:7] @ qp[t-1, 3:7] < 0:
            qp[t, 3:7] *= -1
    # A conservative world AABB contains all rotated height scans + 0.3 m.
    lo, hi = qp[:, :2].min(0) - 1.9, qp[:, :2].max(0) + 1.9
    yaw = 2 * np.arctan2(all_boxes[:, 6], all_boxes[:, 3])
    c, s = np.abs(np.cos(yaw)), np.abs(np.sin(yaw))
    half = np.stack((c*all_boxes[:, 7] + s*all_boxes[:, 8], s*all_boxes[:, 7] + c*all_boxes[:, 8]), -1) / 2
    mask = ((all_boxes[:, :2] + half >= lo) & (all_boxes[:, :2] - half <= hi)).all(-1)
    boxes, ids = all_boxes[mask].copy(), gids[mask]
    if not len(boxes):
        raise ValueError("no physical terrain under route")
    floor = np.min(boxes[:, 2] + boxes[:, 9]/2)
    translation = np.r_[-qp[0, :2], -floor]
    # Detect crop holes before adding the implicit infinite plane. Verify a
    # dense route envelope, including neighbors; unsupported outside domains fail.
    x, y = np.arange(lo[0], hi[0] + .025, .05), np.arange(lo[1], hi[1] + .025, .05)
    grid = np.stack(np.meshgrid(x, y, indexing="ij"), -1).reshape(-1, 2)
    before = heights(grid, boxes)
    if not np.isfinite(before).all():
        raise ValueError("route scan envelope leaves recorded terrain; implicit floor would invent support")
    qp[:, :3] += translation
    boxes[:, :3] += translation
    # A slab whose highest surface coincides with the lowest floor contributes
    # no exposed surface above it and is exactly replaced by ACP's ground.
    above = boxes[:, 2] + boxes[:, 9]/2 > 1e-6
    boxes, ids = boxes[above], ids[above]
    after = heights(grid + translation[:2], boxes, floor=0.)
    error = float(np.max(np.abs(before + translation[2] - after)))
    if error > 1e-3:
        raise ValueError(f"terrain height mismatch {error}")
    category = meta["terrain_type"]
    clip = f"{run.name}_{folder.name}"
    dest = output / category
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{clip}_motion.npz"
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, joint_pos=qp.astype(np.float32), joint_names=names, fps=np.array(50., np.float32),
                        robot_id=np.array("roban"), source_episode=np.array(str(folder.resolve())))
    terrain_path = dest / f"{clip}_terrain.npy"
    np.save(terrain_path, boxes[:, None].astype(np.float32))
    return {"episode": str(folder.resolve()), "accepted": True, "clip": f"{category}/{clip}", "frames": len(qp),
            "source_to_acp_translation": translation.tolist(), "box_count": len(boxes), "source_geom_ids": ids.tolist(),
            "surface_check_max_error_m": error, "split_group": f"{run.name}/tile_{meta['terrain_row']}_{meta['terrain_col']}",
            "motion_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "terrain_sha256": hashlib.sha256(terrain_path.read_bytes()).hexdigest(), "status": "requires_contact_validation"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-category", type=int, default=0, help="Optional pilot export cap; raw episodes remain archived")
    args = ap.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    report = []
    from collections import Counter
    kept = Counter()
    for name in args.runs:
        run = Path(name)
        with np.load(run / "terrain_snapshot.npz", allow_pickle=False) as snapshot:
            boxes, gids = terrain_boxes(snapshot)
        for folder in sorted((run / "raw").iterdir()):
            if not (folder / "episode.json").exists():
                continue
            category = json.loads((folder / "episode.json").read_text())["terrain_type"]
            if args.per_category and kept[category] >= args.per_category:
                continue
            try:
                row = export_episode(run, folder, output, boxes, gids)
            except (ValueError, FileExistsError) as e:
                row = {"episode": str(folder), "accepted": False, "reason": str(e)}
            report.append(row)
            if row["accepted"]:
                kept[category] += 1
    (output / "export_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"accepted": sum(r["accepted"] for r in report),
                      "rejected": dict(Counter(r.get("reason") for r in report if not r["accepted"]))}, indent=2))


if __name__ == "__main__":
    main()
