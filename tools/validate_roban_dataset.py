"""Validate real RoBan ACP references and contact targets before publication."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys

os.environ["ACP_ROBOT"] = "roban"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.mdp import MDP
from supermimic.acp.motion import MotionLib
from supermimic.acp.planner import PlannerCfg, PlannerScan, hindsight_labels, onset_index, height_target_supervision
from supermimic.acp.robot import CONTACT_BODIES, default_joint_pos_list
from supermimic.acp.task import ContactCfg, ContactTask, _runs, debounce, drop_short, from_base
from supermimic.acp.terrain import box_heights


@torch.no_grad()
def validate(root, *, device="cpu"):
    torch.set_num_threads(4)
    kin = RobotKinematics(device)
    motion = MotionLib.from_dir(str(root), kin, device)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    task = ContactTask(ContactCfg(), mdp, str(root), device)
    scan = PlannerScan(PlannerCfg(), device)
    rows = []
    for ci, name in enumerate(motion.names):
        errors = []
        climbing = name.split("/")[0] == "box_climb"
        start, end = int(motion.clip_start[ci]), int(motion.clip_end[ci]) + 1
        f = torch.arange(start, end, device=device)
        S = motion.ref_state(f)
        boxes = motion.clip_boxes[ci, 0]
        with np.load(Path(str(root) + "_contacts") / (name + ".npz"), allow_pickle=False) as d:
            touch = d["contact_info"][0, ..., 0] > .5
            cp = d["contact_pos_w"][0]
            distances = d["contact_dist"][0]
            other = d["other_contact"][0]
            if len(touch) != len(f) or float(d["fps"]) != 50:
                errors.append("contact_time_mismatch")
            if list(d["body_names"]) != list(CONTACT_BODIES):
                errors.append("contact_order")
        clean = drop_short(debounce(touch, 2), 10)
        counts = [int((_runs(clean[:, b])[0] > 0).sum()) for b in range(9)]
        if not climbing and min(counts[:2]) < 3:
            errors.append("fewer_than_3_onsets_per_foot")
        if np.any(other):
            errors.append("non_contact_body_collision")
        if not climbing and np.any(touch[:, 2:]):
            errors.append("nonfoot_contact_in_walk")
        if climbing and not clean[:, 4:6].any(axis=0).all():
            errors.append("missing_sustained_hand_support")
        min_distance = float(distances.min())
        if min_distance < -.01:
            errors.append("penetration_over_1cm")
        on = onset_index(touch, 10, 75)
        target, occ = hindsight_labels(torch.tensor(on[None], device=device), torch.tensor(cp[None], device=device),
                                       S[0][None], S[1][None], scan)
        if not torch.isfinite(target).all():
            errors.append("nonfinite_valid_contact_target")
        if climbing and not occ[0, :, 4:6].any(dim=0).all():
            errors.append("missing_planner_hand_positive")
        bb_full = boxes[None].expand(len(f), -1, -1)
        world_targets = from_base(target[0], S[0], S[1])
        # Probe either side of floating-point boundaries. This permits a top
        # edge contact, but not an unrepresentable contact halfway up a riser.
        top_error = torch.full_like(occ[0], float("inf"), dtype=torch.float32)
        for dx, dy in ((0., 0.), (.002, 0.), (-.002, 0.), (0., .002), (0., -.002)):
            z = box_heights(world_targets[..., :2] + torch.tensor([dx, dy], device=device), bb_full)
            top_error = torch.minimum(top_error, (z-world_targets[..., 2]).abs())
        visible_error = float(top_error[occ[0]].max()) if occ.any() else 0.
        supervision = height_target_supervision(target[0], occ[0], S, bb_full) if climbing else torch.ones_like(occ[0])
        retained = occ[0] & supervision
        surface_error = float(top_error[retained].max()) if retained.any() else 0.
        if surface_error > .02:
            errors.append("planner_target_not_on_visible_top")
        if climbing and not retained[:, 4:6].any(dim=0).all():
            errors.append("missing_supervised_planner_hand_positive")
        pts = scan.points(scan.heights(S, bb_full))
        nearest = (target[0, :, :, None] - pts[:, None]).norm(dim=-1).amin(-1)
        grid_error_p95 = float(torch.quantile(nearest[retained], .95)) if retained.any() else 0.
        near_misses, near_error, sphere_errors, max_cut = 0, 0., [], -np.inf
        for i in range(0, len(f), 32):
            ss = tuple(x[i:i+32] for x in S)
            bb = boxes[None].expand(len(ss[0]), -1, -1)
            fk = kin.forward(*ss)
            task.cfg.near_boxes = 4
            dn = task.body_distance(ss, bb, fk)
            task.cfg.near_boxes = boxes.shape[0]
            da = task.body_distance(ss, bb, fk)
            task.cfg.near_boxes = 4
            active = torch.as_tensor(clean[i:i+32], device=device) if climbing else torch.zeros_like(dn, dtype=torch.bool)
            if not climbing:
                active[:, :2] = True
            near_misses += int(((dn > .01) & (da <= .01) & active).sum())
            if active.any():
                near_error = max(near_error, float((dn - da).abs()[active].max()))
            sphere_errors.extend((da[:, :2].cpu().numpy() - distances[i:i+32, :2]).flatten().tolist())
            allp = kin.forward(*ss, all_bodies=True)[0]
            h = box_heights(allp[..., :2], bb)
            max_cut = max(max_cut, float((h - allp[..., 2]).max()))
        if near_misses:
            errors.append("near4_misses_support")
        # MotionLib warns and re-enables all variants when none fits. Explicitly
        # check the underlying condition; variant_ok alone is not a valid gate.
        if max_cut > .02:
            errors.append("terrain_cuts_body_origin")
        cmd = task.command(S, f, torch.zeros_like(f), torch.full_like(f, 50), noise=False)
        if cmd.shape != (len(f), 93) or not torch.isfinite(cmd).all():
            errors.append("invalid_tracker_command")
        row = {"clip": name, "accepted": not errors, "errors": errors, "frames": len(f),
               "profile": "box_climb" if climbing else "walking", "contact_bodies": list(CONTACT_BODIES),
               "body_onsets": counts, "body_contact_fraction": touch.mean(0).tolist(),
               "clean_body_contact_frames": clean.sum(0).tolist(),
               "planner_body_positive_frames": occ[0].sum(0).tolist(),
               "foot_onsets": counts[:2], "foot_contact_fraction": touch[:, :2].mean(0).tolist(),
               "planner_reference_positive_fraction": occ[0, :, :2].float().mean(0).tolist(),
               "planner_visible_top_max_error_m": visible_error, "planner_grid_nearest_p95_m": grid_error_p95,
               "planner_surface_max_error_m": surface_error,
               "planner_representation": "height_grid_v1", "planner_side_policy": "ignore_both_losses" if climbing else "strict_top",
               "planner_ignored_body_frames": (~supervision).sum(0).tolist(),
               "planner_supervised_positive_frames": retained.sum(0).tolist(),
               "contact_profile": task.cfg.contact_profile,
               "min_contact_distance_m": min_distance, "near4_support_misses": near_misses,
               "near4_max_distance_error_m": near_error,
               "sphere_exact_abs_error_p95_m": float(np.quantile(np.abs(sphere_errors), .95)),
               "max_body_origin_cut_m": max_cut}
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--publish", help="Copy only accepted clips and labels to this NEW training directory")
    ap.add_argument("--per-category", type=int, default=10)
    args = ap.parse_args()
    root = Path(args.motion_dir)
    rows = validate(root, device=args.device)
    report = {"motion_dir": str(root), "clips": rows, "accepted": sum(r["accepted"] for r in rows),
              "reasons": dict(Counter(e for r in rows for e in r["errors"])),
              "scope": "reference geometry/contact validation; not ACP closed-loop success or planner rollout data"}
    if args.publish:
        available = Counter(r["clip"].split("/")[0] for r in rows if r["accepted"])
        categories = {r["clip"].split("/")[0] for r in rows}
        short = {c: available[c] for c in sorted(categories) if available[c] < args.per_category}
        if short:
            report["publication_deferred"] = {"required_per_category": args.per_category, "available": short}
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
            print(f"Publication deferred: {short}; report: {args.report}")
            return
        dest = Path(args.publish)
        contacts = Path(str(dest) + "_contacts")
        if dest.exists() or contacts.exists():
            raise FileExistsError("Publication destinations must be new; never overwrite a training release")
        dest.mkdir(parents=True)
        contacts.mkdir(parents=True)
        kept = Counter()
        manifest = {r["clip"]: r for r in json.loads((root / "export_manifest.json").read_text()) if r["accepted"]}
        published = []
        for r in rows:
            category = r["clip"].split("/")[0]
            if not r["accepted"] or kept[category] >= args.per_category:
                continue
            n = r["clip"]
            (dest / category).mkdir(exist_ok=True)
            (contacts / category).mkdir(exist_ok=True)
            for suffix in ("_motion.npz", "_terrain.npy"):
                shutil.copy2(root / (n + suffix), dest / (n + suffix))
            shutil.copy2(Path(str(root) + "_contacts") / (n + ".npz"), contacts / (n + ".npz"))
            published.append({**manifest[n], "status": "geometry_contact_validated_pilot", "split": "pilot_not_formal_holdout"})
            kept[category] += 1
        (dest / "manifest.json").write_text(json.dumps(published, indent=2) + "\n")
        report["published"] = dict(kept)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(f"Accepted {report['accepted']}/{len(rows)}; report: {args.report}")


if __name__ == "__main__":
    main()
