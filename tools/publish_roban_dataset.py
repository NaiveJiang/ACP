"""Publish a balanced pilot from completed validation reports without relabeling."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--out", default="data/roban_motion")
    ap.add_argument("--per-category", type=int, default=10)
    args = ap.parse_args()
    grouped = defaultdict(lambda: defaultdict(list))
    for report_name in args.reports:
        report_file = Path(report_name)
        report = json.loads(report_file.read_text())
        root = Path(report["motion_dir"])
        manifests = {r["clip"]: r for r in json.loads((root / "export_manifest.json").read_text()) if r["accepted"]}
        for row in report["clips"]:
            if not row["accepted"]:
                continue
            m = manifests[row["clip"]]
            files = [root / (row["clip"] + s) for s in ("_motion.npz", "_terrain.npy")]
            files.append(Path(str(root) + "_contacts") / (row["clip"] + ".npz"))
            for f, key in zip(files[:2], ("motion_sha256", "terrain_sha256")):
                if hashlib.sha256(f.read_bytes()).hexdigest() != m[key]:
                    raise ValueError(f"Changed input after export: {f}")
            if any(f.stat().st_mtime > report_file.stat().st_mtime for f in files):
                raise ValueError("Input changed after validation; rerun validation")
            item = {**m, "quality": row, "validation_report": str(report_file.resolve()),
                    "contacts_sha256": hashlib.sha256(files[2].read_bytes()).hexdigest()}
            grouped[row["clip"].split("/")[0]][m["split_group"]].append((item, files))
    expected = {"flat", "pyramid_stairs", "pyramid_stairs_narrow", "pyramid_inv_stairs", "pyramid_inv_stairs_narrow",
                "variable_height_stairs_up", "variable_height_stairs_down"}
    counts = {c: sum(len(v) for v in grouped[c].values()) for c in sorted(expected)}
    short = {c: n for c, n in counts.items() if n < args.per_category}
    if short:
        raise ValueError(f"Not enough validated clips: {short}")
    chosen = []
    for category in sorted(expected):
        groups = [sorted(grouped[category][g], key=lambda r: r[0]["clip"]) for g in sorted(grouped[category])]
        order = []
        while any(groups):
            for group in groups:
                if group:
                    order.append(group.pop(0))
        chosen.extend(order[:args.per_category])
    dest = Path(args.out)
    contacts = Path(str(dest) + "_contacts")
    if dest.exists() or contacts.exists():
        raise FileExistsError("Use a new publication path; existing datasets are immutable")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=dest.name + ".pending_", dir=dest.parent))
    tc = Path(tempfile.mkdtemp(prefix=contacts.name + ".pending_", dir=dest.parent))
    manifest = []
    for row, files in chosen:
        name = row["clip"]
        (tmp / name).parent.mkdir(parents=True, exist_ok=True)
        (tc / name).parent.mkdir(parents=True, exist_ok=True)
        for src, target in zip(files, (tmp/(name+"_motion.npz"), tmp/(name+"_terrain.npy"), tc/(name+".npz"))):
            shutil.copy2(src, target)
        manifest.append({**row, "status": "geometry_contact_validated_pilot", "split": "pilot_not_formal_holdout"})
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {"clips": len(manifest), "per_category": args.per_category, "available": counts,
               "frames": sum(r["frames"] for r in manifest), "seconds": sum((r["frames"]-1)/50 for r in manifest),
               "split_groups": len({r["split_group"] for r in manifest}), "reports": args.reports,
               "source_policy_references": True, "trained_acp_planner_rollouts": False}
    (tmp / "release.json").write_text(json.dumps(summary, indent=2) + "\n")
    os.environ["ACP_ROBOT"] = "roban"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from supermimic.acp.robot import metadata
    (tmp / "robot_contract.json").write_text(json.dumps(metadata(), indent=2) + "\n")
    tmp.rename(dest)
    tc.rename(contacts)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
