"""Prepare and publish a quota-controlled box addition, with a separate holdout.

Preparation verifies source/label hashes and fixed geometry-group splits. Publish
requires staged validation and interface checks and retains the previous release.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QUOTAS = {("regular", "train"): 36, ("regular", "holdout"): 12,
          ("edge", "train"): 12, ("edge", "holdout"): 4,
          ("initial", "train"): 12, ("initial", "holdout"): 4}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def diverse(rows, count):
    features = np.array([r[0]["geometry"]["dimension_delta_m"] + [r[0]["geometry"]["yaw_delta_rad"]] for r in rows])
    features /= [.1, .05, .05, np.pi/12]
    chosen = [int(np.linalg.norm(features, axis=1).argmax())]
    distance = np.full(len(rows), np.inf)
    while len(chosen) < count:
        distance = np.minimum(distance, np.linalg.norm(features-features[chosen[-1]], axis=1))
        distance[chosen] = -np.inf
        chosen.append(int(distance.argmax()))
    return [rows[i] for i in chosen]


def prepare(stage, report_paths):
    groups, seen = defaultdict(list), set()
    for path in report_paths:
        report = json.loads(path.read_text())
        if report.get("publication_eligible") is False:
            raise ValueError(f"Experimental/ineligible report: {path}")
        root = Path(report["motion_dir"]).resolve()
        exports = {r["clip"]: r for r in json.loads((root/"export_manifest.json").read_text()) if r["accepted"]}
        for quality in report["clips"]:
            if not quality["accepted"]:
                continue
            if quality["planner_representation"] != "height_grid_v1" or quality["planner_side_policy"] != "ignore_both_losses":
                raise ValueError("Requires the agreed height-only planner with masked side targets")
            row = dict(exports[quality["clip"]])
            if row["split_group"] in seen:
                continue
            seen.add(row["split_group"])
            assert row["frames"] == 604 and row["source_success"]
            assert row["physics_dt"] == .001 and row["contact_timeconst_s"] == .005
            files = [root/(row["clip"]+s) for s in ("_motion.npz", "_terrain.npy")]
            files.append(Path(str(root)+"_contacts")/(row["clip"]+".npz"))
            for p, key in zip(files[:2], ("motion_sha256", "terrain_sha256")):
                if sha(p) != row[key]:
                    raise ValueError(f"Changed input: {p}")
            if any(p.stat().st_mtime > path.stat().st_mtime for p in files):
                raise ValueError("Input changed since validation")
            row.update(quality=quality, contacts_sha256=sha(files[2]), validation_report=str(path.resolve()),
                       status="geometry_contact_validated_reference", contact_profile="roban_knee_caps_v1",
                       planner_target_policy="ignore_side_targets_in_both_losses_v1")
            groups[row["stratum"], row["split"]].append((row, files))
    available = {f"{k[0]}/{k[1]}": len(groups[k]) for k in QUOTAS}
    if any(len(groups[k]) < n for k, n in QUOTAS.items()):
        raise ValueError(f"Insufficient accepted clips for the fixed quotas: {available}")
    selected = [r for key, n in QUOTAS.items() for r in diverse(groups[key], n)]
    assert len({r[0]["split_group"] for r in selected}) == 80
    geometry = np.array([r[0]["geometry"]["box_full_size_m"] + [r[0]["geometry"]["box_yaw_rad"]] for r in selected])
    assert len(np.unique(np.round(geometry, 6), axis=0)) == 80
    stage.mkdir(parents=True, exist_ok=False)
    manifests = {"train": [], "holdout": []}
    for row, files in selected:
        root = stage/(row["split"]+"_motion")
        labels = Path(str(root)+"_contacts")
        for source, target in zip(files, (root/(row["clip"]+"_motion.npz"), root/(row["clip"]+"_terrain.npy"), labels/(row["clip"]+".npz"))):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        manifests[row["split"]].append(row)
    for split, rows in manifests.items():
        save(stage/(split+"_motion")/"manifest.json", rows)
        save(stage/(split+"_motion")/"export_manifest.json", rows)
    old_root, old_labels = ROOT/"data/roban_motion", ROOT/"data/roban_motion_contacts"
    dm, dc = stage/"merged_motion", stage/"merged_motion_contacts"
    shutil.copytree(old_root, dm)
    shutil.copytree(old_labels, dc)
    shutil.copytree(stage/"train_motion/box_climb", dm/"box_climb")
    shutil.copytree(stage/"train_motion_contacts/box_climb", dc/"box_climb")
    old = json.loads((old_root/"manifest.json").read_text())
    manifest = old + manifests["train"]
    save(dm/"manifest.json", manifest)
    previous = json.loads((old_root/"release.json").read_text())
    release = dict(previous, release="roban_v3_box_climb", previous_release=previous["release"], clips=len(manifest), added=60,
        per_category=dict(Counter(r["clip"].split("/")[0] for r in manifest)), frames=sum(r["frames"] for r in manifest),
        seconds=sum((r["frames"]-1)/50 for r in manifest), split_groups=len({r["split_group"] for r in manifest}),
        available_new=available, box_climb_holdout_clips=20, holdout_motion_dir="data/roban_box_climb_holdout",
        holdout_contacts_dir="data/roban_box_climb_holdout_contacts", baseline_split="original_200_not_formal_holdout",
        contact_profiles=["legacy", "roban_knee_caps_v1"], planner_representation="height_grid_v1",
        planner_target_policy="ignore_side_targets_in_both_losses_v1", trained_acp_planner_rollouts=False)
    save(dm/"release.json", release)
    save(dm/"dataset_stats.json", release)
    coverage = json.loads((dm/"coverage.json").read_text())
    train_geo = np.array([r["geometry"]["box_full_size_m"] + [r["geometry"]["box_yaw_rad"]] for r in manifests["train"]])
    coverage.append(dict(category="box_climb", clips=60, frames=60*604, seconds=60*603/50, max_frames=604,
        terrain_representations=["yaw_boxes_v1"], geometry_fields=["length_m", "width_m", "height_m", "yaw_rad"],
        geometry_min=train_geo.min(0).tolist(), geometry_max=train_geo.max(0).tolist(),
        strata={"regular":36,"edge":12,"initial":12}, independent_holdout_clips=20))
    save(dm/"coverage.json", coverage)
    save(stage/"holdout_motion/release.json", dict(release="roban_box_climb_holdout_v1", split="holdout", clips=20,
        frames=20*604, seconds=20*603/50, strata={"regular":12,"edge":4,"initial":4}, group_disjoint_from_train=True))
    save(stage/"selection.json", dict(available=available, quotas={"/".join(k):v for k,v in QUOTAS.items()},
        selected=[r[0]["clip"] for r in selected], group_disjoint=True, unique_geometry_count=80,
        method="Greedy geometry-space coverage within preassigned stratum/split; never move rejected groups between splits"))
    print(json.dumps(release, indent=2))


def publish(stage, baseline):
    for split, count in (("train", 60), ("holdout", 20)):
        report = json.loads((stage/(split+"_validation.json")).read_text())
        assert report["accepted"] == count and len(report["clips"]) == count
    interface = json.loads((stage/"merged_planner_interface.json").read_text())
    assert interface["finite_gradients"] and interface["categories"] == 21
    smoke = json.loads((stage/"stage1_smoke.json").read_text())
    assert smoke["finite_losses_and_gradients"] and smoke["updates"] >= 2
    old_hashes = json.loads(baseline.read_text())
    assert all(Path(p).exists() and sha(p) == h for p,h in old_hashes.items())
    for p,h in old_hashes.items():
        relative = Path(p)
        if relative.suffix in (".npz", ".npy"):
            merged = "merged_motion_contacts" if "roban_motion_contacts" in relative.parts else "merged_motion"
            offset = relative.parts.index("roban_motion_contacts" if merged.endswith("contacts") else "roban_motion")
            assert sha(stage/merged/Path(*relative.parts[offset+1:])) == h
    for split in ("train", "holdout"):
        root = stage/(split+"_motion")
        for row in json.loads((root/"manifest.json").read_text()):
            for path,key in ((root/(row["clip"]+"_motion.npz"),"motion_sha256"),
                             (root/(row["clip"]+"_terrain.npy"),"terrain_sha256"),
                             (Path(str(root)+"_contacts")/(row["clip"]+".npz"),"contacts_sha256")):
                assert sha(path) == row[key]
            if split == "train":
                for path,key in ((stage/"merged_motion"/(row["clip"]+"_motion.npz"),"motion_sha256"),
                                 (stage/"merged_motion"/(row["clip"]+"_terrain.npy"),"terrain_sha256"),
                                 (stage/"merged_motion_contacts"/(row["clip"]+".npz"),"contacts_sha256")):
                    assert sha(path) == row[key]
    destinations = [ROOT/"data"/name for name in ("roban_motion","roban_motion_contacts","roban_box_climb_holdout","roban_box_climb_holdout_contacts")]
    sources = [stage/name for name in ("merged_motion","merged_motion_contacts","holdout_motion","holdout_motion_contacts")]
    assert not destinations[2].exists() and not destinations[3].exists()
    backup = stage/"previous_release"
    backup.mkdir(exist_ok=False)
    backups = [backup/p.name for p in destinations[:2]]
    destinations[0].rename(backups[0])
    try:
        destinations[1].rename(backups[1])
    except Exception:
        backups[0].rename(destinations[0]);raise
    moved=[]
    try:
        for source,dest in zip(sources,destinations):
            source.rename(dest);moved.append((source,dest))
    except Exception:
        for source,dest in reversed(moved):dest.rename(source)
        for old,dest in zip(backups,destinations[:2]):old.rename(dest)
        raise
    # Original data arrays must be byte-identical; only release metadata changes.
    data_hashes = {p:h for p,h in old_hashes.items() if Path(p).suffix in (".npz",".npy")}
    assert all(sha(p)==h for p,h in data_hashes.items())
    save(stage/"publication.json", dict(status="published", train_added=60, holdout=20,
        total_training_clips=260, old_data_files_verified=len(data_hashes), old_data_files_changed=0,
        backup=str(backup), destinations=[str(p) for p in destinations], original_metadata_hashes=str(baseline)))
    print("Published 60 training clips and 20 group-disjoint holdout clips; previous release retained.")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare","publish"))
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--reports", nargs="*", type=Path, default=[])
    ap.add_argument("--baseline", type=Path)
    args=ap.parse_args()
    if args.action=="prepare":prepare(args.stage.resolve(),args.reports)
    else:publish(args.stage.resolve(),args.baseline.resolve())


if __name__=="__main__":main()
