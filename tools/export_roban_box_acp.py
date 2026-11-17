"""Export box episodes to quarantine, retaining preassigned geometry groups."""
import argparse
import json
from pathlib import Path

import numpy as np

from export_roban_acp import export_episode, terrain_boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for value in args.runs:
        run = Path(value)
        with np.load(run / "terrain_snapshot.npz") as snapshot:
            boxes, gids = terrain_boxes(snapshot)
        source = json.loads((run / "collection_report.json").read_text())
        for folder in sorted((run / "raw").iterdir()):
            meta = json.loads((folder / "episode.json").read_text())
            try:
                record = export_episode(run, folder, out, boxes, gids, min_frames=604)
            except (ValueError, FileExistsError) as exc:
                record = dict(episode=str(folder.resolve()), accepted=False, reason=str(exc))
            record.update(candidate=meta["candidate"], stratum=meta["stratum"], split=meta["split"],
                          split_group=meta["group"], geometry={k: meta[k] for k in
                          ("box_center_local", "box_full_size_m", "box_yaw_rad", "dimension_delta_m", "yaw_delta_rad")},
                          initial_delta=meta["initial_delta"], source_success=meta["source_success"],
                          checkpoint_sha256=source["checkpoint_sha256"], reference_sha256=source["reference_sha256"],
                          source_commit=source["source_commit"], abi=source["abi"],
                          physics_dt=source.get("physics_dt", .002), contact_timeconst_s=source.get("contact_timeconst_s"))
            records.append(record)
    (out / "export_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(dict(exported=sum(r["accepted"] for r in records), candidates=len(records))))


if __name__ == "__main__":
    main()
