"""Record one median-duration accepted clip per terrain and concatenate a reel."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion-dir", default="data/roban_motion")
    ap.add_argument("--out", default="videos/roban_collection_v1")
    ap.add_argument("--only-added",action="store_true",help="Render the v2 additions only")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    motions = (root / args.motion_dir).resolve()
    output = (root / args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((motions / "manifest.json").read_text())
    if args.only_added:
        manifest=[r for r in manifest if r['status']=='geometry_contact_validated_reference']
    categories = sorted({r["clip"].split("/")[0] for r in manifest})
    selected = []
    for category in categories:
        rows = [r for r in manifest if r["clip"].startswith(category + "/")]
        if category == "flat" and any(r["box_count"] == 0 for r in rows):
            rows = [r for r in rows if r["box_count"] == 0]
        rows = sorted(rows, key=lambda r: (r["frames"], r["clip"]))
        row = rows[len(rows)//2]
        if row["status"] not in ("geometry_contact_validated_pilot","geometry_contact_validated_reference"):
            raise ValueError("Only accepted pilot data may be recorded")
        video = output / (category + ".mp4")
        info = video.with_suffix(".json")
        inputs = [root / "supermimic/acp/demo_reference.py", root / "supermimic/acp/render.py",
                  motions / (row['clip'] + '_motion.npz'), motions / (row['clip'] + '_terrain.npy'),
                  Path(str(motions) + '_contacts') / (row['clip'] + '.npz')]
        heightfield = motions / (row['clip'] + '_heightfield.npz')
        if heightfield.exists():
            inputs.append(heightfield)
        reusable = (video.exists() and info.exists() and video.stat().st_mtime >= max(p.stat().st_mtime for p in inputs)
                    and json.loads(info.read_text())["clips"][0]["clip"] == row["clip"] + "_motion.npz")
        if not reusable:
            subprocess.run([sys.executable, "-m", "supermimic.acp", "--robot", "roban", "demo_reference",
                            "--motion-dir", str(motions), "--clips", row["clip"] + "_motion.npz", "--out", str(video),
                            "--width", "1280", "--height", "720", "--recorded-rollout"], cwd=root,
                           env={**os.environ, "MUJOCO_GL": "egl", "OMP_NUM_THREADS": "1"}, check=True)
        item = {**row, "video": str(video.relative_to(root)), "duration_s": row["frames"] / 50}
        selected.append(item)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(item["duration_s"] / 2), "-i", str(video),
                        "-frames:v", "1", str(output / (category + ".jpg"))], check=True)
    # Only generated, fixed category names enter this concat file.
    listing = output / "concat.txt"
    listing.write_text("".join(f"file '{c}.mp4'\n" for c in categories))
    reel = output / "roban_typical_data.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c", "copy", "-movflags", "+faststart", str(reel)], check=True)
    start = 0.
    for row in selected:
        row["reel_start_s"] = start
        start += row["duration_s"]
    index = {"mode": "recorded_source_policy_trajectory_replay_in_acp", "acp_policy_closed_loop": False,
             "fps": 50, "resolution": [1280, 720], "duration_s": start, "clips": selected}
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    from PIL import Image, ImageOps
    thumbs = [Image.open(output / (c + ".jpg")).convert("RGB") for c in categories]
    sheet = Image.new("RGB", (640 * 2, 360 * ((len(thumbs) + 1)//2)), "#101826")
    for i, thumb in enumerate(thumbs):
        sheet.paste(ImageOps.fit(thumb, (640, 360)), ((i % 2) * 640, (i//2)*360))
    sheet.save(output / "overview.jpg")
    print(json.dumps({"reel": str(reel), "duration_s": start, "clips": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
