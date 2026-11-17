"""Render actual PHP reference clips with this project's G1, terrain and contact code.

This is kinematic playback, not a trained policy or a physics rollout.
Run from the repository root: python -m supermimic.acp.demo_reference
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .robot import JOINT_NAMES, MOTION_DIR, NUM_J, ROBOT_ID
from .make_labels import variant_labels
from .motion import load_php_npz
from .render import TARGET_RGBA, scene_xml
from .render_labels import onset_points
from .terrain import load_terrain


TITLES = {
    "motions-high-climb-76": "High-speed climb / 0.76 m",
    "motions-low-climb-76": "Low-speed climb / 0.76 m",
    "motions-high-step": "High-speed step",
    "motions-low-step": "Low-speed step",
    "motions-locomotion": "Locomotion",
    "flat": "Flat ground",
    "pyramid_stairs": "Wide solid pyramid stairs",
    "pyramid_stairs_narrow": "Narrow solid pyramid stairs",
    "pyramid_inv_stairs": "Wide inverted stairs",
    "pyramid_inv_stairs_narrow": "Narrow inverted stairs",
    "variable_height_stairs_up": "Variable-height stairs / up",
    "variable_height_stairs_down": "Variable-height stairs / down",
    "flat_backward": "Flat / backward walking",
    "flat_left": "Flat / left sidestepping",
    "flat_right": "Flat / right sidestepping",
    "flat_turn_left": "Flat / walking left turn",
    "flat_turn_right": "Flat / walking right turn",
    "hf_pyramid_slope": "Wide downhill slope",
    "hf_pyramid_slope_inv": "Wide uphill slope",
    "hf_pyramid_slope_narrow": "Narrow slope / up then down",
    "hf_pyramid_slope_inv_narrow": "Narrow slope / down then up",
    "random_rough": "Perlin rough ground",
    "threshold": "Repeated thresholds",
    "narrow_beams": "Narrow beam",
    "random_spread_boxes": "Scattered box obstacles",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-dir", type=Path, default=Path(MOTION_DIR))
    parser.add_argument("--out", type=Path, default=Path("videos/reference_demo.mp4"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seconds", type=float, default=None, help="Cap each clip for a smoke test")
    parser.add_argument("--clips", default="*/*_motion.npz")
    parser.add_argument("--recorded-rollout", action="store_true", help="Identify source-policy recorded states, replayed kinematically in ACP")
    args = parser.parse_args()
    files = sorted(args.motion_dir.glob(args.clips))
    if not files:
        parser.error(f"No clips in {args.motion_dir}")
    if args.width % 2 or args.height % 2:
        parser.error("Video dimensions must be even")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp.mp4")
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title_font = ImageFont.truetype(font_path, 28)
    font = ImageFont.truetype(font_path, 19)
    small = ImageFont.truetype(font_path, 16)
    fps = 50
    encoder = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{args.width}x{args.height}", "-r", str(fps), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(temporary),
    ], stdin=subprocess.PIPE)
    records = []
    total_frames = 0
    try:
        for path in files:
            root_p, root_quat, joints, src_fps = load_php_npz(str(path))
            assert src_fps == fps, f"Expected {fps} FPS, got {src_fps}"
            assert np.isfinite(np.concatenate([root_p, root_quat, joints], axis=1)).all()
            boxes = load_terrain(str(path))[0]
            name=path.stem.removesuffix("_motion")+".npz"
            label_file=Path(str(args.motion_dir)+"_contacts")/path.parent.name/name
            if not label_file.exists():
                label_file=args.out.parent/".contact_cache"/path.parent.name/name
                label_file.parent.mkdir(parents=True,exist_ok=True)
                print(f"Computing reference contacts: {path.name}", flush=True)
                np.savez_compressed(label_file, **variant_labels((str(path), boxes, 0.005, 0.1)))
            with np.load(label_file) as labels:
                info,cp=labels["contact_info"],labels["contact_pos_w"]
                if info.ndim==4:info,cp=info[0],cp[0]
                contacts = info[..., 0] > 0.5
                points = onset_points(contacts, cp)
            xml = ET.fromstring(scene_xml(boxes))
            world = xml.find("worldbody")
            for body in list(world):
                if body.get("name", "").startswith("ref_"):
                    world.remove(body)
            model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
            assert model.nq == 7 + NUM_J
            assert [model.joint(i).name for i in range(1, model.njnt)] == JOINT_NAMES
            model.vis.global_.offwidth = args.width
            model.vis.global_.offheight = args.height
            model.vis.headlight.ambient[:] = 0.45
            model.vis.headlight.diffuse[:] = 0.6
            data = mujoco.MjData(model)
            cam = mujoco.MjvCamera()
            delta = root_p[-1, :2] - root_p[0, :2]
            cam.azimuth = float(np.degrees(np.arctan2(delta[1], delta[0])) + 110)
            cam.elevation, cam.distance = -16.0, 3.3
            count = len(joints) if args.seconds is None else min(len(joints), round(args.seconds * fps))
            start_frame = total_frames
            with mujoco.Renderer(model, args.height, args.width) as renderer:
                for t in range(count):
                    data.qpos[:] = np.r_[root_p[t], root_quat[t], joints[t]]
                    mujoco.mj_kinematics(model, data)
                    lookat_z = root_p[t, 2] - 0.12 if args.recorded_rollout else 0.6 * root_p[t, 2] + 0.2
                    cam.lookat[:] = (root_p[t, 0], root_p[t, 1], lookat_z)
                    renderer.update_scene(data, cam)
                    scene = renderer.scene
                    for body in np.flatnonzero(contacts[t]):
                        if np.isfinite(points[t, body]).all() and scene.ngeom < scene.maxgeom:
                            mujoco.mjv_initGeom(
                                scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.03, 0, 0]), points[t, body].astype(np.float64),
                                np.eye(3).ravel(), np.array(TARGET_RGBA[body], dtype=np.float32),
                            )
                            scene.ngeom += 1
                    frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame, "RGBA")
                    draw.rectangle((0, 0, args.width, 106), fill=(16, 24, 38, 235))
                    draw.text((24, 10), f"ACP-{ROBOT_ID.upper()}  |  " + TITLES.get(path.parent.name, path.parent.name), font=title_font, fill="white")
                    caption_mode = ("SOURCE POLICY TRAJECTORY  |  Recorded-state playback in ACP" if args.recorded_rollout
                                    else "REFERENCE MOTION  |  Kinematic playback; no trained policy")
                    draw.text((24, 49), caption_mode, font=font, fill=(255, 211, 120))
                    caption = path.stem.removesuffix("_motion")
                    if draw.textlength(caption, font=small) > args.width - 48:
                        while draw.textlength(caption + "...", font=small) > args.width - 48:
                            caption = caption[:-1]
                        caption += "..."
                    draw.text((24, 78), caption, font=small, fill=(197, 210, 227))
                    draw.rectangle((0, args.height - 42, args.width, args.height), fill=(16, 24, 38, 225))
                    draw.text((24, args.height - 31), "Contact markers: feet red/blue  |  knees orange  |  hands green", font=small, fill="white")
                    draw.text((args.width - 195, args.height - 31), f"{t / fps:5.2f} / {count / fps:.2f} s", font=small, fill="white")
                    encoder.stdin.write(frame.tobytes())
                    total_frames += 1
            records.append({"clip": str(path.relative_to(args.motion_dir)), "start_seconds": start_frame / fps, "frames": count, "duration_seconds": count / fps})
            print(f"Rendered {path.parent.name}: {count} frames", flush=True)
    except BaseException:
        encoder.stdin.close()
        encoder.wait()
        raise
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg encoding failed")
    temporary.replace(args.out)
    report = {"mode": "reference_motion_kinematic_playback", "trained_policy": False,
              "recorded_source_policy_rollout": args.recorded_rollout,
              "fps": fps, "width": args.width, "height": args.height, "frames": total_frames,
              "duration_seconds": total_frames / fps, "clips": records}
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.out}: {total_frames / fps:.2f}s, {total_frames} frames", flush=True)


if __name__ == "__main__":
    main()
