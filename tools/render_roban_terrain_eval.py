"""Render source rollout states against the exact saved MuJoCo source model.

Mesh slopes and heightfields remain intact. Playback is not a re-simulation.
Use MUJOCO_GL=egl .venv_roban_collect/bin/python.
"""
import argparse
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render(run, e, dest):
    model = mujoco.MjModel.from_binary_path(str(run / "source_model.mjb"))
    model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720
    model.vis.quality.shadowsize = 2048
    data = mujoco.MjData(model)
    states = np.load(run / "raw" / e["episode_id"] / "states.npz")
    qpos = states["qpos"]
    assert model.nq == qpos.shape[1] == 28
    for i, name in enumerate(states["joint_names"]):
        assert model.jnt_qposadr[model.joint("robot/"+str(name)).id] == 7+i
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 4.8, 135, -25
    opt = mujoco.MjvOption()
    opt.geomgroup[3:] = 0
    # Source terrain group 0, robot collision group 3, visible meshes group 2.
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-r", "25", "-i", "-", "-an", "-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)], stdin=subprocess.PIPE)
    frames = list(range(0, len(qpos), 2))
    courses = json.loads((run / "courses.json").read_text())
    course = courses[e["terrain_col"] * 4 + e["terrain_row"]]
    assert course["terrain"] == e["terrain_type"]
    detail = f"d={e['difficulty']:.1f}"
    if course.get("controlled_gap_m") is not None:
        detail = f"controlled gap={course['controlled_gap_m']*100:.0f} cm"
    elif "slope" in e["terrain_type"]:
        detail += f" / {np.degrees(np.arctan(.4*e['difficulty'])):.1f} deg"
    elif e["terrain_type"] == "narrow_beams":
        detail += f" / beam width={(.8-.6*e['difficulty'])*100:.0f} cm"
    try:
        for frame in frames:
            data.qpos[:] = qpos[frame]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = qpos[frame, :3] + np.array([.35, 0., -.25])
            renderer.update_scene(data, camera=camera, scene_option=opt)
            scene = renderer.scene
            goal = np.array(e["route_goal"])
            goal[2] += .12
            if scene.ngeom < scene.maxgeom:
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([.08, .08, .08]), goal, np.eye(3).ravel(), np.array([.1, .9, .25, 1.]))
                scene.ngeom += 1
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)
            draw.rectangle((0, 0, 1280, 105), fill=(18, 22, 29))
            draw.text((20, 12), f"RoBan source policy | {e['terrain_type']} | {detail}", font=font, fill="white")
            color = (110, 240, 145) if e["outcome"] == "success" else (255, 175, 100)
            draw.text((20, 47), f"Outcome: {e['outcome']}     Time: {frame/50:.2f} / {(len(qpos)-1)/50:.2f} s", font=small, fill=color)
            draw.text((20, 76), "Recorded closed-loop states | exact source mesh / heightfield | 1x speed", font=small, fill=(190, 200, 215))
            proc.stdin.write(np.asarray(img).tobytes())
            if frame == frames[len(frames)//2]:
                img.save(dest.with_suffix(".jpg"))
    finally:
        proc.stdin.close()
        code = proc.wait()
        renderer.close()
    if code:
        raise RuntimeError(f"ffmpeg exited {code}")
    return dict(file=str(dest), episode=e["episode_id"], terrain=e["terrain_type"], difficulty=e["difficulty"],
                outcome=e["outcome"], frames=len(frames), fps=25, duration_s=len(frames)/25, run=str(run))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episode", nargs="+")
    args = ap.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads((run / "analysis.json").read_text())
    episodes = report["episodes"]
    if args.episode:
        selected = [e for e in episodes if e["episode_id"] in args.episode]
        assert len(selected) == len(args.episode)
    else:
        selected = []
        for name in dict.fromkeys(e["terrain_type"] for e in episodes):
            if name == "flat":
                continue
            for outcome in ("success", "fall_or_termination", "timeout"):
                candidates = [e for e in episodes if e["terrain_type"] == name and e["outcome"] == outcome and e["frames"] >= 51]
                if candidates:
                    hardest = max(e["difficulty"] for e in candidates)
                    group = sorted([e for e in candidates if e["difficulty"] == hardest], key=lambda e: e["frames"])
                    selected.append(group[len(group)//2])
    index = []
    for e in selected:
        dest = out / f"{e['terrain_type']}_d{e['difficulty']:.1f}_{e['outcome']}.mp4"
        index.append(render(run, e, dest))
        (out / "index.json").write_text(json.dumps(index, indent=2)+"\n")
        print("RENDERED", dest, flush=True)


if __name__ == "__main__":
    main()
