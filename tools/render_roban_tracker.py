"""Render recorded ACP tracker trials, including failures, at real speed.

Run with ACP_ROBOT=roban. Reuses the ACP scene builder; does not re-simulate
or replay source-policy motion as the tracked robot. After termination only
the terminal state is shown, followed by a labelled freeze frame.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("ACP_ROBOT", "roban")
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from supermimic.acp.render import TARGET_RGBA, mat_to_quat, scene_xml
from supermimic.acp.robot import JOINT_NAMES

GROUPS = {
    "flat": "平地前行",
    "pyramid_stairs": "普通凸形楼梯",
    "pyramid_stairs_narrow": "窄凸形楼梯",
    "pyramid_inv_stairs": "普通凹形楼梯",
    "pyramid_inv_stairs_narrow": "窄凹形楼梯",
    "variable_height_stairs_up": "变高上楼梯",
    "variable_height_stairs_down": "变高下楼梯",
}
STAIR_GROUPS = [group for group in GROUPS if group != "flat"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", type=Path, required=True)
    ap.add_argument("--evaluation", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--groups", nargs="+", choices=list(GROUPS), default=STAIR_GROUPS,
                    help="Categories to render, in order (default: all six stair categories)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    evaluation = json.loads(args.evaluation.read_text())
    checkpoint = Path(evaluation["ckpt"]).resolve()
    iteration = int(checkpoint.stem.split("_")[-1])
    with np.load(args.traj) as source:
        d = {k: source[k] for k in source.files}
    assert str(d["robot_id"].item()) == "roban"
    assert list(d["joint_names"]) == JOINT_NAMES
    names = list(d["names"])
    failed_names = set(evaluation["failed"])
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title = ImageFont.truetype(font_path, 27)
    small = ImageFont.truetype(font_path, 20)
    width, height, fps, stride = 1280, 720, 25, 2
    index = []
    offset = 0.0
    for group in args.groups:
        label = GROUPS[group]
        candidates = sorted((i for i, name in enumerate(names) if name.startswith(group + "/")),
                            key=lambda i: (int(d["clip_len"][i]), names[i]))
        if group == "flat":
            # Some flat-folder references include obstacles. Use empty
            # terrain, without selecting based on the tracker outcome.
            candidates = [i for i in candidates if not np.any(d["boxes"][i, :, 5] > 0)
                          and (d["boxes"].shape[-1] < 9 or not np.any(d["boxes"][i, :, 8] > 0))]
        if not candidates:
            raise ValueError(f"No matching recorded reference for {group}")
        # Selection uses reference duration, independent of the policy outcome.
        c = candidates[len(candidates) // 2]
        name = names[c]
        alive = d["alive"][:, c]
        end = int(alive.sum())
        assert alive[:end].all() and not alive[end:].any()
        failed = name in failed_names
        assert failed == (end < int(d["clip_len"][c]) - 1)
        terminal = min(end, len(alive) - 1)
        times = list(range(0, terminal + 1, stride))
        if times[-1] != terminal:
            times.append(terminal)
        model = mujoco.MjModel.from_xml_string(scene_xml(d["boxes"][c]))
        model.vis.global_.offwidth, model.vis.global_.offheight = width, height
        model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:] = 0.45, 0.6
        if group == "flat":
            # A finite visible grid and darker material make foot motion clear
            # under this bright lighting; recorded physical states are unchanged.
            model.geom_size[model.geom("floor").id] = (20., 20., .05)
            grid = model.mat("grid").id
            model.mat_rgba[grid, :3] = .55
            model.mat_specular[grid] = 0.
            model.mat_reflectance[grid] = 0.
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height, width)
        option = mujoco.MjvOption()
        option.geomgroup[3:] = 0
        camera = mujoco.MjvCamera()
        camera.elevation = -22
        start = int(d["clip_start"][c])
        heading = np.degrees(np.arctan2(d["ref_root_R"][start, 1, 0], d["ref_root_R"][start, 0, 0]))
        camera.azimuth = heading + 120
        dest = args.out / f"{group}.mp4"
        encoder = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
            "-threads", "2", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(dest)
        ], stdin=subprocess.PIPE)
        nq = 7 + len(JOINT_NAMES)
        color = (255, 153, 125) if failed else (132, 232, 167)
        outcome = "触发失败终止" if failed else "参考片段运行结束"
        try:
            for frame_index, t in enumerate(times):
                fr = int(d["f"][t, c])
                data.qpos[:3], data.qpos[3:7], data.qpos[7:nq] = d["root_p"][t, c], d["root_quat"][t, c], d["q"][t, c]
                data.qpos[nq:nq+3], data.qpos[nq+3:nq+7] = d["ref_root_p"][fr], mat_to_quat(d["ref_root_R"][fr])
                data.qpos[nq+7:] = d["ref_q"][fr]
                mujoco.mj_forward(model, data)
                actual, reference = d["root_p"][t, c], d["ref_root_p"][fr]
                camera.lookat[:] = 0.65 * actual + 0.35 * reference + np.array([0., 0., -0.12])
                camera.distance = max(3.6, 2.8 + 1.2 * np.linalg.norm(actual[:2] - reference[:2]))
                renderer.update_scene(data, camera, scene_option=option)
                scene = renderer.scene
                command_t = min(t, max(0, end - 1))
                for b in np.flatnonzero(d["tgt_valid"][command_t, c]):
                    if scene.ngeom >= scene.maxgeom:
                        break
                    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([.025, 0, 0]), d["tgt_w"][command_t, c, b].astype(float),
                        np.eye(3).ravel(), np.array(TARGET_RGBA[b], dtype=np.float32))
                    scene.ngeom += 1
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, width, 108), fill=(19, 26, 38))
                draw.text((20, 10), f"RoBan tracker · 第 {iteration} 次迭代 · {label}", font=title, fill="white")
                draw.text((20, 50), name, font=small, fill=(200, 211, 228))
                draw.text((20, 78), f"时间 {t/50:.2f} / {(int(d['clip_len'][c])-1)/50:.2f} 秒   |   原速播放   |   本次结果：{outcome}", font=small, fill=color)
                draw.rectangle((0, height-65, width, height), fill=(19, 26, 38))
                draw.text((20, height-59), "实体：tracker 闭环输出    蓝色半透明：参考动作    红/蓝点：左右脚目标", font=small, fill="white")
                draw.text((20, height-30), "训练参考集 · 无噪声 · 参考接触指令 · 尚未训练完成", font=small, fill=(181, 194, 213))
                encoder.stdin.write(image.tobytes())
                if frame_index == len(times) // 2:
                    image.save(dest.with_suffix(".jpg"))
            # Keep the final physical state visible; never simulate/display
            # post-termination motion on the evaluator's parked terrain.
            draw.rectangle((20, 123, 620, 169), fill=(19, 26, 38))
            draw.text((32, 130), f"{outcome} · 末帧定格", font=title, fill=color)
            image.save(args.out / f"{group}_end.jpg")
            for _ in range(fps):
                encoder.stdin.write(image.tobytes())
        finally:
            encoder.stdin.close()
            code = encoder.wait()
            renderer.close()
        if code:
            raise RuntimeError(f"ffmpeg failed: {code}")
        item = dict(file=dest.name, clip=name, category=group, label=label,
                    failed=failed, live_control_steps=end, rollout_duration_s=end/50,
                    reference_duration_s=(int(d["clip_len"][c])-1)/50,
                    frames=len(times)+fps, reel_start_s=offset,
                    duration_s=(len(times)+fps)/fps, terminal_freeze_s=1.0)
        index.append(item)
        offset += item["duration_s"]
        print(json.dumps(item, ensure_ascii=False), flush=True)
    listing = args.out / "concat.txt"
    listing.write_text("".join(f"file '{item['file']}'\n" for item in index))
    kind = "flat_and_stairs" if "flat" in args.groups else "stairs"
    reel = args.out / (index[0]["file"] if len(index) == 1 else f"roban_tracker_{kind}_iter{iteration:06d}.mp4")
    if len(index) > 1:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", "-movflags", "+faststart", str(reel)], check=True)
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  iteration=iteration, trajectory=str(args.traj.resolve()), evaluation=str(args.evaluation.resolve()),
                  selection="median reference duration in each category, independent of outcome; flat excludes obstacle terrain",
                  categories=args.groups,
                  fps=fps, resolution=[width,height], duration_s=offset,
                  actual_acp_tracker_closed_loop=True, reference_ghost=True,
                  contact_markers="actual commanded future targets; not measured contact forces",
                  clips=index, reel=str(reel.resolve()))
    (args.out / "index.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("WROTE", reel, flush=True)


if __name__ == "__main__":
    main()
