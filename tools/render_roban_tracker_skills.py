"""Render preselected tracker trials from detailed evaluation, including failures.

Uses exact post-step physical states, including the terminal transition. The
robot and reference freeze together at termination; no inactive-world states
are rendered. Selection must have been recorded independently of outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

os.environ.setdefault("ACP_ROBOT", "roban")
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from supermimic.acp import heightfield
from supermimic.acp.render import TARGET_RGBA, mat_to_quat, scene_xml
from supermimic.acp.robot import CONTACT_BODIES, JOINT_NAMES


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", type=Path, required=True)
    ap.add_argument("--evaluation", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-reels", action="store_true", help="Write individual clips only")
    ap.add_argument("--contact-intent", action="store_true", help="Show active reference target names and body-to-target direction lines")
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.evaluation.read_text())
    selection = json.loads(args.selection.read_text())
    checkpoint = Path(report["checkpoint"])
    assert report["checkpoint_sha256"] == selection["checkpoint_sha256"] == sha256(checkpoint)
    assert report["protocol"]["reference_commands"]
    assert not report["protocol"]["noise"] and not report["protocol"]["domain_randomization"]
    iteration = int(checkpoint.stem.split("_")[-1])
    with np.load(args.traj) as source:
        d = {key: source[key] for key in source.files}
    assert str(d["robot_id"].item()) == "roban"
    assert list(d["joint_names"]) == JOINT_NAMES
    assert Path(str(d["checkpoint"].item())) == checkpoint
    heightfield.initialize(d["heightfield_paths"].tolist())
    names = d["names"].tolist()
    episodes = {e["clip"]: e for e in report["episodes"]}
    font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title = ImageFont.truetype(font, 27)
    small = ImageFont.truetype(font, 20)
    width, height, fps = 1280, 800, 25
    nq = 7 + len(JOINT_NAMES)
    index, previews = [], []
    for selected in selection["clips"]:
        name = selected["clip"]
        c, e = names.index(name), episodes[name]
        if not e["valid_physics"]:
            raise ValueError(f"Selected rollout has invalid physics: {name}")
        end, reference_steps = e["steps"], e["reference_steps"]
        assert end == int(d["alive"][:, c].sum())
        assert d["alive"][:end, c].all() and not d["alive"][end:, c].any()
        assert reference_steps == int(d["clip_len"][c]) - 1
        start = int(d["clip_start"][c])
        assert np.array_equal(d["post_f"][:end, c], np.arange(start + 1, start + end + 1))
        times = [min(2 * j, end) for j in range(math.ceil(end / 2) + 1)]
        model = mujoco.MjModel.from_xml_string(scene_xml(d["boxes"][c]))
        model.vis.global_.offwidth, model.vis.global_.offheight = width, height
        model.vis.headlight.ambient[:] = .45
        model.vis.headlight.diffuse[:] = .6
        model.geom_size[model.geom("floor").id] = (20., 20., .05)
        grid = model.mat("grid").id
        model.mat_rgba[grid, :3] = .55
        model.mat_specular[grid] = model.mat_reflectance[grid] = 0.
        data = mujoco.MjData(model)
        contact_ids = [model.body(frame).id for frame, _ in CONTACT_BODIES.values()]
        contact_names = ("左脚", "右脚", "左膝", "右膝", "左手", "右手", "左肘", "右肘", "骨盆")
        renderer = mujoco.Renderer(model, height, width)
        option = mujoco.MjvOption()
        option.geomgroup[3:] = 0
        camera = mujoco.MjvCamera()
        camera.elevation = -20
        R = d["ref_root_R"][start]
        camera.azimuth = np.degrees(np.arctan2(R[1, 0], R[0, 0])) + 120
        dest = out / (selected["id"] + ".mp4")
        encoder = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
            "-threads", "2", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(dest)
        ], stdin=subprocess.PIPE)
        color = (255, 153, 125) if e["failed"] else (132, 232, 167)
        outcome = "触发失败终止" if e["failed"] else "运行至参考末尾"
        try:
            for frame_index, t in enumerate(times):
                fr = start + t
                if t == 0:
                    p, quat, q = (d[key][0, c] for key in ("root_p", "root_quat", "q"))
                else:
                    p, quat, q = (d[key][t - 1, c] for key in ("post_root_p", "post_root_quat", "post_q"))
                assert np.isfinite(np.concatenate((p, quat, q))).all()
                rp = d["ref_root_p"][fr]
                data.qpos[:3], data.qpos[3:7], data.qpos[7:nq] = p, quat, q
                data.qpos[nq:nq + 3] = rp
                data.qpos[nq + 3:nq + 7] = mat_to_quat(d["ref_root_R"][fr])
                data.qpos[nq + 7:] = d["ref_q"][fr]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = .65 * p + .35 * rp + np.array([0., 0., -.12])
                camera.distance = max(3.6, 2.8 + 1.2 * np.linalg.norm(p[:2] - rp[:2]))
                renderer.update_scene(data, camera, scene_option=option)
                scene = renderer.scene
                command_t = min(t, end - 1)
                active_targets = np.flatnonzero(d["tgt_valid"][command_t, c])
                for b in active_targets:
                    if scene.ngeom >= scene.maxgeom:
                        break
                    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                       np.array([.025, 0., 0.]), d["tgt_w"][command_t, c, b].astype(float),
                                       np.eye(3).ravel(), np.array(TARGET_RGBA[b], dtype=np.float32))
                    scene.ngeom += 1
                    if args.contact_intent and scene.ngeom < scene.maxgeom:
                        geom = scene.geoms[scene.ngeom]
                        rgba = np.array(TARGET_RGBA[b], dtype=np.float32)
                        rgba[3] = .65
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba)
                        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .003,
                                             data.xpos[contact_ids[b]], d["tgt_w"][command_t, c, b].astype(float))
                        scene.ngeom += 1
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, width, 140 if args.contact_intent else 110), fill=(19, 26, 38))
                draw.text((20, 9), f"ACP tracker · 第 {iteration} 次迭代 · {selected['label']}", font=title, fill="white")
                draw.text((20, 48), name, font=small, fill=(200, 211, 228))
                status = outcome if t == end else "正在执行"
                draw.text((20, 78), f"时间 {t / 50:.2f} / {reference_steps / 50:.2f} 秒   |   原速   |   {status}",
                          font=small, fill=color if t == end else (200, 211, 228))
                if args.contact_intent:
                    active_text = "、".join(contact_names[b] for b in active_targets) or "无"
                    draw.text((20, 108), "未来窗口内的参考接触目标：" + active_text, font=small, fill=(200, 211, 228))
                draw.rectangle((0, height - 92, width, height), fill=(19, 26, 38))
                legend = "实体：tracker 输出  蓝色半透明：参考动作  彩色点/细线：参考接触目标及方向（非接触力）" if args.contact_intent else "实体：tracker 物理闭环输出    蓝色半透明：参考动作    彩色点：接触目标（非接触力）"
                draw.text((20, height - 85), legend,
                          font=small, fill="white")
                xy_error = np.linalg.norm(p[:2] - rp[:2]) * 100
                z_error = abs(p[2] - rp[2]) * 100
                draw.text((20, height - 57), f"当前根位置误差：水平 {xy_error:.1f} cm / 高度 {z_error:.1f} cm    |    终止后定格 1 秒",
                          font=small, fill=(200, 211, 228))
                draw.text((20, height - 29), "训练参考集 · 从起点执行 · 无噪声 · 参考接触指令 · Stage 1 训练进行中",
                          font=small, fill=(181, 194, 213))
                encoder.stdin.write(image.tobytes())
                if frame_index == len(times) // 2:
                    image.save(dest.with_suffix(".jpg"))
                    previews.append(image.copy())
            badge_y = 150 if args.contact_intent else 125
            draw.rectangle((20, badge_y, 650, badge_y + 47), fill=(19, 26, 38))
            draw.text((32, badge_y + 6), f"{outcome} · 实际终止状态定格", font=title, fill=color)
            image.save(out / (selected["id"] + "_end.jpg"))
            for _ in range(fps):
                encoder.stdin.write(image.tobytes())
        finally:
            encoder.stdin.close()
            code = encoder.wait()
            renderer.close()
        if code:
            raise RuntimeError(f"ffmpeg failed: {code}")
        item = {**selected, "category": e["category"], "file": dest.name,
                "completed": e["completed"], "failed": e["failed"], "valid_physics": e["valid_physics"],
                "control_steps": end, "rollout_duration_s": end / 50,
                "reference_duration_s": reference_steps / 50, "reference_progress": e["reference_progress"],
                "root_xy_error_mean_m": e["tracking"]["root_xy_error_m"]["mean"],
                "root_xy_error_at_end_m": e["root_xy_error_at_end_m"],
                "frames": len(times) + fps, "duration_s": (len(times) + fps) / fps}
        index.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    reels = {}
    for group, items in [("stairs", [i for i in index if i["category"] != "box_climb"]),
                         ("box_climb", [i for i in index if i["category"] == "box_climb"]), ("all", index)]:
        if not items or args.no_reels:
            continue
        listing = out / f"concat_{group}.txt"
        listing.write_text("".join(f"file '{item['file']}'\n" for item in items))
        reel = out / f"tracker_iter{iteration:06d}_{group}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", "-movflags", "+faststart", str(reel)], check=True)
        reels[group] = str(reel)
    sheet = Image.new("RGB", (1920, math.ceil(len(previews) / 3) * 400), (19, 26, 38))
    for i, preview in enumerate(previews):
        sheet.paste(preview.resize((640, 400), Image.Resampling.LANCZOS), ((i % 3) * 640, (i // 3) * 400))
    sheet.save(out / "preview.jpg", quality=92)
    result = {"checkpoint": str(checkpoint), "checkpoint_sha256": report["checkpoint_sha256"],
              "iteration": iteration, "selection": selection, "protocol": report["protocol"],
              "trajectory": str(args.traj.resolve()), "trajectory_sha256": sha256(args.traj),
              "evaluation": str(args.evaluation.resolve()), "evaluation_sha256": sha256(args.evaluation),
              "actual_acp_tracker_closed_loop": True, "exact_terminal_states": True,
              "reference_ghost": True, "contact_markers": "commanded targets, not contact forces",
              "body_to_target_lines": args.contact_intent,
              "fps": fps, "resolution": [width, height], "terminal_freeze_s": 1.,
              "clips": index, "reels": reels}
    (out / "index.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(reels, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
