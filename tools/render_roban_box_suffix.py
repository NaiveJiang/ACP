"""Render selected phase resets with nominal/noisy physical rollouts.

Every displayed state is recorded before or immediately after a physics step.
After termination, both robot and reference freeze at that exact transition.
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
from supermimic.acp.render import TARGET_RGBA, scene_xml
from supermimic.acp.robot import CONTACT_BODIES, JOINT_NAMES

BG = (19, 26, 38)
WHITE = (231, 237, 245)
MUTED = (182, 198, 219)
GREEN = (132, 232, 167)
RED = (255, 153, 125)
MODES = [("nominal", "无扰动"), ("training_randomization", "训练范围随机扰动")]
PHASES = [.8, .85, .9]
LABELS = ["箱顶，下箱前", "箱顶，准备下箱", "已经落地，收尾"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clip", default="box_climb/clean_b02_48_c0074")
    ap.add_argument("--phases", type=float, nargs="+", default=PHASES)
    ap.add_argument("--label", default="箱体翻越")
    ap.add_argument("--preview-only", action="store_true")
    ap.add_argument("--speed", type=float, choices=[.25, .5, 1.], default=.25)
    ap.add_argument("--contact-intent", action="store_true", help="Show exact applied target commands and body-to-target lines")
    ap.add_argument("--mode", choices=["comparison", "nominal", "training_randomization"], default="comparison")
    ap.add_argument("--follow-camera", action="store_true", help="Follow long traversals instead of a static box camera")
    args = ap.parse_args()
    if not args.phases or len(set(args.phases)) != len(args.phases) or not all(0 <= p < 1 for p in args.phases):
        ap.error("--phases must contain distinct phases between 0 and 1")
    phases, count = args.phases, len(args.phases)
    modes = [m for m in MODES if args.mode == "comparison" or m[0] == args.mode]
    single_mode = len(modes) == 1
    phase_tag = "_".join(str(round(p * 100)) for p in phases)
    phase_labels = dict(zip(PHASES, LABELS))
    src, out = args.directory.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads((src / "results.json").read_text())
    assert digest(report["checkpoint"]) == report["checkpoint_sha256"]
    iteration = report["iteration"]
    random_reset_noise = report.get("training_randomization", {}).get("reset_noise", True)
    bykey = {(e["mode"], e["clip"], e["start_phase"]): e for e in report["episodes"]}
    datasets = {}
    for mode, _ in modes:
        with np.load(src / (mode + ".npz")) as z:
            datasets[mode] = {k: z[k] for k in z.files}
        d = datasets[mode]
        assert list(d["joint_names"]) == JOINT_NAMES
        assert str(d["robot_id"].item()) == "roban"
        assert str(d["checkpoint"].item()) == report["checkpoint"]
        if args.contact_intent:
            assert {"tgt_w", "tgt_valid"} <= d.keys()
    heightfield.initialize(datasets[modes[0][0]]["heightfield_paths"].tolist())
    entries = []
    for mode, mode_label in modes:
        if mode == "training_randomization" and not random_reset_noise:
            mode_label = "训练扰动（初态噪声关闭）"
        d = datasets[mode]
        for phase in phases:
            label = phase_labels.get(phase, "参考状态重置") if args.clip == "box_climb/clean_b02_48_c0074" else "参考状态重置"
            if phase == 0:
                label = "第一帧起步"
            matched = np.flatnonzero((d["names"] == args.clip) & np.isclose(d["start_phase"], phase))
            assert len(matched) == 1
            c = int(matched[0])
            e = bykey[mode, args.clip, phase]
            assert e["valid_physics"]
            steps = int(d["alive"][:, c].sum())
            assert steps == round(e["elapsed_s"] / .02)
            start = int(d["start_frames"][c])
            assert d["alive"][:steps, c].all() and not d["alive"][steps:, c].any()
            assert np.array_equal(d["post_f"][:steps, c], np.arange(start + 1, start + steps + 1))
            entries.append(dict(mode=mode, mode_label=mode_label, phase=phase, label=label,
                                data=d, c=c, episode=e, steps=steps, start=start))
    W, H, FPS = (1280 if single_mode else 640), 640, 25
    header_h, footer_h = 76, 84 if args.contact_intent else 58
    font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title = ImageFont.truetype(font, 25)
    textfont = ImageFont.truetype(font, 20)
    small = ImageFont.truetype(font, 18)
    nq = 7 + len(JOINT_NAMES)
    base = entries[0]
    assert all(np.array_equal(e["data"]["boxes"][e["c"]], base["data"]["boxes"][base["c"]]) for e in entries), "Shared scene requires identical terrain"
    model = mujoco.MjModel.from_xml_string(scene_xml(base["data"]["boxes"][base["c"]]))
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    model.vis.headlight.ambient[:] = .45
    model.vis.headlight.diffuse[:] = .6
    model.geom_size[model.geom("floor").id] = (20., 20., .05)
    grid = model.mat("grid").id
    model.mat_rgba[grid, :3] = .55
    model.mat_specular[grid] = model.mat_reflectance[grid] = 0.
    data = mujoco.MjData(model)
    contact_ids = [model.body(frame).id for frame, _ in CONTACT_BODIES.values()]
    contact_names = ("左脚", "右脚", "左膝", "右膝", "左手", "右手", "左肘", "右肘", "骨盆")
    renderer = mujoco.Renderer(model, H, W)
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    camera = mujoco.MjvCamera()
    camera.elevation = -17
    d, c = base["data"], base["c"]
    first = int(d["start_frames"][c])
    last = int(d["clip_start"][c] + d["clip_len"][c] - 1)
    travel = d["ref_root_p"][last] - d["ref_root_p"][first]
    camera.azimuth = np.degrees(np.arctan2(travel[1], travel[0])) + 100
    positions = [e["data"]["initial_root_p"][e["c"]][None] for e in entries]
    positions += [e["data"]["post_root_p"][:e["steps"], e["c"]] for e in entries]
    positions += [d["ref_root_p"][first:last + 1]]
    points = np.concatenate(positions)
    assert np.isfinite(points).all()
    camera.lookat[:] = (points.min(0) + points.max(0)) / 2 + np.array([0., 0., -.12])
    camera.distance = 3.7
    cache = {}

    def panel(i, requested_t):
        e = entries[i]
        t = min(requested_t, e["steps"])
        key = (i, t)
        if key in cache:
            return cache[key]
        d, c, start = e["data"], e["c"], e["start"]
        if t == 0:
            p, quat, q = (d[k][c] for k in ("initial_root_p", "initial_root_quat", "initial_q"))
        else:
            p, quat, q = (d[k][t - 1, c] for k in ("post_root_p", "post_root_quat", "post_q"))
        fr = start + t
        rp = d["ref_root_p"][fr]
        data.qpos[:3], data.qpos[3:7], data.qpos[7:nq] = p, quat, q
        data.qpos[nq:nq + 3] = rp
        data.qpos[nq + 3:nq + 7] = d["ref_root_quat"][fr]
        data.qpos[nq + 7:] = d["ref_q"][fr]
        mujoco.mj_forward(model, data)
        if args.follow_camera:
            camera.lookat[:] = .65 * p + .35 * rp + np.array([0., 0., -.12])
        renderer.update_scene(data, camera, scene_option=option)
        active_targets = []
        if args.contact_intent:
            scene = renderer.scene
            command_t = max(0, t - 1)
            active_targets = np.flatnonzero(d["tgt_valid"][command_t, c])
            for b in active_targets:
                target = d["tgt_w"][command_t, c, b].astype(float)
                if scene.ngeom + 2 > scene.maxgeom:
                    break
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                   np.array([.025, 0., 0.]), target, np.eye(3).ravel(), np.array(TARGET_RGBA[b], dtype=np.float32))
                scene.ngeom += 1
                rgba = np.array(TARGET_RGBA[b], dtype=np.float32)
                rgba[3] = .65
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), rgba)
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .003, data.xpos[contact_ids[b]], target)
                scene.ngeom += 1
        im = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(im)
        draw.rectangle((0, 0, W, 125 if args.contact_intent else 94), fill=BG)
        draw.text((14, 7), f"从 {e['phase']:.0%} 开始 · {e['label']}", font=title, fill=WHITE)
        draw.text((14, 43), e["mode_label"], font=textfont, fill=(137, 204, 255) if i < count else (255, 211, 140))
        clip_time = (fr - int(d["clip_start"][c])) / 50
        draw.text((14, 71), f"参考时刻 {clip_time:.2f} s   |   重置后 {t / 50:.2f} s", font=small, fill=MUTED)
        if args.contact_intent:
            active_text = "、".join(contact_names[b] for b in active_targets) or "无"
            draw.text((14, 98), "接触目标：" + active_text, font=small, fill=MUTED)
        draw.rectangle((0, H - 76, W, H), fill=BG)
        terminal = t == e["steps"]
        status = ("失败终止 · 保持末帧" if e["episode"]["failed"] else "到达参考末尾 · 保持末帧") if terminal else "正在跟踪"
        color = RED if terminal and e["episode"]["failed"] else GREEN
        draw.text((14, H - 69), status, font=textfont, fill=color)
        draw.text((14, H - 40), f"水平误差 {np.linalg.norm(p[:2] - rp[:2]) * 100:.1f} cm   高度误差 {abs(p[2] - rp[2]) * 100:.1f} cm",
                  font=small, fill=MUTED)
        draw.rectangle((0, 0, W - 1, H - 1), outline=(76, 93, 116), width=2)
        cache[key] = im
        return im

    def compose(indices, columns, t):
        rows = (len(indices) + columns - 1) // columns
        im = Image.new("RGB", (W * columns, H * rows + header_h + footer_h), BG)
        draw = ImageDraw.Draw(im)
        test_label = "全程跟踪测评" if phases == [0.] else "分段重置测试"
        draw.text((18, 7), f"ACP tracker · 第 {iteration} 次迭代 · {args.label} · {test_label}", font=title, fill=WHITE)
        speed_label = "原速" if args.speed == 1 else f"{args.speed:g}× 慢放"
        draw.text((18, 42), f"{args.clip}   |   {speed_label}   |   同步显示重置后的时间", font=small, fill=MUTED)
        for ordinal, i in enumerate(indices):
            im.paste(panel(i, t), ((ordinal % columns) * W, header_h + (ordinal // columns) * H))
        y = im.height - footer_h
        draw.text((18, y + 4), "白色实体：tracker 物理闭环输出    蓝色半透明：参考动作    终止后两者一起定格", font=small, fill=WHITE)
        randomization_label = ("基于对应参考姿态/速度重置；随机版本包含初态、观测、指令、动作及动力学扰动，各相位独立采样。"
            if random_reset_noise else "右侧：观测、指令、动作及动力学扰动；初态噪声关闭。两组均保留重置穿透修正。")
        if single_mode:
            randomization_label = ("训练配置：观测、指令、动作及动力学扰动；初态噪声" + ("开启" if random_reset_noise else "关闭") + "。保留重置穿透修正。"
                if modes[0][0] == "training_randomization" else "无扰动参考状态重置；保留重置穿透修正。")
        draw.text((18, y + 30), randomization_label,
                  font=small, fill=MUTED)
        if args.contact_intent:
            draw.text((18, y + 56), "彩色点/细线：实际输入策略的接触目标及方向（包含相应指令噪声；非接触力）", font=small, fill=MUTED)
        return im

    streams = [dict(name=f"phase_{round(p * 100)}", indices=[i] if single_mode else [i, i + count],
                    columns=1 if single_mode else 2) for i, p in enumerate(phases)]
    if count > 1 and not single_mode:
        streams += [dict(name=f"nominal_{phase_tag}", indices=list(range(count)), columns=min(count, 3)),
                    dict(name=f"randomized_{phase_tag}", indices=list(range(count, 2 * count)), columns=min(count, 3))]
    try:
        preview_indices = list(range(len(entries)))
        preview_columns = min(len(entries), 3) if count > 1 else len(modes)
        overview = compose(preview_indices, preview_columns, 0)
        overview.save(out / "initial_states.jpg", quality=94)
        compose(preview_indices, preview_columns, 25).save(out / "after_half_second.jpg", quality=94)
        compose(preview_indices, preview_columns, max(e["steps"] for e in entries)).save(out / "terminal_states.jpg", quality=94)
        if args.preview_only:
            return
        for stream in streams:
            stream["last"] = max(entries[i]["steps"] for i in stream["indices"])
            stream["path"] = out / (stream["name"] + ".mp4")
            im = compose(stream["indices"], stream["columns"], 0)
            stream["resolution"] = list(im.size)
            stride = 50 * args.speed / FPS
            stream["times"] = [min(round(j * stride), stream["last"]) for j in range(math.ceil(stream["last"] / stride) + 1)]
            stream["frames"] = len(stream["times"]) + 2 * FPS
            stream["encoder"] = subprocess.Popen([
                "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{im.width}x{im.height}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
                "-threads", "2", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(stream["path"])], stdin=subprocess.PIPE)
            for _ in range(FPS):
                stream["encoder"].stdin.write(im.tobytes())
        for frame_number in range(max(len(s["times"]) for s in streams)):
            for stream in streams:
                if frame_number >= len(stream["times"]):
                    continue
                t = stream["times"][frame_number]
                im = compose(stream["indices"], stream["columns"], t)
                payload = im.tobytes()
                stream["encoder"].stdin.write(payload)
                if frame_number == len(stream["times"]) - 1:
                    im.save(stream["path"].with_suffix(".jpg"), quality=94)
                    for _ in range(FPS):
                        stream["encoder"].stdin.write(payload)
                    stream["encoder"].stdin.close()
                    code = stream["encoder"].wait()
                    if code:
                        raise RuntimeError(f"ffmpeg exit {code}")
                    print("WROTE", stream["path"], flush=True)
    finally:
        renderer.close()
        for stream in streams:
            enc = stream.get("encoder")
            if enc and enc.poll() is None:
                enc.stdin.close()
                enc.wait()
    combined = streams[0]["path"]
    if count > 1:
        listing = out / "concat.txt"
        listing.write_text("".join(f"file '{s['path'].name}'\n" for s in streams[:count]))
        combined = out / f"phases_{phase_tag}_{args.mode}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", "-movflags", "+faststart", str(combined)], check=True)
    result = {"checkpoint": report["checkpoint"], "checkpoint_sha256": report["checkpoint_sha256"],
              "iteration": iteration, "clip": args.clip, "label": args.label, "phases": phases,
              "selection": "Caller-selected clip and phases; no outcome-based replacement.",
              "results": str(src / "results.json"), "results_sha256": digest(src / "results.json"),
              "recording_sha256": {mode: digest(src / (mode + ".npz")) for mode, _ in modes},
              "mode": args.mode, "follow_camera": args.follow_camera,
              "episodes": [e["episode"] for e in entries], "actual_physics_closed_loop": True,
              "exact_terminal_states": True, "reference_frozen_with_robot_at_terminal": True,
              "shared_camera": True, "time_alignment": "Elapsed time since each independent phase reset",
              "playback_speed": args.speed, "fps": FPS, "initial_hold_s": 1., "terminal_hold_s": 1.,
              "contact_targets": "Exact applied policy input in world coordinates, including command noise" if args.contact_intent else None,
              "videos": [{"file": s["path"].name, "frames": s["frames"], "resolution": s["resolution"],
                          "duration_s": s["frames"] / FPS} for s in streams],
              "combined": {"file": combined.name, "frames": sum(s["frames"] for s in streams[:count]),
                           "duration_s": sum(s["frames"] for s in streams[:count]) / FPS}}
    (out / "index.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("WROTE", combined, flush=True)


if __name__ == "__main__":
    main()
