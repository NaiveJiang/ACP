"""Replay published RoBan clips with the saved, instantaneous contact labels.

No dynamics or relabeling: spheres use contact_pos_w at the current frame, and
the status panel and timeline use contact_info verbatim. Outputs stay outside
the training data. Run from the repository root with the ACP Python environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["ACP_ROBOT"] = "roban"
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from supermimic.acp.motion import load_php_npz
from supermimic.acp.render import scene_xml
from supermimic.acp.robot import CONTACT_BODIES, JOINT_NAMES, NUM_J
from supermimic.acp.terrain import load_terrain

W, H, VW, VH = 1600, 1000, 600, 500
NAMES = ("左脚", "右脚", "左膝", "右膝", "左手", "右手", "左肘", "右肘", "骨盆")
COLORS = ((239, 76, 81), (65, 145, 255), (255, 153, 47), (246, 205, 61),
          (59, 212, 132), (41, 210, 221), (175, 119, 245), (233, 128, 210), (224, 218, 168))
TITLES = {"regular": "常规箱体", "edge": "边界尺寸", "initial": "初态扰动"}
BG, MUTED, WHITE = (16, 25, 39), (161, 179, 199), (239, 244, 250)
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONTS = {n: ImageFont.truetype(FONT, n) for n in (18, 20, 22, 24, 30, 34)}


def text(draw, xy, value, size=22, color=WHITE):
    draw.text(xy, value, font=FONTS[size], fill=color)


def runs(mask):
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def contact_annotations(image, scene, points, active):
    """Label projected saved points, including points hidden by the robot mesh."""
    camera = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
    forward, up = np.array(camera.forward), np.array(camera.up)
    right = np.cross(forward, up)
    scale = VH * camera.frustum_near / (camera.frustum_top - camera.frustum_bottom)
    draw, placed = ImageDraw.Draw(image), []
    for b in np.flatnonzero(active):
        v = points[b] - camera.pos
        depth = float(v @ forward)
        if depth <= camera.frustum_near:
            continue
        x = VW / 2 + scale * float(v @ right) / depth
        y = VH * camera.frustum_top / (camera.frustum_top - camera.frustum_bottom) - scale * float(v @ up) / depth
        if not (12 < x < VW - 12 and 48 < y < VH - 12):
            continue
        color = COLORS[b]
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=BG, width=4)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=color, width=2)
        lx = min(VW - 50, max(4, x - 61 if b % 2 == 0 else x + 15))
        ly = min(VH - 29, max(41, y + (10 if b % 2 == 0 else -31)))
        for _ in range(8):
            if not any(lx < rx + 48 and lx + 48 > rx and ly < ry + 27 and ly + 27 > ry for rx, ry in placed):
                break
            ly = ly + 28 if ly + 55 < VH else ly - 28
        draw.line((x, y, lx + 22, ly + 13), fill=color, width=1)
        draw.rounded_rectangle((lx, ly, lx + 48, ly + 27), radius=4, fill=BG)
        text(draw, (lx + 5, ly), NAMES[b], 18, color)
        placed.append((lx, ly))


def render(entry, output, preview_only=False):
    clip = entry["clip"]
    motion_path = ROOT / "data/roban_motion" / (clip + "_motion.npz")
    label_path = ROOT / "data/roban_motion_contacts" / (clip + ".npz")
    terrain_path = ROOT / "data/roban_motion" / (clip + "_terrain.npy")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    hashes = {"motion_sha256": digest(motion_path), "terrain_sha256": digest(terrain_path),
              "contacts_sha256": digest(label_path)}
    assert all(entry[k] == v for k, v in hashes.items())
    root_p, root_q, joints, fps = load_php_npz(str(motion_path))
    with np.load(label_path, allow_pickle=False) as labels:
        assert list(labels["body_names"]) == list(CONTACT_BODIES)
        assert float(labels["fps"]) == fps == 50
        contact = labels["contact_info"][0, ..., 0] > .5
        points = labels["contact_pos_w"][0]
        distances = labels["contact_dist"][0]
        threshold = float(labels["threshold"])
        boxes = labels["boxes"][0]
        profile = str(labels["contact_profile"])
    assert contact.shape == (len(joints), 9)
    assert np.isfinite(points[contact]).all()
    assert np.array_equal(contact, distances <= threshold)
    np.testing.assert_allclose(boxes, load_terrain(str(motion_path))[0])

    tree = ET.fromstring(scene_xml(boxes))
    world = tree.find("worldbody")
    for child in list(world):
        if child.get("name", "").startswith("ref_"):
            world.remove(child)
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    assert model.nq == 7 + NUM_J
    assert [model.joint(i).name for i in range(1, model.njnt)] == JOINT_NAMES
    model.vis.global_.offwidth, model.vis.global_.offheight = VW, VH
    model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:] = .45, .65
    model.vis.quality.shadowsize = 1024
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    wide_lookat = ((root_p[:, 0].min() + root_p[:, 0].max()) / 2, 1.02, .83)
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    option.sitegroup[:] = 0
    renderer = mujoco.Renderer(model, VH, VW)
    T = len(joints)
    duration = (T - 1) / fps
    x0, x1, y0, dy = 126, 1555, 710, 24
    base = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(base)
    title = TITLES[entry["stratum"]]
    text(draw, (24, 9), f"翻越箱子 · 接触标注回放   |   {title}", 34)
    dims = " × ".join(f"{x:.3f}" for x in entry["geometry"]["box_full_size_m"])
    text(draw, (26, 57), f"{clip.split('/')[-1]}   ·   箱体 {dims} m", 22, MUTED)
    text(draw, (26, 90), "已入库的专家轨迹与保存标签   ·   0.5× 慢放   ·   彩球：接触位置；圆圈与文字：位置投影（含遮挡点）", 20, MUTED)
    text(draw, (25, 634), "接触时间轴", 24)
    text(draw, (200, 638), "有色段 = 接触标签为 1；暗色段 = 0；白线 = 当前时刻", 20, MUTED)
    for second in range(0, 13, 2):
        x = x0 + second / duration * (x1 - x0)
        draw.line((x, y0 - 8, x, y0 + 9 * dy), fill=(48, 63, 81), width=1)
        text(draw, (x - 12, y0 - 37), f"{second}s", 18, MUTED)
    for b in range(9):
        y = y0 + b * dy
        text(draw, (28, y - 6), NAMES[b], 20, COLORS[b])
        draw.rectangle((x0, y, x1, y + 15), fill=(35, 47, 65))
        for start, end in runs(contact[:, b]):
            xa = x0 + start / (T - 1) * (x1 - x0)
            xb = x0 + min(end, T - 1) / (T - 1) * (x1 - x0)
            draw.rectangle((xa, y, max(xa + 1, xb), y + 15), fill=COLORS[b])
    text(draw, (25, 938), "接触判定：保存的几何距离 ≤ 5 mm；状态栏显示几何间距，负值表示轻微穿透。", 20, MUTED)
    text(draw, (25, 969), "这里展示原始接触标签；侧面目标按约定在 planner 训练监督中屏蔽。", 18, MUTED)

    def frame(t):
        data.qpos[:] = np.r_[root_p[t], root_q[t], joints[t]]
        mujoco.mj_kinematics(model, data)
        image = base.copy()
        for view, (azimuth, distance, elevation, caption) in enumerate([
            (0., 3.35, -18., "侧视：完整翻越过程"),
            (-145., 2.0, -20., "对侧近景：手、膝接触位置"),
        ]):
            camera.azimuth, camera.distance, camera.elevation = azimuth, distance, elevation
            camera.lookat[:] = wide_lookat if view == 0 else root_p[t] + [0., 0., -.12]
            renderer.update_scene(data, camera=camera, scene_option=option)
            scene = renderer.scene
            for b in np.flatnonzero(contact[t]):
                assert scene.ngeom < scene.maxgeom
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                   np.array([.029, 0., 0.]), points[t, b].astype(np.float64),
                                   np.eye(3).ravel(), np.array([*(np.array(COLORS[b]) / 255), 1.], np.float32))
                scene.ngeom += 1
            view_image = Image.fromarray(renderer.render())
            contact_annotations(view_image, scene, points[t], contact[t])
            vd = ImageDraw.Draw(view_image)
            vd.rectangle((0, 0, VW, 37), fill=BG)
            text(vd, (16, 4), caption, 20)
            image.paste(view_image, (view * VW, 126))
        d = ImageDraw.Draw(image)
        d.rectangle((1206, 126, W - 10, 626), fill=(23, 35, 52))
        text(d, (1224, 139), f"当前接触：{contact[t].sum()} / 9 个部位", 24)
        text(d, (1224, 177), f"{t / fps:5.2f} / {duration:.2f} s   ·   帧 {t:03d}", 22, MUTED)
        for b in range(9):
            y = 221 + b * 39
            color = COLORS[b] if contact[t, b] else (74, 88, 107)
            d.ellipse((1228, y + 9, 1244, y + 25), fill=color)
            text(d, (1259, y), NAMES[b], 24, color)
            value = f"{distances[t, b] * 1000:+.1f} mm" if contact[t, b] else "未接触"
            text(d, (1395, y + 2), value, 22, color)
        text(d, (1224, 586), "颜色与 3D 球及时间轴一致", 20, MUTED)
        x = x0 + t / (T - 1) * (x1 - x0)
        d.line((x, y0 - 7, x, y0 + 9 * dy - 5), fill=WHITE, width=3)
        d.polygon([(x - 6, y0 - 12), (x + 6, y0 - 12), (x, y0 - 5)], fill=WHITE)
        return image

    stem = output / (entry["stratum"] + "_contact_labels")
    keys = [0, 185, 240, 310, 390, 545]
    try:
        frame(240).save(stem.with_suffix(".jpg"), quality=94)
        if preview_only:
            print(f"PREVIEW {stem.with_suffix('.jpg')}", flush=True)
            return None
        video = stem.with_suffix(".mp4")
        temporary = stem.with_suffix(".tmp.mp4")
        encoder = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", "25", "-i", "-", "-an", "-c:v", "libx264", "-threads", "2",
            "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)], stdin=subprocess.PIPE)
        try:
            for t in range(T):
                img = frame(t)
                encoder.stdin.write(img.tobytes())
                if t in keys:
                    img.save(output / f"{entry['stratum']}_frame_{t:03d}.jpg", quality=94)
                if (t + 1) % 200 == 0:
                    print(f"{entry['stratum']} {t + 1}/{T} frames", flush=True)
        finally:
            encoder.stdin.close()
            code = encoder.wait()
        if code:
            raise RuntimeError(f"ffmpeg exited {code}")
        temporary.replace(video)
        sheet = Image.new("RGB", (2400, 1000), BG)
        for k, t in enumerate(keys):
            with Image.open(output / f"{entry['stratum']}_frame_{t:03d}.jpg") as img:
                sheet.paste(img.resize((800, 500), Image.Resampling.LANCZOS), ((k % 3) * 800, (k // 3) * 500))
        sheet.save(output / f"{entry['stratum']}_contact_sequence.jpg", quality=94)
    finally:
        renderer.close()
    assert all(entry[k] == digest(p) for k, p in [("motion_sha256", motion_path),
               ("terrain_sha256", terrain_path), ("contacts_sha256", label_path)])
    report = {"clip": clip, "stratum": entry["stratum"], "video": str(video.resolve()),
              "motion": str(motion_path), "labels": str(label_path), **hashes,
              "data_unchanged": True, "contact_profile": profile, "source_fps": fps,
              "video_fps": 25, "playback_speed": .5, "frames": T, "video_seconds": T / 25,
              "contact_counts_by_body": dict(zip(CONTACT_BODIES, contact.sum(0).tolist())),
              "marker_semantics": "instantaneous saved contact_pos_w at frames with contact_info=1; no onset holding or relabeling",
              "projected_annotations_include_occluded_points": True,
              "mode": "published_expert_reference_kinematic_playback"}
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"VIDEO {video}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "videos/roban_box_climb_contact_labels")
    ap.add_argument("--preview-only", action="store_true")
    ap.add_argument("--strata", nargs="+", default=["regular", "edge", "initial"], choices=list(TITLES))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {r["clip"]: r for r in json.loads((ROOT / "data/roban_motion/manifest.json").read_text())}
    selected = json.loads((ROOT / "artifacts/roban/collection/box_climb_v1/release_staging/video_selection.json").read_text())
    reports = [render(manifest[r["clip"]], args.out, args.preview_only) for r in selected if r["stratum"] in args.strata]
    if not args.preview_only:
        (args.out / "index.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
