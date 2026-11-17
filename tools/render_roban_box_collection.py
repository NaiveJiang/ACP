"""Render actual saved expert states; label quarantine status on every frame."""
import argparse
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--accepted", action="store_true", help="Use only for an episode selected from a passing validation report")
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    folder = run / "raw" / args.episode
    meta = json.loads((folder / "episode.json").read_text())
    raw = np.load(folder / "states.npz")
    qp, qv = raw["qpos"], raw["qvel"]
    m = mujoco.MjModel.from_binary_path(str(run / "source_model.mjb"))
    # Hide only robot collision proxies, keeping real terrain colliders visible.
    for g in range(m.ngeom):
        if m.body(m.geom_bodyid[g]).name.startswith("robot/") and m.geom_group[g] == 3:
            m.geom_group[g] = 5
    m.stat.extent = 5.
    m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
    m.vis.quality.shadowsize = 2048
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=720, width=1280)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.elevation = 3.8, -16
    camera.lookat[:] = (qp[:, :3].min(0) + qp[:, :3].max(0)) / 2 - [0, 0, .15]
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 1, 1, 0, 0]
    option.sitegroup[:] = 0
    fontfile = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font, small = ImageFont.truetype(fontfile, 27), ImageFont.truetype(fontfile, 21)
    dimensions = " × ".join(f"{v:.3f}" for v in meta["box_full_size_m"])
    dyaw = np.rad2deg(meta["yaw_delta_rad"])

    def frame(i, angle):
        d.qpos[:], d.qvel[:] = qp[i], qv[i]
        mujoco.mj_forward(m, d)
        camera.azimuth = angle
        renderer.update_scene(d, camera=camera, scene_option=option)
        img = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 1280, 110), fill=(19, 26, 36))
        draw.text((22, 6), f"RoBan 爬箱采集 · {args.label} · {args.episode}", font=font, fill="white")
        draw.text((22, 46), f"箱体 {dimensions} m   |   偏航增量 {dyaw:+.1f}°   |   {i/50:.2f} / {(len(qp)-1)/50:.2f} s   |   原速 1×", font=small, fill=(200, 215, 230))
        status = "已通过数据验收 · 专家闭环记录 · 侧面目标不参与 planner 监督" if args.accepted else "专家闭环成功；ACP 验收未通过，暂存诊断区"
        draw.text((22, 77), status, font=small, fill=(170, 230, 190) if args.accepted else (255, 192, 100))
        return img

    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", "1280x720", "-r", "25", "-i", "-", "-an", "-c:v", "libx264",
                             "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    count = 0
    try:
        for angle in (0, -45):
            for i in range(0, len(qp), 2):
                img = frame(i, angle)
                proc.stdin.write(np.asarray(img).tobytes())
                count += 1
            print(f"Rendered {args.episode} angle={angle}", flush=True)
        frame(238, -45).save(out.with_suffix(".jpg"))
        sheet = Image.new("RGB", (1920, 720))
        for k, i in enumerate((0, 168, 238, 350, 534, 603)):
            sheet.paste(frame(min(i, len(qp)-1), -45).resize((640, 360)), ((k % 3)*640, (k//3)*360))
        sheet.save(out.with_name(out.stem + "_sequence.jpg"))
    finally:
        proc.stdin.close()
        result = proc.wait()
        renderer.close()
    if result:
        raise RuntimeError(f"ffmpeg exited {result}")
    out.with_suffix(".json").write_text(json.dumps(dict(source_episode=str(folder.resolve()), source_meta=meta,
        status="validated_source_reference" if args.accepted else "source_success_acp_quarantined",
        fps=25, frames=count, duration_s=count/25, views_of_same_episode=2), indent=2) + "\n")
    print(f"VIDEO {out}", flush=True)


if __name__ == "__main__":
    main()
