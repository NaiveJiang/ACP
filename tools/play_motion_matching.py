"""Viser playback of motion references or recorded physical policy evaluations."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import viser
from roban_motion_matching import JOINT_NAMES, load_model


def load_rollout(path):
    clips = {}
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["robot_id"].item()) != "roban":
            raise ValueError("Expected RoBan rollout")
        if archive["joint_names"].tolist() != JOINT_NAMES:
            raise ValueError("Rollout joint order differs from RoBan")
        for index, name in enumerate(archive["names"].tolist()):
            alive = archive["alive"][:, index]
            count = int(alive.sum())
            if count < 1 or not alive[:count].all() or alive[count:].any():
                raise ValueError("Expected a nonempty continuous alive prefix")
            poses = np.c_[
                archive["root_p"][:count, index],
                archive["root_quat"][:count, index],
                archive["q"][:count, index],
            ]
            boxes = archive["boxes"][index]
            active = boxes[boxes[:, 5] > 0]
            if len(active) != 1 or active.shape[1] != 9 or active[0, 8] != 0:
                raise ValueError("This viewer expects one box and no heightfield")
            box = active[0]
            yaw = np.arctan2(box[4], box[3])
            terrain = np.array(
                [
                    box[0],
                    box[1],
                    box[2] - box[7],
                    np.cos(yaw / 2),
                    0,
                    0,
                    np.sin(yaw / 2),
                    2 * box[5],
                    2 * box[6],
                    2 * box[7],
                ]
            )
            completed = (
                not bool(archive["failed"][index])
                if "failed" in archive
                else count >= int(archive["clip_len"][index]) - 1
            )
            status = (
                "policy rollout (completed)"
                if completed
                else "policy rollout (terminated)"
            )
            clips[name] = {
                "joint_pos": poses,
                "stage": np.full(count, status),
                "seam_frames": np.array([], dtype=int),
                "terrain": terrain,
            }
    return clips


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=root / "data/motion_matching_preview"
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--rollout",
        type=Path,
        help="Recorded policy evaluation, not reference animation",
    )
    parser.add_argument(
        "--robot",
        type=Path,
        default=root
        / "data/roban_description/xml/biped_s17_verified_handball_fixedhead.xml",
    )
    args = parser.parse_args()
    manifest = (
        None
        if args.rollout
        else json.loads((args.directory / "manifest.json").read_text())
    )
    model = load_model(args.robot if args.rollout else manifest["robot_xml"])
    data = mujoco.MjData(model)
    addresses = [model.jnt_qposadr[model.joint(name).id] for name in JOINT_NAMES]
    clips = load_rollout(args.rollout) if args.rollout else {}
    for clip in [] if args.rollout else manifest["clips"]:
        name = clip["name"]
        with np.load(args.directory / f"{name}_motion.npz") as archive:
            clips[name] = {
                key: archive[key] for key in ("joint_pos", "stage", "seam_frames")
            }
        clips[name]["terrain"] = np.load(args.directory / f"{name}_terrain.npy")[0, 0]
    if args.check:
        for clip in clips.values():
            for state in clip["joint_pos"]:
                data.qpos[:7], data.qpos[addresses] = state[:7], state[7:]
                mujoco.mj_kinematics(model, data)
                if not np.isfinite(data.xpos).all():
                    raise ValueError("Nonfinite playback pose")
        print(f"Playback check passed: {len(clips)} clips")
        return
    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=20, height=20, cell_size=0.25)
    geom_frames = []
    for index in range(model.ngeom):
        if (
            model.geom_type[index] != mujoco.mjtGeom.mjGEOM_MESH
            or model.geom_group[index] != 1
        ):
            continue
        mesh_id = model.geom_dataid[index]
        vertex_start, vertex_count = (
            model.mesh_vertadr[mesh_id],
            model.mesh_vertnum[mesh_id],
        )
        face_start, face_count = (
            model.mesh_faceadr[mesh_id],
            model.mesh_facenum[mesh_id],
        )
        frame = server.scene.add_frame(f"/robot/geom_{index}", show_axes=False)
        server.scene.add_mesh_simple(
            f"/robot/geom_{index}/mesh",
            vertices=model.mesh_vert[vertex_start : vertex_start + vertex_count],
            faces=model.mesh_face[face_start : face_start + face_count],
            color=(195, 205, 220),
            flat_shading=False,
        )
        geom_frames.append((index, frame))
    box_frame = server.scene.add_frame("/box", show_axes=False)
    box_mesh = server.scene.add_mesh_trimesh(
        "/box/mesh", trimesh.creation.box(extents=(1, 1, 1))
    )
    del box_mesh
    server.gui.add_markdown(
        "## RoBan policy rollout\nRecorded physical simulation, using reference contact plans."
        if args.rollout
        else "## RoBan motion matching\nReference animation only — no dynamics.\n"
        "Box ±20° cases include explicit approach-path warping."
    )
    selector = server.gui.add_dropdown(
        "Sequence",
        options=list(clips),
        initial_value=(
            next(
                (name for name in clips if "full_original_yaw+00" in name),
                next(iter(clips)),
            )
        ),
    )
    play = server.gui.add_checkbox("Play", initial_value=True)
    speed = server.gui.add_slider(
        "Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0
    )
    timeline = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    info = server.gui.add_markdown("")
    previous_name = None
    fractional_frame = 0.0
    previous_time = time.monotonic()
    previous_slider = 0

    @server.on_client_connect
    def connected(client):
        client.camera.position = (7.0, -5.0, 3.0)
        client.camera.look_at = (3.0, 0.0, 0.7)

    while True:
        now = time.monotonic()
        elapsed = min(now - previous_time, 0.1)
        previous_time = now
        name = selector.value
        clip = clips[name]
        count = len(clip["joint_pos"])
        if name != previous_name:
            fractional_frame = 0.0
            timeline.max = count - 1
            timeline.value = previous_slider = 0
            box = clip["terrain"]
            box_frame.position = box[:3]
            box_frame.wxyz = box[3:7]
            server.scene.add_mesh_trimesh(
                "/box/mesh", trimesh.creation.box(extents=box[7:10])
            )
            path = clip["joint_pos"][::5, :3]
            server.scene.add_line_segments(
                "/root_path",
                points=np.stack((path[:-1], path[1:]), axis=1),
                colors=(80, 180, 100),
                line_width=3.0,
            )
            previous_name = name
        if timeline.value != previous_slider:
            fractional_frame = float(timeline.value)
        if play.value:
            fractional_frame = (fractional_frame + elapsed * 50 * speed.value) % count
        frame = int(fractional_frame)
        timeline.value = previous_slider = frame
        state = clip["joint_pos"][frame]
        data.qpos[:7], data.qpos[addresses] = state[:7], state[7:]
        mujoco.mj_kinematics(model, data)
        for index, handle in geom_frames:
            body = model.geom_bodyid[index]
            rotation = data.xmat[body].reshape(3, 3)
            handle.position = data.xpos[body] + rotation @ model.geom_pos[index]
            quaternion = np.zeros(4)
            mujoco.mju_mulQuat(quaternion, data.xquat[body], model.geom_quat[index])
            handle.wxyz = quaternion
        info.content = (
            f"**{clip['stage'][frame]}** · {frame / 50:.2f} / {(count - 1) / 50:.2f} s\n\n"
            f"Seams: {clip['seam_frames'].tolist()}\n\n"
            + (
                "Recorded policy simulation; replay loops after completion/termination."
                if args.rollout
                else "Inspect feet/support and seams. Not dynamics-validated."
            )
        )
        time.sleep(1 / 50)


if __name__ == "__main__":
    main()
