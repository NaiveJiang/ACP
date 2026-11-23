"""Bake RoBan LAFAN1 poses into a Blender file for visual review."""
import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "logs/lafan1_walk_reference_001/reference/manifest.json"


def read_clips(args, robot):
    if args.csv is not None:
        pose = robot.load(args.csv, args.fps)
        stop = len(pose) if args.stop is None else args.stop
        if not 0 <= args.start < stop <= len(pose):
            raise ValueError(f"Require 0 <= start < stop <= {len(pose)}")
        yield args.csv.stem, args.start, stop, pose[args.start:stop], ""
        return

    manifest = args.manifest or DEFAULT_MANIFEST
    for clip in json.loads(manifest.read_text())["clips"]:
        path = manifest.parent / (clip["case"] + "_labels.npz")
        with np.load(path) as data:
            variant = ""
            if "mirrored" in clip:
                variant = clip["category"] + (" mirrored" if clip["mirrored"] else "")
            yield (clip["raw_motion"], clip["source_row_start"],
                   clip["source_row_stop"], data["qpos"], variant)


def export_payload(args, path):
    import mujoco

    from supermimic.acp.footstep.motion_matching import Robot

    robot = Robot()
    model = robot.model
    root_body = int(model.jnt_bodyid[robot.free])
    bodies, visuals, arrays = [], [], {}
    for body in range(1, model.nbody):
        joint = int(model.body_jntadr[body])
        info = dict(name=model.body(body).name, parent=int(model.body_parentid[body]),
                    pos=model.body_pos[body].tolist(), quat=model.body_quat[body].tolist())
        if joint >= 0 and body != root_body:
            if model.body_jntnum[body] != 1 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"Expected one hinge on {info['name']}")
            info.update(joint=int(model.jnt_qposadr[joint]), anchor=model.jnt_pos[joint].tolist(),
                        axis=model.jnt_axis[joint].tolist(), joint_name=model.joint(joint).name)
        bodies.append(info)

    for geom in range(model.ngeom):
        mesh = int(model.geom_dataid[geom])
        if model.geom_group[geom] != 1 or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mat = model.geom_matid[geom]
        rgba = model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[geom]
        visuals.append(dict(name=model.mesh(mesh).name, mesh=mesh, body=int(model.geom_bodyid[geom]),
                            pos=model.geom_pos[geom].tolist(), quat=model.geom_quat[geom].tolist(),
                            rgba=rgba.tolist()))
        v, f = model.mesh_vertadr[mesh], model.mesh_faceadr[mesh]
        arrays[f"verts_{mesh}"] = model.mesh_vert[v:v+model.mesh_vertnum[mesh]]
        arrays[f"faces_{mesh}"] = model.mesh_face[f:f+model.mesh_facenum[mesh]]

    marks, poses = [], []
    frame = 1
    for source, start, stop, pose, variant in read_clips(args, robot):
        if pose.ndim != 2 or pose.shape[1] != model.nq or not len(pose):
            raise ValueError(f"Invalid RoBan poses in {source}")
        pose = pose.copy()
        # Place each clip's starting point at the scene origin for review.
        pose[:, robot.root:robot.root+2] -= pose[0, robot.root:robot.root+2].copy()
        q = pose[:, robot.root+3:robot.root+7]
        flip = np.where(np.sum(q[1:]*q[:-1], axis=1) < 0., -1., 1.)
        q[1:] *= np.cumprod(flip)[:, None]
        marks.append(dict(source=source, start=start, stop=stop, frame=frame, length=len(pose), variant=variant))
        poses.append(pose)
        frame += len(pose)
    if not poses:
        raise ValueError("No motion clips to preview")

    fps = args.fps if args.csv else json.loads(
        ((args.manifest or DEFAULT_MANIFEST).parent.parent / "metadata.json").read_text())["sample_hz"]
    source_kind = "raw CSV" if args.csv else "training reference"
    if any(mark["variant"] for mark in marks):
        source_kind = "Motion Matching preview"
    info = dict(bodies=bodies, visuals=visuals, marks=marks, root_body=root_body, root_qpos=robot.root,
                fps=fps, source_kind=source_kind)
    np.savez_compressed(path, poses=np.concatenate(poses), info=json.dumps(info), **arrays)


def make_blend(payload, output):
    import bpy
    from mathutils import Vector

    def empty(name, parent=None, position=(0., 0., 0.)):
        obj = bpy.data.objects.new(name, None)
        scene.collection.objects.link(obj)
        obj.empty_display_size = .04
        obj.parent = parent
        obj.location = position
        return obj

    def curves(obj, channels):
        action = bpy.data.actions.new(obj.name + " motion")
        slot = action.slots.new("OBJECT", obj.name)
        strip = action.layers.new("LAFAN1").strips.new(type="KEYFRAME")
        bag = strip.channelbag(slot, ensure=True)
        obj.animation_data_create()
        obj.animation_data.action = action
        obj.animation_data.action_slot = slot
        for name, channel, values in channels:
            curve = bag.fcurves.new(name, index=channel)
            curve.keyframe_points.add(len(values))
            coords[:, 1] = values
            curve.keyframe_points.foreach_set("co", coords.ravel())
            curve.keyframe_points.foreach_set("interpolation", interpolation)
            curve.update()

    scene = bpy.context.scene
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    info = json.loads(payload["info"].item())
    poses = payload["poses"]
    coords = np.zeros((len(poses), 2), dtype=np.float32)
    coords[:, 0] = np.arange(1, len(poses)+1)
    modes = bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items
    interpolation = np.full(len(poses), modes["LINEAR"].value, dtype=np.int32)
    for mark in info["marks"][1:]:
        interpolation[mark["frame"]-2] = modes["CONSTANT"].value

    bodies = {0: None}
    for index, body in enumerate(info["bodies"], 1):
        parent = bodies[body["parent"]]
        if index == info["root_body"]:
            root = empty(body["name"], parent)
            root.rotation_mode = "QUATERNION"
            q = info["root_qpos"]
            curves(root, [("location", i, poses[:, q+i]) for i in range(3)]
                   + [("rotation_quaternion", i, poses[:, q+3+i]) for i in range(4)])
            obj = root
        else:
            rest = empty(body["name"], parent, body["pos"])
            rest.rotation_mode = "QUATERNION"
            rest.rotation_quaternion = body["quat"]
            obj = rest
            if "joint" in body:
                pivot = empty(body["joint_name"], rest, body["anchor"])
                pivot.rotation_mode = "AXIS_ANGLE"
                pivot.rotation_axis_angle = (0., *body["axis"])
                curves(pivot, [("rotation_axis_angle", 0, poses[:, body["joint"]])])
                obj = empty(body["name"] + " visual frame", pivot, -Vector(body["anchor"]))
        bodies[index] = obj

    materials = {}
    for visual in info["visuals"]:
        mesh_id = visual["mesh"]
        mesh = bpy.data.meshes.new(visual["name"])
        mesh.from_pydata(payload[f"verts_{mesh_id}"].tolist(), [], payload[f"faces_{mesh_id}"].tolist())
        mesh.update()
        obj = bpy.data.objects.new(visual["name"], mesh)
        scene.collection.objects.link(obj)
        obj.parent = bodies[visual["body"]]
        obj.location = visual["pos"]
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = visual["quat"]
        rgba = tuple(visual["rgba"])
        if rgba not in materials:
            color = bpy.data.materials.new(f"RoBan material {len(materials)+1}")
            color.diffuse_color = rgba
            materials[rgba] = color
        mesh.materials.append(materials[rgba])

    follow = empty("Camera follows RoBan XY")
    follow.hide_set(True)
    copy = follow.constraints.new("COPY_LOCATION")
    copy.target = root
    copy.use_z = False
    camera = bpy.data.objects.new("Review camera", bpy.data.cameras.new("Review camera"))
    scene.collection.objects.link(camera)
    camera.parent = follow
    camera.location = (2.8, -3.4, 1.8)
    camera.rotation_euler = (Vector((0., 0., .64))-camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 62
    scene.camera = camera

    scene.name = "RoBan LAFAN1 " + info["source_kind"]
    scene.render.fps = round(info["fps"])
    scene.render.fps_base = scene.render.fps / info["fps"]
    scene.frame_end = len(poses)
    legend = bpy.data.texts.new("LAFAN1 source index")
    legend.write("Blender frames are inclusive. CSV rows are 0-based, with the stop row excluded.\n\n")
    for mark in info["marks"]:
        start, stop, frame = mark["start"], mark["stop"], mark["frame"]
        source = mark["source"]
        variant = f"{mark['variant']} | " if mark["variant"] else ""
        scene.timeline_markers.new(f"{variant}{source} [{start}:{stop}]", frame=frame)
        legend.write(f"{frame:6d}-{frame+mark['length']-1:<6d}  {variant}{source}.csv [{start}:{stop}]\n")
    scene.frame_set(1)
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                space = area.spaces.active
                space.region_3d.view_perspective = "CAMERA"
                space.shading.color_type = "MATERIAL"
                space.overlay.show_extras = False
    bpy.ops.wm.save_as_mainfile(filepath=str(output), compress=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path, help="Preview a raw retargeted CSV")
    source.add_argument("--manifest", type=Path, help="Preview the prepared references listed in the manifest")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blender", default=shutil.which("blender") or str(Path.home()/"blender-5.2.0-linux-x64/blender"))
    parser.add_argument("--start", type=int, default=0, help="First raw data row")
    parser.add_argument("--stop", type=int, help="Exclusive last raw data row")
    parser.add_argument("--fps", type=float, default=30., help="Raw CSV sample rate")
    args = parser.parse_args()
    if not math.isfinite(args.fps) or args.fps < 1.:
        parser.error("--fps must be at least 1")
    if (args.start != 0 or args.stop is not None) and args.csv is None:
        parser.error("--start and --stop require --csv")
    output = args.out.resolve()
    if output.suffix != ".blend" or output.exists():
        parser.error("--out must be a new .blend file")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lafan-blender-") as folder:
        payload = Path(folder)/"scene.npz"
        export_payload(args, payload)
        subprocess.run([args.blender, "-b", "--factory-startup", "--python-exit-code", "1",
                        "--python", __file__, "--", str(payload), str(output)], check=True)


if __name__ == "__main__":
    if "--" in sys.argv:
        payload, output = sys.argv[sys.argv.index("--")+1:]
        with np.load(payload) as data:
            make_blend(data, output)
    else:
        main()
