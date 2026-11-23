"""Align planted RoBan soles with the box top in an ascent reference."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def sole_vertices(model, body):
    meshes = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body or model.geom_group[geom] != 3:
            continue
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("Expected a mesh collision sole")
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        rotation = Rotation.from_quat(model.geom_quat[geom, [1, 2, 3, 0]])
        meshes.append(
            rotation.apply(model.mesh_vert[start : start + count])
            + model.geom_pos[geom]
        )
    vertices = np.concatenate(meshes)
    return vertices[vertices[:, 2] <= vertices[:, 2].min() + 0.002]


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value**2 * (3.0 - 2.0 * value)


def align_ascent_soles(model, poses, center, half):
    """Preserve foot heading and planar position, shifting only stance leg joints."""
    data = mujoco.MjData(model)
    body_ids = [model.body(f"leg_{side}6_link").id for side in "lr"]
    soles = [sole_vertices(model, body) for body in body_ids]
    anchors = [
        np.r_[vertices.mean(axis=0)[:2], vertices[:, 2].min()] for vertices in soles
    ]
    leg_ids = [
        [model.joint(f"leg_{side}{index}_joint").id for index in range(1, 7)]
        for side in "lr"
    ]
    top = center[2] + half[2]
    weights = np.zeros((len(poses), 2))
    targets = []
    for frame, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        frame_targets = []
        for foot, (body, vertices, anchor) in enumerate(
            zip(body_ids, soles, anchors, strict=True)
        ):
            rotation = data.xmat[body].reshape(3, 3).copy()
            point = data.xpos[body] + rotation @ anchor
            sole = vertices @ rotation.T + data.xpos[body]
            margin = (half[:2] - np.abs(sole[:, :2] - center[:2])).min()
            gap = sole[:, 2].min() - top
            tilt = np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))
            weights[frame, foot] = (
                smoothstep(margin / 0.02)
                * smoothstep((0.018 - abs(gap)) / 0.010)
                * smoothstep((np.deg2rad(14) - tilt) / np.deg2rad(8))
            )
            heading = np.arctan2(rotation[1, 0], rotation[0, 0])
            flat = Rotation.from_euler("z", heading)
            difference = (flat * Rotation.from_matrix(rotation).inv()).as_rotvec()
            frame_targets.append((point, rotation, difference))
        targets.append(frame_targets)

    result = poses.copy()
    jac_pos, jac_rot = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for frame, pose in enumerate(poses):
        data.qpos[:] = pose
        for foot, (body, anchor, ids) in enumerate(
            zip(body_ids, anchors, leg_ids, strict=True)
        ):
            weight = weights[frame, foot]
            if weight <= 1e-6:
                continue
            origin, rotation, difference = targets[frame][foot]
            target = origin.copy()
            target[2] += weight * (top + 0.0005 - target[2])
            orientation = Rotation.from_rotvec(
                weight * difference
            ) * Rotation.from_matrix(rotation)
            joint_pos, joint_vel = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
            lower, upper = model.jnt_range[ids].T
            for _ in range(12):
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                current = data.xmat[body].reshape(3, 3)
                point = data.xpos[body] + current @ anchor
                twist = (orientation * Rotation.from_matrix(current).inv()).as_rotvec()
                error = np.r_[target - point, 0.15 * twist]
                if np.linalg.norm(error) < 1e-5:
                    break
                mujoco.mj_jac(model, data, jac_pos, jac_rot, point, body)
                jacobian = np.vstack(
                    (jac_pos[:, joint_vel], 0.15 * jac_rot[:, joint_vel])
                )
                update = np.linalg.solve(
                    jacobian.T @ jacobian + 1e-5 * np.eye(6), jacobian.T @ error
                )
                data.qpos[joint_pos] = np.clip(
                    data.qpos[joint_pos] + np.clip(update, -0.04, 0.04), lower, upper
                )
        result[frame] = data.qpos
    return result
