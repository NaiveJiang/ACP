"""Read solved physics contacts; never regenerate forces from a reference pose."""

import mujoco
import mujoco_warp
import numpy as np
import warp as wp

from .env import DECIMATION
from .robot import CONTACT_BODIES, contact_geom_ids, get_spec

RAW_COLUMNS = (
    "state_frame",
    "substep",
    "robot_geom",
    "terrain_geom",
    "body",
    "surface",
    "distance_m",
    "normal_force_n",
    "px",
    "py",
    "pz",
    "nx",
    "ny",
    "nz",
    "fx",
    "fy",
    "fz",
    "tx",
    "ty",
    "tz",
)


def aggregate_contacts(records, threshold=5.0, substeps=DECIMATION):
    count = len(CONTACT_BODIES)
    normal_force = np.zeros(count)
    force = np.zeros((count, 3))
    position = np.full((count, 3), np.nan)
    normal = np.full((count, 3), np.nan)
    distance = np.full(count, np.nan)
    surface = np.full(count, -1, dtype=np.int32)
    for body in range(count):
        group = records[records[:, 4] == body]
        if not len(group):
            continue
        normal_force[body] = group[:, 7].sum() / substeps
        force[body] = group[:, 14:17].sum(0) / substeps
        strongest = group[group[:, 7].argmax()]
        position[body], normal[body] = strongest[8:11], strongest[11:14]
        distance[body], surface[body] = strongest[6], int(strongest[5])
    return {
        "normal_force": normal_force,
        "force": force,
        "position": position,
        "normal": normal,
        "distance": distance,
        "surface": surface,
        "contact": normal_force >= threshold,
    }


class SolvedContactRecorder:
    def __init__(self, env, threshold=5.0):
        if env.N != 1:
            raise ValueError(
                "Offline contact capture uses one world to preserve episode identity"
            )
        self.threshold = threshold
        self.frame = 0
        self.chunks = []
        canonical = (
            get_spec(primitive_soles=True) if env.primitive_soles else get_spec()
        ).compile()
        model = env.sim.mj_model
        self.owner = {}
        for body, group in enumerate(contact_geom_ids(canonical, "roban_knee_caps_v1")):
            for geom in group:
                name = canonical.geom(geom).name
                full = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{name}"
                )
                if full < 0:
                    raise ValueError(f"Missing simulation contact geometry: {name}")
                self.owner[full] = body
        self.robot_geoms = set(env.robot.indexing.geom_ids.cpu().tolist())
        self.surface = {int(geom): 0 for geom in env._terrain_geoms.cpu().tolist()}
        self.surface.update(
            {
                int(geom): index + 1
                for index, geom in enumerate(env.box_geoms.cpu().tolist())
            }
        )
        capacity = env.sim.wp_data.naconmax
        self.ids = wp.array(
            np.arange(capacity, dtype=np.int32), dtype=wp.int32, device=env.device
        )
        self.forces = wp.zeros(capacity, dtype=wp.spatial_vector, device=env.device)

    def begin(self, frame):
        self.frame = frame
        self.chunks = []

    def __call__(self, env, substep):
        data = env.sim.wp_data
        count = int(data.nacon.numpy()[0])
        if count > data.naconmax:
            raise RuntimeError(
                "Contact capacity overflow; refusing an incomplete force capture"
            )
        if count == 0:
            return
        mujoco_warp.contact_force(env.sim.wp_model, data, self.ids, True, self.forces)
        forces = self.forces.numpy()[:count]
        pairs = data.contact.geom.numpy()[:count]
        positions = data.contact.pos.numpy()[:count]
        frames = data.contact.frame.numpy()[:count]
        distances = data.contact.dist.numpy()[:count]
        rows = []
        for index, (first, second) in enumerate(pairs):
            if first in self.robot_geoms and second in self.surface:
                robot, terrain, sign = first, second, -1
            elif second in self.robot_geoms and first in self.surface:
                robot, terrain, sign = second, first, 1
            else:
                continue
            normal = frames[index, 0] * sign
            force = forces[index, :3] * sign
            torque = forces[index, 3:] * sign
            magnitude = max(0.0, float(np.dot(force, normal)))
            if not np.isfinite(
                np.r_[positions[index], normal, force, torque, distances[index]]
            ).all():
                raise FloatingPointError("Nonfinite solved contact")
            rows.append(
                [
                    self.frame,
                    substep,
                    robot,
                    terrain,
                    self.owner.get(int(robot), -2),
                    self.surface[int(terrain)],
                    distances[index],
                    magnitude,
                    *positions[index],
                    *normal,
                    *force,
                    *torque,
                ]
            )
        if rows:
            self.chunks.append(np.asarray(rows))

    def finish(self):
        records = (
            np.concatenate(self.chunks)
            if self.chunks
            else np.empty((0, len(RAW_COLUMNS)))
        )
        return aggregate_contacts(records, self.threshold), records
