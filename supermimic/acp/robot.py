"""One robot per ACP process; select before importing the learning modules.

Use ``python -m supermimic.acp --robot roban <command> ...`` or set ACP_ROBOT.
Network dimensions are import-time constants, so runtime switching is unsupported.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

ROBOT_ID = os.environ.get("ACP_ROBOT", "g1").lower()
if ROBOT_ID not in ("g1", "roban"):
    raise ValueError(f"Unknown ACP_ROBOT={ROBOT_ID!r}; expected g1 or roban")
_backend = importlib.import_module(f".{ROBOT_ID}", __package__)
JOINT_NAMES = list(_backend.G1_JOINT_NAMES if ROBOT_ID == "g1" else _backend.JOINT_NAMES)
NUM_J = len(JOINT_NAMES)
MJCF_PATH, MOTION_DIR, MOTION_GLOB = _backend.MJCF_PATH, _backend.MOTION_DIR, _backend.MOTION_GLOB
TRACKED_BODIES, CONTACT_BODIES = _backend.TRACKED_BODIES, _backend.CONTACT_BODIES
CONTACT_GEOMS = getattr(_backend, "CONTACT_GEOMS", {})
NB = len(CONTACT_BODIES)
ROOT_BODY = getattr(_backend, "ROOT_BODY", "pelvis")
TORSO_BODY = getattr(_backend, "TORSO_BODY", "torso_link")
COM_BODY = getattr(_backend, "COM_BODY", TORSO_BODY)
STAND_HEIGHT = getattr(_backend, "STAND_HEIGHT", .78)
SOFT_JOINT_LIMIT_FACTOR = getattr(_backend, "SOFT_JOINT_LIMIT_FACTOR", .9)
get_spec, robot_cfg = _backend.get_spec, _backend.robot_cfg
action_scale, pd_params = _backend.action_scale, _backend.pd_params
default_joint_pos_list = _backend.default_joint_pos_list


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    joint_names: tuple[str, ...]
    root_body: str
    torso_body: str
    com_body: str
    tracked_bodies: tuple[str, ...]
    control_model: str


ROBOT = RobotSpec(ROBOT_ID, tuple(JOINT_NAMES), ROOT_BODY, TORSO_BODY, COM_BODY,
                  tuple(TRACKED_BODIES), getattr(_backend, "CONTROL_MODEL", "g1_implicit_pd_v1"))


def contact_geom_ids(model, profile="legacy"):
    """Non-overlapping, explicit geom ownership shared by rewards and labels."""
    if profile not in ("legacy", "roban_knee_caps_v1"):
        raise ValueError(f"Unknown contact profile: {profile}")
    if profile == "roban_knee_caps_v1" and ROBOT_ID != "roban":
        raise ValueError("RoBan knee-cap profile requires the RoBan asset")
    groups, claimed = [], set()
    for name, (_, links) in CONTACT_BODIES.items():
        if name in CONTACT_GEOMS:
            ids = [model.geom(n).id for n in CONTACT_GEOMS[name]]
        else:
            bodies = {model.body(n).id for n in links}
            ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies
                   and (model.geom_contype[g] or model.geom_conaffinity[g])]
        if profile == "roban_knee_caps_v1" and name in ("left_knee", "right_knee"):
            # The proximal knee cap is attached to the thigh, immediately above
            # the knee hinge. Keep its true moving frame; never attach it rigidly
            # to the shin in the differentiable reward geometry.
            short = "l" if name == "left_knee" else "r"
            ids.append(model.geom(f"leg_{short}3_link_collision1").id)
        if not ids or claimed.intersection(ids):
            raise ValueError(f"Invalid or overlapping collision group: {name}")
        claimed.update(ids)
        groups.append(ids)
    return groups


def friction_torque(velocity):
    if ROBOT_ID == "roban":
        import torch
        return _backend.FRICTION_STATIC * torch.tanh(velocity / _backend.ACTIVATION_VEL)
    return 0.0


def metadata(s17_actuators=False, *, primitive_soles=False, legacy_primitive_soles=False):
    """Portable checkpoint contract (no absolute asset or dataset paths)."""
    data = {"id": ROBOT_ID, "joint_names": JOINT_NAMES, "root_body": ROOT_BODY,
        "torso_body": TORSO_BODY, "com_body": COM_BODY, "tracked_bodies": TRACKED_BODIES,
        "contacts": CONTACT_BODIES, "contact_geoms": CONTACT_GEOMS, "contact_version": 2,
        "control_model": ROBOT.control_model, "control_dt": .02, "physics_dt": .005,
        "kp_kd_effort_velocity": pd_params(), "default_q": default_joint_pos_list(),
        "action_scale": [action_scale(n) for n in JOINT_NAMES], "stand_height": STAND_HEIGHT,
        "armature": [(_backend.PARAMS[n][4] if ROBOT_ID == "roban" else _backend.motor_of(n)[2]) for n in JOINT_NAMES],
        "passive_damping_override": .02 if ROBOT_ID == "roban" else None,
        "smooth_friction_static": getattr(_backend, "FRICTION_STATIC", 0.),
        "friction_activation_velocity": getattr(_backend, "ACTIVATION_VEL", None),
        "soft_joint_limit_factor": SOFT_JOINT_LIMIT_FACTOR,
        "mjcf_sha256": hashlib.sha256(Path(MJCF_PATH).read_bytes()).hexdigest(),
        "state_dim": 13 + 2 * NUM_J, "xi_dim": NUM_J + 4,
        "friction_policy": "acp_uniform_geom_friction", "delay_steps": 0}
    if s17_actuators:
        if ROBOT_ID != "roban":
            raise ValueError("S17 actuator metadata requires the RoBan robot")
        data.update(_backend.s17_control_metadata())
    if primitive_soles:
        if ROBOT_ID != "roban":
            raise ValueError("Primitive-sole metadata requires the RoBan robot")
        data["primitive_sole"] = _backend.primitive_sole_metadata()
        source_xml = Path(_backend.PRIMITIVE_MJCF_PATH)
        sha = hashlib.sha256(source_xml.read_bytes()).hexdigest()
        if not legacy_primitive_soles or sha != _backend.LEGACY_PRIMITIVE_MJCF_SHA256:
            data["primitive_mjcf_sha256"] = sha
    # Normalize tuples for stable equality after JSON/torch serialization.
    data = json.loads(json.dumps(data))
    data["fingerprint"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    return data
