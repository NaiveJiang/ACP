"""RoBan S2, 21 joints. Adapted from kuavo_amp_mjlab feb3a70a.

The legacy ACP profile uses direct PD with smooth dry friction. An optional
S17 profile uses the native teacher's motor parameters and torque-speed curves.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from xml.etree import ElementTree as ET

import mujoco
import torch
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg, pd_torque
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from .s17_actuation import ACTION_SCALE, CURVES, DELAY_STEPS, GAINS
from .s17_actuation import limit_torque, torque_curve

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO, "data")
DESCRIPTION_DIR = os.environ.get("ACP_ROBAN_DESCRIPTION", os.path.join(DATA_DIR, "roban_description"))
MJCF_PATH = os.path.join(DESCRIPTION_DIR, "xml", "biped_s17_verified_handball_fixedhead.xml")
PRIMITIVE_MJCF_PATH = os.path.join(
    os.path.dirname(__file__), "assets", "biped_s17_verified_handball_fixedhead_primitive_soles.xml"
)
# This authored XML compiles to the same model as the former runtime replacement.
LEGACY_PRIMITIVE_MJCF_SHA256 = "7d16fcca6a98b552225dc4599bb39146eb6344264b992235ac1fc0814c180fcb"
MOTION_DIR = os.path.join(DATA_DIR, "roban_motion")
MOTION_GLOB = "*/*_motion.npz"
JOINT_NAMES = (["waist_yaw_joint"] + [f"leg_{s}{i}_joint" for s in "lr" for i in range(1, 7)]
               + [f"zarm_{s}{i}_joint" for s in "lr" for i in range(1, 5)])
NUM_J = len(JOINT_NAMES)
ROOT_BODY, TORSO_BODY, COM_BODY = "base_link", "base_link", "waist_yaw_link"
STAND_HEIGHT = 0.81
SOFT_JOINT_LIMIT_FACTOR = 0.95
CONTROL_MODEL = "explicit_pd_smooth_friction_no_delay_v1"
FRICTION_STATIC, ACTIVATION_VEL = 0.2, 0.1
TRACKED_BODIES = [
    "base_link", "leg_l2_link", "leg_l4_link", "leg_l6_link",
    "leg_r2_link", "leg_r4_link", "leg_r6_link", "waist_yaw_link",
    "zarm_l2_link", "zarm_l4_link", "left_hand_contact",
    "zarm_r2_link", "zarm_r4_link", "right_hand_contact",
]
CONTACT_BODIES = {
    "left_foot": ("leg_l6_link", ("leg_l6_link",)),
    "right_foot": ("leg_r6_link", ("leg_r6_link",)),
    "left_knee": ("leg_l4_link", ("leg_l4_link",)),
    "right_knee": ("leg_r4_link", ("leg_r4_link",)),
    "left_hand": ("left_hand_contact", ("zarm_l4_link",)),
    "right_hand": ("right_hand_contact", ("zarm_r4_link",)),
    "left_elbow": ("zarm_l4_link", ("zarm_l4_link",)),
    "right_elbow": ("zarm_r4_link", ("zarm_r4_link",)),
    "pelvis": ("waist_yaw_link", ("waist_yaw_link",)),
}
# Geom-level ownership overrides prevent the hand sphere from belonging to the elbow.
def _elbow_geometries(side):
    if not os.path.isfile(MJCF_PATH):
        return (f"{side}_elbow1_collision", f"{side}_elbow2_collision")
    short = "l" if side == "left" else "r"
    body = ET.parse(MJCF_PATH).getroot().find(f".//body[@name='zarm_{short}4_link']")
    if body is None:
        raise ValueError(f"Missing {side} forearm in RoBan asset")
    count = sum(
        not (geom.get("contype") == "0" and geom.get("conaffinity") == "0")
        and geom.get("type", "sphere") != "sphere"
        for geom in body.findall("geom")
    )
    if count < 1:
        raise ValueError(f"No {side} forearm collision geometry")
    return tuple(f"{side}_elbow{index + 1}_collision" for index in range(count))


CONTACT_GEOMS = {
    f"{side}_hand": (f"{side}_hand_collision",) for side in ("left", "right")
} | {
    f"{side}_elbow": _elbow_geometries(side)
    for side in ("left", "right")
}
NB = len(CONTACT_BODIES)

# kp, kd, effort (Nm), speed (rad/s), reflected armature, action scale.
_WAIST = (40.179, 2.557, 80., 12., .031787, .25)
_LEG = [
    (90.179, 3.557, 150., 14.6, .0493, .25),
    (150.098, 8.308, 150., 14.6, .0493, .25),
    (40.179, 3.557, 70., 12., .031787, .25),
    (150.098, 8.308, 150., 14.6, .0493, .35),
    (34.25, 1.907, 74., 17., .036, .25),
    (34.25, 1.907, 74., 17., .036, .25),
]
_ARM = [(14.25, .907, 14.1, 10.5, .015, .25)] + [
    (14.25, .907, 37., 15., .018, scale) for scale in (.05, .05, .25)
]
PARAMS = dict(zip(JOINT_NAMES, [_WAIST] + _LEG * 2 + _ARM * 2))
DEFAULT_JOINT_POS = dict(zip(JOINT_NAMES, [0., -.285, -.0194, -.201, .404, -.221, 0.,
    .285, .0194, .201, .404, -.221, 0., 0., .05, 0., -.2, 0., -.05, 0., -.2]))


def default_joint_pos_list():
    return [DEFAULT_JOINT_POS[n] for n in JOINT_NAMES]


def action_scale(name):
    return PARAMS[name][5]


def pd_params():
    return tuple([PARAMS[n][i] for n in JOINT_NAMES] for i in range(4))


def s17_control_metadata():
    return {
        "control_model": "s17_torque_speed_delayed_pd_v1",
        "kp_kd_effort_velocity": (
            [gains[0] for gains in GAINS],
            [gains[1] for gains in GAINS],
            *pd_params()[2:],
        ),
        "armature": [gains[2] for gains in GAINS],
        "action_scale": [ACTION_SCALE] * NUM_J,
        "delay_steps": DELAY_STEPS,
        "torque_speed_curves_rpm": CURVES,
        "xi_dim": NUM_J + 4 + DELAY_STEPS * NUM_J,
    }


def primitive_sole_metadata():
    root = ET.parse(PRIMITIVE_MJCF_PATH).getroot()
    geoms = root.findall(".//body[@name='leg_l6_link']/geom[@class='collision']")
    positions = [tuple(map(float, geom.attrib["pos"].split())) for geom in geoms]
    return {
        "center_z": positions[0][2],
        "geoms": [
            (geom.attrib["type"], pos[0], tuple(map(float, geom.attrib["size"].split())))
            for geom, pos in zip(geoms, positions, strict=True)
        ],
    }


def get_spec(*, self_collision=False, primitive_soles=False) -> mujoco.MjSpec:
    xml = PRIMITIVE_MJCF_PATH if primitive_soles else MJCF_PATH
    root = ET.parse(xml).getroot()
    root.find("compiler").set("meshdir", os.path.abspath(os.path.join(os.path.dirname(MJCF_PATH), "../meshes")))
    root.remove(root.find("actuator"))
    # The source training asset also drops the unused radar branch.
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "body" and child.get("name") == "head_radar":
                parent.remove(child)
    asset = root.find("asset")
    for mesh in list(asset):
        if mesh.tag == "mesh" and mesh.get("name") == "head_radar":
            asset.remove(mesh)
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    for body in spec.bodies:
        hand_side = {"l_handball": "left", "r_handball": "right"}.get(body.name)
        collisions = [g for g in body.geoms if g.contype or g.conaffinity]
        for i, geom in enumerate(collisions, 1):
            geom.name = f"{body.name}_collision{i}"
            if not self_collision:
                geom.contype, geom.conaffinity = 0, 1
            geom.group = 3
        if hand_side is not None and collisions:
            if len(collisions) != 1:
                raise ValueError("Handball body must have one collision geometry")
            body.add_body(name=f"{hand_side}_hand_contact", pos=(0., 0., 0.))
            collisions[0].name = f"{hand_side}_hand_collision"
        for short, side in (("l", "left"), ("r", "right")):
            if body.name == f"zarm_{short}4_link":
                elbows = 0
                for geom in collisions:
                    if geom.type == mujoco.mjtGeom.mjGEOM_SPHERE:
                        geom.name = f"{side}_hand_collision"
                        body.add_body(name=f"{side}_hand_contact", pos=geom.pos.copy())
                    else:
                        elbows += 1
                        geom.name = f"{side}_elbow{elbows}_collision"
    # Also apply passive parameters to CPU models used for replay/validation.
    for name in JOINT_NAMES:
        joint = spec.joint(name)
        joint.armature = PARAMS[name][4]
        joint.damping[:] = (.02, 0., 0.)
        joint.frictionloss = 0.
    return spec


@dataclass(kw_only=True)
class RobanPdActuatorCfg(IdealPdActuatorCfg):
    def build(self, entity, target_ids, target_names):
        return RobanPdActuator(self, entity, target_ids, target_names)


class RobanPdActuator(IdealPdActuator):
    @staticmethod
    def control_law(params, cmd):
        tau = pd_torque(params["stiffness"], params["damping"], cmd)
        tau = tau - FRICTION_STATIC * torch.tanh(cmd.vel / ACTIVATION_VEL)
        return torch.clamp(tau, -params["force_limit"], params["force_limit"])


@dataclass(kw_only=True)
class S17PdActuatorCfg(RobanPdActuatorCfg):
    speed_limit: float
    curve: tuple[tuple[float, float], ...] | None

    def build(self, entity, target_ids, target_names):
        return S17PdActuator(self, entity, target_ids, target_names)


class S17PdActuator(RobanPdActuator):
    def initialize(self, mj_model, model, data, device):
        super().initialize(mj_model, model, data, device)
        self.speed, self.ratio = torque_curve(
            self.cfg.curve, self.cfg.speed_limit, device
        )

    def compute(self, cmd):
        return limit_torque(
            super().compute(cmd), cmd.vel, self.force_limit, self.speed, self.ratio
        )


def robot_cfg(s17_actuators=False, *, self_collision=False, primitive_soles=False):
    if s17_actuators:
        groups = {}
        for name, gains, curve in zip(JOINT_NAMES, GAINS, CURVES, strict=True):
            groups.setdefault((*gains, *PARAMS[name][2:4], curve), []).append(name)
        acts = tuple(
            S17PdActuatorCfg(
                target_names_expr=tuple(names), stiffness=kp, damping=kd,
                armature=inertia, effort_limit=effort, speed_limit=speed, curve=curve,
                frictionloss=0.0, viscous_damping=0.02,
            )
            for (kp, kd, inertia, effort, speed, curve), names in groups.items()
        )
    else:
        acts = tuple(RobanPdActuatorCfg(target_names_expr=(n,), stiffness=PARAMS[n][0], damping=PARAMS[n][1],
            effort_limit=PARAMS[n][2], armature=PARAMS[n][4], frictionloss=0., viscous_damping=.02) for n in JOINT_NAMES)
    return EntityCfg(spec_fn=partial(get_spec, self_collision=self_collision,
                                    primitive_soles=primitive_soles),
        init_state=EntityCfg.InitialStateCfg(pos=(0., 0., STAND_HEIGHT), joint_pos=DEFAULT_JOINT_POS, joint_vel={".*": 0.}),
        articulation=EntityArticulationInfoCfg(actuators=acts, soft_joint_pos_limit_factor=SOFT_JOINT_LIMIT_FACTOR))
