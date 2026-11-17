"""RoBan S2, 21 joints. Adapted from kuavo_amp_mjlab feb3a70a.

The ACP variant uses direct PD with smooth dry friction, no command delay,
and terrain-only collisions. It does not import the upstream mjlab fork.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from xml.etree import ElementTree as ET

import mujoco
import torch
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg, pd_torque
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO, "data")
MJCF_PATH = os.path.join(DATA_DIR, "roban_description", "xml", "biped_s17_verified_handball_fixedhead.xml")
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
CONTACT_GEOMS = {
    f"{side}_hand": (f"{side}_hand_collision",) for side in ("left", "right")
} | {
    f"{side}_elbow": (f"{side}_elbow1_collision", f"{side}_elbow2_collision")
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


def get_spec() -> mujoco.MjSpec:
    root = ET.parse(MJCF_PATH).getroot()
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
        collisions = [g for g in body.geoms if g.contype or g.conaffinity]
        for i, geom in enumerate(collisions, 1):
            geom.name = f"{body.name}_collision{i}"
            geom.contype, geom.conaffinity, geom.group = 0, 1, 3
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


def robot_cfg():
    acts = tuple(RobanPdActuatorCfg(target_names_expr=(n,), stiffness=PARAMS[n][0], damping=PARAMS[n][1],
        effort_limit=PARAMS[n][2], armature=PARAMS[n][4], frictionloss=0., viscous_damping=.02) for n in JOINT_NAMES)
    return EntityCfg(spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0., 0., STAND_HEIGHT), joint_pos=DEFAULT_JOINT_POS, joint_vel={".*": 0.}),
        articulation=EntityArticulationInfoCfg(actuators=acts, soft_joint_pos_limit_factor=SOFT_JOINT_LIMIT_FACTOR))
