import os

os.environ["ACP_ROBOT"] = "roban"

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from supermimic.acp.collect_mimic import (
    contact_targets,
    save_success,
    success_reasons,
    terrain_array,
)
from supermimic.acp.env import surface_friction_values, validate_surface_friction
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.mdp import (
    MDP,
    OBS_DIM,
    XI_DIM,
    RewardCfg,
    flatten_state,
    normalize_xi,
)
from supermimic.acp.motion import MotionLib, measured_velocities
from supermimic.acp.reference_mimic import (
    ReferenceCfg,
    ReferencePolicy,
    ReferenceTask,
    check_interface,
)
from supermimic.acp.robot import (
    CONTACT_BODIES,
    JOINT_NAMES,
    NUM_J,
    default_joint_pos_list,
)
from supermimic.acp.rollout_contacts import RAW_COLUMNS, aggregate_contacts
from supermimic.acp.task import ContactCfg, ContactTask


def test_reference_command_clamps_to_own_clip_and_contains_joint_targets():
    frames = 8
    rotation = torch.eye(3).repeat(frames, 1, 1)
    motion = SimpleNamespace(
        clip_id=torch.tensor([0] * 4 + [1] * 4),
        clip_end=torch.tensor([3, 7]),
        root_p=torch.zeros(frames, 3),
        root_R=rotation,
        root_v=torch.zeros(frames, 3),
        root_w=torch.zeros(frames, 3),
        q=torch.arange(frames).float()[:, None].expand(-1, NUM_J),
        qd=torch.zeros(frames, NUM_J),
        boxes=lambda frames, variants: torch.zeros(len(frames), 1, 9),
    )
    task = ReferenceTask(
        ReferenceCfg(), SimpleNamespace(motion=motion, default_q=torch.zeros(NUM_J))
    )
    state = (torch.zeros(1, 3), torch.eye(3)[None])
    command = task.command(state, torch.tensor([2]), torch.tensor([0]))
    targets = command[:, : 6 * task.frame_dim].reshape(1, 6, task.frame_dim)
    torch.testing.assert_close(
        targets[0, :, 15 : 15 + NUM_J], torch.full((6, NUM_J), 3.0)
    )
    assert task.cmd_dim == command.shape[1]


def test_reference_skip_and_projection_match_physical_joint_coordinates():
    policy = ReferencePolicy(
        OBS_DIM,
        57,
        torch.zeros(NUM_J),
        torch.full((NUM_J,), -2.0),
        torch.full((NUM_J,), 2.0),
        hist=1,
    )
    for parameter in policy.parameters():
        parameter.data.zero_()
    goal = torch.zeros(1, 57)
    goal[:, 15 : 15 + NUM_J] = 0.1
    action = policy(torch.zeros(1, 1, OBS_DIM), goal)
    torch.testing.assert_close(
        action * policy.reference_scale, torch.full((1, NUM_J), 0.1)
    )
    assert torch.all(
        policy.project_action(torch.full_like(action, 1e6)) <= policy.action_hi
    )
    with pytest.raises(ValueError, match="reference Mimic"):
        check_interface({"task_interface": "contact"})


def test_measured_velocities_are_not_replaced_by_pose_differences():
    velocity = np.ones((3, 6 + NUM_J)) * 2
    archive = {
        "measured_velocities": True,
        "fps": np.array(50.0),
        "joint_vel": velocity,
    }
    result = measured_velocities(archive, torch.eye(3).repeat(3, 1, 1), "cpu")
    assert result is not None
    torch.testing.assert_close(result[2], torch.full((3, NUM_J), 2.0))
    archive["fps"] = np.array(25.0)
    with pytest.raises(ValueError, match="50 Hz"):
        measured_velocities(archive, torch.eye(3).repeat(3, 1, 1), "cpu")


def test_force_labels_use_all_substeps_and_retain_a_real_contact_point():
    records = np.zeros((4, len(RAW_COLUMNS)))
    records[:, 4], records[:, 5] = 0, 1
    records[:, 7] = [4, 8, 12, 16]
    records[:, 8] = [1, 2, 3, 4]
    records[:, 13] = 1
    records[:, 16] = records[:, 7]
    signal = aggregate_contacts(records, threshold=9)
    assert signal["normal_force"][0] == 10
    assert signal["contact"][0]
    assert not signal["contact"][1]
    np.testing.assert_allclose(signal["position"][0], [4, 0, 0])
    assert np.isnan(signal["position"][1]).all()


def test_short_force_spikes_do_not_become_future_targets():
    contacts = np.zeros((35, 9), dtype=bool)
    contacts[5, 0] = True
    contacts[15:30, 1] = True
    positions = np.zeros((35, 9, 3))
    _, onset, targets = contact_targets(contacts, positions)
    assert (onset[:, 0] == -1).all()
    assert onset[0, 1] == 15
    assert np.isfinite(targets).all()


def test_incomplete_or_fallen_rollouts_cannot_pass_success_gate():
    pose = np.array([0, 0, 0.77, 1, 0, 0, 0])
    assert not success_reasons(True, False, True, pose, pose)
    assert "incomplete_clip" in success_reasons(True, False, False, pose, pose)
    assert "tracking_termination" in success_reasons(True, True, True, pose, pose)
    assert "invalid_physics" in success_reasons(False, False, True, pose, pose)


def test_surface_friction_can_change_independently_and_reaches_context():
    validate_surface_friction((0.9, 0.9), (0.4, 0.7))
    ground, box, robot = surface_friction_values(32, "cpu", (0.9, 0.9), (0.4, 0.7))
    torch.testing.assert_close(torch.maximum(ground, robot), torch.full((32,), 0.9))
    torch.testing.assert_close(torch.maximum(box, robot), box)
    assert box.min() >= 0.4 and box.max() <= 0.7 and box.min() < box.max()
    context = torch.zeros(32, XI_DIM + 1)
    context[:, 0] = ground
    context[:, -1] = box
    normalized = normalize_xi(context)
    torch.testing.assert_close(
        normalized[:, :XI_DIM], normalize_xi(context[:, :XI_DIM])
    )
    torch.testing.assert_close(normalized[:, -1], (box - 0.75) / 0.45)
    with pytest.raises(ValueError, match="both ground and box"):
        validate_surface_friction((0.9, 0.9), None)


def test_knee_tracking_cost_has_corrective_gradient_for_both_knees():
    kin = RobotKinematics("cpu")
    default = torch.tensor(default_joint_pos_list())
    position = torch.tensor([[0.0, 0.0, 0.77]])
    rotation = torch.eye(3)[None]
    velocity = torch.zeros(1, 3)
    joint_velocity = torch.zeros(1, NUM_J)
    joints = default[None].clone()
    knee_ids = [JOINT_NAMES.index(name) for name in ("leg_l4_joint", "leg_r4_joint")]
    joints[0, knee_ids] -= 0.3
    joints.requires_grad_()
    state = (position, rotation, velocity, velocity, joints, joint_velocity)
    body_p, body_r, body_v, body_w = kin.forward(
        position,
        rotation,
        velocity,
        velocity,
        default[None],
        joint_velocity,
    )
    motion = SimpleNamespace(
        root_p=position,
        root_R=rotation,
        q=default[None],
        body_p=body_p,
        body_R=body_r,
        body_v=body_v,
        body_w=body_w,
    )
    frames = torch.tensor([0])
    standard = MDP(kin, motion, default)
    weighted = MDP(kin, motion, default, RewardCfg(w_knee_tracking=1.0))
    difference = weighted.reward(
        state, frames, joint_velocity, joint_velocity
    ) - standard.reward(
        state,
        frames,
        joint_velocity,
        joint_velocity,
    )
    difference.backward()
    assert difference.item() < 0
    torch.testing.assert_close(joints.grad[0, knee_ids], torch.full((2,), 0.02))


def test_success_export_loads_as_motion_and_force_based_contact_task(tmp_path):
    kin = RobotKinematics("cpu")
    count = 32
    default = torch.tensor(default_joint_pos_list())
    state = (
        torch.tensor([[0.0, 0.0, 0.77]]).expand(count, -1),
        torch.eye(3).repeat(count, 1, 1),
        torch.zeros(count, 3),
        torch.zeros(count, 3),
        default.repeat(count, 1),
        torch.zeros(count, NUM_J),
    )
    states = flatten_state(state).numpy()
    actions = np.zeros((count - 1, NUM_J))
    records = np.zeros((1, len(RAW_COLUMNS)))
    records[0, 0] = 1
    records[0, 4], records[0, 7], records[0, 13] = 0, 100, 1
    contact = aggregate_contacts(records)
    signals = [contact for _ in range(count - 1)]
    boxes = np.zeros((1, 9))
    save_success(
        tmp_path, "flat/test", states, actions, signals, records, boxes, kin, 5.0
    )
    motion = MotionLib.from_dir(str(tmp_path / "motions"), kin, "cpu")
    assert motion.num_frames == count - 1
    torch.testing.assert_close(motion.qd, torch.zeros(count - 1, NUM_J))
    mdp = MDP(kin, motion, default)
    task = ContactTask(
        ContactCfg(contacts_dir=str(tmp_path / "contacts")),
        mdp,
        str(tmp_path / "motions"),
        "cpu",
    )
    assert task.labels.shape == (count - 1, 1, len(CONTACT_BODIES))
    with np.load(tmp_path / "motions/flat/test_motion.npz") as archive:
        assert archive["joint_names"].tolist() == JOINT_NAMES
        assert len(archive["previous_action"]) == len(archive["joint_pos"])
        assert len(archive["action"]) == len(archive["joint_pos"]) - 1
    with np.load(tmp_path / "contacts/flat/test.npz") as archive:
        assert archive["raw_contacts"][0, 0] == 0
    assert terrain_array(boxes).shape == (0, 1, 10)
