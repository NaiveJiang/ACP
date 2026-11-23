import importlib
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from play_motion_matching import load_rollout
from roban_motion_matching import (
    JOINT_NAMES,
    angular_velocity,
    ease_from_rest,
    inertialize,
    mirror_pose,
    quaternions,
    rotations,
    turn_approach,
)


def pose_sequence():
    pose = np.zeros((100, 28))
    pose[:, 0] = np.arange(100) * 0.005
    pose[:, 2] = 0.77
    pose[:, 3] = 1
    pose[:, 7:] = np.random.default_rng(5).normal(0, 0.1, (100, 21))
    return pose


def test_mirror_is_involution():
    pose = pose_sequence()
    pose[:, 3:7] = quaternions(
        Rotation.from_euler("xyz", np.tile([0.1, 0.2, 0.3], (100, 1)))
    )
    restored = mirror_pose(mirror_pose(pose, 0.237), 0.237)
    np.testing.assert_allclose(restored[:, :3], pose[:, :3], atol=1e-12)
    np.testing.assert_allclose(restored[:, 7:], pose[:, 7:], atol=1e-12)
    np.testing.assert_allclose(
        rotations(restored[:, 3:7]).as_matrix(),
        rotations(pose[:, 3:7]).as_matrix(),
        atol=1e-12,
    )


def test_turning_preserves_endpoint_and_joint_poses():
    pose = pose_sequence()
    turned = turn_approach(pose, 0.3)
    np.testing.assert_allclose(turned[-1], pose[-1])
    np.testing.assert_array_equal(turned[:, 7:], pose[:, 7:])
    assert np.isfinite(turned).all()


def test_world_angular_velocity_ignores_quaternion_sign():
    quaternion = quaternions(Rotation.from_euler("z", np.arange(100) * 0.01))
    quaternion[::2] *= -1
    velocity = angular_velocity(quaternion)
    np.testing.assert_allclose(velocity[:, 2], 0.5, atol=1e-12)
    np.testing.assert_allclose(velocity[:, :2], 0, atol=1e-12)


def test_stand_down_eases_into_reference_without_changing_later_poses():
    pose = pose_sequence()
    eased = ease_from_rest(pose)
    np.testing.assert_allclose(eased[0], pose[0], atol=1e-12)
    np.testing.assert_allclose(eased[40:], pose[20:], atol=1e-12)
    assert 0 < eased[1, 0] - eased[0, 0] < (pose[1, 0] - pose[0, 0]) / 10


def test_php_inertialization_starts_at_previous_pose():
    php = Path(__file__).resolve().parents[2] / "php_parkour/motion_matching"
    sys.path.insert(0, str(php))
    upstream = importlib.import_module("motion_matching.utils.animation_state")
    previous, following = pose_sequence(), pose_sequence()
    following[:, 0] += 2
    result = inertialize(previous, following, upstream)
    np.testing.assert_allclose(result[0, :3], previous[-1, :3], atol=1e-12)
    np.testing.assert_allclose(result[0, 7:], previous[-1, 7:], atol=1e-12)
    assert np.isfinite(result).all()


def test_recorded_policy_playback_crops_dead_frames_and_converts_box_center(tmp_path):
    path = tmp_path / "rollout.npz"
    quaternion = np.zeros((5, 1, 4))
    quaternion[..., 0] = 1
    position = np.zeros((5, 1, 3))
    position[2:] = np.nan
    np.savez(
        path,
        robot_id="roban",
        joint_names=np.asarray(JOINT_NAMES),
        names=["full/test"],
        root_p=position,
        root_quat=quaternion,
        q=np.zeros((5, 1, 21)),
        alive=np.array([[True], [True], [False], [False], [False]]),
        boxes=np.array([[[3.0, 0.6, 0.547, 1.0, 0.0, 1.5, 1.0, 0.2735, 0.0]]]),
        clip_len=np.array([10]),
    )
    clip = load_rollout(path)["full/test"]
    assert len(clip["joint_pos"]) == 2
    assert np.isfinite(clip["joint_pos"]).all()
    np.testing.assert_allclose(clip["terrain"][:3], [3.0, 0.6, 0.2735])
    np.testing.assert_allclose(clip["terrain"][7:], [3.0, 2.0, 0.547])
    assert "terminated" in clip["stage"][0]
