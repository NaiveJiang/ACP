import json
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
from supermimic.acp.footstep.motion_matching import Robot
from supermimic.acp.footstep.play_lmm_tracker import GamepadSteering
from supermimic.acp.footstep.lmm_contact import (FORMAT as FOOTHOLD_FORMAT, LOOKAHEAD, POSE_FEATURES, Preview,
                                              check_seed_codec, decode_events, label_rows, load_head, world_events)
from supermimic.acp.footstep.learned_motion import LearnedPlayback
from supermimic.acp.footstep.finetune_lmm_contacts import WorldPlan, expand_policy, extract_rows, measured_events
from supermimic.acp.footstep.lmm_contact_scenarios import SCHEMA as SCENARIO_SCHEMA, read_scenarios
from supermimic.acp.footstep.rollout_controller import digest
from supermimic.acp.footstep.rollout_controller import encode_event_plan, split_rollouts
from supermimic.acp.frax_rollout import prepare_references, touchdown_windows
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


def test_local_reference_command_ignores_independent_simulated_xy_and_yaw():
    mdp = SimpleNamespace(motion=None, default_q=torch.zeros(NUM_J))
    task = ReferenceTask(ReferenceCfg(root_reference="local"), mdp)
    identity = torch.eye(3)[None]
    turn = torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
    state = (torch.tensor([[0., 0., .8]]), identity)
    displaced = (torch.tensor([[15., -7., .8]]), turn)
    anchor = (torch.tensor([[4., 2., .78]]), identity)
    joints = torch.zeros(1, 1, NUM_J)
    window = (torch.tensor([[[4.2, 2.1, .79]]]), turn[:, None], torch.tensor([[[.5, 0., 0.]]]),
              torch.tensor([[[0., 0., .3]]]), joints, joints)
    boxes = torch.zeros(1, 1, 9)
    command = task.command_from_reference(state, window, boxes, anchor=anchor)
    other = task.command_from_reference(displaced, window, boxes, anchor=anchor)
    torch.testing.assert_close(command, other)
    torch.testing.assert_close(command[:, :3], torch.tensor([[.2, .1, -.01]]))


def test_local_reference_reward_and_critic_ignore_planar_drift_but_keep_height():
    kin = RobotKinematics("cpu")
    q = torch.tensor(default_joint_pos_list())[None]
    p = torch.tensor([[0., 0., .80]])
    R = torch.eye(3)[None]
    v = torch.tensor([[.4, .2, 0.]])
    w = torch.tensor([[.1, .2, .3]])
    qd = torch.zeros_like(q)
    ref = (p, R, v, w, q, qd)
    bp, bR, bv, bw = kin.forward(*ref)
    motion = SimpleNamespace(root_p=p, root_R=R, q=q, qd=qd, body_p=bp, body_R=bR, body_v=bv, body_w=bw)
    mdp = MDP(kin, motion, q[0], RewardCfg(local_reference_root=True))
    turn = torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
    state = (p + torch.tensor([[3., -2., 0.]]), turn @ R, (turn @ v[..., None]).squeeze(-1),
             (turn @ w[..., None]).squeeze(-1), q, qd)
    frames = torch.tensor([0])
    torch.testing.assert_close(mdp.reward(ref, frames, qd, qd), mdp.reward(state, frames, qd, qd))
    boxes, xi = torch.zeros(1, 1, 9), torch.zeros(1, XI_DIM)
    torch.testing.assert_close(mdp.critic_features(ref, frames, xi, boxes),
                               mdp.critic_features(state, frames, xi, boxes), atol=1e-6, rtol=1e-5)
    higher = (p + torch.tensor([[0., 0., .10]]), *ref[1:])
    assert mdp.reward(higher, frames, qd, qd) < mdp.reward(ref, frames, qd, qd)


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


def test_frax_touchdown_windows_keep_all_events_and_mask_censored_tails():
    n = 100
    contact = np.zeros((n, 2), dtype=bool)
    contact[:10, 0] = True  # Already supporting at reset.
    contact[11:13, 1] = True  # Brief force spike.
    contact[20:30, 1] = True
    contact[24, 1] = False  # A short dropout inside support.
    contact[75:80, 0] = True  # Inclusive 1.5 s boundary.
    contact[76:83, 1] = True
    contact[97:, 0] = True  # Too short to confirm at the end of the episode.
    position = np.zeros((n, 2, 3))
    position[:, 1, :2] = [1, 1]
    sole_quat = np.broadcast_to([1, 0, 0, 0], (n, 2, 4))
    root = np.zeros((n, 7))
    root[:, [3, 6]] = np.sqrt(0.5)  # Root yaw is +90 degrees.

    labels = touchdown_windows(contact, position, sole_quat, root, min_support=5)
    np.testing.assert_array_equal(labels["event_frame"], [20, 75, 76])
    np.testing.assert_array_equal(labels["event_foot"], [1, 0, 1])
    np.testing.assert_array_equal(labels["future_event_bounds"][:2], [[0, 2], [0, 3]])
    np.testing.assert_allclose(labels["next_time_s"][0], [1.5, 0.4])
    np.testing.assert_allclose(labels["next_pos_base"][0, 1], [1, -1, 0], atol=1e-7)
    np.testing.assert_allclose(labels["next_heading_base"][0, 1], [-1, 0], atol=1e-7)
    assert labels["sample_valid"][18] and not labels["sample_valid"][19]
    assert labels["next_event_known"][20].all()  # Both positive events are observable.
    assert not labels["next_event_known"][85].any()


def test_frax_default_selects_all_cases_and_merges_short_final_segment(tmp_path):
    robot = Robot()
    upstream = tmp_path / "upstream"
    reference = upstream / "reference"
    reference.mkdir(parents=True)
    (upstream / "metadata.json").write_text(json.dumps({"sample_hz": 60}))
    frames = 289  # 4.8 seconds, leaving a short 0.8 s remainder for 2 s episodes.
    clips = []
    for case in ("walk", "walk_mirror"):
        clips.append(dict(case=case, category="walk", raw_motion="walk1_subject5",
                          mirrored=case.endswith("mirror"), frames=frames))
        qpos = np.broadcast_to(robot.model.qpos0, (frames, robot.model.nq)).copy()
        qpos[:, robot.root] = np.arange(frames) / 60
        np.savez(reference / f"{case}_labels.npz", qpos=qpos, source_row=np.arange(frames))
    (reference / "manifest.json").write_text(json.dumps(dict(
        schema="roban_motion_matching_preview", retargeter="frax", robot="biped_s17_ver", clips=clips,
    )))

    root, episodes = prepare_references(upstream, tmp_path / "output", None, 2, 5, 50)
    assert list(episodes) == [f"{case}/{case}_{i:04d}" for case in ("walk", "walk_mirror") for i in (0, 1)]
    assert episodes["walk/walk_0000"]["reference_end_frame_50hz"] == 100
    assert episodes["walk/walk_0000"]["command_end_frame_50hz"] == 150
    with np.load(root / "walk/walk_0000_motion.npz") as clip:
        assert len(clip["joint_pos"]) == 151
    assert episodes["walk/walk_0001"]["reference_start_frame_50hz"] == 100
    assert episodes["walk/walk_0001"]["reference_end_frame_50hz"] == 240
    assert episodes["walk/walk_0001"]["command_end_frame_50hz"] == 240
    with np.load(root / "walk/walk_0001_motion.npz") as clip:
        assert clip["fps"].item() == 50
        np.testing.assert_allclose(clip["joint_pos"][:, 0], np.arange(100, 241) / 50, atol=1e-6)
        np.testing.assert_allclose(clip["upstream_frame"], np.arange(100, 241) * 1.2)


def test_cpp_style_gamepad_keeps_speed_and_redirects_the_lmm_query():
    steering = GamepadSteering(.72, 1.55, 2.8, .20)
    mode = dict(walk=False, orbit=False, hold=False, camera_yaw=0.)
    forward = steering.update((1., 0., 0., 0.), **mode)
    reverse = steering.update((-1., 0., 0., 0.), **mode)
    np.testing.assert_allclose(forward.drive, reverse.drive)
    assert abs(reverse.heading) == np.pi
    command, angle = reverse.resolve(0.)
    assert command[2] == 2.8
    position, heading = reverse.trajectory(angle, np.zeros(2), 0.)
    assert position[-1, 0] < 0. and heading[-1, 0] < -.9
    neutral = steering.update((0., 0., 0., 0.), **mode)
    assert neutral.drive == (0., 0.) and neutral.heading == reverse.heading
    hold = steering.update((-1., 0., 0., 0.), **(mode | {"hold": True}))
    assert hold.drive == (-1.55, 0.) and hold.heading == reverse.heading


def test_cpp_style_right_stick_aim_is_remembered_for_forward_drive():
    steering = GamepadSteering(.72, 1.55, 2.8, .20)
    mode = dict(walk=False, orbit=False, hold=False, camera_yaw=.3)
    aimed = steering.update((0., 0., 1., 0.), **mode)
    assert aimed.heading == .3 + np.pi / 2
    assert steering.update((1., 0., 0., 0.), **mode).heading == aimed.heading
    assert abs(steering.update((-1., 0., 0., 0.), **mode).heading) == np.pi


def test_lmm_joint_exploration_only_reaches_committed_steps():
    player = LearnedPlayback.__new__(LearnedPlayback)
    player.state = 0
    seen = []
    player.on_committed_projection = seen.append
    player.pose_and_velocity = lambda state: (np.asarray([state]), 0., 0.)

    def step(state, command, root_velocity=None, *, on_commit=None):
        if on_commit is not None:
            on_commit(state)
        return state + 1

    player.step = step
    np.testing.assert_array_equal(player.forecast_reference(np.zeros(3), 2)[:, 0], [0, 1, 2])
    assert not seen and player.state == 0
    player.advance_reference(np.zeros(3), 2)
    assert seen == [0, 1] and player.state == 2


def test_rollout_event_command_reanchors_to_current_heading_and_masks_padding():
    root = (torch.tensor([[1., 2., .8]]), torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]]))
    valid = torch.tensor([[True, False]])
    foot = torch.tensor([[1, -1]])
    time = torch.tensor([[.3, float("nan")]])
    position = torch.tensor([[[2., 3., .1], [float("nan")] * 3]])
    yaw = torch.tensor([[0., float("nan")]])
    plan = encode_event_plan(root, valid, foot, time, position, yaw).reshape(1, 2, 9)
    torch.testing.assert_close(plan[0, 0], torch.tensor([1., 0., 1., .3, 1., -1., -.7, -1., 0.]))
    torch.testing.assert_close(plan[0, 1], torch.zeros(9))


def test_rollout_split_keeps_mirrors_and_segments_together():
    clips = [dict(take=take, category=category) for take, category in (
        ("walk1", "walk"), ("walk1", "walk"), ("run1", "run"), ("run2", "run"), ("idle1", "idle"),
    )]
    split = split_rollouts(clips, .2, 4)
    train = {part["take"] for part in split["train"]}
    val = {part["take"] for part in split["val"]}
    assert not train & val
    assert {part["category"] for part in split["train"]} == {"walk", "run", "idle"}


def test_lmm_footstep_decoder_keeps_multiple_events_and_confirms_window_edge():
    logits = torch.full((1, LOOKAHEAD + 1, 2), -1.)
    logits[:, 0:6, 0] = 1.  # Initial support is not a touchdown.
    logits[:, 8:17, 0] = 1.  # Bridge two missing frames.
    logits[:, 23:30, 0] = 1.
    logits[:, 44:47, 1] = 1.  # Too short.
    logits[:, 75:80, 1] = 1.
    valid, foot, dt, _, _ = decode_events(logits, torch.zeros(1, LOOKAHEAD + 1, 2, 3),
                                          torch.zeros_like(logits), 4)
    torch.testing.assert_close(valid[0], torch.tensor([True, True, False, False]))
    torch.testing.assert_close(foot[valid], torch.tensor([0, 1]))
    torch.testing.assert_close(dt[valid], torch.tensor([.46, 1.5]))


def test_lmm_plan_locks_an_imminent_world_footprint_across_replans():
    plan = WorldPlan(2, "cpu")

    def event(dt, x):
        return (torch.tensor([[True]]), torch.tensor([[0]]), torch.tensor([[dt]]),
                torch.tensor([[[x, 0., 0.]]]), torch.zeros(1, 1))

    plan.replace(0, event(.20, 1.))
    plan.replace(5, event(.14, 3.))
    assert len(plan.events) == 1 and plan.events[0][0] == 10
    state = (torch.tensor([[.2, 0., 0.]]), torch.eye(3)[None])
    command = plan.command(5, state).reshape(1, 2, 9)
    torch.testing.assert_close(command[0, 0, 3:5], torch.tensor([.10, .8]))
    assert not plan.command(10, state).bool().any()


def test_lmm_event_slot_expansion_preserves_the_original_actor_inputs():
    weights = {"enc.0.weight": torch.ones(2, 9), "upper_enc.0.weight": torch.ones(2, 9),
               "goal_norm.mean": torch.ones(9), "goal_norm.var": torch.arange(9.)}
    expanded = expand_policy(weights, 1, 2)
    for name, original in weights.items():
        torch.testing.assert_close(expanded[name][..., :9], original)
        torch.testing.assert_close(expanded[name][..., 9:], torch.full_like(original, float(name == "goal_norm.var")))


def test_lmm_closed_loop_targets_are_held_command_windows():
    frames = 200
    samples = LOOKAHEAD + 1
    commands = np.zeros((frames, 3), dtype=np.float32)
    commands[20:, 0] = .6
    history = [torch.full((1, 10, 69), float(t)) for t in range(frames)]
    feet = np.zeros((frames, 2, 3), dtype=np.float32)
    quat = np.broadcast_to([1., 0., 0., 0.], (frames, 2, 4)).astype(np.float32)
    support = np.zeros((frames, 2), dtype=bool)
    support[50:65, 1] = True
    preview = Preview(torch.zeros(3, samples, POSE_FEATURES), torch.zeros(3, samples, 2, 3),
                      torch.zeros(3, samples, 2))
    root_pose = np.tile([0., 0., 0., 1., 0., 0., 0.], (frames, 1)).astype(np.float32)
    batch = extract_rows(history, preview, root_pose, feet, quat, support, commands, {10: 0, 40: 1, 130: 2})
    assert batch["history"].shape[0] == 1  # Crossing command change and short tail are excluded.
    assert batch["history"][0, 0, 0] == 40
    assert batch["event"][0, 10, 1]


def test_lmm_contact_query_uses_actual_root_instead_of_lmm_world_drift():
    point = torch.tensor([[[.1, 0., -.75]]])
    event = (torch.tensor([[True]]), torch.tensor([[0]]), torch.tensor([[.2]]), point, torch.zeros(1, 1))
    actual = (torch.tensor([[1., 4., .75]]), torch.eye(3)[None])
    predicted = world_events(actual, event)
    torch.testing.assert_close(predicted[3], torch.tensor([[[1.1, 4., 0.]]]))
    # At the next control step, the previously locked point stays on the ground.
    plan = WorldPlan(1, "cpu")
    plan.replace(0, predicted)
    later = (torch.tensor([[1.03, 4., .75]]), actual[1])
    torch.testing.assert_close(plan.command(1, later)[0, 4], torch.tensor(.07))
    # Both supervision and deployment use the measured yaw as well as translation.
    turn = torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
    rotated = world_events((actual[0], turn), event)
    torch.testing.assert_close(rotated[3], torch.tensor([[[1., 4.1, 0.]]]))
    pose = torch.tensor([[1., 4., .75, 2**-.5, 0., 0., 2**-.5]])
    preview = Preview(torch.zeros(1, 1, POSE_FEATURES), torch.zeros(1, 1, 1, 3), torch.zeros(1, 1, 1))
    flag = torch.ones(1, 1, 1, dtype=torch.bool)
    labels = label_rows(preview, pose, flag, rotated[3][:, None], rotated[4][:, None], flag, flag[..., 0])
    torch.testing.assert_close(labels["foot_position"], point[:, None], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(labels["foot_heading"], torch.tensor([[[[0., 1.]]]]), atol=1e-6, rtol=1e-5)


def test_lmm_student_preceding_force_does_not_invent_reset_touchdowns():
    support = np.zeros((35, 2), dtype=bool)
    support[1:12, 0] = True
    support[19:30, 0] = True
    position = np.zeros((len(support), 2, 3))
    quats = np.broadcast_to([1., 0., 0., 0.], (len(support), 2, 4))
    events = measured_events(support, position, quats, reset_unmeasured=True)
    np.testing.assert_array_equal(events["event_frame"], [19])
    np.testing.assert_array_equal(events["foot_support"], support)


def test_lmm_seed_codec_must_be_the_same_encoder(tmp_path):
    model, codec = tmp_path / "model", tmp_path / "codec"
    model.mkdir()
    codec.mkdir()
    for filename in ("model.pt", "stepper.pt", "projector.pt"):
        (model / filename).write_bytes(b"frozen weights")
    (codec / "model.pt").write_bytes(b"other encoder on the same dataset")
    with pytest.raises(ValueError, match="exact encoder"):
        check_seed_codec(model, codec)


@pytest.mark.parametrize("format", ["roban_lmm_timed_footholds_v1", FOOTHOLD_FORMAT])
def test_lmm_foothold_checkpoint_requires_robot_root_frame(tmp_path, format):
    checkpoint = tmp_path / "head.pt"
    torch.save(dict(format=format, event_frame="lmm_root"), checkpoint)
    with pytest.raises(ValueError, match="must use robot-root labels"):
        load_head(checkpoint, tmp_path, "cpu")


def test_lmm_prepared_scenarios_reject_modified_queries(tmp_path):
    query, motion = tmp_path / "walk_queries.npz", tmp_path / "references/walk/walk_motion.npz"
    motion.parent.mkdir(parents=True)
    motion.write_bytes(b"generated poses")
    query.write_bytes(b"commands and forecasts")
    (tmp_path / "manifest.json").write_text(json.dumps(dict(schema=SCENARIO_SCHEMA, clips={
        "walk/walk": dict(queries=query.name, motion_sha256=digest(motion), queries_sha256=digest(query)),
    })))
    assert "walk/walk" in read_scenarios(tmp_path)[1]
    query.write_bytes(b"changed commands")
    with pytest.raises(ValueError, match="files changed"):
        read_scenarios(tmp_path)


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
