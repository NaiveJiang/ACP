"""Prepare Frax upstream episodes and export aligned, solver-labelled footstep samples."""

import json
import re

import numpy as np
import torch

from ..rotations import mat_to_quat
from .footstep.event_planner import WINDOW
from .footstep.motion_matching import FOOT_OFFSET, Robot, heading, local, upstream_manifest
from .mdp import unflatten_state
from .motion import CONTROL_DT, _resample
from .robot import CONTACT_BODIES, JOINT_NAMES, ROBOT_ID
from .task import _runs, debounce


SCHEMA = "frax_upstream_footstep_rollout_v1"
FEET = ("left_foot", "right_foot")
FPS = round(1 / CONTROL_DT)
DEBOUNCE_STEPS = 2


def confirmation_tail(min_support):
    return WINDOW + min_support + DEBOUNCE_STEPS


def prepare_references(upstream, output, cases, window_seconds, min_support, lookahead_steps):
    """Keep Frax ground and the upstream time base; reset only at explicit episode boundaries."""
    manifest = upstream_manifest(upstream)
    if manifest.get("retargeter") != "frax" or manifest.get("robot") != "biped_s17_ver":
        raise ValueError(f"Expected Frax S17-ver references at {upstream}")
    fps = float(json.loads((upstream / "metadata.json").read_text())["sample_hz"])
    if fps != 60:
        raise ValueError(f"Expected 60 Hz Frax upstream references, got {fps:g} Hz")

    clips = {clip["case"]: clip for clip in manifest["clips"]}
    cases = tuple(dict.fromkeys(cases)) if cases else tuple(clips)
    if unknown := set(cases) - clips.keys():
        raise ValueError(f"Unknown cases {sorted(unknown)}. Available: {', '.join(clips)}")
    if not cases:
        raise ValueError(f"No clips in Frax upstream library {upstream}")
    robot = Robot()
    columns = np.r_[np.arange(robot.root, robot.root + 7), robot.joint_qpos]
    root = output / "references"
    episodes = {}
    for case in cases:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", case):
            raise ValueError(f"Invalid upstream case name {case!r}")
        clip = clips[case]
        source = upstream / "reference" / f"{case}_labels.npz"
        with np.load(source, allow_pickle=False) as archive:
            poses, source_row = archive["qpos"], archive["source_row"]
        if (poses.shape != (clip["frames"], robot.model.nq) or len(poses) < 2
                or source_row.shape != (len(poses),)
                or not all(np.isfinite(x).all() for x in (poses, source_row))):
            raise ValueError(f"Invalid upstream archive {source}")
        poses = torch.as_tensor(poses[:, columns], dtype=torch.float32)
        if not torch.allclose(poses[:, 3:7].norm(dim=1), torch.ones(len(poses)), atol=1e-3):
            raise ValueError(f"Invalid root quaternion in {source}")
        # Reuse the tracker resampler so the stored reference matches what the actor sees.
        p, quat, joints = _resample(poses[:, :3], poses[:, 3:7], poses[:, 7:], fps)
        qpos = torch.cat((p, quat, joints), dim=1).numpy()
        upstream_frame = np.arange(len(qpos)) * (fps / FPS)
        rows = np.interp(upstream_frame, np.arange(len(source_row)), source_row)
        folder = root / case
        folder.mkdir(parents=True)
        end = len(qpos) - 1
        limit = round(window_seconds * FPS) if window_seconds else end
        bounds = list(range(0, end, limit))
        # Merge a short final remainder into the previous episode.
        if len(bounds) > 1 and end - bounds[-1] < confirmation_tail(min_support):
            bounds.pop()
        for episode, (lo, hi) in enumerate(zip(bounds, bounds[1:] + [end])):
            stem = f"{case}_{episode:04d}"
            command_end = min(hi + lookahead_steps, end)
            stop = command_end + 1
            np.savez_compressed(folder / f"{stem}_motion.npz", joint_pos=qpos[lo:stop],
                                joint_names=JOINT_NAMES, robot_id=ROBOT_ID, fps=FPS,
                                upstream_frame=upstream_frame[lo:stop], source_row=rows[lo:stop])
            np.save(folder / f"{stem}_terrain.npy", np.empty((0, 1, 10)))
            episodes[f"{case}/{stem}"] = {
                "case": case, "category": clip["category"], "raw_motion": clip["raw_motion"],
                "mirrored": clip["mirrored"], "upstream_file": str(source.resolve()),
                "reference_start_frame_50hz": lo, "reference_end_frame_50hz": hi,
                "command_end_frame_50hz": command_end,
            }
    return root, episodes


def pose_velocity(state):
    p, rotation, v, w, q, qd = state
    local_w = (rotation.transpose(-1, -2) @ w[..., None]).squeeze(-1)
    return (torch.cat((p, mat_to_quat(rotation), q), -1).cpu().numpy(),
            torch.cat((v, local_w, qd), -1).cpu().numpy())


def align_attempt(states, reference_frames, contacts, motion, clip, kin, observation):
    """State and pose at frame t, force measured over (t - dt, t]."""
    state = unflatten_state(torch.as_tensor(states, dtype=torch.float32, device=kin.device))
    qpos, qvel = pose_velocity(state)
    foot_links = [kin.body_names.index(CONTACT_BODIES[name][0]) for name in FEET]
    body_p, body_r, _, _ = kin.forward(*state, all_bodies=True)
    foot_r = body_r[:, foot_links]
    offset = torch.as_tensor(FOOT_OFFSET, device=kin.device, dtype=foot_r.dtype)
    sole = body_p[:, foot_links] + foot_r @ offset
    feet = [list(CONTACT_BODIES).index(name) for name in FEET]
    indices = torch.as_tensor(reference_frames, device=kin.device)
    ref_qpos, ref_qvel = pose_velocity(motion.ref_state(indices))
    relative = np.asarray(reference_frames) - int(motion.clip_start[clip])
    with np.load(motion.files[clip], allow_pickle=False) as source:
        upstream_frame, source_row = (source[key][relative] for key in ("upstream_frame", "source_row"))
    n = len(states)
    measured = np.arange(n) > 0
    normal_force = np.full((n, 2), np.nan, dtype=np.float32)
    force_world = np.full((n, 2, 3), np.nan, dtype=np.float32)
    contact = np.zeros((n, 2), dtype=bool)
    normal_force[1:] = np.stack([item["normal_force"][feet] for item in contacts])
    force_world[1:] = np.stack([item["force"][feet] for item in contacts])
    contact[1:] = np.stack([item["contact"][feet] for item in contacts])
    return dict(qpos=qpos, qvel=qvel, observation=observation, reference_qpos=ref_qpos,
                reference_qvel=ref_qvel, reference_episode_frame=relative,
                upstream_frame=upstream_frame, source_row=source_row,
                sole_pos_w=sole.cpu().numpy(), sole_quat_w=mat_to_quat(foot_r).cpu().numpy(),
                foot_normal_force_n=normal_force, foot_force_w=force_world,
                foot_contact_raw=contact, force_measured=measured)


def touchdown_windows(contact, sole_pos, sole_quat, root_qpos, min_support):
    """Return the touchdown table, per-frame window ranges, and first target for each foot."""
    support = debounce(contact, DEBOUNCE_STEPS)
    events = []
    for foot in range(len(FEET)):
        starts, stops = _runs(support[:, foot])
        # The first measured step may already be supporting the robot after reset.
        events.extend((start, foot) for start, stop in zip(starts, stops)
                      if start > 0 and stop - start >= min_support)
    event_frame, event_foot = np.asarray(sorted(events), dtype=np.int64).reshape(-1, 2).T
    position = sole_pos[event_frame, event_foot]
    yaw = heading(sole_quat[event_frame, event_foot])
    frames = np.arange(len(contact))
    ranges = np.stack((np.searchsorted(event_frame, frames, side="right"),
                       np.searchsorted(event_frame, frames + WINDOW, side="right")), axis=-1)
    # Allow the support confirmation and gap filling to look past the last query frame.
    complete = frames + confirmation_tail(min_support) <= len(contact)

    first = np.full((len(frames), 2), -1, dtype=np.int64)
    for foot in range(len(FEET)):
        ids = np.flatnonzero(event_foot == foot)
        lookup = np.searchsorted(event_frame[ids], frames, side="right")
        rows = np.flatnonzero(lookup < len(ids))
        found = ids[lookup[rows]]
        within = found < ranges[rows, 1]
        first[rows[within], foot] = found[within]
    valid = first >= 0
    target_p = np.zeros((len(frames), 2, 3), dtype=np.float32)
    target_h = np.zeros((len(frames), 2, 2), dtype=np.float32)
    target_dt = np.zeros((len(frames), 2), dtype=np.float32)
    row, foot = np.nonzero(valid)
    selected = first[row, foot]
    root_yaw = heading(root_qpos[row, 3:7])
    target_p[row, foot] = local(position[selected] - root_qpos[row, :3], root_yaw)
    angle = yaw[selected] - root_yaw
    target_h[row, foot] = np.stack((np.sin(angle), np.cos(angle)), axis=-1)
    target_dt[row, foot] = (event_frame[selected] - row) * CONTROL_DT
    return dict(foot_support=support, event_frame=event_frame,
                event_foot=event_foot, event_pos_w=position, event_yaw_w=yaw,
                future_event_bounds=ranges, sample_valid=complete, next_event_id=first,
                next_event_valid=valid, next_event_known=valid | complete[:, None],
                next_pos_base=target_p, next_heading_base=target_h, next_time_s=target_dt)


def save_samples(output, name, alignment, actions, min_support, threshold):
    # Reset has no solved forces. All resulting event indices start at the first measured frame.
    arrays = {key: value[1:] for key, value in alignment.items()}
    labels = touchdown_windows(arrays["foot_contact_raw"], arrays["sole_pos_w"],
                               arrays["sole_quat_w"], arrays["qpos"], min_support)
    path = output / "samples" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, schema=SCHEMA, fps=FPS, joint_names=JOINT_NAMES, foot_names=FEET,
                        time_s=np.arange(1, len(actions) + 1) * CONTROL_DT,
                        previous_action=actions, action=actions[1:],
                        force_threshold_n=threshold, min_support_steps=min_support,
                        horizon_steps=WINDOW, debounce_steps=DEBOUNCE_STEPS,
                        sole_offset=FOOT_OFFSET, **arrays, **labels)
    return str(path.relative_to(output)), int(labels["sample_valid"].sum()), len(labels["event_foot"])
