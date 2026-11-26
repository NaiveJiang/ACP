"""Run a trained Mimic separately and export successful, force-labelled rollouts."""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..rotations import mat_to_quat, quat_to_mat
from . import networks
from .checkpoint import load_checkpoint
from .env import PHYSICS_DT, TrackingSim, evaluation_surface_friction
from .kinematics import RobotKinematics
from .mdp import MDP, RewardCfg, flatten_state, unflatten_state
from .motion import CONTROL_DT, MotionLib
from .reference_mimic import ReferenceCfg, ReferenceTask, build_policy, check_interface
from .robot import (
    CONTACT_BODIES,
    JOINT_NAMES,
    ROBOT_ID,
    default_joint_pos_list,
    metadata,
)
from .rollout_contacts import RAW_COLUMNS, SolvedContactRecorder
from .s17_actuation import DELAY_STEPS
from .task import debounce, drop_short, next_onsets


def success_reasons(
    valid,
    terminated,
    completed,
    final,
    reference,
    position_tolerance=0.3,
    height_tolerance=0.15,
    *,
    track_translation=True,
):
    reasons = []
    if not valid:
        reasons.append("invalid_physics")
    if terminated:
        reasons.append("tracking_termination")
    if not completed:
        reasons.append("incomplete_clip")
    if track_translation and np.linalg.norm(final[:2] - reference[:2]) > position_tolerance:
        reasons.append("final_xy_error")
    if track_translation and abs(final[2] - reference[2]) > height_tolerance:
        reasons.append("final_height_error")
    final_rotation = quat_to_mat(torch.tensor(final[3:7])).numpy()
    reference_rotation = quat_to_mat(torch.tensor(reference[3:7])).numpy()
    if np.dot(final_rotation[2], reference_rotation[2]) < 0.9:
        reasons.append("final_attitude_error")
    final_heading = np.arctan2(final_rotation[1, 0], final_rotation[0, 0])
    reference_heading = np.arctan2(reference_rotation[1, 0], reference_rotation[0, 0])
    if (
        abs(
            np.arctan2(
                np.sin(final_heading - reference_heading),
                np.cos(final_heading - reference_heading),
            )
        )
        > 0.35
    ):
        reasons.append("final_heading_error")
    return reasons


def terrain_array(boxes):
    output = []
    for box in boxes:
        if box[8] != 0:
            raise ValueError("Capture currently supports boxes, not heightfields")
        if box[5] <= 0:
            continue
        yaw = np.arctan2(box[4], box[3])
        output.append(
            [
                box[0],
                box[1],
                box[2] - box[7],
                np.cos(yaw / 2),
                0,
                0,
                np.sin(yaw / 2),
                2 * box[5],
                2 * box[6],
                2 * box[7],
            ]
        )
    return np.asarray(output).reshape(-1, 1, 10)


def contact_targets(contact, positions, min_steps=10, horizon=75):
    labels = debounce(contact, 2)
    onsets = next_onsets(drop_short(labels, min_steps), horizon)
    valid = onsets >= 0
    targets = np.zeros_like(positions)
    for body in range(contact.shape[1]):
        targets[valid[:, body], body] = positions[onsets[valid[:, body], body], body]
    if not np.isfinite(targets[valid]).all():
        raise ValueError("A measured contact onset has no finite solver contact point")
    return labels, onsets, targets


@torch.no_grad()
def save_success(output, name, states, actions, contact, raw, boxes, kin, threshold):
    state = unflatten_state(
        torch.as_tensor(states[1:], dtype=torch.float32, device=kin.device)
    )
    position, rotation, velocity, angular, joints, joint_velocity = state
    local_angular = (rotation.transpose(-1, -2) @ angular[..., None]).squeeze(-1)
    poses = torch.cat((position, mat_to_quat(rotation), joints), -1).cpu().numpy()
    velocities = torch.cat((velocity, local_angular, joint_velocity), -1).cpu().numpy()
    body_p, body_r, body_v, body_w = kin.forward(*state, all_bodies=True)
    body_names = kin.body_names
    contact_indices = [body_names.index(frame) for frame, _ in CONTACT_BODIES.values()]
    signals = {key: np.stack([sample[key] for sample in contact]) for key in contact[0]}
    labels, onsets, targets = contact_targets(signals["contact"], signals["position"])
    exported_raw = raw.copy()
    exported_raw[:, 0] -= 1
    motion_path = output / "motions" / (name + "_motion.npz")
    label_path = output / "contacts" / (name + ".npz")
    motion_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        motion_path,
        fps=np.array(50.0),
        robot_id=ROBOT_ID,
        joint_names=np.asarray(JOINT_NAMES),
        joint_pos=poses,
        joint_vel=velocities,
        measured_velocities=np.array(True),
        body_pos_w=body_p.cpu().numpy(),
        body_quat_w=mat_to_quat(body_r).cpu().numpy(),
        body_lin_vel_w=body_v.cpu().numpy(),
        body_ang_vel_w=body_w.cpu().numpy(),
        body_names=np.asarray(body_names),
        action=actions[1:],
        previous_action=actions,
        time_s=np.arange(1, len(states)) * CONTROL_DT,
        velocity_convention="world linear, root-local angular, measured joint velocity",
    )
    np.save(
        motion_path.with_name(motion_path.name.replace("_motion.npz", "_terrain.npy")),
        terrain_array(boxes),
    )
    np.savez_compressed(
        label_path,
        robot_id=ROBOT_ID,
        body_names=np.asarray(list(CONTACT_BODIES)),
        link_names=np.asarray([frame for frame, _ in CONTACT_BODIES.values()]),
        fps=50.0,
        body_pos_w=body_p[:, contact_indices].cpu().numpy(),
        body_quat_w=mat_to_quat(body_r[:, contact_indices]).cpu().numpy(),
        contact_info=labels[None, ..., None].astype(np.float32),
        contact_info_raw=signals["contact"][None, ..., None],
        contact_pos_w=signals["position"][None],
        contact_normal_w=signals["normal"][None],
        contact_dist=signals["distance"][None],
        contact_surface=signals["surface"][None],
        normal_force=signals["normal_force"][None],
        force_world=signals["force"][None],
        next_contact_frame=onsets,
        next_contact_pos_w=targets,
        next_contact_valid=onsets >= 0,
        boxes=boxes[None],
        force_threshold_n=threshold,
        contact_profile="roban_knee_caps_v1",
        terrain_label_method="solved_physics_contact_force",
        source_motion=name,
        raw_contacts=exported_raw,
        raw_columns=np.asarray(RAW_COLUMNS),
        contact_time_alignment="mean solved force over preceding four 5ms substeps",
        time_s=np.arange(1, len(states)) * CONTROL_DT,
        raw_frame_indexing="0-based exported state; substeps precede that state",
    )
    return str(motion_path.relative_to(output)), str(label_path.relative_to(output))


def state_array(state):
    return flatten_state(state)[0].detach().cpu().numpy()


def write_manifest(output, report):
    pending = output / "manifest.json.tmp"
    pending.write_text(json.dumps(report, indent=2) + "\n")
    pending.replace(output / "manifest.json")


@torch.no_grad()
def collect(args):
    checkpoint = load_checkpoint(args.ckpt, map_location=args.device, weights_only=True)
    check_interface(checkpoint)
    if args.ema and "pi_ema" not in checkpoint:
        raise ValueError("Checkpoint has no EMA policy")
    cfg = ReferenceCfg(**checkpoint["reference_cfg"])
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    networks.AMP_DTYPE = torch.bfloat16 if checkpoint["args"]["amp"] == "bf16" else None
    kin = RobotKinematics(args.device)
    episodes = None
    if args.upstream_roban is not None or args.planned_reference is not None:
        from . import frax_rollout

        args.output.mkdir(parents=True, exist_ok=False)
        if args.planned_reference is None:
            motion_dir, episodes = frax_rollout.prepare_references(
                args.upstream_roban, args.output, args.case, args.window_seconds,
                args.min_support_steps, max(cfg.offsets),
            )
        else:
            from .footstep.lmm_contact_scenarios import read_scenarios

            motion_dir, episodes = read_scenarios(args.planned_reference)
    else:
        motion_dir = args.motion_dir or checkpoint["args"]["motion_dir"]
    motion = MotionLib.from_dir(motion_dir, kin, args.device)
    ground, box = evaluation_surface_friction(checkpoint["args"])
    delay_steps = DELAY_STEPS if checkpoint["args"].get("s17_actuators") else 0
    self_collision_weight = checkpoint["args"].get("self_collision_weight", 0.0)
    primitive_soles = checkpoint["args"].get("primitive_soles", False)
    mdp = MDP(
        kin, motion, torch.tensor(default_joint_pos_list(), device=args.device),
        RewardCfg(w_self_collision=self_collision_weight, local_reference_root=cfg.root_reference == "local",
                  body_reference=cfg.body_reference),
        independent_surface_friction=ground is not None, delay_steps=delay_steps,
    )
    task = ReferenceTask(cfg, mdp)
    policy = build_policy(mdp, task, checkpoint["args"]["hist"]).to(args.device)
    policy.load_state_dict(checkpoint["pi_ema"] if args.ema else checkpoint["pi"])
    policy.eval()
    if episodes is None:
        args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest(),
        "robot": metadata(s17_actuators=bool(delay_steps), primitive_soles=primitive_soles),
        "policy": "ema" if args.ema else "online",
        "args": vars(args).copy(),
        "reference_cfg": dataclasses.asdict(cfg),
        "attempts": [],
        "successful_rollouts": 0,
        "completed": False,
        "force_method": "mujoco_warp.contact_force after each physics step; no CPU re-solve",
        "physics_dt": PHYSICS_DT,
        "s17_actuators": bool(delay_steps),
        "self_collision_weight": self_collision_weight,
        "primitive_soles": primitive_soles,
        "surface_friction": {
            "ground": 1.0 if ground is None else ground,
            "box": 1.0 if box is None else box,
        },
        "control_dt": CONTROL_DT,
        "initial_state": "unmeasured force; excluded from exported training arrays",
    }
    report["args"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in report["args"].items()
    }
    if episodes is not None:
        report["sample_schema"] = frax_rollout.SCHEMA
        report["valid_samples"] = 0
        report["touchdown_events"] = 0
    write_manifest(args.output, report)
    env = TrackingSim(
        1, motion, mdp, args.device, eval_mode=True, reference_reset_action=True,
        ground_friction_range=None if ground is None else (ground, ground),
        box_friction_range=None if box is None else (box, box),
        primitive_soles=primitive_soles,
    )
    recorder = SolvedContactRecorder(env, args.force_threshold)
    env.substep_callback = recorder
    clips = range(min(args.max_clips or len(motion.names), len(motion.names)))
    for clip in clips:
        start, end = int(motion.clip_start[clip]), int(motion.clip_end[clip])
        episode = episodes[motion.names[clip]] if episodes is not None else None
        if episode is not None:
            # Leave extra frames accessible to the WBC command without executing them twice.
            end = start + episode["reference_end_frame_50hz"] - episode["reference_start_frame_50hz"]
        for attempt in range(args.attempts):
            variant = motion.first_variants(torch.tensor([clip], device=args.device))
            env.done_eval.zero_()
            env.failed_eval.zero_()
            env.reset(
                torch.tensor([0], device=args.device),
                torch.tensor([start], device=args.device),
                variant,
            )
            states, actions, contacts, raw_records = (
                [state_array(env.state())],
                [],
                [],
                [],
            )
            reference_frames = [start]
            observations = []
            history = env.observe()[:, None].repeat(1, policy.hist, 1)
            initial_action = env.a_prev[0].cpu().numpy().copy()
            valid = True
            validity, terminations = [], []
            terminated = False
            overlaps = []
            limit = min(end - start, args.max_steps or end - start)
            for step in range(limit):
                if step:
                    history = torch.cat((history[:, 1:], env.observe()[:, None]), 1)
                action = policy(history, task.command(env.state(), env.f, env.v))
                action = policy.project_action(
                    action + args.action_noise * torch.randn_like(action)
                )
                recorder.begin(len(states))
                transition = env.step(action)
                if episodes is not None:
                    observations.append(transition["o"][0].cpu().numpy())
                if "self_overlap" in transition and bool(transition["valid"][0]):
                    overlaps.append(float(transition["self_overlap"][0]))
                contact, raw = recorder.finish()
                states.append(transition["s1"][0].cpu().numpy())
                actions.append(action[0].cpu().numpy())
                contacts.append(contact)
                raw_records.append(raw)
                reference_frames.append(int(env.f[0]))
                valid &= bool(transition["valid"][0])
                terminated |= bool(transition["term"][0])
                validity.append(bool(transition["valid"][0]))
                terminations.append(bool(transition["term"][0]))
                if bool(transition["done"][0]):
                    break
            states, actions = np.asarray(states), np.asarray(actions)
            raw = np.concatenate(raw_records)
            final_reference = (
                torch.cat((motion.root_p[end], mat_to_quat(motion.root_R[end])))
                .cpu()
                .numpy()
            )
            reasons = success_reasons(
                valid,
                terminated,
                reference_frames[-1] == end,
                states[-1, :7],
                final_reference,
                track_translation=not cfg.body_reference,
            )
            joint_error = float(
                np.sqrt(
                    np.mean(
                        (
                            states[-1, 13 : 13 + len(JOINT_NAMES)]
                            - motion.q[end].cpu().numpy()
                        )
                        ** 2
                    )
                )
            )
            velocity_error = float(
                np.linalg.norm(states[-1, 7:10] - motion.root_v[end].cpu().numpy())
            )
            if joint_error > 0.4:
                reasons.append("final_joint_error")
            if velocity_error > 0.75:
                reasons.append("final_velocity_error")
            if args.max_steps and len(actions) < end - start:
                reasons.append("truncated_by_max_steps")
            name = f"{motion.names[clip]}_attempt{attempt:03d}"
            alignment = {}
            if episodes is not None:
                observations.append(env.observe()[0].cpu().numpy())
                alignment = frax_rollout.align_attempt(
                    states, reference_frames, contacts, motion, clip, kin, np.asarray(observations)
                )
            path = args.output / "attempts" / (name + ".npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                state=states,
                action=actions,
                reference_frame=reference_frames,
                valid=valid,
                successful=not reasons,
                rejection_reasons=np.asarray(reasons),
                raw_contacts=raw,
                raw_columns=np.asarray(RAW_COLUMNS),
                previous_action=np.concatenate((initial_action[None], actions)),
                transition_valid=np.asarray(validity),
                transition_terminated=np.asarray(terminations),
                time_s=np.arange(len(states)) * CONTROL_DT,
                initial_contact_measured=False,
                **alignment,
            )
            entry = {
                "source_clip": motion.names[clip],
                "attempt": attempt,
                "steps": len(actions),
                "successful": not reasons,
                "rejection_reasons": reasons,
                "raw": str(path.relative_to(args.output)),
                "final_joint_rms_rad": joint_error,
                "final_velocity_error_m_s": velocity_error,
            }
            if self_collision_weight:
                entry["max_self_overlap_m"] = max(overlaps, default=None)
            if episode is not None:
                entry["upstream"] = episode
                xy_error = np.linalg.norm(alignment["qpos"][:, :2] - alignment["reference_qpos"][:, :2], axis=1)
                entry["max_xy_error_m"] = float(xy_error.max())
            if not reasons:
                boxes = motion.clip_boxes[clip, int(variant[0])].cpu().numpy()
                entry["motion"], entry["contacts"] = save_success(
                    args.output,
                    name,
                    states,
                    actions,
                    contacts,
                    raw,
                    boxes,
                    kin,
                    args.force_threshold,
                )
                report["successful_rollouts"] += 1
                if episodes is not None:
                    path, count, events = frax_rollout.save_samples(
                        args.output, name, alignment, actions, args.min_support_steps, args.force_threshold
                    )
                    entry.update(samples=path, valid_samples=count, touchdown_events=events)
                    report["valid_samples"] += count
                    report["touchdown_events"] += events
            report["attempts"].append(entry)
            write_manifest(args.output, report)
            print(
                f"{name}: {'exported' if not reasons else 'rejected: ' + ', '.join(reasons)}",
                flush=True,
            )
    env.substep_callback = None
    report["completed"] = True
    write_manifest(args.output, report)
    print(
        f"Exported {report['successful_rollouts']} successful rollouts to {args.output}"
    )
    if episodes is not None:
        print(f"Exported {report['valid_samples']} full 1.5 s windows and {report['touchdown_events']} touchdown events")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--motion-dir")
    source.add_argument("--upstream-roban", type=Path, help="Frax S17 upstream library")
    source.add_argument("--planned-reference", type=Path, help="LMM command-scenario reference library")
    parser.add_argument("--case", action="append", help="Upstream case to collect; repeat to select multiple")
    parser.add_argument("--window-seconds", type=float, default=8.0,
                        help="Duration of each upstream rollout; 0 keeps entire source clips")
    parser.add_argument("--min-support-steps", type=int, default=5,
                        help="Minimum sustained loaded steps for an upstream touchdown")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--force-threshold", type=float, default=5.0)
    parser.add_argument("--action-noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if (
        args.attempts < 1
        or args.max_steps < 0
        or args.max_clips < 0
        or not np.isfinite(args.force_threshold) or args.force_threshold <= 0
        or not np.isfinite(args.action_noise) or args.action_noise < 0
    ):
        parser.error("Invalid capture limits, force threshold or action noise")
    if args.upstream_roban is not None or args.planned_reference is not None:
        if ROBOT_ID != "roban":
            parser.error("Frax S17 collection requires ACP_ROBOT=roban")
        if args.min_support_steps < 3:
            parser.error("--min-support-steps must be at least 3")
        from .frax_rollout import FPS, confirmation_tail

        seconds = args.window_seconds
        if (not np.isfinite(seconds) or seconds < 0
                or (seconds and round(seconds * FPS) < confirmation_tail(args.min_support_steps))):
            parser.error("--window-seconds must be 0 or cover the 1.5 s horizon and support confirmation")
    elif args.case:
        parser.error("--case requires --upstream-roban")
    report = collect(args)
    if (args.upstream_roban is not None or args.planned_reference is not None) and not report["valid_samples"]:
        parser.exit(1, "No complete 1.5 s training windows; see manifest.json for diagnostics.\n")


if __name__ == "__main__":
    main()
