"""Stage 2: PPO for the controller and measured-contact supervision for the LMM head."""

import argparse
from collections import deque
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch import nn
from torch.nn.functional import smooth_l1_loss

from .backend import ROLLOUT_FORMAT, backend_for_format
from .event_planner import WINDOW
from .event_rollout import FootContactForces
from .fast.models import FastKinematics
from .lmm_contact import (EVENT_FRAME, HISTORY, LOOKAHEAD, OBS, FootstepPreview, Preview, angle_of, decode_events,
                          label_rows, lmm_identity, load_head, recorded_history, world_events)
from .lmm_contact_scenarios import prepare_scenarios, read_scenarios
from .rollout_controller import EVENT_FIELDS, RolloutMotion, digest, encode_event_plan, rollout_clips
from .motion_matching import heading
from .train_lmm_contacts import save, stack_rows, slice_rows
from ..frax_rollout import touchdown_windows
from ..env import DECIMATION
from ..mdp import unflatten_state
from ..motion import CONTROL_DT
from ..robot import default_joint_pos_list, metadata
from ...rotations import mat_to_quat


class WorldPlan:
    """Latch imminent targets across replans and retain just-due targets for the reward."""
    def __init__(self, slots, device):
        self.slots, self.device = slots, device
        self.events = []

    def replace(self, t, proposal):
        valid, side, dt, points, yaw = (item[0].detach().cpu().numpy() for item in proposal)
        locked = [event for event in self.events if t - 2 <= event[0] <= t + 8]
        for flag, foot, delay, point, heading in zip(valid, side, dt, points, yaw):
            due = t + round(float(delay) / CONTROL_DT)
            if flag and not any(old[1] == foot and abs(old[0] - due) <= 8 for old in locked):
                locked.append((due, int(foot), point, heading))
        self.events = sorted(locked, key=lambda x: (x[0], x[1]))

    def tensors(self, rows, t):
        times, sides, positions, angles = zip(*rows) if rows else ([], [], [], [])
        mask = np.arange(self.slots) < len(rows)
        pad = self.slots - len(rows)
        time = np.pad((np.asarray(times) - t) * CONTROL_DT, (0, pad))
        side = np.pad(np.asarray(sides, dtype=np.int64), (0, pad))
        pos = np.pad(np.asarray(positions).reshape(-1, 3), ((0, pad), (0, 0)))
        yaw = np.pad(np.asarray(angles), (0, pad))
        return tuple(torch.as_tensor(x, device=self.device)[None] for x in
                     (mask, side, time.astype(np.float32), pos.astype(np.float32), yaw.astype(np.float32)))

    def command(self, t, state):
        rows = [event for event in self.events if t < event[0] <= t + WINDOW][:self.slots]
        return encode_event_plan(state, *self.tensors(rows, t))

    def reward(self, state, f, t, task):
        c = task.cfg
        due = [event for event in self.events if 0 <= t - event[0] <= 2][:self.slots]
        reference = task.reward_terms(state, f)
        old = c.position_weight * reference["position"] + c.yaw_weight * reference["yaw"]
        if not due:
            return -CONTROL_DT * old
        side = torch.tensor([x[1] for x in due], device=self.device)
        target = torch.as_tensor(np.stack([x[2] for x in due]), device=self.device)
        yaw = torch.as_tensor(np.asarray([x[3] for x in due]), device=self.device)
        sole, R, gap = task.foot_geometry(state)
        xy2 = (sole[:, side, :2] - target[:, :2]).square().sum(-1)
        error = angle_of(R[:, side]) - yaw
        near = torch.sigmoid((c.contact_distance - gap[:, side]) / c.contact_temperature)
        landing = .5 * ((c.position_weight * torch.exp(-xy2 / c.position_std**2)
                        + c.yaw_weight * torch.exp(-2 * (1 - error.cos()) / c.yaw_std**2)) * near).sum(-1)
        return CONTROL_DT * (landing - old)


def load_scenario_data(parts, episodes, kin, device):
    preview = FootstepPreview(kin)
    scenarios = []
    for part in parts:
        item = episodes[part["case"]]
        with np.load(item["queries"], allow_pickle=False) as data:
            commands, frames, poses = (data[k] for k in ("command", "frames", "forecast"))
        if len(commands) != part["frames"] + 1 or poses.shape[1] != LOOKAHEAD + 1:
            raise ValueError(f"Scenario and teacher rollout length differ: {part['case']}")
        query_features = preview(torch.as_tensor(poses, device=device))
        # Exported sample frame zero corresponds to scenario command frame one.
        scenarios.append((commands[1:], {int(frame) - 1: i for i, frame in enumerate(frames) if frame >= 1},
                          query_features))
    return scenarios


def extract_rows(history, preview, qpos, feet, yaw, contact, commands, query_rows, *, events=None):
    n = len(feet)
    if n <= LOOKAHEAD:
        return None
    labels = measured_events(contact, feet, yaw) if events is None else events
    event = np.zeros((n, 2), dtype=bool)
    event[labels["event_frame"], labels["event_foot"]] = True
    starts = np.asarray([t for t in query_rows if t > 0 and t + LOOKAHEAD < n], dtype=np.int64)
    index = starts[:, None] + np.arange(LOOKAHEAD + 1)
    if not len(starts):
        return None
    # The forecast held its input command. Later command changes invalidate the matching label window.
    keep = (commands[index] == commands[starts, None]).all(axis=(1, 2))
    starts, index = starts[keep], index[keep]
    if not len(starts):
        return None
    device = preview.feature.device
    q = [query_rows[t] for t in starts]
    chosen = Preview(preview.feature[q], preview.sole[q], preview.yaw[q])
    sole = torch.as_tensor(np.asarray(feet)[index], device=device)
    query_root = torch.as_tensor(np.asarray(qpos)[starts, :7], device=device)
    result = label_rows(chosen, query_root, torch.as_tensor(labels["foot_support"][index], device=device), sole,
                        torch.as_tensor(heading(np.asarray(yaw)[index]), device=device),
                        torch.as_tensor(event[index], device=device),
                        torch.ones(index.shape, dtype=torch.bool, device=device))
    result["history"] = torch.cat([history[t] for t in starts])
    return {key: value.cpu() for key, value in result.items()}


def teacher_rows(part, scenario):
    commands, queries, preview = scenario
    with np.load(part["sample"], allow_pickle=False) as s:
        data = {key: s[key] for key in ("qpos", "qvel", "previous_action", "sole_pos_w", "sole_quat_w", "foot_contact_raw")}
    history = recorded_history(data, np.arange(len(commands))).split(1)
    return extract_rows(history, preview, data["qpos"], data["sole_pos_w"], data["sole_quat_w"],
                        data["foot_contact_raw"], commands, queries)


def measured_events(contact, feet, quat, *, reset_unmeasured=False):
    skip = int(reset_unmeasured)
    root = np.tile([0., 0., 0., 1., 0., 0., 0.], (len(feet) - skip, 1))
    events = touchdown_windows(np.asarray(contact)[skip:], np.asarray(feet)[skip:],
                               np.asarray(quat)[skip:], root, 5)
    if skip:
        events["event_frame"] += skip
        events["foot_support"] = np.pad(events["foot_support"], ((1, 0), (0, 0)))
    keys = ("foot_support", "event_frame", "event_foot", "event_pos_w", "event_yaw_w")
    return {key: events[key] for key in keys}


def realized_events(committed, measured, frames):
    # Ignore the first frame (no pre-reset solver force) and the last confirmation tail.
    targets = [(t, side, point, yaw) for t, side, point, yaw in committed if 1 <= t < frames - 5]
    actual = [(t, side, point, yaw) for t, side, point, yaw in zip(
        measured['event_frame'], measured['event_foot'], measured['event_pos_w'], measured['event_yaw_w'], strict=True)
              if 1 <= t < frames - 5]
    count = len(actual)
    errors = []
    for t, side, point, yaw in targets:
        candidates = [(abs(t - other[0]), i) for i, other in enumerate(actual) if side == other[1]]
        if not candidates:
            continue
        dt, index = min(candidates)
        other = actual[index]
        xy = float(np.linalg.norm(point[:2] - other[2][:2]))
        angle = abs(float(np.arctan2(np.sin(yaw - other[3]), np.cos(yaw - other[3]))))
        if dt <= 6 and xy <= .12 and angle <= .45:
            errors.append([dt * CONTROL_DT, xy, angle])
            actual.pop(index)
    error = np.mean(errors, axis=0).tolist() if errors else [None] * 3
    return dict(planned_touchdowns=len(targets), actual_touchdowns=count, realized_touchdowns=len(errors),
                touchdown_time_mae_s=error[0], touchdown_xy_mae_m=error[1], touchdown_yaw_mae_rad=error[2])


def policy_value(value, policy, history, command):
    return value(torch.cat((policy.obs_norm(history[:, -1]), policy.goal_norm(command)), dim=-1)).squeeze(-1)


@torch.no_grad()
def collect_episode(env, task, policy, baseline, head, scenario, clip, planned, exploration, threshold):
    device, motion = env.device, env.motion
    commands, queries, preview = scenario
    start, end = int(motion.clip_start[clip]), int(motion.clip_end[clip])
    env.done_eval.zero_()
    env.failed_eval.zero_()
    env.reset(torch.zeros(1, device=device, dtype=torch.long), torch.tensor([start], device=device))
    sensor = FootContactForces(env)
    force = torch.zeros(1, 2, device=device)

    def substep(_, step):
        force.add_(sensor.normal_force() / DECIMATION)

    env.substep_callback = substep
    locked = WorldPlan(motion.event_slots, device)
    history = env.observe()[:, None].expand(-1, HISTORY, -1).clone()
    foot_contact = torch.zeros(1, 2, device=device, dtype=torch.bool)
    rows, feet, angles, contacts, histories = [], [], [], [], []
    poses, normal, slip = [], [], []
    prior_force = torch.full_like(force, float('nan'))
    expected = progress = expected_turn = turned = 0.
    fallen = False
    force_matches = 0.
    committed = []
    try:
        for t in range(end - start):
            state = env.state()
            f = env.f.clone()
            if t in queries and planned:
                i = queries[t]
                p = Preview(preview.feature[i:i+1], preview.sole[i:i+1], preview.yaw[i:i+1])
                event = decode_events(*head(**p.rows(), history=history), motion.event_slots)
                locked.replace(t, world_events(state, event))
            command = locked.command(t, state) if planned else task.command(state, f, None)
            if planned:
                committed.extend(event for event in locked.events if event[0] == t + 1)
            mean = policy(history, command)
            action = mean + exploration * torch.randn_like(mean)
            old_value = policy_value(baseline, policy, history, command)
            force.zero_()
            transition = env.step(policy.clamp_action(action), contact_reward_delta=(
                lambda S, f: locked.reward(S, f, t + 1, task)) if planned else None)
            q1 = unflatten_state(transition["s1"])
            sole, R, _, velocity = task.ground.foot_state(state)
            histories.append(history)
            poses.append(torch.cat((state[0], mat_to_quat(state[1]), state[4]), dim=-1)[0].cpu().numpy())
            feet.append(sole[0].cpu().numpy())
            angles.append(mat_to_quat(R)[0].cpu().numpy())
            contacts.append(foot_contact[0].cpu().numpy())
            normal.append(prior_force[0].cpu().numpy())
            speed = velocity[..., :2].norm(dim=-1)
            slip.append(torch.where(foot_contact, speed, torch.nan)[0].cpu().numpy())
            prior_force = force.clone()
            foot_contact = force >= threshold
            support = motion.contact[f + 1]
            measured = (foot_contact == support).float().mean(-1)
            geometry = task.reward_terms(q1, f + 1)["contact"]
            reward = transition["r"] + CONTROL_DT * task.cfg.contact_weight * (measured - geometry)
            force_matches += float(measured[0])
            rows.append(dict(history=history, command=command, action=action, mean=mean, value=old_value,
                             reward=reward - .5 * transition["term"].float()))
            line = motion.root_p[f + 1, :2] - motion.root_p[f, :2]
            length = float(line.norm())
            expected += length
            progress += float(((q1[0][:, :2] - state[0][:, :2]) * line).sum()) / max(length, 1e-6)
            desired = angle_of(motion.root_R[f + 1]) - angle_of(motion.root_R[f])
            actual = angle_of(q1[1]) - angle_of(state[1])
            desired = torch.atan2(desired.sin(), desired.cos())
            expected_turn += float(desired.abs())
            turned += float(torch.atan2(actual.sin(), actual.cos()) * desired.sign())
            if bool(transition["term"][0]):
                fallen = True
                break
            history = torch.cat((history[:, 1:], env.observe()[:, None]), dim=1)
    finally:
        env.substep_callback = None
    if bool(sensor.overflow):
        raise RuntimeError("Contact force capacity overflow in closed-loop rollout")
    instructed = np.abs(commands[:len(rows)]).max(axis=0)
    moving = bool((np.linalg.norm(instructed[:2]) >= .10 and expected >= .20)
                  or (instructed[2] >= .10 and expected_turn >= .30))
    accepted = (not fallen and len(rows) == end - start
                and (expected < .20 or progress >= .45 * expected)
                and (expected_turn < .30 or turned >= .45 * expected_turn))
    events = measured_events(contacts, feet, angles, reset_unmeasured=True)
    landing = realized_events(committed, events, len(rows))
    plan_followed = (landing["realized_touchdowns"] >= max(1, .5 * landing["planned_touchdowns"],
                                                         .5 * landing["actual_touchdowns"]))
    samples = (extract_rows(histories, preview, poses, feet, angles, contacts, commands, queries, events=events)
               if accepted and (not planned or plan_followed or not moving) else None)
    finite_slip = np.asarray(slip)[np.isfinite(slip)]
    metrics = dict(planned=planned, accepted=accepted, moving=moving, steps=len(rows), fallen=fallen,
                   progress_m=progress, reference_progress_m=expected, turned_rad=turned,
                   reference_turn_rad=expected_turn, support_match=force_matches / len(rows),
                   stance_point_speed_mps=float(finite_slip.mean()) if len(finite_slip) else None,
                   head_windows=0 if samples is None else len(samples["history"]), **landing)
    capture = dict(qpos=poses, sole_pos_w=feet, sole_quat_w=angles, command=commands[:len(rows)],
                   foot_normal_force_n=normal, stance_point_speed_mps=slip,
                   time_s=np.arange(len(rows)) * CONTROL_DT, planned=planned,
                   planned_frame=np.asarray([event[0] for event in committed], dtype=np.int64),
                   planned_foot=np.asarray([event[1] for event in committed], dtype=np.int64),
                   planned_pos_w=np.asarray([event[2] for event in committed]).reshape(-1, 3),
                   planned_yaw_w=np.asarray([event[3] for event in committed]),
                   **{key: events[key] for key in ('event_frame', 'event_foot', 'event_pos_w', 'event_yaw_w')})
    return {key: torch.cat([row[key] for row in rows]) for key in rows[0]}, samples, metrics, capture


def ppo_batch(rollouts, sigma):
    combined = []
    for episode in rollouts:
        reward, value = episode["reward"], episode["value"]
        next_value = torch.cat((value[1:], torch.zeros_like(value[:1])))
        delta = reward + .99 * next_value - value
        advantage = torch.zeros_like(reward)
        running = torch.zeros_like(reward[0])
        for step in range(len(reward) - 1, -1, -1):
            running = delta[step] + .99 * .95 * running
            advantage[step] = running
        episode["target"] = value + advantage
        episode["advantage"] = advantage
        episode["logp"] = - .5 * ((episode["action"] - episode["mean"]) / sigma).square().sum(-1)
        combined.append(episode)
    batch = stack_rows(combined)
    advantage = batch["advantage"]
    batch["advantage"] = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(.01)
    return batch


def update_controller(batch, actor, value, optimizer, sigma, epochs, batch_size, *, teacher=None, anchor_weight=0.):
    last = None
    for _ in range(epochs):
        for ids in torch.randperm(len(batch["action"]), device=batch["action"].device).split(batch_size):
            sample = {key: x[ids] for key, x in batch.items()}
            mean = actor(sample["history"], sample["command"])
            logp = -.5 * ((sample["action"] - mean) / sigma).square().sum(-1)
            ratio = (logp - sample["logp"]).clamp(-10, 10).exp()
            advantage = sample["advantage"]
            actor_loss = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean()
            critic_loss = smooth_l1_loss(policy_value(value, actor, sample["history"], sample["command"]), sample["target"])
            last = actor_loss + .5 * critic_loss
            if teacher is not None:
                with torch.no_grad():
                    original = teacher(sample["history"], sample["command"])
                last = last + anchor_weight * ((mean - original) / sigma).square().mean()
            optimizer.zero_grad(set_to_none=True)
            last.backward()
            torch.nn.utils.clip_grad_norm_([*actor.parameters(), *value.parameters()], 1., error_if_nonfinite=True)
            optimizer.step()
    return float(last.detach())


def atomic_checkpoint(path, payload):
    pending = path.with_suffix(".tmp")
    torch.save(payload, pending)
    pending.replace(path)


def expand_policy(weights, old_slots, slots):
    """Keep the old actor function while adding initially inactive event inputs."""
    width = (slots - old_slots) * len(EVENT_FIELDS)
    if not width:
        return weights
    weights = dict(weights)
    for name in ("enc.0.weight", "upper_enc.0.weight", "goal_norm.mean", "goal_norm.var"):
        weights[name] = torch.nn.functional.pad(weights[name], (0, width), value=float(name == "goal_norm.var"))
    return weights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("head", "controller", "lmm", "out", "prepared", "teacher", "scenarios"):
        p.add_argument("--" + flag, type=Path, required=flag in ("head", "controller", "lmm", "out"))
    p.add_argument("--seconds", type=float, default=9.)
    p.add_argument("--plan-stride", type=int, default=5)
    p.add_argument("--rounds", type=int, default=300)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--head-updates", type=int, default=2)
    p.add_argument("--start-share", type=float, default=.15)
    p.add_argument("--ramp", type=float, default=.10)
    p.add_argument("--action-noise", type=float, default=.07)
    p.add_argument("--force-threshold", type=float, default=5.)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if (min(args.rounds, args.episodes, args.epochs, args.batch, args.plan_stride, args.head_updates) < 1
            or not 0 < args.start_share <= 1 or not 0 < args.ramp <= 1 or not 0 < args.action_noise <= .3
            or not np.isfinite(args.seconds) or args.seconds < 2. or not np.isfinite(args.force_threshold)
            or args.force_threshold <= 0 or args.out.exists() or (args.prepared is None and args.teacher is None)):
        p.error("Use a new output, valid rollout settings, and --teacher or --prepared")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    identity = lmm_identity(args.lmm)
    head, head_ck = load_head(args.head, args.lmm, args.device)
    checkpoint = torch.load(args.controller, map_location=args.device, weights_only=True)
    config = dict(checkpoint["config"])
    if (checkpoint.get("format") != ROLLOUT_FORMAT or config["hist"] != HISTORY
            or checkpoint["robot"]["fingerprint"] != metadata()["fingerprint"]):
        raise ValueError("Stage 2 requires a compatible 10-frame rollout controller checkpoint")
    if "lmm_foothold_head" in checkpoint:
        if checkpoint.get("lmm_event_frame") != EVENT_FRAME:
            raise ValueError("Stage 2 checkpoint uses incompatible foothold coordinates; start from the rollout contact tracker")
        if checkpoint.get("lmm") != identity:
            raise ValueError("Stage 2 controller checkpoint and LMM weights do not match")
    args.out.mkdir(parents=True)
    prepared = args.prepared or args.out / "prepared"
    scenario_path = prepared / "scenarios"
    dataset = prepared / "teacher"
    if args.prepared is None:
        prepare_scenarios(args.lmm, scenario_path, args.seconds, args.plan_stride, args.scenarios)
        subprocess.run([sys.executable, "-m", "supermimic.acp.collect_mimic",
                        "--ckpt", str(args.teacher), "--planned-reference", str(scenario_path),
                        "--output", str(dataset), "--device", args.device, "--force-threshold",
                        str(args.force_threshold)], check=True)
    scenario_manifest = json.loads((scenario_path / "manifest.json").read_text())
    if scenario_manifest.get("lmm") != identity:
        raise ValueError("Prepared scenarios and current LMM do not match")
    teacher_report = json.loads((dataset / "manifest.json").read_text())
    if Path(teacher_report["args"]["planned_reference"]).resolve() != scenario_path.resolve():
        raise ValueError("Prepared teacher data and scenario library do not match")
    if teacher_report['args']['force_threshold'] != args.force_threshold:
        raise ValueError('Prepared teacher data uses a different --force-threshold')
    _, episode_info = read_scenarios(scenario_path)
    for entry in teacher_report["attempts"]:
        source = entry["upstream"]
        current = episode_info[entry["source_clip"]]
        if any(source.get(key) != current.get(key) for key in ("motion_sha256", "queries_sha256")):
            raise ValueError(f"Teacher rollout and generated commands disagree: {entry['source_clip']}")
    parts = rollout_clips([dataset], LOOKAHEAD + 2)
    old_slots = config["event_slots"]
    slots = max(old_slots, *(part["event_slots"] for part in parts))
    config["event_slots"] = slots
    if slots != old_slots:
        print(f"Expanded controller event capacity from {old_slots} to {slots}", flush=True)
    kin = FastKinematics(args.device)
    motion = RolloutMotion(parts, kin, args.device, slots)
    backend = backend_for_format(ROLLOUT_FORMAT, event_slots=slots)
    task = backend.Task(motion, kin, backend.RewardCfg(**checkpoint["reward"]))
    mdp = backend.MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=args.device), task)
    policy = backend.Policy(config["hist"], config["policy_hidden"], event_slots=slots).to(args.device)
    policy.load_state_dict(expand_policy(checkpoint["pi"], old_slots, slots))
    value = nn.Sequential(nn.Linear(OBS + backend.command_dim, 128), nn.SiLU(), nn.Linear(128, 1)).to(args.device)
    optimizer = torch.optim.AdamW([*policy.parameters(), *value.parameters()], lr=3e-5)
    head_opt = torch.optim.AdamW(head.parameters(), lr=5e-5)
    scenarios = load_scenario_data(parts, episode_info, kin, args.device)
    oracle_rows = [rows for part, scenario in zip(parts, scenarios, strict=True)
                   if (rows := teacher_rows(part, scenario)) is not None]
    if not oracle_rows:
        raise ValueError("No teacher query has a complete 1.5 s unchanged-command window")
    oracle = stack_rows(oracle_rows)
    env = backend.Env(1, motion, mdp, args.device, eval_mode=True, reset_noise=False, reference_reset_action=True)
    source_hash = digest(args.controller)
    share = args.start_share
    handover_trials = deque(maxlen=12)
    replay = None
    latest = args.out / 'latest_rollouts'
    latest.mkdir()
    with (args.out / "train.jsonl").open("w") as log:
        for iteration in range(1, args.rounds + 1):
            rollouts, measured, reports = [], [], []
            for episode in range(args.episodes):
                clip = (iteration * args.episodes + episode - args.episodes) % len(parts)
                use_plan = bool(rng.random() < share)
                batch, support, report, capture = collect_episode(
                    env, task, policy, value, head, scenarios[clip], clip, use_plan,
                    args.action_noise, args.force_threshold)
                rollouts.append(batch)
                if support is not None:
                    measured.append(support)
                path = latest / f'episode_{episode:03d}.npz'
                pending = path.with_suffix('.tmp.npz')
                np.savez_compressed(pending, iteration=iteration, **capture)
                pending.replace(path)
                reports.append(dict(case=parts[clip]["case"], recording=str(path.relative_to(args.out)), **report))
            actor_loss = update_controller(ppo_batch(rollouts, args.action_noise), policy, value, optimizer,
                                           args.action_noise, args.epochs, args.batch)
            loss_value = None
            if measured:
                combined = stack_rows(([replay] if replay is not None else []) + measured)
                replay = {key: x[-4096:] for key, x in combined.items()}
            for _ in range(args.head_updates):
                ids = torch.randint(len(oracle["history"]), (args.batch,))
                loss = head.loss(slice_rows(oracle, ids, args.device))
                if replay is not None:
                    ids = torch.randint(len(replay["history"]), (args.batch,))
                    loss = .5 * (loss + head.loss(slice_rows(replay, ids, args.device)))
                head_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                head_opt.step()
                loss_value = float(loss.detach())
            attempts = [row for row in reports if row["planned"] and row["moving"]]
            outcomes = [row["accepted"] and row["head_windows"] > 0 for row in attempts]
            handover_trials.extend(outcomes)
            tested, passed = len(handover_trials), sum(handover_trials)
            following = min(1., share + args.ramp) if tested >= 3 and passed >= .6 * tested else share
            if following > share:
                handover_trials.clear()
            report = dict(iteration=iteration, planned_share=share, next_share=following, controller_loss=actor_loss,
                          planner_loss=loss_value, teacher_windows=len(oracle["history"]),
                          student_windows=0 if replay is None else len(replay["history"]),
                          moving_planned_trials=len(attempts), accepted_moving_plans=sum(outcomes),
                          handover_trials=tested, handover_passes=passed,
                          episodes=reports)
            print(json.dumps(report), flush=True)
            log.write(json.dumps(report) + "\n")
            log.flush()
            share = following
            save(args.out / "head.pt", head, identity, head_ck["split"], stage=2, step=iteration, metrics=report)
            saved = dict(format=ROLLOUT_FORMAT, config=config, robot=checkpoint["robot"],
                         command_dim=backend.command_dim, observation_dim=OBS, reward=checkpoint["reward"],
                         args=checkpoint["args"], split=checkpoint["split"], iteration=iteration,
                         training_engine="lmm_rollout_ppo_v1", pi=policy.state_dict(), pi_live=policy.state_dict(),
                         lmm_foothold_head=head.state_dict(), lmm_head_hidden=head.hidden,
                         lmm_event_frame=EVENT_FRAME,
                         lmm=identity, stage2=report, controller_parent_sha256=source_hash)
            atomic_checkpoint(args.out / "last.pt", saved)


if __name__ == "__main__":
    main()
