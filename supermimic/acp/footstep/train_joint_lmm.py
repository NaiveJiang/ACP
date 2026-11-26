"""Jointly fine-tune the LMM Projector and S17 full-body tracker in MuJoCo."""

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn.functional import smooth_l1_loss

from .finetune_lmm_contacts import atomic_checkpoint, policy_value, ppo_batch, update_controller
from .learned_motion import checksum
from .motion_matching import rotate
from .play_lmm_tracker import GamepadSteering, load_tracker
from ..mdp import MAX_STEP_REWARD, OBS_DIM, unflatten_state
from ..motion import CONTROL_DT


# Times are fractions of an episode. Axes are the SDL stick convention used by play.
TRAIN = (
    ("walk_start_stop", True, [(0., (0, 0, 0, 0)), (.13, (1, 0, 0, 0)), (.70, (0, 0, 0, 0))]),
    ("run_start_stop", False, [(0., (0, 0, 0, 0)), (.13, (1, 0, 0, 0)), (.73, (0, 0, 0, 0))]),
    ("walk_turn", True, [(0., (1, 0, 0, 0)), (.34, (.6, .8, 0, 0)), (.76, (0, 0, 0, 0))]),
    ("run_reverse", False, [(0., (1, 0, 0, 0)), (.40, (-1, 0, 0, 0)), (.83, (0, 0, 0, 0))]),
    ("run_aim_right", False, [(0., (.9, 0, 0, 0)), (.33, (.9, 0, -1, 0)), (.64, (.6, 0, 0, 0))]),
    ("walk_speed_switch", True, [(0., (0, 0, 0, 0)), (.12, (.65, 0, 0, 0)), (.44, (1, 0, 0, 0)),
                                 (.71, (.55, 0, 0, 0))]),
)
VALIDATION = (
    ("walk_reverse_stop", True, [(0., (.8, 0, 0, 0)), (.36, (-1, .1, 0, 0)), (.82, (0, 0, 0, 0))]),
    ("run_left_turn_stop", False, [(0., (0, 0, 0, 0)), (.12, (1, 0, 0, 0)), (.45, (.42, .68, 0, 0)),
                                  (.82, (0, 0, 0, 0))]),
)


def locomotion_score(state, intent):
    p, R, velocity, _, _, _ = state
    yaw = float(torch.atan2(R[0, 1, 0], R[0, 0, 0]))
    target = yaw if intent.heading is None else intent.heading
    desired = p.new_tensor(rotate(intent.drive, target))
    delta = float((velocity[0, :2] - desired).norm())
    heading_error = abs(math.remainder(target - yaw, math.tau))
    # When turning, emphasize facing the destination before matching world velocity.
    heading = .5 + .5 * math.cos(heading_error)
    tracking = math.exp(-(delta / .65)**2) * (.4 + .6 * heading)
    return tracking, heading, delta, heading_error


@torch.no_grad()
def episode(tracker, ctrl_value, plan_value, program, frames, ctrl_sigma, plan_sigma, *, mirror=False):
    env, policy, playback = tracker.env, tracker.policy, tracker.playback
    name, walk, controls = program
    inputs, decisions, states = [], [], []
    steering = GamepadSteering(.72, 1.55, 2.8, .20)
    planner = tracker.reset()
    history = None
    timings, axes = zip(*controls, strict=True)
    timings = np.asarray(timings)
    actual_frame = 0
    committed = 0

    def on_commit(query, mean):
        nonlocal committed
        committed += 1
        if not plan_sigma:
            return mean
        sampled = mean + plan_sigma * torch.randn_like(mean)
        # The LMM uses inference_mode for forecasting. PPO needs ordinary input tensors.
        with torch.inference_mode(False):
            decisions.append((actual_frame, *(x.detach().clone() for x in (query, mean, sampled))))
        return sampled

    playback.on_committed_projection = on_commit
    failed = False
    try:
        for t in range(frames):
            actual_frame = t
            state = env.state()
            selected = axes[int(np.searchsorted(timings, t / frames, side="right") - 1)]
            if mirror:
                selected = (selected[0], -selected[1], -selected[2], selected[3])
            intent = steering.update(selected, walk=walk, orbit=False, hold=False, camera_yaw=0.)
            fields = planner.plan(intent)
            o = env.observe()[:, None]
            history = o.expand(-1, policy.hist, -1) if history is None else torch.cat((history[:, 1:], o), dim=1)
            command, window = tracker.command(state, fields)
            mean = policy(history, command).float()
            sampled = mean + ctrl_sigma * torch.randn_like(mean) if ctrl_sigma else mean
            transition = tracker.step(sampled, window, score=True)
            outcome = unflatten_state(transition["s1"])
            failed = bool(transition["term"][0])
            tracking, heading, speed_error, heading_error = locomotion_score(outcome, intent)
            style = (transition["r"] / MAX_STEP_REWARD).clamp(-2., 1.)
            reward = CONTROL_DT * (.5 * style + tracking + .35 * heading)
            if failed:
                reward = reward - .7
            states.append((float(style[0]), speed_error, heading_error, float(reward[0])))
            if ctrl_sigma:
                inputs.append(dict(history=history, command=command, action=sampled, mean=mean,
                                   value=policy_value(ctrl_value, policy, history, command), reward=reward))
            if failed:
                break
    finally:
        playback.on_committed_projection = None
    records = None
    if inputs:
        records = {key: torch.cat([row[key] for row in inputs]) for key in inputs[0]}
    learn_plan = None
    if records is not None and decisions:
        index, query, mean, action = zip(*decisions, strict=True)
        return_to_go = []
        running = 0.
        for row in reversed(states):
            running = row[3] + .99 * running
            return_to_go.append(running)
        target = torch.tensor(return_to_go[::-1])[list(index)]
        query, mean, action = map(torch.stack, (query, mean, action))
        estimate = plan_value(query).squeeze(-1)
        learn_plan = dict(query=query, action=action, target=target,
                          advantage=target - estimate, old_logp=normal_logp(action, mean, plan_sigma))
    metrics = np.asarray(states)
    report = dict(scenario=name, steps=len(states), failed=failed, projections=committed,
                  style=float(metrics[:, 0].mean()), velocity_mae_mps=float(metrics[:, 1].mean()),
                  heading_mae_deg=float(np.rad2deg(metrics[:, 2].mean())),
                  # Normalize by the intended duration so stopping early cannot improve selection.
                  task_return=float(metrics[:, 3].sum() / (frames * CONTROL_DT)))
    return records, learn_plan, report


def normal_logp(action, mean, sigma):
    return -.5 * ((action - mean) / sigma).square().sum(-1)


def update_projector(projector, value, teacher, optimizer, episodes, sigma, epochs, batch_size):
    batch = {key: torch.cat([row[key] for row in episodes]) for key in episodes[0]}
    advantage = batch["advantage"]
    batch["advantage"] = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(.01)
    loss = None
    for _ in range(epochs):
        for ids in torch.randperm(len(advantage)).split(batch_size):
            sample = {key: tensor[ids] for key, tensor in batch.items()}
            query = sample["query"]
            mean = projector(query)
            ratio = (normal_logp(sample["action"], mean, sigma) - sample["old_logp"]).clamp(-10, 10).exp()
            gain = torch.minimum(ratio * sample["advantage"], ratio.clamp(.8, 1.2) * sample["advantage"])
            with torch.no_grad():
                anchor = teacher(query)
            critic = smooth_l1_loss(value(query).squeeze(-1), sample["target"])
            prior = ((mean - anchor) / sigma).square().mean()
            loss = -gain.mean() + .5 * critic + .05 * prior
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([*projector.parameters(), *value.parameters()], 1., error_if_nonfinite=True)
            optimizer.step()
    return float(loss.detach())


def validate(tracker, ctrl_value, plan_value, frames, average):
    actor = tracker.policy
    tracker.policy = average
    try:
        return [episode(tracker, ctrl_value, plan_value, case, frames, 0., 0.)[2] for case in VALIDATION]
    finally:
        tracker.policy = actor


@torch.no_grad()
def update_average(average, actor, decay):
    for target, parameter in zip(average.parameters(), actor.parameters(), strict=True):
        target.lerp_(parameter, 1. - decay)


def quality(metrics):
    return (-sum(item["failed"] for item in metrics), float(np.mean([item["task_return"] for item in metrics])))


def save_pair(directory, tracker, average, base_lmm, projector_payload, iteration):
    model = directory / "model"
    model.mkdir(parents=True, exist_ok=True)
    for name in ("model.pt", "stepper.pt"):
        path = model / name
        if not path.exists():
            shutil.copyfile(base_lmm / name, path)
    atomic_checkpoint(model / "projector.pt", {**projector_payload, "projector": tracker.playback.projector.state_dict()})
    checkpoint = tracker.checkpoint
    policy = {name: tensor.detach().cpu() for name, tensor in tracker.policy.state_dict().items()}
    ema = {name: tensor.detach().cpu() for name, tensor in average.state_dict().items()}
    payload = dict(robot=checkpoint["robot"], args=checkpoint["args"],
                   task_interface=checkpoint["task_interface"], reference_cfg=asdict(tracker.task.cfg),
                   iteration=iteration, pi=policy, pi_ema=ema,
                   training_engine="lmm_joint_ppo_v1", lmm_projector_sha256=checksum(model / "projector.pt"))
    atomic_checkpoint(directory / "controller.pt", payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lmm", "checkpoint", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=6.)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--ctrl-noise", type=float, default=.035)
    parser.add_argument("--projector-noise", type=float, default=.025)
    parser.add_argument("--ctrl-lr", type=float, default=1e-5)
    parser.add_argument("--projector-lr", type=float, default=1e-5)
    parser.add_argument("--ema-decay", type=float, default=.95)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--root-reference", choices=("local", "world", "body-heading"), default="local")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    scalars = (args.seconds, args.ctrl_noise, args.projector_noise, args.ctrl_lr, args.projector_lr)
    if (min(args.iterations, args.episodes, args.epochs, args.batch, args.eval_every) < 1
            or not all(math.isfinite(x) and x > 0. for x in scalars)
            or not 0. <= args.ema_decay < 1. or args.seconds < 1. or args.out.exists()):
        parser.error("Use positive settings and a new --out directory")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tracker = load_tracker(args.checkpoint, args.lmm, args.device, True, score=True,
                           root_reference=args.root_reference)
    actor = tracker.policy
    projector = tracker.playback.projector
    teacher = copy.deepcopy(actor).requires_grad_(False)
    average = copy.deepcopy(actor).requires_grad_(False)
    prior = copy.deepcopy(projector).requires_grad_(False)
    ctrl_value = nn.Sequential(nn.Linear(OBS_DIM + tracker.task.cmd_dim, 128), nn.SiLU(),
                               nn.Linear(128, 1)).to(args.device)
    plan_value = nn.Sequential(nn.Linear(len(tracker.playback.input_mean), 128), nn.SiLU(), nn.Linear(128, 1))
    ctrl_opt = torch.optim.AdamW([*actor.parameters(), *ctrl_value.parameters()], lr=args.ctrl_lr)
    plan_opt = torch.optim.AdamW([*projector.parameters(), *plan_value.parameters()], lr=args.projector_lr)
    projector_payload = torch.load(args.lmm / "projector.pt", map_location="cpu", weights_only=True)
    frames = round(args.seconds / CONTROL_DT)
    args.out.mkdir(parents=True)
    (args.out / "args.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")

    with (args.out / "train.jsonl").open("w") as log:
        def report(row):
            record = json.dumps(row)
            print(record, flush=True)
            log.write(record + "\n")
            log.flush()

        before = validate(tracker, ctrl_value, plan_value, frames, average)
        best = quality(before)
        save_pair(args.out / "best", tracker, average, args.lmm, projector_payload, 0)
        report(dict(iteration=0, validation=before))
        for i in range(1, args.iterations + 1):
            start = time.monotonic()
            controllers, plans, reports = [], [], []
            for j in range(args.episodes):
                case = TRAIN[((i - 1) * args.episodes + j) % len(TRAIN)]
                ctrl, plan, metrics = episode(tracker, ctrl_value, plan_value, case, frames, args.ctrl_noise,
                                              args.projector_noise, mirror=bool(rng.integers(2)))
                controllers.append(ctrl)
                if plan is not None:
                    plans.append(plan)
                reports.append(metrics)
            if not plans:
                raise RuntimeError("No committed LMM projections observed in the joint rollout batch")
            ctrl_loss = update_controller(ppo_batch(controllers, args.ctrl_noise), actor, ctrl_value, ctrl_opt,
                                          args.ctrl_noise, args.epochs, args.batch, teacher=teacher, anchor_weight=.025)
            plan_loss = update_projector(projector, plan_value, prior, plan_opt, plans,
                                         args.projector_noise, args.epochs, args.batch)
            update_average(average, actor, args.ema_decay)
            row = dict(iteration=i, controller_loss=ctrl_loss, projector_loss=plan_loss,
                       episodes=reports, wall_s=time.monotonic() - start)
            if i % args.eval_every == 0 or i == args.iterations:
                checks = validate(tracker, ctrl_value, plan_value, frames, average)
                row["validation"] = checks
                if quality(checks) > best:
                    best = quality(checks)
                    save_pair(args.out / "best", tracker, average, args.lmm, projector_payload, i)
            save_pair(args.out / "last", tracker, average, args.lmm, projector_payload, i)
            report(row)


if __name__ == "__main__":
    main()
