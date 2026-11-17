"""Oracle two-foot tracker trained with simulator replay + differentiable world-model rollouts."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import time


def main(variant="velocity"):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--cache-dir", default="data/cache/footstep_v2")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--iterations", type=int, default=20000)
    p.add_argument("--hist", type=int, default=5)
    p.add_argument("--horizons", type=float, nargs="+", default=[.1, .2, .3, .5, .8, 1.])
    p.add_argument("--horizon", type=int, default=16, help="Imagination length in 20 ms control steps")
    p.add_argument("--wm-hidden", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--policy-hidden", type=int, nargs="+", default=[512, 256, 128])
    p.add_argument("--buffer-steps", type=int, default=512)
    p.add_argument("--warmup-steps", type=int, default=64)
    p.add_argument("--wm-window", type=int, default=8)
    p.add_argument("--wm-batch", type=int, default=512)
    p.add_argument("--wm-batches", type=int, default=4)
    p.add_argument("--wm-only-iters", type=int, default=500)
    p.add_argument("--pi-batch", type=int, default=512)
    p.add_argument("--collect-steps", type=int, default=3)
    p.add_argument("--critic-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gamma", type=float, default=.99)
    p.add_argument("--lam", type=float, default=.95)
    p.add_argument("--action-noise", type=float, default=.1)
    p.add_argument("--reset-noise", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--val-fraction", type=float, default=.2)
    p.add_argument("--max-clips", type=int, default=0, help="Smoke tests only: restrict training clips")
    p.add_argument("--max-geometry-error", type=float, default=.05, help="Reject incompatible sole FK, metres")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--resume", help="Continue own tracker checkpoint in a new log directory; refill replay")
    args = p.parse_args()
    counts = (args.num_envs, args.iterations, args.hist, args.horizon, args.buffer_steps, args.warmup_steps,
              args.wm_window, args.wm_batch, args.wm_batches, args.pi_batch, args.collect_steps,
              args.critic_epochs, args.ckpt_every, args.log_every, args.threads, *args.wm_hidden, *args.policy_hidden)
    if min(counts) < 1 or args.buffer_steps < args.warmup_steps or args.warmup_steps < args.wm_window:
        p.error("Positive counts and buffer_steps >= warmup_steps >= wm_window required")
    if args.hist < 2 or args.wm_only_iters < 0 or args.max_clips < 0:
        p.error("hist >= 2, wm_only_iters >= 0, max_clips >= 0 required")
    if not 0 < args.gamma < 1 or not 0 <= args.lam <= 1 or args.lr <= 0 or args.max_geometry_error <= 0:
        p.error("Invalid discount, learning rate or geometry tolerance")
    if os.environ.get("ACP_ROBOT", "roban") != "roban":
        p.error("This independent task supports only RoBan")
    os.environ["ACP_ROBOT"] = "roban"
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    from .data import prepare, split_clips
    from .reference import FootstepMotion
    from .backend import backend_for_format, VELOCITY_FORMAT, CONTACT_FORMAT
    if variant not in ("velocity", "contact"):
        raise ValueError("Unknown tracker variant")
    backend = backend_for_format(CONTACT_FORMAT if variant == "contact" else VELOCITY_FORMAT)
    FootstepTask, FootstepMDP, TrackerPolicy = backend.Task, backend.MDP, backend.Policy
    OBS_DIM, COMMAND_DIM = backend.obs_dim, backend.command_dim
    from .env import FootstepTrackingSim
    from ..buffer import ReplayBuffer
    from ..kinematics import RobotKinematics
    from ..mdp import INV_DIM, normalize_xi, unflatten_state
    from ..networks import WorldModel, Critic
    from ..robot import default_joint_pos_list, metadata, NUM_J
    from ..wm_training import rollout, regression_loss, UpdateGuard
    from ..sim_numerics import TransitionGuard

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dev = args.device
    out = Path(args.log_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out/"args.json").write_text(json.dumps(vars(args), indent=2))
    clips = prepare(args.data, args.cache_dir)
    split = split_clips(clips, args.val_fraction, args.seed)
    config = dict(hist=args.hist, horizons=args.horizons, wm_hidden=args.wm_hidden, policy_hidden=args.policy_hidden)
    ck = None
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=True)
        if ck.get("format") != backend.format or ck["config"] != config:
            raise ValueError("Resume requires matching tracker format, hist, horizons and network dimensions")
        if ck["robot"]["fingerprint"] != metadata()["fingerprint"]:
            raise ValueError("Resume robot definition differs")
        fingerprints = {c["fingerprint"] for v in ck["split"].values() for c in v}
        current = {c["fingerprint"] for c in clips}
        if current != fingerprints:
            raise ValueError("Resume datasets differ")
        cached = {c["fingerprint"]: c["cache"] for c in clips}
        split = {k: [dict(c, cache=cached[c["fingerprint"]]) for c in v] for k, v in ck["split"].items()}
    kin = RobotKinematics(dev)
    motion = FootstepMotion(split["train"], kin, dev, args.horizons, args.max_clips)
    geometry = dict(mean_m=sum(r["mean_m"] for r in motion.geometry_reports)/len(motion.geometry_reports),
                    max_m=max(r["max_m"] for r in motion.geometry_reports), clips=motion.geometry_reports,
                    note="Capture uses WAQ14; training uses existing ACP RoBan asset, not identical dynamics")
    (out/"geometry.json").write_text(json.dumps(geometry, indent=2))
    if geometry["max_m"] > args.max_geometry_error:
        raise ValueError("Training asset and captured sole FK differ excessively; see geometry.json")
    if int(motion.clip_len.min()) <= args.horizon+1:
        raise ValueError("Reference block too short for query margin plus imagination horizon")
    task = FootstepTask(motion, kin)
    mdp = FootstepMDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev), task)
    policy = TrackerPolicy(args.hist, args.policy_hidden).to(dev)
    wm = WorldModel(mdp.scan_dim, args.wm_hidden).to(dev)
    critic = Critic(getattr(mdp, "critic_input_dim", INV_DIM+mdp.critic_extra+COMMAND_DIM)).to(dev)
    opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=args.lr, betas=(.9, .95))
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
    start_it, sim_steps = 0, 0
    if ck:
        for key, obj in (("pi", policy), ("wm", wm), ("critic", critic),
                         ("opt_pi", opt_pi), ("opt_wm", opt_wm), ("opt_v", opt_v)):
            obj.load_state_dict(ck[key])
        start_it, sim_steps = ck["iteration"], ck["sim_steps"]
        if args.iterations <= start_it:
            raise ValueError("--iterations must exceed saved iteration")
    env = FootstepTrackingSim(args.num_envs, motion, mdp, dev, gamma=args.gamma, reset_noise=args.reset_noise)
    buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev, extra={"o": (OBS_DIM, torch.float32)})
    history = torch.zeros(args.num_envs, args.hist, OBS_DIM, device=dev)
    query = task.sample_queries(args.num_envs)
    wm_guard, sim_guard = UpdateGuard(str(out)), TransitionGuard(str(out))
    error_floor = torch.cat((torch.full((3,), .003), torch.full((3,), .005), torch.full((3,), .02),
                             torch.full((3,), .05), torch.full((NUM_J,), .003), torch.full((NUM_J,), .05))).to(dev)
    report = dict(format=backend.format, config=config, split=split, train_names=motion.names,
                  train_frames=motion.num_frames, command_dim=COMMAND_DIM, observation_dim=OBS_DIM,
                  reward=asdict(task.cfg), robot=metadata())
    (out/"task.json").write_text(json.dumps(report, indent=2))
    print(f"Oracle tracker: {len(motion.names)} clips, {motion.num_frames} frames, obs={OBS_DIM}, command={COMMAND_DIM}, "
          f"sole FK max={geometry['max_m']:.4f}m", flush=True)
    recent_rewards, recent_terms, recent_lengths = [], [], []

    @torch.no_grad()
    def collect(n, iteration):
        nonlocal history, sim_steps, query
        for _ in range(n):
            obs = env.observe()
            history = torch.where((env.ep_step == 0)[:, None, None], obs[:, None].expand_as(history),
                                  torch.cat((history[:, 1:], obs[:, None]), 1))
            command = task.command(env.state(), env.f, query)
            policy.obs_norm.update(obs)
            policy.goal_norm.update(command)
            action = (policy(history, command)+args.action_noise*torch.randn_like(env.a_prev)).clamp(-5, 5)
            transition = env.step(action)
            transition["goal_steps"] = query.clone()
            sim_guard.observe(transition, motion, sim_steps, iteration)
            buffer.add(transition)
            S, S1 = unflatten_state(transition["s"]), unflatten_state(transition["s1"])
            good = transition["valid"]
            wm.in_norm.update(wm.inputs(S, action, normalize_xi(transition["xi"]),
                                        mdp.scan(S, motion.boxes(transition["f"], transition["v"])))[good])
            wm.out_norm.update(wm.targets(S, S1)[good])
            sim_steps += args.num_envs
            recent_rewards.append(float(transition["r"].mean()))
            if "ep_len" in transition:
                recent_lengths.extend(transition["ep_len"].tolist())
                recent_terms.extend(transition["term"][transition["done"]].float().tolist())
            done = transition["done"]
            query[done] = task.sample_queries(int(done.sum()))

    def world_update(iteration):
        window = buffer.sample_windows(args.wm_batch, args.wm_window)
        predicted = rollout(wm, mdp, motion, window["s"], window["a"], window["xi"],
                            window["f"], window["v"], window["mask"])
        if not window["mask"].any():
            raise RuntimeError("No valid world-model transitions")
        loss, _ = regression_loss(wm, predicted, window["s1"], window["mask"], error_floor)
        wm_guard.step(loss, wm, opt_wm, window, iteration)
        return float(loss.detach()/args.wm_window)

    def starts():
        selections, histories, count = [], [], 0
        for _ in range(64):
            s, h = buffer.sample_starts(2*args.pi_batch, args.hist)
            ok = s["valid"] & (s["f"]+args.horizon <= motion.clip_end[motion.clip_id[s["f"]]])
            chosen = ok.nonzero(as_tuple=False).squeeze(-1)[:args.pi_batch-count]
            selections.append({k: v[chosen] for k, v in s.items()})
            histories.append(h["o"][chosen])
            count += len(chosen)
            if count == args.pi_batch:
                return {k: torch.cat([s[k] for s in selections]) for k in selections[0]}, torch.cat(histories)
        raise RuntimeError("Cannot find enough episode-safe policy starts")

    def policy_update():
        s, obs_hist = starts()
        S = unflatten_state(s["s"])
        f, xi, qs, prev = s["f"], s["xi"], s["goal_steps"], s["a_prev"]
        xi_n, v = normalize_xi(xi), s["v"]
        boxes = motion.boxes(f, v)
        states, frames, rewards, dones = [S], [f], [], []
        for obj in (wm, critic):
            obj.requires_grad_(False)
        try:
            for _ in range(args.horizon):
                action = policy(obs_hist, task.command(S, f, qs))
                S1 = wm.step(S, action, xi_n, mdp.scan(S, boxes))
                f1 = f+1
                rewards.append(mdp.reward(S1, f1, action, prev))
                dones.append(mdp.soft_termination(S1, f1))
                obs = mdp.add_noise(mdp.obs(S1, action, xi))
                obs_hist = torch.cat((obs_hist[:, 1:], obs[:, None]), 1)
                S, f, prev = S1, f1, action
                states.append(S); frames.append(f)
            r, d = torch.stack(rewards), torch.stack(dones)
            alive = torch.cumprod(1-d, dim=0)
            before = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
            discount = (args.gamma**torch.arange(args.horizon, device=dev))[:, None]
            objective = (discount*before*r).sum(0)+args.gamma**args.horizon*alive[-1]*critic(mdp.critic_input(S, f, qs, xi_n))
            loss = -objective.mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite imagined return")
            opt_pi.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
            if not torch.isfinite(grad):
                raise RuntimeError("Nonfinite tracker gradient")
            opt_pi.step()
        finally:
            for obj in (wm, critic):
                obj.requires_grad_(True)
        with torch.no_grad():
            features = torch.stack([mdp.critic_input(tuple(x.detach() for x in state), frame, qs, xi_n)
                                    for state, frame in zip(states, frames)])
            critic.norm.update(features)
            value = critic(features)
            target, targets = value[-1], []
            for t in range(args.horizon-1, -1, -1):
                target = r[t].detach()+args.gamma*(1-d[t].detach())*((1-args.lam)*value[t+1]+args.lam*target)
                targets.append(target)
            targets = torch.stack(targets[::-1]).flatten()
            features = features[:-1].flatten(0, 1)
        for _ in range(args.critic_epochs):
            vloss = (critic(features)-targets).square().mean()
            opt_v.zero_grad(set_to_none=True)
            vloss.backward()
            gn = torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.)
            if not torch.isfinite(gn):
                raise RuntimeError("Nonfinite critic gradient")
            opt_v.step()
        return float(loss.detach()), float(vloss.detach()), float(grad), float(alive[-1].mean().detach())

    def save(iteration):
        payload = dict(format=report["format"], robot=report["robot"], config=config, split=split,
                       args=vars(args), reward=asdict(task.cfg), iteration=iteration, sim_steps=sim_steps,
                       wm=wm.state_dict(), pi=policy.state_dict(), critic=critic.state_dict(),
                       opt_wm=opt_wm.state_dict(), opt_pi=opt_pi.state_dict(), opt_v=opt_v.state_dict())
        temp = out/"last.tmp"
        torch.save(payload, temp)
        temp.replace(out/"last.pt")

    collect(args.warmup_steps, start_it)
    start = time.monotonic()
    with (out/"train.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "wall_s", "sim_steps", "wm_loss", "pi_loss", "critic_loss", "pi_gradient", "imag_survival",
                         "sim_reward", "episode_length", "term_fraction"])
        for iteration in range(start_it+1, args.iterations+1):
            collect(args.collect_steps, iteration)
            losses = [world_update(iteration) for _ in range(args.wm_batches)]
            values = policy_update() if iteration > args.wm_only_iters else (None, None, None, None)
            if iteration % args.log_every == 0 or iteration == args.iterations:
                average = lambda xs: sum(xs)/len(xs) if xs else None
                row = [iteration, time.monotonic()-start, sim_steps, average(losses), *values,
                       average(recent_rewards), average(recent_lengths), average(recent_terms)]
                writer.writerow(row); file.flush()
                print("it {} wm {:.4g} pi {} survival {} sim_r {}".format(iteration, row[3], values[0], values[3], row[8]), flush=True)
                recent_rewards.clear(); recent_lengths.clear(); recent_terms.clear()
            if iteration % args.ckpt_every == 0 or iteration == args.iterations:
                save(iteration)
    print("Saved oracle tracker: "+str(out/"last.pt"), flush=True)


if __name__ == "__main__":
    main()
