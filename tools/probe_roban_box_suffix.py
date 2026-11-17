"""Evaluate tracker clips at fixed start phases without changing training."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("ACP_ROBOT", "roban")
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch
import warp as wp
from supermimic.acp import networks
from supermimic.acp.checkpoint import load_checkpoint
from supermimic.acp.env import TrackingSim, BOX_PARK, FRICTION, JOINT_OFFSET, COM_RANGE
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.mdp import MDP, OBS_DIM, RewardCfg
from supermimic.acp.motion import MotionLib
from supermimic.acp.networks import Policy
from supermimic.acp.robot import NUM_J, COM_BODY, default_joint_pos_list
from supermimic.acp.task import ContactTask, ContactCfg, from_base
from supermimic.rotations import mat_to_quat


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--planner", type=Path, help="Planner checkpoint for closed-loop contact commands")
    ap.add_argument("--eta-obs", type=float, default=1., help="Probability of planner targets when --planner is set")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phases", type=float, nargs="+", default=[.65, .75, .8, .85, .9, .95])
    ap.add_argument("--record-dir", type=Path, help="Save initial and exact post-step physical states for video")
    ap.add_argument("--clip", help="Evaluate just this clip (relative name without _motion.npz)")
    ap.add_argument("--category", default="box_climb", help="Clip category, or 'all'; ignored with --clip")
    ap.add_argument("--modes", nargs="+", choices=["nominal", "training_randomization"],
                    default=["nominal", "training_randomization"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--motion-dir", type=Path, help="Evaluation dataset override; never edits the checkpoint")
    ap.add_argument("--contacts-dir", type=Path, help="Contact labels for the evaluation dataset")
    ap.add_argument("--metrics", action="store_true", help="Measure tracking, contacts and endpoint accuracy")
    ap.add_argument("--friction", type=float, help="Override terrain and robot friction after randomization")
    ap.add_argument("--goal-s", type=float, help="Override sampled/fixed goal lookahead in seconds")
    ap.add_argument("--reset_noise", action=argparse.BooleanOptionalAction, default=None,
                    help="Override physical reset noise in randomized mode; default follows the checkpoint's training setting")
    args = ap.parse_args()
    if not 0. <= args.eta_obs <= 1.:
        ap.error("--eta-obs must be between 0 and 1")
    assert all(0 <= p < 1 for p in args.phases)
    if args.friction is not None and args.friction <= 0:
        ap.error("--friction must be positive")
    if args.goal_s is not None and args.goal_s <= 0:
        ap.error("--goal-s must be positive")
    dev = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    ck = load_checkpoint(args.ckpt, map_location=dev)
    training_reset_noise = ck["args"].get("reset_noise", True) if args.reset_noise is None else args.reset_noise
    ca = ck.get("init_args", ck["args"])
    motion_dir = str(args.motion_dir or ca["motion_dir"])
    contact_cfg = dict(ck["ctc_cfg"])
    if args.contacts_dir or args.motion_dir:
        contact_cfg["contacts_dir"] = str(args.contacts_dir or (motion_dir.rstrip('/') + '_contacts'))
    kin = RobotKinematics(dev)
    if args.clip:
        selected = Path(motion_dir) / (args.clip + "_motion.npz")
        if not selected.is_file():
            ap.error(f"Clip not found: {selected}")
        motion = MotionLib([str(selected)], kin, dev, root=motion_dir)
    else:
        motion = MotionLib.from_dir(motion_dir, kin, dev)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev),
              RewardCfg(term_window=round(ck["args"].get("term_window", 0.) / .02)))
    ctc = ContactTask(ContactCfg(**contact_cfg), mdp, motion_dir, dev)
    networks.AMP_DTYPE = torch.bfloat16 if ck["args"].get("amp") == "bf16" else None
    H = ca["hist"]
    pi = Policy(OBS_DIM, ctc.cmd_dim, hist=H).to(dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    if args.planner:
        from supermimic.acp.mdp import REF_ORI_DIMS
        from supermimic.acp.planner import ContactPlanner, PlannerCfg, PlannerScan, plan
        pk = load_checkpoint(args.planner, map_location=dev)
        pcfg = PlannerCfg(**pk["planner_cfg"])
        scan = PlannerScan(pcfg, dev)
        planner = ContactPlanner(OBS_DIM - REF_ORI_DIMS, pcfg).to(dev)
        planner.load_state_dict(pk["planner"])
        planner.eval()
    ids = [i for i, n in enumerate(motion.names)
           if args.clip or args.category == "all" or n.split("/")[0] == args.category]
    if not ids:
        ap.error("No matching clips selected")
    clips = torch.tensor(ids * len(args.phases), device=dev)
    phases = torch.tensor(args.phases, device=dev).repeat_interleave(len(ids))
    lengths = motion.clip_len[clips] - 1
    starts = motion.clip_start[clips] + (phases * lengths).long()
    ends = motion.clip_end[clips]
    N = len(clips)
    env = TrackingSim(N, motion, mdp, dev, eval_mode=True)
    env_ids = torch.arange(N, device=dev)
    torso = env.robot.indexing.body_ids[env.robot.find_bodies(COM_BODY)[0][0]]
    results = []
    initial_condition_hashes = {}
    # Keep nominal first when both modes are requested: it uses the initial
    # nominal dynamics before any randomized model fields are installed.
    for mode in [m for m in ("nominal", "training_randomization") if m in args.modes]:
        torch.manual_seed(args.seed)
        noisy = mode != "nominal"
        env.done_eval.zero_()
        env.failed_eval.zero_()
        env.reset_noise = noisy and training_reset_noise
        if noisy:
            env.xi[:, 0].uniform_(*FRICTION)
            env.xi[:, 1:1 + NUM_J].uniform_(-JOINT_OFFSET, JOINT_OFFSET)
            com = torch.stack([torch.empty(N, device=dev).uniform_(lo, hi) for lo, hi in COM_RANGE], -1)
            env.xi[:, 1 + NUM_J:] = com
            env.sim.model.body_ipos[:, torso] = env.sim.get_default_field("body_ipos")[torso] + com
            env.sim.model.geom_friction[:, env._terrain_geoms, 0] = FRICTION[0]
            env.sim.model.geom_friction[:, env.robot.indexing.geom_ids, 0] = env.xi[:, :1]
        if args.friction is not None:
            env.xi[:, 0] = args.friction
            env.sim.model.geom_friction[:, env._terrain_geoms, 0] = args.friction
            env.sim.model.geom_friction[:, env.robot.indexing.geom_ids, 0] = args.friction
        variants = motion.sample_variants(clips) if noisy else motion.first_variants(clips)
        env.reset(env_ids, starts.clone(), variants)
        if args.record_dir:
            args.record_dir.mkdir(parents=True, exist_ok=True)
            initial = env.state()
            recorded = {"initial_root_p": initial[0].cpu().numpy().copy(),
                        "initial_root_quat": mat_to_quat(initial[1]).cpu().numpy().copy(),
                        "initial_q": initial[4].cpu().numpy().copy()}
            trajectory = {k: [] for k in ("post_root_p", "post_root_quat", "post_q", "post_f", "alive", "tgt_w", "tgt_valid")}
        initial_lift = env.state()[0][:, 2] - motion.root_p[starts, 2]
        gs = ctc.sample_goal_steps(N) if noisy else torch.full((N,), 50, device=dev)
        if args.goal_s is not None:
            gs.fill_(round(args.goal_s / .02))
        digest = hashlib.sha256()
        for x in (*env.state(), env.xi, starts, variants, gs):
            digest.update(x.detach().cpu().contiguous().numpy().tobytes())
        initial_condition_hashes[mode] = digest.hexdigest()
        metrics = None
        if args.metrics:
            from tools.roban_tracker_metrics import TrackerMetrics
            metrics = TrackerMetrics(ctc, N, dev)
        steps = torch.zeros(N, dtype=torch.long, device=dev)
        last_f = starts.clone()
        xy_sum = torch.zeros(N, device=dev)
        z_sum = torch.zeros(N, device=dev)
        bad = torch.zeros(N, dtype=torch.bool, device=dev)
        overflow_seen = torch.zeros(N, dtype=torch.int32, device=dev)
        last_z = torch.zeros(N, device=dev)
        last_tilt = torch.zeros(N, device=dev)
        max_action_before_clamp = torch.tensor(0., device=dev)
        hist = None
        while not bool(env.done_eval.all()):
            alive = ~env.done_eval.clone()
            inactive = (~alive).nonzero().flatten()
            if len(inactive):
                pose = torch.zeros(len(inactive), 7, device=dev)
                pose[:, 2], pose[:, 3] = 20., 1.
                env.robot.write_root_link_pose_to_sim(pose, env_ids=inactive)
                env.robot.write_root_link_velocity_to_sim(torch.zeros(len(inactive), 6, device=dev), env_ids=inactive)
                env.robot.write_joint_position_to_sim(env.default_q.expand(len(inactive), -1), env_ids=inactive)
                env.robot.write_joint_velocity_to_sim(torch.zeros(len(inactive), NUM_J, device=dev), env_ids=inactive)
            S = env.state()
            o = env.observe()
            if noisy:
                o = mdp.add_noise(o)
            hist = o[:, None].repeat(1, H, 1) if hist is None else torch.cat((hist[:, 1:], o[:, None]), 1)
            tw = ctc.next_pos[env.f, ctc.rv(env.v)]
            tv = ctc.next_valid[env.f, ctc.rv(env.v)]
            if args.planner:
                pw, pv = plan(planner, scan, pcfg, ctc, hist, S, env.f, env.v, gs, noise=noisy)
                # Keep RNG draws identical for reference and planner comparison arms.
                use = torch.rand(N, device=dev) < args.eta_obs
                tw = torch.where(use[:, None, None], pw, tw)
                tv = torch.where(use[:, None], pv, tv)
                cmd = ctc.command(S, env.f, env.v, gs, noise=noisy, targets_w=tw, valid=tv)
            else:
                cmd = ctc.command(S, env.f, env.v, gs, noise=noisy)
            if metrics is not None:
                metrics.command(env.f, env.v, tw, tv)
            if args.record_dir:
                # Record the exact target command seen by the policy, including command noise.
                targets = from_base(cmd[:, :27].reshape(N, 9, 3), S[0], S[1])
                trajectory["tgt_w"].append(targets.cpu().numpy().copy())
                trajectory["tgt_valid"].append(tv.cpu().numpy().copy())
            action = pi(hist, cmd)
            if noisy:
                action += ck["args"]["action_noise"] * torch.randn_like(action)
                max_action_before_clamp = torch.maximum(max_action_before_clamp, action[alive].abs().max())
                action = action.clamp(-20, 20)
            action[inactive] = 0.
            overflow = wp.to_torch(env.sim.wp_data.overflow)
            overflow.zero_()
            transition = env.step(action)
            overflow_seen |= torch.where(alive, overflow, 0)
            S1 = env.state()
            f1 = transition["f"] + 1
            if args.record_dir:
                for key, value in [("post_root_p", S1[0]), ("post_root_quat", mat_to_quat(S1[1])),
                                   ("post_q", S1[4]), ("post_f", f1), ("alive", alive)]:
                    trajectory[key].append(value.cpu().numpy().copy())
            finite = transition["valid"]
            if metrics is not None:
                metrics.observe(S1, f1, env.v, alive & finite, action)
            bad |= alive & ~finite
            steps += alive
            last_f = torch.where(alive, f1, last_f)
            xy = (S1[0][:, :2] - motion.root_p[f1, :2]).norm(dim=-1)
            z = (S1[0][:, 2] - motion.root_p[f1, 2]).abs()
            tilt = (S1[1][:, 2, 2] - motion.root_R[f1, 2, 2]).abs()
            xy_sum += torch.where(alive & finite, xy, 0.)
            z_sum += torch.where(alive & finite, z, 0.)
            last_z = torch.where(alive & finite, z, last_z)
            last_tilt = torch.where(alive & finite, tilt, last_tilt)
            env.f = torch.minimum(env.f, ends - 1)
            finished = (alive & env.done_eval).nonzero().flatten()
            if len(finished):
                env.sim.model.geom_pos[finished[:, None], env.box_geoms[None, :]] = torch.tensor(BOX_PARK, device=dev)
                if env.hf_geom is not None:
                    env.sim.model.geom_pos[finished, env.hf_geom] = torch.tensor((0., 0., -100.), device=dev)
        mode_rows = []
        measured = metrics.finish() if metrics is not None else None
        for j, c in enumerate(clips.cpu().tolist()):
            valid = not bool(bad[j]) and not int(overflow_seen[j])
            failed = bool(env.failed_eval[j])
            row = {"mode": mode, "clip": motion.names[c], "start_phase": args.phases[j // len(ids)],
                   "reset_noise": bool(env.reset_noise),
                   "start_time_s": int(starts[j] - motion.clip_start[c]) * .02,
                   "elapsed_s": int(steps[j]) * .02, "remaining_steps": int(ends[j] - starts[j]),
                   "completed": not failed and int(last_f[j]) == int(ends[j]), "failed": failed,
                   "curriculum_reached_99pct": bool((last_f[j] - motion.clip_start[c]) >= .99 * lengths[j]) and not bool(bad[j]),
                   "terminal_phase": float((last_f[j] - motion.clip_start[c]) / lengths[j]),
                   "valid_physics": valid, "overflow_flags": int(overflow_seen[j]),
                   "root_xy_error_mean_m": float(xy_sum[j] / steps[j]),
                   "root_z_error_mean_m": float(z_sum[j] / steps[j]),
                   "terminal_root_z_error_m": float(last_z[j]), "terminal_tilt_error": float(last_tilt[j]),
                   "initial_lift_plus_reset_noise_m": float(initial_lift[j]),
                   "goal_lookahead_s": float(gs[j]) * .02,
                   "friction": float(env.xi[j, 0]),
                   "joint_offsets_rad": env.xi[j, 1:1 + NUM_J].cpu().tolist(),
                   "torso_com_offset_m": env.xi[j, 1 + NUM_J:].cpu().tolist(),
                   "mode_max_action_before_clamp": float(max_action_before_clamp) if noisy else None}
            if measured is not None:
                row.update(measured[j])
                row["valid_physics"] = valid and row["finite_tracking_metrics"]
                row["completed_endpoint_20cm"] = bool(row["completed"] and row["valid_physics"]
                    and row["terminal_root_xy_error_m"] is not None and row["terminal_root_xy_error_m"] <= .2)
            mode_rows.append(row)
        results.extend(mode_rows)
        if args.record_dir:
            from supermimic.acp import heightfield
            from supermimic.acp.robot import JOINT_NAMES, ROBOT_ID
            np.savez_compressed(args.record_dir / (mode + ".npz"), **recorded,
                **{key: np.stack(value) for key, value in trajectory.items()},
                names=np.array([motion.names[c] for c in clips.cpu().tolist()]),
                start_phase=phases.cpu().numpy(), start_frames=starts.cpu().numpy(),
                clip_start=motion.clip_start[clips].cpu().numpy(), clip_len=motion.clip_len[clips].cpu().numpy(),
                boxes=motion.clip_boxes[clips, variants].cpu().numpy(),
                ref_root_p=motion.root_p.cpu().numpy(), ref_root_quat=mat_to_quat(motion.root_R).cpu().numpy(),
                ref_q=motion.q.cpu().numpy(), heightfield_paths=np.array([a["path"] for a in heightfield.ASSETS]),
                robot_id=np.array(ROBOT_ID), joint_names=np.array(JOINT_NAMES),
                checkpoint=np.array(str(args.ckpt.resolve())), mode=np.array(mode), seed=np.array(args.seed),
                reset_noise=np.array(env.reset_noise),
                planner_checkpoint=np.array(str(args.planner.resolve()) if args.planner else ""),
                eta_obs=np.array(args.eta_obs if args.planner else 0.))
        print(json.dumps({"mode": mode, "phases": [
            {"phase": p, "completed": sum(r["completed"] for r in mode_rows if r["start_phase"] == p),
             "total": len(ids), "invalid_physics": sum(not r["valid_physics"] for r in mode_rows if r["start_phase"] == p)}
            for p in args.phases]}, ensure_ascii=False), flush=True)
    report = {"checkpoint": str(args.ckpt.resolve()), "checkpoint_sha256": hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
              "planner_checkpoint": str(args.planner.resolve()) if args.planner else None,
              "planner_checkpoint_sha256": hashlib.sha256(args.planner.read_bytes()).hexdigest() if args.planner else None,
              "eta_obs": args.eta_obs if args.planner else 0.,
              "command_source": "planner" if args.planner and args.eta_obs == 1 else "mixed" if args.planner and args.eta_obs > 0 else "reference",
              "initial_condition_sha256": initial_condition_hashes,
              "iteration": ck["iteration"], "seed": args.seed, "phases": args.phases, "requested_clip": args.clip,
              "category": args.category, "modes": args.modes,
              "motion_dir": str(Path(motion_dir).resolve()),
              "contacts_dir": str(Path(ctc.cfg.contacts_dir or (motion_dir.rstrip('/') + '_contacts')).resolve()),
              "metrics": args.metrics, "friction_override": args.friction, "goal_s_override": args.goal_s,
              "metric_protocol": "Metrics include the failure transition and exclude invalid/inactive states. Contact is geometric distance <= touch_dist, not force. Contact precision/recall use the checkpoint contact profile and debounced reference labels. Placement uses the selected intended targets before command noise over the last five command frames; initial contacts are not onsets. Foot errors/speeds measure link origins, so support speed includes foot rolling. Completion plus endpoint <=0.2 m is an additional diagnostic, not the training termination rule.",
              "training_randomization": {"reset_noise": bool(training_reset_noise),
                  "reset_noise_source": "checkpoint" if args.reset_noise is None else "cli_override",
                  "target_noise": ctc.cfg.target_noise, "scan_noise": ctc.cfg.scan_noise,
                  "action_noise": ck["args"]["action_noise"]},
              "record_directory": str(args.record_dir.resolve()) if args.record_dir else None,
              "protocol": "Nominal resets to reference pose/velocity. Both modes use reset penetration lifting; one trial per clip/phase/mode with the same policy. Randomized mode includes observation/command/action noise, friction, COM, joint offsets and sampled goal horizons; physical reset noise is controlled by training_randomization.reset_noise. Recorded contact targets are the exact policy command transformed back to world coordinates; post-step states display the preceding applied command.",
              "episodes": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
