"""Evaluate a contact tracker (supermimic.acp.train_stage1 / train_stage2 checkpoint) on every clip of its motion set.

Protocol of supermimic.evaluate: each clip plays once from its first frame on its first valid terrain variant, no
randomization and no noise, and fails if the rollout triggers the training termination; normalized return is the
motion-tracking reward sum over its maximum. The contact command is noise-free with the goal 1 s ahead. Contact
metrics count, over the frames the rollout is alive, the bodies touching the terrain (sphere approximation within
``touch_dist``) against the reference labels: precision and recall per body. ``--planner`` gives the tracker the
contact planner's targets (a train_planner_offline or train_stage2 checkpoint; noise-free height scan) instead of the
reference's, with probability ``--eta_obs`` per step and clip. Placement error: per body, the distance from where each
new contact lands to the nearest target given over the last :data:`supermimic.acp.task.PLACE_WINDOW` steps.

    python -m supermimic.acp.evaluate --ckpt logs/acp_stage2/ckpt_040000.pt --planner logs/acp_stage2/ckpt_040000.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
reference_source = parser.add_mutually_exclusive_group()
reference_source.add_argument("--motion-dir", help="Reference motions to evaluate instead of the checkpoint's motion directory")
reference_source.add_argument("--upstream-roban", type=Path, help="Mapped upstream Motion Matching library")
reference_source.add_argument("--forecast", type=Path, help="LMM forecast.npz, with its sibling planned/ directory")
parser.add_argument("--clip", help="One clip, such as walk/walk for --motion-dir or walk for --upstream-roban")
parser.add_argument("--start-seconds", type=float, default=0., help="Start inside the selected upstream clip")
parser.add_argument("--viewer", action="store_true", help="Play a clip or the entire upstream library in a live MuJoCo window")
parser.add_argument("--ema", action="store_true", help="Evaluate the saved EMA policy")
parser.add_argument("--ground-friction", type=float, default=None)
parser.add_argument("--box-friction", type=float, default=None)
parser.add_argument("--goal_s", type=float, default=1.0)
parser.add_argument("--out", type=str, default=None, help="results json (default: next to the checkpoint)")
parser.add_argument("--record", type=str, default=None, help="save trajectories (npz)")
parser.add_argument("--planner", type=str, default=None, help="contact targets from this checkpoint's planner")
parser.add_argument("--eta_obs", type=float, default=1.0, help="with --planner: per step and clip, the planner's targets with this probability")
parser.add_argument("--term_window", type=float, default=None, help="s: termination window either side of the frame; default: the checkpoint's, else 0")
parser.add_argument("--train_noise", nargs="*", default=None, choices=["obs", "command", "action"],
                    help="the training noise (all three without a value): observations; contact targets and height scans;"
                         " actions (0.1)")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
if args.viewer and not (args.clip or args.upstream_roban or args.forecast):
    parser.error("--viewer requires --clip, --upstream-roban, or --forecast")
if args.forecast and args.clip:
    parser.error("--forecast already specifies a single planned clip")
if not math.isfinite(args.start_seconds) or args.start_seconds < 0 or (
    args.start_seconds and not (args.upstream_roban and args.clip)
):
    parser.error("--start-seconds must be nonnegative and requires one upstream clip")
_noise = set() if args.train_noise is None else set(args.train_noise or ("obs", "command", "action"))
NOISE_OBS, NOISE_CMD, NOISE_ACT = "obs" in _noise, "command" in _noise, "action" in _noise

import numpy as np
import torch
from supermimic.acp.checkpoint import load_checkpoint

import supermimic.acp.networks as networks
from supermimic.acp.env import TrackingSim, evaluation_surface_friction
from supermimic.acp.robot import CONTACT_BODIES, MOTION_DIR, NB, ROBOT_ID, JOINT_NAMES, default_joint_pos_list
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.mdp import MAX_STEP_REWARD, MDP, OBS_DIM, REF_ORI_DIMS, RewardCfg
from supermimic.acp.motion import CONTROL_DT, MotionLib
from supermimic.acp.networks import Policy
from supermimic.acp.planner import ContactPlanner, PlannerCfg, PlannerScan, plan
from supermimic.acp.task import PLACE_WINDOW, ContactCfg, ContactTask, placement_error
from supermimic.acp.reference_mimic import ReferenceCfg, ReferenceTask, build_policy
from supermimic.acp.s17_actuation import DELAY_STEPS
from supermimic.rotations import mat_to_quat

torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(args.seed)
dev = "cuda:0"
ck = load_checkpoint(args.ckpt, map_location=dev)
a_ = ck["args"]
ia = ck.get("init_args", a_)
motion_dir = args.motion_dir or ia.get("motion_dir") or MOTION_DIR
kin = RobotKinematics(dev)
if (args.clip or args.upstream_roban or args.forecast) and ck.get("task_interface") != "reference_mimic_v1":
    parser.error("--clip, --upstream-roban and --forecast require a full-body reference checkpoint")
forecast = None
if args.forecast:
    root = args.forecast.parent / "planned"
    motion = MotionLib([str(root / "velocity_demo_motion.npz")], kin, dev, root=str(root))
    offsets = ck["reference_cfg"]["offsets"]
    with np.load(args.forecast, allow_pickle=False) as source:
        if not np.array_equal(source["offsets"], offsets):
            parser.error(f"Forecast offsets must match this controller's offsets {offsets}")
        shapes = (("root_position", (3,)), ("root_rotation", (3, 3)), ("root_linear_velocity", (3,)),
                  ("root_angular_velocity", (3,)), ("joint_position", (len(JOINT_NAMES),)),
                  ("joint_velocity", (len(JOINT_NAMES),)))
        arrays = [source[name] for name, _ in shapes]
        if any(array.shape != (motion.num_frames, len(offsets), *shape) or not np.isfinite(array).all()
               for array, (_, shape) in zip(arrays, shapes, strict=True)):
            parser.error("Invalid forecast shapes or nonfinite frames in the LMM forecast")
        forecast = tuple(torch.as_tensor(array, dtype=torch.float32, device=dev) for array in arrays)
elif args.upstream_roban:
    from supermimic.acp.footstep.motion_matching import Database, Robot, upstream_clips

    if ROBOT_ID != "roban":
        parser.error("--upstream-roban requires ACP_ROBOT=roban")
    directory = args.upstream_roban
    robot = Robot()
    try:
        sample_hz = float(json.loads((directory/"metadata.json").read_text())["sample_hz"])
        if not math.isfinite(sample_hz) or sample_hz <= 0:
            raise ValueError("The upstream sample rate must be positive and finite")
        database = Database.from_upstream_preview(robot, directory, sample_hz, case=args.clip,
                                                  high_step_running=False)
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    cases = [args.clip] if args.clip else [clip["case"] for clip in upstream_clips(directory)]
    columns = np.r_[np.arange(robot.root, robot.root+7), robot.joint_qpos]
    with TemporaryDirectory(prefix="roban-upstream-") as temp:
        paths = []
        for index, case in enumerate(cases):
            poses = database.pose[database.clip == index]
            if args.start_seconds > (len(poses)-1)/sample_hz-CONTROL_DT:
                parser.error(f"--start-seconds leaves less than one control step in {case!r}")
            start = int(args.start_seconds*sample_hz)
            path = Path(temp)/f"{case}_motion.npz"
            np.savez(path, joint_pos=poses[start:, columns].astype(np.float32),
                     joint_names=np.array(robot.joints), robot_id=np.array(ROBOT_ID), fps=np.float32(sample_hz))
            np.save(Path(temp)/f"{case}_terrain.npy", np.empty((0, 1, 10), dtype=np.float32))
            paths.append(str(path))
        motion = MotionLib(paths, kin, dev, root=temp)
elif args.clip:
    from supermimic.acp.robot import MOTION_GLOB

    directory = Path(motion_dir)
    files = {str(file.relative_to(directory)).removesuffix("_motion.npz"): file
             for file in directory.glob(MOTION_GLOB)}
    if args.clip not in files:
        parser.error(f"Unknown clip {args.clip!r} in {directory}. Available clips: {', '.join(sorted(files))}")
    motion = MotionLib([str(files[args.clip])], kin, dev, root=str(directory))
else:
    motion = MotionLib.from_dir(motion_dir, kin, dev)
term_win = a_.get("term_window", 0.0) if args.term_window is None else args.term_window
delay_steps = DELAY_STEPS if a_.get("s17_actuators") else 0
ground_friction, box_friction = evaluation_surface_friction(
    a_, args.ground_friction, args.box_friction
)
friction_override = ground_friction is not None or box_friction is not None
ground_friction = 1.0 if ground_friction is None else ground_friction
box_friction = 1.0 if box_friction is None else box_friction
reference_task = ck.get("task_interface") == "reference_mimic_v1"
reference_cfg = ReferenceCfg(**ck["reference_cfg"]) if reference_task else None
local_reference_root = reference_cfg is not None and reference_cfg.root_reference == "local"
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev),
          RewardCfg(term_window=int(round(term_win / CONTROL_DT)),
                    w_knee_tracking=a_.get("knee_tracking_weight", 0.0),
                    w_self_collision=a_.get("self_collision_weight", 0.0),
                    local_reference_root=local_reference_root,
                    body_reference=reference_cfg is not None and reference_cfg.body_reference),
          independent_surface_friction=friction_override, delay_steps=delay_steps)
ctc = (ReferenceTask(reference_cfg, mdp) if reference_task
       else ContactTask(ContactCfg(**ck["ctc_cfg"]), mdp, motion_dir, dev))
Hh = ia["hist"]
networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
pi = (build_policy(mdp, ctc, Hh) if reference_task else Policy(OBS_DIM, ctc.cmd_dim, hist=Hh)).to(dev)
if args.ema and "pi_ema" not in ck:
    raise ValueError("Checkpoint has no EMA policy")
pi.load_state_dict(ck["pi_ema"] if args.ema else ck["pi"])
pi.eval()

if args.planner:
    if reference_task:
        raise ValueError("The reference Mimic baseline does not accept a contact planner")
    pk = load_checkpoint(args.planner, map_location=dev)
    pcfg = PlannerCfg(**pk["planner_cfg"])
    scan = PlannerScan(pcfg, dev)
    planner = ContactPlanner(OBS_DIM - REF_ORI_DIMS, pcfg).to(dev)
    planner.load_state_dict(pk["planner"])
    planner.eval()

C = len(motion.names)
env = TrackingSim(1 if args.viewer else C, motion, mdp, dev, eval_mode=True, reference_reset_action=reference_task,
                  ground_friction_range=(ground_friction,) * 2 if friction_override else None,
                  box_friction_range=(box_friction,) * 2 if friction_override else None,
                  primitive_soles=a_.get("primitive_soles", False))


@torch.no_grad()
def evaluate(viewer=None, clips=None, record_path=None) -> dict | None:
    if clips is None:
        clips = list(range(len(motion.names)))
    names = [motion.names[index] for index in clips]
    folders = sorted({name.split("/")[0] for name in names})
    folder_of = torch.tensor([folders.index(name.split("/")[0]) for name in names], device=dev)
    C = len(clips)
    ids = torch.arange(C, device=dev)
    env.done_eval.zero_()
    env.failed_eval.zero_()
    clip_start = motion.clip_start[clips]
    clip_end = motion.clip_end[clips]
    clip_len = motion.clip_len[clips]
    env.reset(ids, clip_start)
    gs = torch.full((C,), int(round(args.goal_s / CONTROL_DT)), device=dev)
    ret = torch.zeros(C, device=dev)
    overlap_peak = torch.zeros(C, device=dev)
    overlap_count = torch.zeros(C, dtype=torch.long, device=dev)
    tp, n_touch, n_label = (torch.zeros(NB, device=dev) for _ in range(3))
    prev_touch = torch.ones(C, NB, dtype=torch.bool, device=dev)
    tw_win = torch.zeros(PLACE_WINDOW, C, NB, 3, device=dev)
    tv_win = torch.zeros(PLACE_WINDOW, C, NB, dtype=torch.bool, device=dev)
    place = []  # (C, NB) per step: placement error at new contacts, NaN elsewhere
    hist = None
    traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": [], "tgt_w": [], "tgt_valid": []}
    while not bool(env.done_eval.all()):
        S = env.state()
        if viewer is not None and not viewer.show(S, env.f):
            return None
        o = env.observe()
        if NOISE_OBS:
            o = mdp.add_noise(o)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        if reference_task:
            tw = torch.zeros(C, NB, 3, device=dev)
            tv = torch.zeros(C, NB, device=dev, dtype=torch.bool)
        else:
            tw, tv = ctc.next_pos[env.f, env.v], ctc.next_valid[env.f, env.v]
        if args.planner:
            pw, pv = plan(planner, scan, pcfg, ctc, hist, S, env.f, env.v, gs, noise=NOISE_CMD)
            use = torch.rand(C, device=dev) < args.eta_obs
            tw, tv = torch.where(use[:, None, None], pw, tw), torch.where(use[:, None], pv, tv)
        if forecast is not None:
            window = tuple(array[env.f] for array in forecast)
            anchor = (motion.root_p[env.f], motion.root_R[env.f]) if local_reference_root else None
            cmd = ctc.command_from_reference(S, window, motion.boxes(env.f, env.v), anchor=anchor)
        elif reference_task:
            cmd = ctc.command(S, env.f, env.v, gs)
        else:
            cmd = ctc.command(S, env.f, env.v, gs, noise=NOISE_CMD, targets_w=tw, valid=tv)
        alive = ~env.done_eval.clone()
        tw_win, tv_win = torch.cat((tw_win[1:], tw[None])), torch.cat((tv_win[1:], tv[None]))
        if record_path:
            traj["root_p"].append(S[0].cpu()), traj["root_quat"].append(mat_to_quat(S[1]).cpu())
            traj["q"].append(S[4].cpu()), traj["f"].append(env.f.cpu()), traj["alive"].append(alive.cpu())
            traj["tgt_w"].append(tw.cpu()), traj["tgt_valid"].append(tv.cpu())
        a = pi(hist, cmd)
        action = a + .1 * torch.randn_like(a) if NOISE_ACT else a
        out = env.step(pi.project_action(action) if reference_task else action)
        ret += torch.where(alive, out["r"], 0.0)
        if "self_overlap" in out:
            overlap = torch.where(alive & out["valid"], out["self_overlap"], 0.0)
            overlap_peak = torch.maximum(overlap_peak, overlap)
            overlap_count += overlap > mdp.self_collision_penalty.tolerance
        f1 = out["f"] + 1
        S1 = env.state()
        boxes1 = motion.boxes(f1, out["v"])
        if not reference_task:
            touch = (ctc.body_distance(S1, boxes1) <= ctc.cfg.touch_dist) & alive.unsqueeze(-1)
            pe = placement_error(ctc.contact_point(S1, boxes1), tw_win, tv_win)
            place.append(torch.where(touch & ~prev_touch & pe.isfinite(), pe, torch.nan))
            prev_touch = touch
            label = ctc.labels[f1, out["v"]] & alive.unsqueeze(-1)
            tp += (touch & label).sum(0)
            n_touch += touch.sum(0)
            n_label += label.sum(0)
        env.f = torch.minimum(env.f, clip_end - 1)  # finished clips keep stepping; keep f+1 in range
        # Finished worlds no longer contribute to scores. Park their obstacles
        # before the next physics step: a fallen robot left on a heightfield can
        # otherwise exceed Warp's per-pair contact limit while other clips run.
        finished = (alive & env.done_eval).nonzero(as_tuple=False).flatten()
        env.park_finished_terrains(finished)
    if viewer is not None:
        failed = bool(env.failed_eval[0])
        if failed and not bool(out["valid"][0]):
            S1 = S
        follow_reference = bool(failed and args.upstream_roban)
        if not viewer.show(S1, f1, follow_reference=follow_reference):
            return None
        if follow_reference:
            print(f"{names[0]}: tracking terminated at {int(f1[0]-clip_start[0])*CONTROL_DT:.2f} s. "
                  "Continuing the reference to the end.", flush=True)
            remaining = torch.arange(int(f1[0])+1, int(clip_end[0])+1, device=dev)
            for frame in remaining.split(1):
                if not viewer.show(S1, frame, follow_reference=True):
                    return None
        print(f"{names[0]}: {'tracking terminated' if failed else 'tracking completed'}.", flush=True)
    if record_path:
        np.savez_compressed(record_path, robot_id=np.array(ROBOT_ID), joint_names=np.array(JOINT_NAMES), names=np.array(names), variant=env.v.cpu().numpy(),
                            failed=env.failed_eval.cpu().numpy(),
                            boxes=motion.clip_boxes[clips, env.v].cpu().numpy(),
                            **{k: torch.stack(v).numpy() for k, v in traj.items()}, ref_root_p=motion.root_p.cpu().numpy(),
                            ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                            clip_start=clip_start.cpu().numpy(), clip_len=clip_len.cpu().numpy())  # fmt: skip
    norm = ret / ((clip_len - 1).float() * MAX_STEP_REWARD)
    place = torch.stack(place).flatten(0, 1) if place else torch.empty(0, NB, device=dev)
    pl = [place[:, b][~place[:, b].isnan()] for b in range(NB)]
    rnd = lambda x: round(x.item(), 3)  # noqa: E731
    failed = env.failed_eval.float()
    return {
        "ckpt": args.ckpt,
        "policy_variant": "ema" if args.ema else "online",
        "s17_actuators": bool(delay_steps),
        "primitive_soles": env.primitive_soles,
        **({"self_overlap": [
            {"clip": name, "max_m": float(peak), "frames_over_1mm": int(count)}
            for name, peak, count in zip(
                names, overlap_peak.tolist(), overlap_count.tolist(), strict=True
            )
        ]} if mdp.self_collision_penalty is not None else {}),
        "surface_friction": {"ground": ground_friction, "box": box_friction},
        "interface": "reference_mimic" if reference_task else "planner" if args.planner else "reference",
        **({"forecast": str(args.forecast)} if forecast is not None else {}),
        "inactive_terrain_parked": True,
        "eta_obs": args.eta_obs if args.planner else 0.0,
        "term_window": term_win,
        "norm_return": round(norm.mean().item(), 4),
        "n_failed": int(failed.sum().item()),
        "failed": [names[i] for i in failed.nonzero().flatten().tolist()],
        "per_folder": {g: {"norm_return": round(norm[folder_of == i].mean().item(), 4), "fail_rate": round(failed[folder_of == i].mean().item(), 3)}
                       for i, g in enumerate(folders)},  # fmt: skip
        "contacts": {n: {"precision": round((tp[b] / n_touch[b]).item(), 3) if n_touch[b] > 0 else None,
                         "recall": round((tp[b] / n_label[b]).item(), 3) if n_label[b] > 0 else None,
                         "label_frames": int(n_label[b].item()), "touch_frames": int(n_touch[b].item()),
                         "placement_err": rnd(pl[b].mean()) if len(pl[b]) else None,
                         "placement_median": rnd(pl[b].median()) if len(pl[b]) else None,
                         "placement_within_0.1": rnd((pl[b] <= 0.1).float().mean()) if len(pl[b]) else None,
                         "onsets": len(pl[b])} for b, n in enumerate(CONTACT_BODIES)} if not reference_task else {},  # fmt: skip
    }


if args.viewer:
    from supermimic.acp.reference_viewer import ReferenceViewer

    variant = motion.first_variants(motion.clip_id[motion.clip_start])
    results = []
    with ReferenceViewer(motion, motion.boxes(motion.clip_start, variant)[0].cpu().numpy(),
                         primitive_soles=env.primitive_soles) as viewer:
        for index, name in enumerate(motion.names):
            seconds = (int(motion.clip_len[index])-1)*CONTROL_DT
            print(f"[{index+1}/{C}] Playing {name} for {seconds:.2f} s of reference.", flush=True)
            record = args.record
            if record and C > 1:
                path = Path(record)
                record = path.with_name(f"{path.stem}_{name}.npz")
            result = evaluate(viewer, [index], record_path=record)
            if result is None:
                raise SystemExit(0)
            results.append(result)
        print("Playback finished. Close the viewer to finish.", flush=True)
        viewer.wait_until_closed()
    res = results[0]
    if C > 1:
        res = {**res, "norm_return": round(sum(r["norm_return"] for r in results)/C, 4),
               "n_failed": sum(r["n_failed"] for r in results),
               "failed": [name for r in results for name in r["failed"]],
               "per_folder": {name: stats for r in results for name, stats in r["per_folder"].items()}}
        if "self_overlap" in res:
            res["self_overlap"] = [item for r in results for item in r["self_overlap"]]
else:
    res = evaluate(record_path=args.record)
print(json.dumps({k: v for k, v in res.items() if k != "failed"}, indent=1))
print(f"failed {res['n_failed']}/{C}: {res['failed'][:10]}{' ...' if res['n_failed'] > 10 else ''}")
if args.out or not args.viewer:
    tag = res["interface"] + ("" if args.eta_obs == 1 or not args.planner else f"_eta{args.eta_obs:g}") + (f"_tw{term_win:g}" if term_win else "")
    out = args.out or args.ckpt.replace(".pt", f"_eval_{tag}.json")
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)
