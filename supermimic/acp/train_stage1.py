"""Stage 1: the contact-target tracker trained from scratch with SuperMimic (Algorithm A.1) on the G1 box terrain.

The policy a = dec(enc(c), o_{t-9:t}) sees proprioception and the contact command c only (next-contact targets on
the terrain, the goal command and the paper's 9 x 7 local height scan; ``--blind``: no scan), while its reward and termination grade
it on the full reference and the critic sees the reference errors and the height scan. The world model sees the height
scan of every state. The reward is the contact paper's Stage 1 reward (:class:`supermimic.acp.task.PaperReward`), or
with ``--reward supermimic`` SuperMimic's motion-tracking reward plus the asymmetric contact-label term.
Contact training defaults to the backward curriculum. Reference BlinkMimic defaults to
clip-conditioned return, failure and EMA probe sampling. ``--start_from_beginning`` uses
the first frame of a uniformly sampled clip.
``--no-reset_noise`` disables physical initial-state perturbations; all other randomization is unchanged.
``--init`` fine-tunes another run (its networks, optimizers and curriculum; its learning-rate schedule continues).

Usage: python -m supermimic.acp.train_stage1 --log_dir logs/acp_stage1
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import time
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=60_000)
parser.add_argument("--buffer_steps", type=int, default=1000)
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--hist", type=int, default=10)
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batches", type=int, default=10)
parser.add_argument("--wm_batch", type=int, default=None,
                    help="WM replay windows per batch (RoBan Mimic: 1024)")
parser.add_argument("--pi_batch", type=int, default=None,
                    help="Actor starts per batch (RoBan Mimic: 512)")
parser.add_argument("--lr", type=float, default=None,
                    help="Base optimizer learning rate (RoBan Mimic: 4e-4)")
parser.add_argument("--wm_mirror", type=float, default=None,
                    help="Fraction of WM windows to mirror (RoBan Mimic: 0.5)")
parser.add_argument("--pi_lr_multiplier", type=float, default=None,
                    help="Initial actor LR relative to base (RoBan Mimic: 4)")
parser.add_argument("--lr_decay_return", type=float, default=0.75)
parser.add_argument("--lr_decay_by", type=int, default=1400)
parser.add_argument("--lr_decay_length", type=int, default=300)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--warmup_steps", type=int, default=64, help="sim steps before any update")
parser.add_argument("--wm_only_iters", type=int, default=500, help="world-model-only iterations at the start")
parser.add_argument("--lr_decay_iters", type=int, default=250_000)
parser.add_argument("--ckpt_every", type=int, default=2000)
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--motion_dir", type=str, default=None, help="default: data/php_parkour")
parser.add_argument("--contacts_dir", type=str, default=None, help="supermimic.acp.make_labels output (default: <motion_dir>_contacts)")
parser.add_argument("--target_noise", type=float, default=0.05)
parser.add_argument("--scan_noise", type=float, default=0.05, help="--no-blind: uniform noise on the command's height scan")
parser.add_argument("--target_point", type=str, default="terrain", choices=["link", "terrain"], help="contact target: the terrain point the body touches, or its link")
parser.add_argument("--target_min_steps", type=int, default=10, help="targets only from contacts of at least this many frames")
parser.add_argument("--blind", action=argparse.BooleanOptionalAction, default=False, help="command without the paper's 9 x 7 local height scan (targets and goal only)")
parser.add_argument("--reward", type=str, default="paper", choices=["paper", "supermimic"], help="the contact paper's Stage 1 reward, or SuperMimic's tracking reward + contact labels")
parser.add_argument("--w_label", type=float, default=1.0, help="--reward supermimic: weight of the contact-label term")
parser.add_argument("--backward_curriculum", action=argparse.BooleanOptionalAction,
                    default=None, help="explicitly select the legacy episode-start rule")
parser.add_argument("--start_from_beginning", action="store_true", help="always reset at the first frame of a uniformly sampled clip; overrides backward curriculum and start-bin sampling")
parser.add_argument("--reset_noise", action=argparse.BooleanOptionalAction, default=True,
                    help="perturb reference root pose/velocities and joint positions at reset; independent of observation/action noise and domain randomization")
parser.add_argument("--log_dir", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--init", type=str, default=None, help="fine-tune: start from this run's last.pt")
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
parser.add_argument("--learner", choices=("supermimic", "blink"), default="supermimic")
parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--wm_hidden", type=int, nargs="+", default=None)
parser.add_argument("--pi_ema", type=float, default=.995)
parser.add_argument("--task", choices=("contact", "reference"), default="contact")
parser.add_argument("--reference_offsets", type=int, nargs="+", default=[1, 5, 10, 20, 30, 50])
parser.add_argument("--reference_action_skip", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--reference_root", choices=("world", "local"), default="world",
                    help="Reference root commands with or without a global planar path")
parser.add_argument("--knee_tracking_weight", type=float, default=0.0,
                    help="Additional smooth joint-angle tracking cost for both RoBan knees")
parser.add_argument("--ground_friction_range", type=float, nargs=2, metavar=("LOW", "HIGH"))
parser.add_argument("--box_friction_range", type=float, nargs=2, metavar=("LOW", "HIGH"))
parser.add_argument("--s17_actuators", action="store_true",
                    help="Use the native S17 gains, torque-speed curves and target delay")
parser.add_argument("--self_collision_weight", type=float, default=0.0,
                    help="Enable RoBan self-contact and penalize mesh overlap in Mimic")
parser.add_argument("--primitive_soles", action="store_true",
                    help="Use two cylinders and a box per RoBan sole at the mesh height")
parser.add_argument("--blink_reset_sampling", action=argparse.BooleanOptionalAction,
                    default=None, help="Use Kuavo-style failure and EMA probe resets")
parser.add_argument("--failure_reset_share", type=float, default=0.30)
parser.add_argument("--probe_share", type=float, default=0.15)
parser.add_argument("--start_reset_fraction", type=float, default=0.30)
parser.add_argument("--probe_start", type=int, default=1000)
parser.add_argument("--probe_every", type=int, default=125)
parser.add_argument("--probe_envs", type=int, default=16)
parser.add_argument("--probe_preceding_bins", type=int, default=2)
args = parser.parse_args()
reference_task = args.task == "reference"
if reference_task and args.learner != "blink":
    parser.error("The reference baseline requires --learner blink")
if not math.isfinite(args.knee_tracking_weight) or args.knee_tracking_weight < 0:
    parser.error("knee_tracking_weight must be finite and nonnegative")
if args.knee_tracking_weight and not reference_task:
    parser.error("knee_tracking_weight requires the reference task")
if args.reference_root == "local" and not reference_task:
    parser.error("--reference_root local requires the reference task")
if args.reference_offsets != sorted(set(args.reference_offsets)) or min(args.reference_offsets) < 0:
    parser.error("Reference offsets must be sorted, unique and nonnegative")
if not 0 <= args.pi_ema < 1:
    parser.error("pi_ema must be in [0, 1)")
if args.wm_hidden is not None and (
    min(args.wm_hidden) < 1 or (args.learner == "blink" and len(args.wm_hidden) < 2)
):
    parser.error("World-model widths must be positive; Blink needs two hidden layers")
if args.blink_reset_sampling is None:
    args.blink_reset_sampling = (
        reference_task and args.backward_curriculum is None and not args.start_from_beginning
    )
if args.blink_reset_sampling and (
    not reference_task or args.backward_curriculum is True or args.start_from_beginning
):
    parser.error("Blink reset sampling requires the reference task and its own reset rule")
for fraction in (args.failure_reset_share, args.probe_share, args.start_reset_fraction):
    if not 0 <= fraction <= 1:
        parser.error("Reset mixture probabilities must be in [0, 1]")
if min(args.probe_every, args.probe_envs) < 1 or min(
    args.probe_start, args.probe_preceding_bins
) < 0:
    parser.error("Invalid EMA probe schedule or lookback")
if args.start_from_beginning or args.blink_reset_sampling:
    args.backward_curriculum = False
elif args.backward_curriculum is None:
    args.backward_curriculum = True

import torch
from supermimic.acp.checkpoint import load_checkpoint, save_checkpoint
import torch.nn as nn

import supermimic.acp.networks as networks
from supermimic.acp.buffer import ReplayBuffer
from supermimic.acp.blink_learning_rate import BlinkActorRate
from supermimic.acp.blink_probe import BlinkProbe, isolated_probe_rng
from supermimic.acp.blink_reset_sampling import BlinkResetCfg
from supermimic.acp.blink_wm_mirror import BlinkWindowMirror
from supermimic.acp.env import DECIMATION, TrackingSim, validate_surface_friction
from supermimic.acp.robot import MOTION_DIR, NUM_J, ROBOT_ID, default_joint_pos_list
from supermimic.acp.s17_actuation import DELAY_STEPS, append_targets
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.mdp import INV_DIM, MAX_STEP_REWARD, MDP, OBS_DIM, RewardCfg, normalize_xi, unflatten_state
from supermimic.acp.motion import CONTROL_DT, MotionLib
from supermimic.acp.networks import Critic, Policy, WorldModel
from supermimic.acp.wm_training import UpdateGuard, regression_loss, rollout
from supermimic.acp.sim_numerics import TransitionGuard
from supermimic.acp.task import ContactCfg, ContactTask, PaperReward, PaperRewardCfg
from supermimic.acp.blink_contact import BlinkContactWorldModel, blink_optimizer, update_ema
from supermimic.acp.reference_mimic import ReferenceCfg, ReferenceTask, build_policy

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(args.seed)
validate_surface_friction(args.ground_friction_range, args.box_friction_range)
independent_friction = args.ground_friction_range is not None
roban_mimic = reference_task and ROBOT_ID == "roban"
for name, kuavo, legacy in (
    ("wm_batch", 1024, 2048), ("pi_batch", 512, 4096), ("lr", 4e-4, 2e-4),
    ("wm_mirror", 0.5, 0.0), ("pi_lr_multiplier", 4.0, 1.0),
):
    if getattr(args, name) is None:
        setattr(args, name, kuavo if roban_mimic else legacy)
if not 0.0 <= args.wm_mirror <= 1.0 or (args.wm_mirror and not roban_mimic):
    parser.error("--wm_mirror requires a fraction in [0, 1] on RoBan Mimic")
if not (
    math.isfinite(args.lr) and args.lr > 0
    and math.isfinite(args.pi_lr_multiplier) and args.pi_lr_multiplier > 0
    and math.isfinite(args.lr_decay_return)
    and args.lr_decay_iters > 0 and args.lr_decay_length > 0 and args.lr_decay_by >= 0
):
    parser.error("Invalid learning-rate schedule")
if args.s17_actuators and (ROBOT_ID != "roban" or not reference_task):
    parser.error("--s17_actuators requires the RoBan BlinkMimic reference task")
if not math.isfinite(args.self_collision_weight) or args.self_collision_weight < 0:
    parser.error("--self_collision_weight must be finite and nonnegative")
if args.self_collision_weight and (ROBOT_ID != "roban" or not reference_task):
    parser.error("--self_collision_weight requires the RoBan BlinkMimic reference task")
if args.primitive_soles and (ROBOT_ID != "roban" or not reference_task):
    parser.error("--primitive_soles requires the RoBan BlinkMimic reference task")
delay_steps = DELAY_STEPS if args.s17_actuators else 0
networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
dev = "cuda:0"
os.makedirs(args.log_dir, exist_ok=bool(args.resume) if args.learner == "blink" else True)
json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)

motion_dir = args.motion_dir or MOTION_DIR
kin = RobotKinematics(dev)
motion = MotionLib.from_dir(motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev),
          RewardCfg(w_knee_tracking=args.knee_tracking_weight,
                    w_self_collision=args.self_collision_weight,
                    local_reference_root=args.reference_root == "local"),
          independent_surface_friction=independent_friction, delay_steps=delay_steps)
ctc_cfg = (ReferenceCfg(tuple(args.reference_offsets), args.reference_action_skip, args.reference_root) if reference_task
           else ContactCfg(contacts_dir=args.contacts_dir, target_noise=args.target_noise, scan_noise=args.scan_noise, w_label=args.w_label,
                           target_point=args.target_point, target_min_steps=args.target_min_steps, blind=args.blind))
ctc = ReferenceTask(ctc_cfg, mdp) if reference_task else ContactTask(ctc_cfg, mdp, motion_dir, dev)
reward_cfg = PaperRewardCfg() if args.reward == "paper" and not reference_task else None
paper_reward = PaperReward(reward_cfg, ctc) if reward_cfg is not None else None
MAX_R = (MAX_STEP_REWARD if reference_task else
         reward_cfg.max_step_reward if reward_cfg is not None else MAX_STEP_REWARD + args.w_label * CONTROL_DT)
Hh, HZ = args.hist, args.horizon

blink = args.learner == "blink"
hidden = args.wm_hidden or ([1024] * 3 if blink else [2048] * 3)
wm = (BlinkContactWorldModel if blink else WorldModel)(mdp.scan_dim, hidden, context_dim=mdp.context_dim).to(dev)
pi = (build_policy(mdp, ctc, Hh) if reference_task else Policy(OBS_DIM, ctc.cmd_dim, hist=Hh)).to(dev)
critic = Critic(INV_DIM + mdp.critic_extra + ctc.cmd_dim).to(dev)
opt_wm = blink_optimizer(wm, [wm.net], args.lr) if blink else torch.optim.Adam(wm.parameters(), lr=args.lr)
opt_pi = blink_optimizer(pi, [pi.enc, pi.dec], args.lr) if blink else torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
pi_ema = deepcopy(pi).requires_grad_(False).eval() if blink else None
actor_rate = BlinkActorRate(
    args.pi_lr_multiplier, args.lr_decay_return, args.lr_decay_by, args.lr_decay_length
) if reference_task else None
window_mirror = BlinkWindowMirror(dev, delay_steps) if args.wm_mirror else None
start_it, wall_offset, sim_steps, lr_it0 = 0, 0.0, 0, 0
wm_only_iters = 0 if args.init else args.wm_only_iters
if args.resume or args.init:
    ck = load_checkpoint(args.resume or args.init, map_location=dev)
    expected_interface = "reference_mimic_v1" if reference_task else "contact"
    if ck.get("task_interface", "contact") != expected_interface:
        raise ValueError("Cannot resume across reference and contact policy interfaces")
    if ck["args"].get("s17_actuators", False) != args.s17_actuators:
        raise ValueError("Checkpoint actuator profile differs from the requested run")
    if reference_task and ReferenceCfg(**ck["reference_cfg"]) != ctc_cfg:
        raise ValueError("Reference conditioning differs from checkpoint")
    for key in ("ground_friction_range", "box_friction_range"):
        if ck["args"].get(key) != getattr(args, key):
            raise ValueError(f"Checkpoint {key} differs from the requested run")
    if args.resume and ck["args"].get("knee_tracking_weight", 0.0) != args.knee_tracking_weight:
        raise ValueError("Checkpoint knee tracking weight differs from the requested run")
    if args.resume and ck["args"].get("self_collision_weight", 0.0) != args.self_collision_weight:
        raise ValueError("Checkpoint self-collision weight differs from the requested run")
    if args.resume and ck["args"].get("primitive_soles", False) != args.primitive_soles:
        raise ValueError("Checkpoint sole collision geometry differs from the requested run")
    if ck["args"].get("learner", "supermimic") != args.learner:
        raise ValueError("Checkpoint learner differs; Blink and SuperMimic are not interchangeable")
    wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck["pi"]), critic.load_state_dict(ck["critic"])
    opt_wm.load_state_dict(ck["opt_wm"]), opt_pi.load_state_dict(ck["opt_pi"]), opt_v.load_state_dict(ck["opt_v"])
    if pi_ema is not None:
        pi_ema.load_state_dict(ck["pi_ema"])
    if args.resume:
        start_it, wall_offset, sim_steps, lr_it0 = ck["iteration"], ck["wall_time"], ck["sim_steps"], ck.get("lr_it0", 0)
        if actor_rate is not None and "blink_actor_rate" in ck:
            actor_rate.load_state_dict(ck["blink_actor_rate"])
            print("Restored Blink actor learning-rate schedule", flush=True)
        elif actor_rate is not None and start_it >= args.lr_decay_by:
            actor_rate.decay_start = args.lr_decay_by
    else:  # the learning-rate schedule continues from the initial run's iteration
        lr_it0 = ck["iteration"] + ck.get("lr_it0", 0)


# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((NUM_J,), 3e-3), torch.full((NUM_J,), 5e-2))).to(dev)  # fmt: skip


def _wm_rollout(s, a, xi, f, v, mask, boxes):
    return rollout(
        wm, mdp, motion, s, a, xi, f, v, mask, delay_steps=delay_steps, boxes=boxes
    )


def _wm_err(S_hat, s1, mask):
    return regression_loss(wm, S_hat, s1, mask, ERR_FLOOR)


def _imag_step(S, o_hist, f, v, gs, xi, xi_n, boxes):
    a = pi(o_hist, ctc.command(S, f, v, gs, noise=True))
    S1 = wm.step(S, a, xi_n, mdp.scan(S, boxes))
    o1 = mdp.add_noise(mdp.obs(S1, a, xi))
    return S1, a, torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1), f + 1


def _reward_term(S, S_prev, f, v, a, a_prev, a_prev2, xi):
    if reference_task:
        r = mdp.reward(S, f, a, a_prev)
    elif paper_reward is not None:
        r = paper_reward(S, S_prev, f, v, a, a_prev, a_prev2, xi)
    else:
        r = mdp.reward(S, f, a, a_prev) + ctc.cfg.w_label * ctc.label_reward(S, f, v, motion.boxes(f, v)) * CONTROL_DT
    return r, mdp.soft_termination(S, f)


wm_rollout = torch.compile(_wm_rollout) if args.compile else _wm_rollout
wm_err = torch.compile(_wm_err) if args.compile else _wm_err
imag_step = torch.compile(_imag_step) if args.compile else _imag_step
# max_fusion_size 32: Triton 3.5.0 fails to compile the fully fused backward kernel of this reward
reward_term = torch.compile(_reward_term, options={"max_fusion_size": 32}) if args.compile else _reward_term
critic_input_c = torch.compile(ctc.critic_input) if args.compile else ctc.critic_input


def lr_at(it: int) -> float:
    it += lr_it0
    if it >= args.lr_decay_iters:
        return args.lr * 0.03
    return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


# ------------------------------------------------------------------------------------------------
wm_guard = UpdateGuard(args.log_dir)
sim_guard = TransitionGuard(args.log_dir)


def wm_update():
    w = buffer.sample_windows(args.wm_batch, args.wm_window)
    boxes = None
    if window_mirror is not None:
        w, boxes = window_mirror.windows(
            w, motion.boxes(w["f"], w["v"]), args.wm_mirror
        )
        w["boxes"] = boxes
    S_hat = wm_rollout(w["s"], w["a"], w["xi"], w["f"], w["v"], w["mask"], boxes)
    loss, e_last = wm_err(S_hat, w["s1"], w["mask"])
    wm_guard.step(loss, wm, opt_wm, w, it + 1)
    m = w["mask"][:, -1]
    return loss.detach() / args.wm_window, e_last.detach().sum() / m.sum().clamp(min=1)


def sample_starts(B: int):
    """Start points with H reference frames left in their clip, the action before their previous one (``a_prev2``)
    and the observations the policy saw before them."""
    starts, hists, n = [], [], 0
    while n < B:
        s, h = buffer.sample_starts(2 * B, Hh)
        ok = s["valid"] & (s["f"] + HZ <= motion.clip_end[motion.clip_id[s["f"]]])
        idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
        starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
        n += len(idx)
    cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
    s, h = cat(starts), cat(hists)
    s["a_prev2"] = h["a_prev"][:, -2]  # the history stays inside the episode: zeros right after a reset
    return s, h["o"]


def policy_update():
    s, o_hist = sample_starts(args.pi_batch)
    S = unflatten_state(s["s"])
    xi, f, v, a_prev, a_prev2, gs = s["xi"], s["f"], s["v"], s["a_prev"], s["a_prev2"], s["goal_steps"]
    xi_n = normalize_xi(xi, delay_steps)
    contexts = [xi_n]
    boxes = motion.boxes(f, v)  # the rollout stays inside its clip
    set_requires_grad(wm, False)
    set_requires_grad(critic, False)
    B = args.pi_batch
    S0, f0 = S, f
    Ss, Sps, As, Aps, Ap2s, F1s = [], [], [], [], [], []
    for t in range(HZ):
        Sps.append(S), Aps.append(a_prev), Ap2s.append(a_prev2)
        a_prev2 = a_prev
        S, a_prev, o_hist, f = imag_step(S, o_hist, f, v, gs, xi, xi_n, boxes)
        Ss.append(S), As.append(a_prev), F1s.append(f)
        if delay_steps:
            head = xi_n[..., :mdp.surface_context_dim]
            fifo = xi_n[..., mdp.surface_context_dim:].reshape(B, delay_steps, NUM_J)
            following = append_targets(fifo, a_prev, DECIMATION).flatten(1)
            xi_n = torch.cat((head, following), dim=-1)
        contexts.append(xi_n)
    flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
    S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
    S_prev = tuple(flat([St[i] for St in Sps]) for i in range(6))
    xi_all = xi.unsqueeze(0).expand(HZ, -1, -1).flatten(0, 1)
    v_all = v.repeat(HZ)
    r, d = reward_term(S_all, S_prev, flat(F1s), v_all, flat(As), flat(Aps), flat(Ap2s), xi_all)
    r, d = r.view(HZ, B), d.view(HZ, B)
    alive = torch.cumprod(1 - d, 0)  # c_{t+1}
    c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
    disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
    Jr = (disc * c_t * r).sum(0)
    c = alive[-1]
    r_sum = (c_t * r).detach().sum(0)
    rews, dones = r.detach(), d.detach()
    Jr = Jr + (args.gamma**HZ) * c * critic(ctc.critic_input(S, f, v, gs, xi_n))
    loss = -Jr.mean()
    opt_pi.zero_grad(set_to_none=True)
    loss.backward()
    gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    nn.utils.clip_grad_value_(pi.parameters(), 0.1)
    if torch.isfinite(gn):
        opt_pi.step()
        if pi_ema is not None:
            update_ema(pi_ema, pi, args.pi_ema)
    set_requires_grad(wm, True)
    set_requires_grad(critic, True)
    # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
    with torch.no_grad():
        S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
        f_c = torch.cat((f0.unsqueeze(0), torch.stack(F1s)), 0).flatten()
        v_c, gs_c = v.repeat(HZ + 1), gs.repeat(HZ + 1)
        xi_c = flat(contexts)
        X = critic_input_c(S_c, f_c, v_c, gs_c, xi_c).view(HZ + 1, B, -1)
    critic.norm.update(X)
    with torch.no_grad():
        V = critic(X)
        R = V[-1]
        targets = []
        for t in range(HZ - 1, -1, -1):
            R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
            targets.append(R)
        targets = torch.stack(targets[::-1])  # (H, B)
    Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
    n = Xf.shape[0]
    mb = n // args.critic_minibatches
    for _ in range(args.critic_epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(args.critic_minibatches):
            idx = perm[i * mb : (i + 1) * mb]
            v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
            opt_v.zero_grad(set_to_none=True)
            v_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt_v.step()
    imag_r = (r_sum / HZ / MAX_R).mean()
    return loss.detach(), v_loss.detach(), imag_r, gn.detach(), c.mean().detach()


# ------------------------------------------------------------------------------------------------
env_options = {
    "reference_reset_action": reference_task,
    "ground_friction_range": args.ground_friction_range,
    "box_friction_range": args.box_friction_range,
    "primitive_soles": args.primitive_soles,
}
reset_cfg = (
    BlinkResetCfg(args.failure_reset_share, args.probe_share, args.start_reset_fraction)
    if args.blink_reset_sampling else None
)
env = TrackingSim(
    args.num_envs, motion, mdp, dev, gamma=args.gamma,
    backward_curriculum=args.backward_curriculum,
    start_phase=(0.0, 0.0) if args.start_from_beginning else None,
    reset_noise=args.reset_noise, blink_reset=reset_cfg, **env_options,
)
sampler = env.blink_sampler
probe = None
if sampler is not None:
    if args.resume and "blink_reset" in ck:
        sampler.load_state_dict(ck["blink_reset"])
        env.reset(torch.arange(args.num_envs, device=dev))
        print("Restored Blink reset sampling state", flush=True)
    elif args.resume:
        print("Checkpoint predates Blink reset sampling; accumulating new statistics",
              flush=True)
    print(f"Blink reset sampling enabled: {reset_cfg}", flush=True)
if args.start_from_beginning:
    env.phase.zero_()  # checkpoint/logged start phases must reflect the enforced first-frame reset
    assert torch.equal(env.f, motion.clip_start[motion.clip_id[env.f]])
    print(f"Episode starts: first frame for all {len(motion.names)} clips; backward curriculum disabled", flush=True)
elif env.curriculum and (args.resume or args.init) and "curriculum_phase" in ck:
    env.phase.copy_(ck["curriculum_phase"])
    env.reset(torch.arange(args.num_envs, device=dev))
print(f"Reset initial-state noise: {env.reset_noise}; backward curriculum: {env.curriculum}",
      flush=True)
# clip groups for the curriculum log: the motion folders
GROUPS = sorted({n.split("/")[0] for n in motion.names})
clip_group = torch.tensor([GROUPS.index(n.split("/")[0]) for n in motion.names], device=dev)
buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev, extra={"xi": (mdp.context_dim, torch.float32)})
obs_hist = torch.zeros(args.num_envs, Hh, OBS_DIM, device=dev)
goal_steps = ctc.sample_goal_steps(args.num_envs)


@torch.no_grad()
def collect(n_steps: int, stats: dict):
    global obs_hist, sim_steps
    for _ in range(n_steps):
        o = env.observe()
        fresh = env.ep_step == 0
        obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
        cmd = ctc.command(env.state(), env.f, env.v, goal_steps, noise=True)
        pi.obs_norm.update(o)
        pi.goal_norm.update(cmd)
        a = pi(obs_hist, cmd)
        a = (pi.project_action(a + args.action_noise * torch.randn_like(a)) if reference_task
             else torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20))
        out = env.step(a)
        out["goal_steps"] = goal_steps.clone()
        sim_guard.observe(out, motion, sim_steps, globals().get("it", start_it - 1) + 1)
        buffer.add(out)
        S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
        good = out["valid"]
        wm_inputs = wm.inputs(
            S0, out["a"], normalize_xi(out["xi"], delay_steps),
            mdp.scan(S0, motion.boxes(out["f"], out["v"])),
        )
        wm.in_norm.update(wm_inputs[good])
        wm.out_norm.update(wm.targets(S0, S1)[good])
        if blink:
            wm.inc_norm.update(WorldModel.targets(S0, S1)[good])
        sim_steps += args.num_envs
        # contact labels in the simulator: touch per body vs the reference label
        if not reference_task:
            f1 = out["f"] + 1
            touch = (ctc.body_distance(S1, motion.boxes(f1, out["v"])) <= ctc.cfg.touch_dist) & good[:, None]
            label = ctc.labels[f1, out["v"]] & good[:, None]
            stats["tp"].append((touch & label).sum()), stats["n_touch"].append(touch.sum()), stats["n_label"].append(label.sum())
        stats["r"].append(out["r"].mean() / MAX_STEP_REWARD)
        if args.self_collision_weight:
            depth = torch.where(good, out["self_overlap"], 0.0)
            overlapping = depth > mdp.self_collision_penalty.tolerance
            stats["self_overlap"].append(overlapping.sum())
            stats["self_overlap_max"].append(depth.max())
            stats["self_overlap_count"].append(good.sum())
        if "ep_len" in out:
            stats["len"].append(out["ep_len"].sum()), stats["n_ep"].append(out["ep_len"].new_tensor(len(out["ep_len"])))
            stats["ret"].append(out["ep_ret_norm"].sum()), stats["term"].append(out["term"][out["done"]].float().sum())
        done = out["done"]
        goal_steps[done] = ctc.sample_goal_steps(int(done.sum()))


stats_keys = ("r", "len", "ret", "n_ep", "term", "tp", "n_touch", "n_label")
if args.self_collision_weight:
    stats_keys += ("self_overlap", "self_overlap_max", "self_overlap_count")
stats = {k: [] for k in stats_keys}
collect(args.warmup_steps, stats)  # the replay buffer is not checkpointed, so a resumed run refills it too
if (actor_rate is not None and stats["r"]
        and not (args.resume and "blink_actor_rate" in ck)):
    actor_rate.training_return = float(torch.stack(stats["r"]).mean())
csv_header = (
    "iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,"
    "imag_survival,sim_rew,ep_len,ep_ret_norm,term_frac,touch_precision,touch_recall,"
    "it_per_s,lr,phase_zero_frac"
    + "".join(f",phase_{g}" for g in GROUPS)
    + (",self_overlap_frac,self_overlap_max_m" if args.self_collision_weight else "")
    + (",reset_failure_weight,reset_probe_weight" if sampler is not None else "")
    + (",pi_lr" if actor_rate is not None else "")
)
csv_path = os.path.join(args.log_dir, "train_log.csv")
if os.path.isfile(csv_path):
    with open(csv_path) as existing:
        header = existing.readline().strip()
    if header and header != csv_header:
        suffix = hashlib.sha256(csv_header.encode()).hexdigest()[:8]
        csv_path = os.path.join(args.log_dir, f"train_log_{suffix}.csv")
new_log = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
log_f = open(csv_path, "a")
if new_log:
    log_f.write(csv_header + "\n")
t_start = time.time() - wall_offset
t_log = time.time()
step_acc = 0.0
agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": []}
for it in range(start_it, args.iterations):
    lr = lr_at(it)
    pi_lr = lr if actor_rate is None else lr * actor_rate.factor(it)
    for o, rate in ((opt_wm, lr), (opt_pi, pi_lr), (opt_v, lr)):
        for gparam in o.param_groups:
            gparam["lr"] = rate
    step_acc += 1.0 / args.utd
    n = int(step_acc)
    step_acc -= n
    first_reward = len(stats["r"])
    collect(n, stats)
    if actor_rate is not None and n:
        actor_rate.training_return = float(torch.stack(stats["r"][first_reward:]).mean())
    for _ in range(args.wm_batches):
        l, e8 = wm_update()
    agg["wm"].append(l), agg["wm8"].append(e8)
    if it >= wm_only_iters:
        for k, v in zip(("pi", "v", "ir", "gn", "surv"), policy_update()):
            agg[k].append(v)
    if (
        sampler is not None and args.probe_share > 0 and it >= wm_only_iters
        and it + 1 >= args.probe_start and (it + 1) % args.probe_every == 0
    ):
        with isolated_probe_rng(dev):
            if probe is None:
                probe_mdp = MDP(
                    kin, motion, mdp.default_q, mdp.cfg,
                    independent_surface_friction=independent_friction,
                    delay_steps=delay_steps,
                )
                probe = BlinkProbe(args.probe_envs, probe_mdp, ctc_cfg, env_options)
            failures = probe.run(pi_ema)
        sampler.set_focus(failures, args.probe_preceding_bins)
        print(f"Blink EMA probe at {it + 1}: {len(failures)} failures", flush=True)
    if (it + 1) % args.log_every == 0:
        mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
        total = lambda k: float(sum(float(v) for v in stats[k]))  # noqa: E731
        ratio = lambda a, b: total(a) / max(total(b), 1)  # noqa: E731
        ips = args.log_every / (time.time() - t_log)
        t_log = time.time()
        wall_h = (time.time() - t_start) / 3600
        row = [it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
               mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), ratio("len", "n_ep"), ratio("ret", "n_ep"),
               ratio("term", "n_ep"), ratio("tp", "n_touch"), ratio("tp", "n_label"), ips, lr]  # fmt: skip
        phases, groups = env.phase, clip_group
        if sampler is not None:
            clips = motion.clip_id[env.f]
            phases = (env.f - env.ep_step - motion.clip_start[clips]) / (
                motion.clip_len[clips] - 1
            )
            groups = clip_group[clips]
        phase_g = [float(phases[groups == i].mean()) for i in range(len(GROUPS))]
        row += [float((phases == 0).float().mean())] + phase_g
        if args.self_collision_weight:
            row += [
                ratio("self_overlap", "self_overlap_count"),
                max((float(value) for value in stats["self_overlap_max"]), default=0.0),
            ]
        if sampler is not None:
            row += [float(sampler.failures.sum()), float(sampler.focus.sum())]
        if actor_rate is not None:
            row.append(pi_lr)
        log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
        log_f.flush()
        print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} "
              f"surv {row[9]:.3f} | sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {row[13]:.2f} "
              f"touch P/R {row[14]:.3f}/{row[15]:.3f} | phase0 {row[18]:.2f}"
              + "".join(f" {g.removeprefix('motions-')} {p:.2f}" for g, p in zip(GROUPS, phase_g)) + f" | {ips:.2f} it/s", flush=True)  # fmt: skip
        stats = {k: [] for k in stats_keys}
        agg = {k: [] for k in agg}
    if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations:
        ck = {"wm": wm.state_dict(), "pi": pi.state_dict(), "critic": critic.state_dict(), "opt_wm": opt_wm.state_dict(),
              "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": it + 1,
              "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
              "ctc_cfg": dataclasses.asdict(ctc_cfg), "curriculum_phase": env.phase.clone(), "lr_it0": lr_it0,
              "reward_cfg": dataclasses.asdict(reward_cfg) if reward_cfg is not None else None,
              "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        if pi_ema is not None:
            ck["pi_ema"] = pi_ema.state_dict()
            ck["training_engine"] = "blink_reference_mimic_v1" if reference_task else "blink_whole_body_contact_v1"
        ck["task_interface"] = "reference_mimic_v1" if reference_task else "contact"
        if reference_task:
            ck["reference_cfg"] = dataclasses.asdict(ctc_cfg)
            ck["ctc_cfg"] = None
        if sampler is not None:
            ck["blink_reset"] = sampler.state_dict()
        if actor_rate is not None:
            ck["blink_actor_rate"] = actor_rate.state_dict()
        last = os.path.join(args.log_dir, "last.pt")
        save_checkpoint(ck, last + ".tmp")
        os.replace(last + ".tmp", last)
        small = {k: ck[k] for k in ("pi", "wm", "iteration", "wall_time", "sim_steps", "args", "ctc_cfg", "saved_at")}
        if pi_ema is not None:
            small["pi_ema"] = pi_ema.state_dict()
        small["task_interface"] = ck["task_interface"]
        if reference_task:
            small["reference_cfg"] = ck["reference_cfg"]
        path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
        save_checkpoint(small, path + ".tmp")
        os.replace(path + ".tmp", path)
