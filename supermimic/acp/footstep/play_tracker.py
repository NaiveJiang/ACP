"""Visual play of a two-foot oracle tracker, with the same mjlab physics as training."""
from __future__ import annotations

import argparse
from collections import deque
import copy
import json
import math
import os
from pathlib import Path
import threading
import time


def resolve_checkpoint(value):
    if value in ("latest", "latest-contact"):
        pattern = "contact_tracker_train*/last.pt" if value == "latest-contact" else "footstep_tracker_train*/last.pt"
        files = list(Path("logs").glob(pattern))
        if not files:
            raise FileNotFoundError("No trained tracker checkpoint; pass --checkpoint PATH")
        return max(files, key=lambda p: p.stat().st_mtime).resolve()
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path/"last.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def select_intervals(checkpoint, split, category):
    intervals = checkpoint["split"][split]
    if category:
        intervals = [c for c in intervals if c["category"] == category]
    if not intervals:
        raise ValueError("No matching reference clips")
    return intervals


def main(default_checkpoint="latest"):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=default_checkpoint, help="Checkpoint file, run directory, or latest")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--category", default="", help="Exact category, e.g. forward, backward, left_ccw")
    p.add_argument('--velocity',nargs=3,type=float,metavar=('VX','VY','WZ'),help='Joint policy constant velocity command; default follows recorded command schedule')
    p.add_argument("--heading-feedback", action="store_true", help="Correct heading drift with --planner-model")
    p.add_argument("--velocity-switch", nargs=4, type=float, action="append", default=[],
                   metavar=("SECONDS","VX","VY","WZ"), help="Scheduled speed change for --planner-model")
    p.add_argument("--gamepad", action="store_true",
                   help="Drive --planner-model with an SDL gamepad; center requests a stop")
    p.add_argument("--gamepad-library", help="Path to libSDL3 for --gamepad")
    plan_mode = p.add_mutually_exclusive_group()
    plan_mode.add_argument("--recorded-timing", action="store_true",
                           help="Use recorded support timing with learned positions and headings")
    plan_mode.add_argument("--recorded-plan", action="store_true",
                   help="Compare with a speed-matched recorded contact sequence from the training split")
    phase_x = p.add_mutually_exclusive_group()
    phase_x.add_argument("--phase-x-model", help="Override recorded x with the phase-conditioned model")
    phase_x.add_argument("--phase-x-samples", help="Collect supervised x samples in a finite recorded-plan run")
    p.add_argument("--clock-model", help="MLP timing checkpoint for --auto-clock")
    p.add_argument("--auto-clock", action="store_true", help="Generate support timing and phase from calibrated forward gait periods")
    p.add_argument("--recorded-split", choices=("train", "val"), default="train",
                   help="Split supplying the recorded contact sequence")
    phase_x.add_argument("--planner-model", help="Packaged clock and phased foot pose planner")
    phase_x.add_argument("--phase-pose-model", help="Phase-conditioned XYZ and heading checkpoint")
    phase_x.add_argument("--phase-pose-samples", help="Collect supervised support-foot poses")
    p.add_argument("--phase-pose-components", choices=("recorded","x","y","z","xy","xyz","yaw","all"), default="all",
                   help="Which pose components to predict, using recorded targets for the others")
    p.add_argument("--recorded-case", help="Choose a specific recorded contact sequence")
    p.add_argument("--clip-index", type=int, default=0, help="Index after category filtering")
    p.add_argument("--list-clips", action="store_true", help="List matching reference clips without starting simulation")
    p.add_argument("--query-dt", type=float, default=.3, help="Query time for legacy tracker checkpoints")
    p.add_argument("--phase", type=float, default=0., help="Starting fraction of the executable reference block")
    p.add_argument("--seconds", type=float, default=60., help="Total simulated playback seconds; 0 runs until closed")
    p.add_argument("--speed", type=float, default=1., help="Playback speed relative to wall time")
    p.add_argument("--contact-trace", type=Path, help="Save targets and measured feet during finite joint playback")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--pace", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True, help="Reset on fall/end and replay selected clip")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    if args.gamepad and (not args.planner_model or args.velocity is not None or args.velocity_switch):
        p.error("--gamepad requires --planner-model and replaces --velocity and --velocity-switch")
    if args.gamepad_library and not args.gamepad:
        p.error("--gamepad-library requires --gamepad")
    if (not math.isfinite(args.phase) or not 0 <= args.phase < 1 or args.clip_index < 0
            or not math.isfinite(args.seconds) or args.seconds < 0
            or not math.isfinite(args.speed) or args.speed <= 0):
        p.error("Need phase in [0,1), nonnegative clip/seconds, positive finite speed")
    if os.environ.get("ACP_ROBOT", "roban") != "roban":
        p.error("RoBan task only")
    os.environ["ACP_ROBOT"] = "roban"
    os.environ["MUJOCO_GL"] = "egl" if args.headless else "glfw"
    import torch
    checkpoint = resolve_checkpoint(args.checkpoint)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    from .backend import BODY_GOAL_FORMAT, LEG_COLLISION_FORMAT, NEXT_CONTACT_FORMAT, VELOCITY_FORMAT
    supported = (VELOCITY_FORMAT, "two_foot_contact_only_tracker_v2", "two_foot_stable_contact_tracker_v3",
                 "two_foot_upper_supervised_tracker_v4", "two_foot_gait_style_tracker_v5", LEG_COLLISION_FORMAT,
                 "two_foot_joint_planner_controller_v7", "two_foot_consistent_joint_controller_v8",
                 "two_foot_fixed_controller_planner_v9", BODY_GOAL_FORMAT, NEXT_CONTACT_FORMAT)
    if ck.get("format") not in supported:
        raise ValueError("Need a two-foot tracker checkpoint, not a planner or legacy ACP checkpoint")
    if default_checkpoint == "latest-contact" and ck["format"] not in supported[1:]:
        raise ValueError("This entrypoint expects a contact-conditioned controller")
    intervals = select_intervals(ck, args.split, args.category)
    if args.list_clips:
        for i, c in enumerate(intervals):
            print(f"{i:4d}  {c['category']:16s} {Path(c['source']).parent.name}/{c['case']} [{c['start']}:{c['end']}]")
        return
    if args.clip_index >= len(intervals):
        p.error(f"clip-index outside 0..{len(intervals)-1}; use --list-clips")
    config = ck["config"]
    next_contact = ck["format"] == NEXT_CONTACT_FORMAT
    matches = [0] if next_contact else [i for i, dt in enumerate(config["horizons"]) if abs(dt-args.query_dt) < 1e-6]
    if not matches:
        p.error("--query-dt must be one of "+str(config["horizons"]))
    if not args.headless and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        p.error("No desktop display; run in a desktop terminal or use --headless")
    import mujoco
    import numpy as np
    from .reference import FootstepMotion
    from .backend import backend_for_format
    from .env import FootstepTrackingSim
    from ..kinematics import RobotKinematics
    from ..robot import default_joint_pos_list, metadata
    from ..motion import CONTROL_DT
    from ...rotations import quat_to_mat, yaw_mat
    backend = backend_for_format(ck["format"])
    joint=getattr(backend,'joint',False)
    external_plan = ck["format"] == LEG_COLLISION_FORMAT and bool(
        args.planner_model or args.recorded_plan or args.auto_clock)
    if next_contact and (args.auto_clock or args.heading_feedback or args.gamepad or args.recorded_timing
                         or args.phase_x_model or args.phase_pose_model or args.phase_x_samples
                         or args.phase_pose_samples or args.clock_model):
        p.error("This checkpoint uses next-contact targets. Use --velocity or --recorded-plan.")
    plan_driven = joint or external_plan or next_contact
    if args.velocity is not None and (not plan_driven or not all(math.isfinite(x) for x in args.velocity)):
        p.error('--velocity requires a planner or recorded-plan mode and three finite values')
    FootstepTask, FootstepMDP, TrackerPolicy = backend.Task, backend.MDP, backend.Policy
    OBS_DIM, ContactRewardCfg = backend.obs_dim, backend.RewardCfg
    if ck["robot"]["fingerprint"] != metadata()["fingerprint"]:
        raise ValueError("Checkpoint robot/control definition differs")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    kin = RobotKinematics(args.device)
    motion = FootstepMotion([intervals[args.clip_index]], kin, args.device, config["horizons"],
                            reserve_steps=getattr(backend,"reserve_steps",0))
    task = FootstepTask(motion, kin, ContactRewardCfg(**ck["reward"]))
    mdp = FootstepMDP(kin, motion, torch.tensor(default_joint_pos_list(), device=args.device), task)
    velocity_override = torch.tensor(args.velocity or [0.,0.,0.],device=args.device)
    if joint and args.velocity is not None:
        mdp.velocity_override = velocity_override
    ground = getattr(task,"ground",None)
    if external_plan or next_contact:
        from .consistent_joint import FootGroundGeometry
        ground = FootGroundGeometry(kin)
    policy = TrackerPolicy(config["hist"], config["policy_hidden"],**(dict(planner_config=config['planner_config']) if joint else {})).to(args.device).eval()
    policy.load_state_dict(ck["pi"])
    env = getattr(backend,'Env',FootstepTrackingSim)(1, motion, mdp, args.device, eval_mode=True, reset_noise=False)
    query = torch.tensor([matches[0]], device=args.device)
    frame0 = motion.clip_start[0]+int(args.phase*(int(motion.clip_len[0])-1))
    live_reference_limit = motion.clip_end[0]-2
    switches = []
    if args.heading_feedback and not args.planner_model:
        p.error("--heading-feedback requires --planner-model")
    if args.velocity_switch and ((not args.planner_model and not next_contact) or args.velocity is None):
        p.error("--velocity-switch requires --planner-model and --velocity")
    for time_s,*velocity in args.velocity_switch:
        if not all(math.isfinite(v) for v in (time_s,*velocity)) or time_s <= 0:
            p.error("Velocity switches require finite commands and positive times")
        frame = round(time_s/CONTROL_DT)
        if frame < 1 or switches and frame <= switches[-1][0]:
            p.error("Provide velocity switches in strictly increasing control steps")
        switches.append((frame,velocity))
    integrated_planner = event_model = stick = gamepad = gamepad_gait = None
    next_planner = None
    if next_contact:
        from .next_contact import FORMAT as next_format, NextContactPlanner, world_plan as next_world_plan
        bundle = torch.load(args.planner_model,map_location="cpu",weights_only=True) if args.planner_model else ck
        if args.recorded_plan and (args.planner_model or args.velocity is not None):
            p.error("Recorded next-contact playback follows the selected clip's commands")
        if not args.recorded_plan and (args.planner_model or args.velocity is not None or "event_planner" in ck):
            saved = bundle.get("event_planner", {})
            if (bundle.get("robot",{}).get("fingerprint") != ck["robot"]["fingerprint"]
                    or saved.get("format") != next_format or saved.get("hist") != config["hist"]):
                p.error("Provide a compatible next-contact Stage 2 checkpoint for learned planning")
            next_planner = NextContactPlanner(saved["hist"]).to(args.device).eval()
            next_planner.load_state_dict(saved["weights"])
    elif args.planner_model:
        if (not plan_driven or (args.velocity is None and not args.gamepad) or args.auto_clock
                or args.clock_model or args.recorded_plan or args.recorded_timing):
            p.error("--planner-model requires a v6 or joint tracker and --velocity or --gamepad")
        from .omni_planner import OmniPlanner
        bundle = torch.load(args.planner_model,map_location="cpu",weights_only=True)
        fingerprint = (bundle.get("robot",{}).get("fingerprint") if "event_planner" in bundle
                       else bundle.get("robot_fingerprint"))
        if fingerprint != ck["robot"]["fingerprint"]:
            p.error("Packaged planner robot differs from the tracker")
        if "event_planner" in bundle:
            from .event_planner import EventPlanner, FORMAT
            saved = bundle["event_planner"]
            if (saved["format"]!=FORMAT or saved["hist"]!=config["hist"] or not external_plan
                    or args.gamepad or args.heading_feedback or switches
                    or not any(abs(args.query_dt-h)<1e-6 for h in (.3,.5))):
                p.error("Event playback requires a matching v6 checkpoint, fixed --velocity and a 0.3/0.5 s query")
            event_model = EventPlanner(saved["hist"]).to(args.device).eval()
            event_model.load_state_dict(saved["weights"])
        else:
            integrated_planner = OmniPlanner(bundle,args.device,heading_feedback=args.heading_feedback)
        if args.gamepad:
            from .gamepad import Gamepad, GamepadGait, StickCommand
            stick = StickCommand(bundle["clock"]["commands"])
            if not stick.ranges:
                p.error("No single-axis gamepad commands in the clock's measured command range")
            if ground is None:
                p.error("--gamepad requires a tracker with ground geometry")
            gamepad_gait = GamepadGait(integrated_planner, ground, stick)
        if switches:
            from .mlp_omni_clock import load_clock
            for _,velocity in switches:
                try:
                    integrated_planner.validate_direction_change(args.velocity,velocity)
                    load_clock(bundle["clock"],velocity)
                except ValueError as error:
                    p.error(str(error))
    recorded = None
    sample_path = args.phase_x_samples or args.phase_pose_samples
    learned_pose = args.phase_pose_model
    experiment = args.phase_x_model or learned_pose or sample_path
    fully_learned = bool(learned_pose and args.auto_clock and args.phase_pose_components == "all")
    if experiment and not (args.recorded_plan or fully_learned):
        p.error("Phase experiments require --recorded-plan or full pose prediction with --auto-clock")
    if sample_path and (args.seconds <= 0 or args.split != "train" or args.recorded_split != "train" or (args.auto_clock and args.phase_x_samples)):
        p.error("Collect samples on a finite recorded-plan rollout in the training split")
    if args.recorded_timing and not hasattr(policy, "planning_prediction"):
        p.error("--recorded-timing requires a planner that exposes undecoded positions and headings")
    if (args.recorded_plan or args.recorded_timing) and not next_contact:
        if not plan_driven or args.velocity is None:
            p.error("Recorded contact modes require a v6 or joint checkpoint and --velocity")
        minimum_frames = 100 + int(motion.horizon_steps.max())
        for part in ck["split"][args.recorded_split]:
            if args.recorded_case and part["case"] != args.recorded_case:
                continue
            velocity = np.load(Path(part["cache"]) / "velocity.npy", mmap_mode="r")[part["start"]:part["end"]]
            match = np.max(np.abs(velocity - args.velocity), axis=1) < 1e-5
            edges = np.flatnonzero(np.diff(np.r_[False, match, False]))
            runs = [(a, b) for a, b in edges.reshape(-1, 2) if b-a >= minimum_frames]
            if runs:
                start, end = runs[0]
                clip = dict(part, start=int(part["start"] + start), end=int(part["start"] + end))
                recorded = FootstepMotion([clip], kin, args.device, config["horizons"])
                print(f"Recorded contact sequence: {part['case']} [{clip['start']}:{clip['end']}]")
                break
        if recorded is None:
            p.error("No sufficiently long recorded training sequence matches this velocity")
    if experiment:
        from .phase_x import cycle_features, features, network
        phases = cycle_features(recorded.contact) if recorded is not None else None
        samples = []
        if args.phase_x_model:
            model_data = torch.load(args.phase_x_model, map_location=args.device, weights_only=True)
            x_model = network(len(model_data["mean"])).to(args.device).eval()
            x_model.load_state_dict(model_data["weights"])
        if learned_pose:
            from .phase_pose import FORMAT, network as pose_network, predict as predict_pose
            pose_ck = torch.load(learned_pose, map_location=args.device, weights_only=True)
            if pose_ck.get("format") != FORMAT:
                p.error("Expected a phase-pose checkpoint")
            pose_model = pose_network(len(pose_ck["mean"])).to(args.device).eval()
            pose_model.load_state_dict(pose_ck["weights"])
    gru_clock = None
    if args.clock_model and not args.auto_clock:
        p.error("--clock-model requires --auto-clock")
    if args.auto_clock:
        clock_ck = (torch.load(args.clock_model, map_location="cpu", weights_only=True)
                    if args.clock_model else {})
        omni = clock_ck.get("format") == "omni_mlp_clock_v1"
        if not plan_driven or not (args.phase_x_model or learned_pose or args.phase_pose_samples) or not args.velocity:
            p.error("--auto-clock requires a phase model and an explicit velocity")
        if omni and not (learned_pose or args.phase_pose_samples):
            p.error("Omnidirectional clock requires --phase-pose-model")
        if not omni and (args.velocity[0] <= 0 or any(args.velocity[1:])):
            p.error("Forward clock requires positive vx and zero vy and vyaw")
        from .gait_clock import calibrate, ForwardClock
        if args.clock_model:
            if omni:
                from .mlp_omni_clock import load_clock
                clock = load_clock(clock_ck, args.velocity)
            elif clock_ck.get("format") == "forward_gru_duration_clock_v1":
                from .gru_duration_clock import DurationClock
                gru_clock = DurationClock(clock_ck, args.device, args.velocity[0], args.query_dt)
                print("Standalone GRU duration clock")
            elif clock_ck.get("format") == "forward_gru_rate_clock_v2":
                from .gru_rate_clock import RateClock
                gru_clock = RateClock(clock_ck, args.device, args.velocity[0])
                print("MLP clock with bounded GRU cadence correction")
            elif clock_ck.get("format") == "forward_gru_clock_v1":
                from .gru_clock import OnlineClock
                gru_clock = OnlineClock(clock_ck, args.device, args.query_dt)
                print("GRU clock using causal velocity and proprioceptive history")
            else:
                from .mlp_clock import load_clock
                clock = load_clock(args.clock_model, args.velocity[0])
        else:
            clock = ForwardClock(calibrate(ck["split"]["train"]), args.velocity[0])
        if gru_clock is None:
            print(f"Automatic clock: period={clock.period:.3f}s startup={clock.delay:.3f}s")
        if recorded is not None:
            template_phase = torch.atan2(phases[:, 0], phases[:, 1])
            template_state = recorded.contact[:, 0].long() + 2*recorded.contact[:, 1].long()
            if not {1, 2, 3}.issubset(template_state.unique().tolist()):
                p.error("Geometry template must contain both single supports and double support")
            template_frames = torch.arange(len(phases), device=args.device)

    recorded_step = 0
    recorded_origin = alignment = initial_local = initial_heading = None
    history = None
    planned_world=planned_yaw=planned_contact=None
    events = deque()
    paused, follow, markers = False, True, True
    viewer, render_model, render_data = None, None, None
    viewer_threads = []
    steps = falls = invalid = repeats = 0
    velocity_sum = torch.zeros(3, device=args.device)
    support_switches = torch.zeros((), device=args.device, dtype=torch.long)
    foot_gaps = []
    measure_gaps = args.headless and args.seconds > 0 and ground is not None
    limit = int(math.ceil(args.seconds/CONTROL_DT)) if args.seconds else None
    pace = not args.headless if args.pace is None else args.pace

    if args.contact_trace and (not plan_driven or args.seconds <= 0 or ground is None or args.contact_trace.exists()):
        p.error("--contact-trace requires finite plan-driven playback and a new output file")
    contact_trace = []
    episode_step = 0
    switch_index = 0
    live_velocity = args.velocity
    segment_velocity = []
    segment_frames = []
    support_trace = []
    cadence_trace = []

    def reset():
        nonlocal episode_step,switch_index,live_velocity
        episode_step,switch_index,live_velocity = 0,0,args.velocity
        if switches:
            velocity_override.copy_(velocity_override.new_tensor(args.velocity))
        nonlocal history,planned_world,planned_yaw,planned_contact
        env.reset(torch.tensor([0], device=args.device), frames=frame0.reshape(1))
        env.refresh()
        if gamepad_gait is not None:
            gamepad_gait.reset(env.state())
        nonlocal recorded_step, recorded_origin, alignment, initial_local, initial_heading
        recorded_step = 0
        if gru_clock is not None:
            gru_clock.reset()
        if args.recorded_plan and recorded is not None:
            recorded_origin = env.state()[0].clone()
            alignment = yaw_mat(env.state()[1]) @ yaw_mat(recorded.root_R[:1]).transpose(-1, -2)
        if args.auto_clock or (integrated_planner is not None and gamepad_gait is None):
            root = env.state()
            if ground is None:
                feet, rotation, _ = task.foot_geometry(root)
            else:
                feet, rotation, _, _ = ground.foot_state(root)
            basis = yaw_mat(root[1]).transpose(-1,-2)[:, None]
            initial_local = (basis @ (feet-root[0][:,None])[...,None]).squeeze(-1)
            local_rot = basis @ rotation
            angle = torch.atan2(local_rot[...,1,0],local_rot[...,0,0])
            initial_heading = torch.stack((angle.sin(),angle.cos()),-1)
            if integrated_planner is not None:
                body_yaw = float(torch.atan2(root[1][0,1,0],root[1][0,0,0])) if args.heading_feedback else None
                integrated_planner.reset(initial_local,initial_heading,body_yaw=body_yaw)
        env.done_eval.zero_(); env.failed_eval.zero_()
        history = None
        planned_world=planned_yaw=planned_contact=None

    def render():
        # Render actual mjlab state; never replay reference qpos into the displayed robot.
        render_data.qpos[:] = env.sim.data.qpos[0].detach().cpu().numpy()
        render_data.qvel[:] = env.sim.data.qvel[0].detach().cpu().numpy()
        render_data.time = steps*CONTROL_DT
        mujoco.mj_forward(render_model, render_data)
        with viewer.lock():
            if follow:
                viewer.cam.lookat[:] = env.state()[0][0].detach().cpu().numpy()+np.array([0., 0., -.1])
            viewer.user_scn.ngeom = 0
            if markers:
                if not plan_driven:
                    target = int(env.f[0]+motion.horizon_steps[query[0]])
                for foot, color in ((0, (1., .2, .2, 1.)), (1, (.2, .4, 1., 1.))):
                    if plan_driven:
                        if planned_contact is None or not bool(planned_contact[0,foot]):continue
                        point=planned_world[0,foot].cpu().numpy().copy()
                        angle=float(planned_yaw[0,foot])
                    elif not bool(motion.contact[target, foot]):
                        continue
                    else:
                        point=motion.feet[target,foot,:3].cpu().numpy().copy()
                        quat=motion.feet[target,foot,3:7].cpu().numpy();w,x,y,z=quat
                        angle=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
                    point[2] += .02
                    i = viewer.user_scn.ngeom
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE,
                                       np.array([.025]*3), point, np.eye(3).ravel(), np.array(color))
                    end = point+np.array([.15*np.cos(angle), .15*np.sin(angle), 0.])
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[i+1], mujoco.mjtGeom.mjGEOM_ARROW,
                                       np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(color))
                    mujoco.mjv_connector(viewer.user_scn.geoms[i+1], mujoco.mjtGeom.mjGEOM_ARROW, .01, point, end)
                    viewer.user_scn.ngeom += 2
        viewer.sync()

    reset()
    plan_label = ("Recorded timing + learned geometry" if args.recorded_timing else
                  "Recorded sequence" if args.recorded_plan else "Learned planner" if joint else "Oracle")
    if args.phase_x_model:
        plan_label = "Recorded timing, y/z and heading + learned x"
    if args.auto_clock:
        plan_label = "Automatic clock + phase-aligned recorded y/z and heading + learned x"
    if integrated_planner is not None:
        plan_label = "Packaged omni planner"
    if event_model is not None:
        plan_label = "Next-contact event planner"
    if args.phase_pose_samples and args.auto_clock:
        plan_label = "Automatic clock + recorded geometry for supervision"
    if learned_pose:
        clock_label = "Automatic clock" if args.auto_clock else "Recorded timing"
        geometry_label = ("recorded geometry" if args.phase_pose_components == "recorded" else
                          f"learned {args.phase_pose_components}")
        plan_label = f"{clock_label} + {geometry_label}"
    if next_contact:
        plan_label = "ACP next-contact planner" if next_planner is not None else "Recorded next touchdowns"
    timing_label = "" if next_contact else f" query: {args.query_dt}s"
    print(f"Checkpoint: {checkpoint} (iteration {ck['iteration']})\nReference: {motion.names[0]}\n"
          f"{plan_label}{timing_label}; Space pause, R reset, F follow camera, P targets, Esc/Q quit", flush=True)
    if not args.headless:
        import mujoco.viewer
        render_model = copy.copy(env.sim.mj_model)
        render_data = mujoco.MjData(render_model)
        before_launch = set(threading.enumerate())
        viewer = mujoco.viewer.launch_passive(render_model, render_data, key_callback=events.append)
        viewer_threads = [t for t in threading.enumerate() if t not in before_launch]
        with viewer.lock():
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.5, 135., -15.
        render()
    try:
        if stick is not None:
            try:
                gamepad = Gamepad(args.gamepad_library)
            except (OSError, RuntimeError) as error:
                p.error(str(error))
            print("Gamepad: left stick forward/side, right stick yaw. Center to stop. "
                  "Direction changes go through standing. A or R resets; center before moving.", flush=True)
        with torch.inference_mode():
            while (limit is None or steps < limit) and (viewer is None or viewer.is_running()):
                start = time.monotonic()
                quit_requested = False
                while events:
                    key = events.popleft()
                    if key in (81, 113, 256):
                        quit_requested = True
                    elif key == 32:
                        paused = not paused
                    elif key in (82, 114):
                        reset()
                    elif key in (70, 102):
                        follow = not follow
                    elif key in (80, 112):
                        markers = not markers
                if quit_requested:
                    break
                if gamepad is not None:
                    try:
                        axes, reset_pressed = gamepad.read()
                    except RuntimeError as error:
                        p.error(str(error))
                    if reset_pressed:
                        reset()
                    if not paused:
                        gamepad_request = stick.update(axes)
                if paused:
                    if viewer:
                        render()
                    time.sleep(.02)
                    continue
                if gamepad_gait is not None:
                    # Keep reference lookups in range without resetting live physics.
                    env.f.clamp_(max=live_reference_limit)
                obs = env.observe()
                history = (obs[:, None].expand(-1, config["hist"], -1).clone() if history is None
                           else torch.cat((history[:, 1:], obs[:, None]), 1))
                if joint:query.copy_(task.pack_queries(env.f,query))
                state=env.state()
                if external_plan:
                    command = torch.cat((velocity_override[None],motion.horizons[query,None]),dim=-1)
                else:
                    command = task.command(state,env.f,query)
                if next_contact:
                    if next_planner is not None:
                        if switch_index < len(switches) and episode_step == switches[switch_index][0]:
                            velocity_override.copy_(velocity_override.new_tensor(switches[switch_index][1]))
                            switch_index += 1
                        velocity = (velocity_override[None] if args.velocity is not None
                                    else motion.requested_velocity[env.f])
                        command = next_planner.plan(history,velocity)
                    else:
                        velocity = motion.requested_velocity[env.f]
                    valid,planned_world,planned_yaw = next_world_plan(command,state)
                    next_valid = valid.bool()
                    if planned_contact is not None:
                        support_switches.add_((next_valid != planned_contact).any())
                    planned_contact = next_valid
                    if switches:
                        support_trace.append(next_valid[0].tolist())
                    action = policy(history,command)
                elif plan_driven:
                    if joint and args.velocity is not None:command[:,:3]=velocity_override
                    if event_model is not None:
                        plan = event_model.plan(history,command[:,:3],command[:,3:4],state,task)
                    elif integrated_planner is not None:
                        if switch_index < len(switches) and episode_step == switches[switch_index][0]:
                            live_velocity = switches[switch_index][1]
                            switch_index += 1
                            velocity_override.copy_(command.new_tensor(live_velocity))
                            command[:, :3] = velocity_override
                            print(f"Speed change at {episode_step * CONTROL_DT:.2f}s: {live_velocity}",flush=True)
                        body_yaw = float(torch.atan2(state[1][0,1,0],state[1][0,0,0])) if args.heading_feedback else None
                        if gamepad_gait is None:
                            plan = integrated_planner.step(obs,live_velocity,args.query_dt,body_yaw=body_yaw)
                        else:
                            plan, live_velocity = gamepad_gait.step(obs,state,gamepad_request,args.query_dt,body_yaw)
                            command[0, :3] = command.new_tensor(live_velocity)
                        if switches:
                            support_trace.append(plan[0,1:3].int().tolist())
                            cadence_trace.append(integrated_planner.cadence_updates)
                    elif recorded is None and not args.auto_clock:
                        plan=policy.contact_plan(history,command)
                    else:
                        index = recorded_step + int(motion.horizon_steps[matches[0]])
                        if args.auto_clock:
                            phi, code, waiting = (gru_clock.query(obs, command) if gru_clock is not None else
                                                  clock.query(recorded_step * CONTROL_DT, args.query_dt))
                            angle = 2 * math.pi * phi
                            future_phase = command.new_tensor([math.sin(angle), math.cos(angle)])
                            support = torch.tensor([[bool(code & 1), bool(code & 2)]], device=args.device)
                            if recorded is not None and not fully_learned:
                                # Pick geometry at the matching contact phase, near this cycle.
                                candidates = (template_state == code)
                                distance = 1 - torch.cos(template_phase - angle)
                                rank = distance + .0002 * (template_frames-index).abs()
                                index = int(rank.masked_fill(~candidates, float("inf")).argmin())
                        else:
                            support = recorded.contact[index].reshape(1, 2)
                            future_phase = phases[index] if experiment else None
                        if fully_learned:
                            local = heading = None
                        elif args.recorded_timing:
                            prediction = policy.planning_prediction(history, command)
                            local = prediction["position"]
                            heading = prediction["heading"]
                        else:
                            feet = recorded.feet[index]
                            xyz = recorded_origin[:, None] + (alignment[:, None] @
                                  (feet[:, :3] - recorded.root_p[0])[..., None]).squeeze(-1)
                            local = (yaw_mat(state[1]).transpose(-1, -2)[:, None] @
                                     (xyz - state[0][:, None])[..., None]).squeeze(-1)
                            rotation = yaw_mat(state[1]).transpose(-1, -2)[:, None] @ alignment[:, None] @ quat_to_mat(feet[:, 3:7])
                            angle = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
                            heading = torch.stack((angle.sin(), angle.cos()), -1)
                        if experiment:
                            inputs = features(history, command, support, future_phase)
                            if args.phase_x_samples:
                                samples.append((inputs.cpu(), local[..., 0].cpu(), support.cpu()))
                            elif args.phase_pose_samples:
                                # Hold poses are generated separately during clock startup.
                                if not args.auto_clock or not waiting:
                                    samples.append((inputs.cpu(),local.cpu(),heading.cpu(),support.cpu()))
                            elif learned_pose:
                                position, facing = predict_pose(pose_model, pose_ck, inputs)
                                components = args.phase_pose_components
                                if components == "all":
                                    local, heading = position, facing
                                elif components == "yaw":
                                    heading = facing
                                elif components != "recorded":
                                    axes = ["xyz".index(axis) for axis in components]
                                    local[..., axes] = position[..., axes]
                            else:
                                local[..., 0] = x_model((inputs-model_data["mean"])/model_data["scale"])
                        if args.auto_clock and waiting:
                            local = initial_local
                            heading = initial_heading
                        plan = torch.cat((command[:, 3:], support, (local * support[..., None]).flatten(1),
                                          (heading * support[..., None]).flatten(1)), -1)
                        recorded_step += 1
                    H=yaw_mat(state[1])
                    planned_world=state[0][:,None]+(H[:,None]@plan[:,3:9].reshape(-1,2,3,1)).squeeze(-1)
                    planned_yaw=torch.atan2(state[1][:,1,0],state[1][:,0,0])[:,None]+torch.atan2(plan[:,[9,11]],plan[:,[10,12]])
                    next_support = plan[:, 1:3] > .5
                    if planned_contact is not None:
                        support_switches.add_((next_support != planned_contact).any())
                    planned_contact = next_support
                    action=policy.controller_action(history,plan) if joint else policy(history,plan)
                else:action=policy(history,command)
                body_velocity = (yaw_mat(state[1]).transpose(-1, -2) @ state[2][..., None])[0, :2, 0]
                measured_velocity = torch.cat((body_velocity, state[3][0, 2:3]))
                velocity_sum.add_(measured_velocity)
                if switches:
                    segment_velocity.append(measured_velocity.tolist())
                    segment_frames.append((episode_step,switch_index))
                if measure_gaps or args.contact_trace:
                    feet,foot_rotation,gaps,point_velocity = ground.foot_state(state)
                    if measure_gaps:
                        foot_gaps.append(gaps[0])
                    if args.contact_trace:
                        foot_yaw = torch.atan2(foot_rotation[0,:,1,0],foot_rotation[0,:,0,0])
                        contact_trace.append(dict(episode=repeats,frame=episode_step,segment=switch_index,
                            command=(velocity[0] if next_contact else command[0,:3]).tolist(),
                            **{("event_valid" if next_contact else "support"): planned_contact[0].tolist()},
                            goal=planned_world[0].tolist(), goal_yaw=planned_yaw[0].tolist(),
                            feet=feet[0].tolist(), foot_yaw=foot_yaw.tolist(), gaps=gaps[0].tolist(),
                            contact_point_velocity=point_velocity[0].tolist(),
                            root=state[0][0].tolist(), body_forward=yaw_mat(state[1])[0,:,0].tolist()))
                episode_step += 1
                transition = env.step(action)
                steps += 1
                falls += int(transition["term"].sum())
                invalid += int((~transition["valid"]).sum())
                if viewer:
                    render()
                if steps % 50 == 0:
                    shown_velocity = (velocity[0] if next_contact else command[0,:3] if plan_driven
                                else motion.velocity_command[transition["f"][0]]).cpu().tolist()
                    target_bits = planned_contact[0] if plan_driven else command[0,backend.contact_offset:backend.contact_offset+2]
                    target_label = "event_valid" if next_contact else "target_contact"
                    speed_label = "velocity_command" if plan_driven or ck["format"]==VELOCITY_FORMAT else "recorded_speed_diagnostic"
                    print(f"t={steps*CONTROL_DT:.1f}s {speed_label}={[round(v, 3) for v in shown_velocity]} "
                          f"measured_velocity={[round(v, 3) for v in measured_velocity.tolist()]} "
                          f"{target_label}={target_bits.int().cpu().tolist()} resets={repeats}", flush=True)
                if bool(transition["done"][0]) or (recorded is not None and recorded_step >= int(recorded.clip_len[0])):
                    print("Reset: "+("fall/termination" if bool(transition["term"][0]) else "reference end"), flush=True)
                    if not args.loop:
                        break
                    repeats += 1
                    reset()
                if pace:
                    time.sleep(max(0., CONTROL_DT/args.speed-(time.monotonic()-start)))
    finally:
        if gamepad is not None:
            gamepad.close()
        if viewer:
            viewer.close()
            # launch_passive uses a daemon render thread. close() only requests
            # exit; let it release GL before Python's glfw.terminate atexit hook.
            for thread in viewer_threads:
                thread.join(timeout=5.)
    if sample_path and samples:
        fields = ("features","position","heading","support") if args.phase_pose_samples else ("features","target","support")
        with open(sample_path, "xb") as stream:
            torch.save(dict(zip(fields, map(torch.cat, zip(*samples)))), stream)
    if args.contact_trace:
        with args.contact_trace.open("x") as f:
            timing = {"event_window_s": 1.5} if next_contact else {"query_steps": round(args.query_dt/CONTROL_DT)}
            json.dump(dict(**timing,frames=contact_trace),f)
    changes_key = "event_validity_changes" if next_contact else "planned_support_switches"
    diagnostics = dict(plan_source=plan_label, **{changes_key: int(support_switches)}) if plan_driven else {}
    oracle_targets = (next_planner is None if next_contact else
                      not plan_driven or (args.recorded_plan and not (args.phase_x_model or learned_pose)))
    if switches:
        switch_trace = dict(frames=segment_frames,velocity=segment_velocity)
        if next_contact:
            switch_trace["event_valid"] = support_trace
        else:
            switch_trace.update(support=support_trace,cadence_updates=cadence_trace)
        diagnostics["speed_switch_trace"] = switch_trace
    if foot_gaps:
        diagnostics["foot_clearance_p95_m"] = torch.stack(foot_gaps).quantile(.95, dim=0).tolist()
    print(json.dumps(dict(steps=steps, simulated_seconds=steps*CONTROL_DT, terminations=falls,
                          mean_forward_speed_mps=float(velocity_sum[0]) / max(steps, 1),
                          mean_velocity=(velocity_sum / max(steps, 1)).tolist(),
                          invalid_transitions=invalid, repeats=repeats, oracle_targets=oracle_targets, **diagnostics)), flush=True)


if __name__ == "__main__":
    main()
