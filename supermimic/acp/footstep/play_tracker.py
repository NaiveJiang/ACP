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
    p.add_argument("--clip-index", type=int, default=0, help="Index after category filtering")
    p.add_argument("--list-clips", action="store_true", help="List matching reference clips without starting simulation")
    p.add_argument("--query-dt", type=float, default=.3, help="One of the checkpoint's trained query times")
    p.add_argument("--phase", type=float, default=0., help="Starting fraction of the executable reference block")
    p.add_argument("--seconds", type=float, default=60., help="Total simulated playback seconds; 0 runs until closed")
    p.add_argument("--speed", type=float, default=1., help="Playback speed relative to wall time")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--pace", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True, help="Reset on fall/end and replay selected clip")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
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
    supported = ("two_foot_oracle_tracker_v1", "two_foot_contact_only_tracker_v2", "two_foot_stable_contact_tracker_v3")
    if ck.get("format") not in supported:
        raise ValueError("Need a two-foot tracker checkpoint, not a planner or legacy ACP checkpoint")
    if default_checkpoint == "latest-contact" and ck["format"] not in supported[1:]:
        raise ValueError("This entrypoint expects the 13-D contact-only controller")
    intervals = select_intervals(ck, args.split, args.category)
    if args.list_clips:
        for i, c in enumerate(intervals):
            print(f"{i:4d}  {c['category']:16s} {Path(c['source']).parent.name}/{c['case']} [{c['start']}:{c['end']}]")
        return
    if args.clip_index >= len(intervals):
        p.error(f"clip-index outside 0..{len(intervals)-1}; use --list-clips")
    config = ck["config"]
    matches = [i for i, dt in enumerate(config["horizons"]) if abs(dt-args.query_dt) < 1e-6]
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
    backend = backend_for_format(ck["format"])
    FootstepTask, FootstepMDP, TrackerPolicy = backend.Task, backend.MDP, backend.Policy
    OBS_DIM, ContactRewardCfg = backend.obs_dim, backend.RewardCfg
    if ck["robot"]["fingerprint"] != metadata()["fingerprint"]:
        raise ValueError("Checkpoint robot/control definition differs")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    kin = RobotKinematics(args.device)
    motion = FootstepMotion([intervals[args.clip_index]], kin, args.device, config["horizons"])
    task = FootstepTask(motion, kin, ContactRewardCfg(**ck["reward"]))
    mdp = FootstepMDP(kin, motion, torch.tensor(default_joint_pos_list(), device=args.device), task)
    policy = TrackerPolicy(config["hist"], config["policy_hidden"]).to(args.device).eval()
    policy.load_state_dict(ck["pi"])
    env = FootstepTrackingSim(1, motion, mdp, args.device, eval_mode=True, reset_noise=False)
    query = torch.tensor([matches[0]], device=args.device)
    frame0 = motion.clip_start[0]+int(args.phase*(int(motion.clip_len[0])-1))
    history = None
    events = deque()
    paused, follow, markers = False, True, True
    viewer, render_model, render_data = None, None, None
    viewer_threads = []
    steps = falls = invalid = repeats = 0
    limit = int(math.ceil(args.seconds/CONTROL_DT)) if args.seconds else None
    pace = not args.headless if args.pace is None else args.pace

    def reset():
        nonlocal history
        env.reset(torch.tensor([0], device=args.device), frames=frame0.reshape(1))
        env.refresh()
        env.done_eval.zero_(); env.failed_eval.zero_()
        history = None

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
                target = int(env.f[0]+motion.horizon_steps[query[0]])
                for foot, color in ((0, (1., .2, .2, 1.)), (1, (.2, .4, 1., 1.))):
                    if not bool(motion.contact[target, foot]):
                        continue
                    point = motion.feet[target, foot, :3].cpu().numpy().copy()
                    point[2] += .02
                    i = viewer.user_scn.ngeom
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE,
                                       np.array([.025]*3), point, np.eye(3).ravel(), np.array(color))
                    quat = motion.feet[target, foot, 3:7].cpu().numpy()
                    w, x, y, z = quat
                    angle = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
                    end = point+np.array([.15*np.cos(angle), .15*np.sin(angle), 0.])
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[i+1], mujoco.mjtGeom.mjGEOM_ARROW,
                                       np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(color))
                    mujoco.mjv_connector(viewer.user_scn.geoms[i+1], mujoco.mjtGeom.mjGEOM_ARROW, .01, point, end)
                    viewer.user_scn.ngeom += 2
        viewer.sync()

    reset()
    print(f"Checkpoint: {checkpoint} (iteration {ck['iteration']})\nReference: {motion.names[0]}\n"
          f"Oracle query: {args.query_dt}s; Space pause, R reset, F follow camera, P targets, Esc/Q quit", flush=True)
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
                if paused:
                    if viewer:
                        render()
                    time.sleep(.02)
                    continue
                obs = env.observe()
                history = (obs[:, None].expand(-1, config["hist"], -1).clone() if history is None
                           else torch.cat((history[:, 1:], obs[:, None]), 1))
                command = task.command(env.state(), env.f, query)
                transition = env.step(policy(history, command))
                steps += 1
                falls += int(transition["term"].sum())
                invalid += int((~transition["valid"]).sum())
                if viewer:
                    render()
                if steps % 50 == 0:
                    velocity = motion.velocity_command[transition["f"][0]].cpu().tolist()
                    offset = backend.contact_offset
                    speed_label = "velocity_command" if backend.command_dim == 16 else "recorded_speed_diagnostic"
                    print(f"t={steps*CONTROL_DT:.1f}s {speed_label}={[round(v, 3) for v in velocity]} "
                          f"target_contact={command[0, offset:offset+2].int().cpu().tolist()} resets={repeats}", flush=True)
                if bool(transition["done"][0]):
                    print("Reset: "+("fall/termination" if bool(transition["term"][0]) else "reference end"), flush=True)
                    if not args.loop:
                        break
                    repeats += 1
                    reset()
                if pace:
                    time.sleep(max(0., CONTROL_DT/args.speed-(time.monotonic()-start)))
    finally:
        if viewer:
            viewer.close()
            # launch_passive uses a daemon render thread. close() only requests
            # exit; let it release GL before Python's glfw.terminate atexit hook.
            for thread in viewer_threads:
                thread.join(timeout=5.)
    print(json.dumps(dict(steps=steps, simulated_seconds=steps*CONTROL_DT, terminations=falls,
                          invalid_transitions=invalid, repeats=repeats, oracle_targets=True)), flush=True)


if __name__ == "__main__":
    main()
