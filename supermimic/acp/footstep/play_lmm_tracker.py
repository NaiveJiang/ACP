"""Drive the S17 full-body policy with live LMM windows in MuJoCo."""

import argparse
from contextlib import ExitStack
from dataclasses import dataclass, replace
import math
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
import time

import numpy as np
import torch

from .gamepad import Gamepad
from .learned_motion import BoundedHeadingCommand, HeadingCommand, LearnedPlayback, checksum
from .motion_matching import Robot
from .s17_alignment import spring_angle
from .reference_planner import ReferencePlanner
from .. import networks
from ..checkpoint import load_checkpoint
from ..env import TrackingSim, evaluation_surface_friction
from ..kinematics import RobotKinematics
from ..mdp import MDP, RewardCfg, unflatten_state
from ..motion import CONTROL_DT, MotionLib
from ..reference_mimic import ReferenceCfg, ReferenceTask, build_policy, check_interface
from ..robot import ROBOT_ID, default_joint_pos_list
from ..s17_actuation import DELAY_STEPS
from ...rotations import mat_to_quat, quat_to_mat, yaw_mat


class GamepadSteering:
    """Translate SDL stick input into motion commands for the LMM."""

    def __init__(self, walk_speed, run_speed, turn_speed, deadzone):
        self.walk_speed = walk_speed
        self.run_speed = run_speed
        self.turn_speed = turn_speed
        self.deadzone = deadzone
        self.reset(0.)

    def reset(self, yaw):
        self.heading = yaw
        self.aimed = False
        self.drive_at_aim = np.zeros(2)
        self.walk = self.walk_rate = 0.
        self.direction_basis = yaw
        self.turning = False

    def forward_speed(self, walk):
        self.walk, self.walk_rate = spring_angle(self.walk, self.walk_rate, float(walk),
                                                  2. * math.log(2.) / .1, CONTROL_DT)
        return self.run_speed + (self.walk_speed - self.run_speed) * self.walk

    def update(self, axes, *, walk, orbit, hold, camera_yaw):
        shaped = np.asarray(axes[:2])
        shaped = np.where(np.abs(shaped) > self.deadzone, shaped * np.abs(shaped), 0.)
        forward = self.forward_speed(walk)
        side_speed = 1.15 + (.75 - 1.15) * self.walk
        requested = shaped * [forward, side_speed] / max(1., np.abs(shaped).sum())
        speed = float(np.linalg.norm(requested))
        direction = requested / speed if speed else np.zeros(2)
        aim = np.array([axes[3], axes[2]])
        if not orbit and not hold and np.linalg.norm(aim) > self.deadzone:
            self.aimed = True
            self.drive_at_aim = direction
            self.heading = camera_yaw + math.atan2(aim[1], aim[0])
        elif not hold and speed:
            same = math.cos(math.pi / 6.)
            if self.aimed and direction[0] >= same:
                self.drive_at_aim = direction
            elif np.dot(direction, self.drive_at_aim) < same:
                self.aimed = False
            if not self.aimed:
                self.heading = math.atan2(direction[1], direction[0])
        drive = tuple(requested) if hold else (speed, 0.)
        return HeadingCommand(drive, self.heading, self.turn_speed)

    def update_direction_basis(self, axes, *, walk, robot_yaw):
        """Turn in place with the right stick and latch measured yaw on release."""
        forward = self.forward_speed(walk)
        drive, side, turn = axes[:3]
        if abs(turn) > self.deadzone:
            self.turning = True
            return np.array([0., 0., self.turn_speed * turn * abs(turn)])
        if self.turning:
            self.direction_basis = robot_yaw
            self.turning = False
        radius = math.hypot(drive, side)
        if radius <= self.deadzone:
            return np.zeros(3)
        speed = forward * min(1., radius)**2
        heading = self.direction_basis + math.atan2(side, drive)
        return HeadingCommand((speed, 0.), math.remainder(heading, math.tau), self.turn_speed)


class OmnidirectionalSteering(GamepadSteering):
    trigger_threshold = .20
    trigger_release = .15
    turn_tolerance = math.radians(3.)

    def reset(self, yaw):
        super().reset(yaw)
        self.last_yaw = self.yaw = yaw
        self.target_yaw = None
        self.completed_yaw = None
        self.turn_origin = yaw
        self.dpad = (False, False, False)
        self.trigger_pressed = False
        self.trigger_side = None
        self.trigger_peak = 0.

    def update_omnidirectional(self, axes, *, walk, robot_yaw, dpad, triggers):
        self.yaw += math.remainder(robot_yaw - self.last_yaw, math.tau)
        self.last_yaw = robot_yaw
        forward = self.forward_speed(walk)
        stick = np.asarray(axes[:2])
        radius = np.linalg.norm(stick)
        drive = (min(1., radius)**2 / radius * stick * [forward, 1.15 + (.75 - 1.15) * self.walk]
                 if radius > self.deadzone else np.zeros(2))
        manual_turn = axes[2]
        previous_dpad, self.dpad = self.dpad, dpad
        threshold = self.trigger_release if self.trigger_pressed else self.trigger_threshold
        pressed = max(triggers) > threshold
        new_trigger = pressed and not self.trigger_pressed
        self.trigger_pressed = pressed

        if abs(manual_turn) > self.deadzone:
            self.target_yaw = None
            self.completed_yaw = None
            self.trigger_side = None
            return np.r_[drive, self.turn_speed * manual_turn * abs(manual_turn)]
        if any(dpad):
            self.trigger_side = None
            self.update_dpad_turn(dpad, previous_dpad)
        else:
            self.update_trigger_turn(triggers, new_trigger)
        if self.target_yaw is None:
            return np.r_[drive, 0.]
        remaining = self.target_yaw - self.yaw
        reached = self.turn_sign * remaining <= self.turn_tolerance
        if reached or self.turn_ticks == 0:
            if reached:
                self.completed_yaw = self.target_yaw
            self.target_yaw = None
            return np.r_[drive, 0.]
        self.turn_ticks -= 1
        # Track the selected side even if a half turn starts across the yaw wrap.
        error = float(np.clip(remaining, -math.pi, math.pi))
        return HeadingCommand(tuple(drive), robot_yaw + error, self.turn_speed, error)

    def start_turn(self, angle):
        self.target_yaw = self.turn_origin + angle
        self.completed_yaw = None
        self.turn_sign = math.copysign(1., angle)
        # Allow time to step into the turn and stop issuing it if the policy stalls.
        self.turn_ticks = math.ceil((2. + 2. * abs(angle) / self.turn_speed) / CONTROL_DT)

    def reference_command(self, command):
        goal = self.target_yaw if self.target_yaw is not None else self.completed_yaw
        if goal is None:
            return command
        drive = command.drive if isinstance(command, HeadingCommand) else tuple(command[:2])
        return BoundedHeadingCommand(drive, goal, self.turn_speed)

    def update_dpad_turn(self, buttons, previous):
        if not any(previous):
            self.turn_origin = self.yaw
        if not any(now and not old for now, old in zip(buttons, previous, strict=True)):
            return
        left, right, down = buttons
        side = int(left) - int(right)
        if side or down:
            self.start_turn(math.atan2(side, -int(down)))

    def update_trigger_turn(self, triggers, new_trigger):
        if new_trigger:
            self.trigger_side = int(triggers[1] > triggers[0])
            self.trigger_peak = self.trigger_threshold
            self.turn_origin = self.yaw
        if self.trigger_side is None:
            return
        amount = triggers[self.trigger_side]
        if amount <= self.trigger_release:
            self.trigger_side = None
        # A held trigger can extend the turn. Small changes in its reported value leave the target in place.
        elif amount > self.trigger_peak and (new_trigger or amount - self.trigger_peak >= .02 or amount == 1.):
            self.trigger_peak = amount
            travel = (amount - self.trigger_threshold) / (1. - self.trigger_threshold)
            signed_angle = (1 - 2 * self.trigger_side) * (1. + travel) * math.pi / 2
            self.start_turn(signed_angle)


class LiveMDP(MDP):
    """Use the trained policy's observations and fall check during open-ended playback."""

    def reward(self, state, frames, action, previous_action, *, return_self_overlap=False):
        value = action.new_zeros(len(action))
        return (value, None) if return_self_overlap else value


def initial_motion(planner, robot, kin, device):
    pose = planner.playback.forecast_reference(np.zeros(3), 0)[:, planner.columns]
    with TemporaryDirectory(prefix="roban-lmm-live-") as directory:
        path = Path(directory) / "standing_motion.npz"
        np.savez(path, fps=50., joint_names=robot.joints, robot_id=ROBOT_ID,
                 joint_pos=np.repeat(pose, 2, axis=0).astype(np.float32))
        np.save(Path(directory) / "standing_terrain.npy", np.empty((0, 1, 10), dtype=np.float32))
        return MotionLib([str(path)], kin, device, root=directory)


def body_yaw(state):
    R = state[1]
    return float(torch.atan2(R[0, 1, 0], R[0, 0, 0]))


def stop_playback(signum, frame):
    raise SystemExit(0)


@dataclass
class LiveTracker:
    playback: LearnedPlayback
    task: ReferenceTask
    policy: torch.nn.Module
    env: TrackingSim
    checkpoint: dict

    def reset(self):
        device = self.env.device
        ids = torch.zeros(1, dtype=torch.long, device=device)
        seed = self.playback.seed_state
        self.playback.state = seed
        self.env.reset(ids, frames=ids)
        self.env.done_eval.zero_()
        self.env.failed_eval.zero_()
        return ReferencePlanner(self.playback, offsets=self.task.cfg.offsets)

    def command(self, state, fields):
        components = fields.components()
        if self.task.cfg.body_reference:
            components = components[1:]
        window = tuple(torch.as_tensor(x, device=self.env.device, dtype=torch.float32)[None] for x in components)
        if self.task.cfg.body_reference:
            # The simulator's reference buffer still uses the common root/body tuple.
            window = (torch.zeros_like(window[1]), *window)
        anchor = None
        if self.task.cfg.root_reference == "local":
            current = window[0].new_tensor(fields.current_pose[:7])[None]
            anchor = (current[:, :3], quat_to_mat(current[:, 3:7]))
        return self.task.command_from_reference(state, window, self.env.boxes(), anchor=anchor), window

    def visible_reference(self, state, fields, fallen):
        pose = fields.future_pose[0] if fallen else fields.current_pose
        if self.task.cfg.root_reference == "world":
            return pose
        pose = pose.copy()
        pose[:2] = state[0][0, :2].cpu().numpy()
        if self.task.cfg.body_reference:
            return pose
        current = quat_to_mat(state[0].new_tensor(fields.current_pose[3:7]))
        rotation = quat_to_mat(state[0].new_tensor(pose[3:7]))
        adjustment = yaw_mat(state[1][0]) @ yaw_mat(current).T
        pose[3:7] = mat_to_quat(adjustment @ rotation).cpu().numpy()
        return pose

    def step(self, action, window, *, score=False):
        env, motion = self.env, self.task.motion
        ref = tuple(x[:, 0] for x in window)
        # The two-frame buffer gives the physics and fall check a fresh target each step.
        motion.root_p[1].copy_(ref[0][0])
        motion.root_R[1].copy_(ref[1][0])
        if score:
            for table, value in zip((motion.root_v, motion.root_w, motion.q, motion.qd), ref[2:], strict=True):
                table[1].copy_(value[0])
            body = self.task.mdp.kin.forward(*ref)
            for table, value in zip((motion.body_p, motion.body_R, motion.body_v, motion.body_w), body, strict=True):
                table[1].copy_(value[0])
        env.f.zero_()
        env.done_eval.zero_()
        return env.step(self.policy.project_action(action))


def load_tracker(checkpoint_path, lmm, device, ema, *, score=False, root_reference="auto"):
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu", weights_only=True)
    check_interface(checkpoint)
    if (expected := checkpoint.get("lmm_projector_sha256")) is not None and checksum(lmm / "projector.pt") != expected:
        raise ValueError("Jointly trained controller and LMM projector are from different checkpoints")
    settings = checkpoint["args"]
    reference_data = dict(checkpoint["reference_cfg"])
    if root_reference != "auto":
        saved_mode = reference_data.get("root_reference")
        if (saved_mode == "body-heading") != (root_reference == "body-heading"):
            raise ValueError("Body-heading playback requires a checkpoint trained with the same reference interface")
        reference_data["root_reference"] = root_reference
    elif "root_reference" not in reference_data:
        # Checkpoints exported by the original joint trainer used the world path.
        reference_data["root_reference"] = "world" if checkpoint.get("training_engine") == "lmm_joint_ppo_v1" else "local"
    reference = ReferenceCfg(**reference_data)
    key = "pi_ema" if ema else "pi"
    if key not in checkpoint:
        raise ValueError(f"Checkpoint does not contain {key}")
    networks.AMP_DTYPE = torch.bfloat16 if settings.get("amp") == "bf16" else None
    robot = Robot()
    playback = LearnedPlayback(lmm, robot)
    planner = ReferencePlanner(playback, offsets=reference.offsets)
    kin = RobotKinematics(device)
    motion = initial_motion(planner, robot, kin, device)
    ground, box = evaluation_surface_friction(settings)
    mdp_cls = MDP if score else LiveMDP
    mdp = mdp_cls(kin, motion, torch.tensor(default_joint_pos_list(), device=device),
                  RewardCfg(w_self_collision=settings.get("self_collision_weight", 0.),
                            local_reference_root=reference.root_reference == "local",
                            body_reference=reference.body_reference),
                  independent_surface_friction=ground is not None,
                  delay_steps=DELAY_STEPS if settings.get("s17_actuators") else 0)
    task = ReferenceTask(reference, mdp)
    policy = build_policy(mdp, task, settings["hist"]).to(device)
    policy.load_state_dict(checkpoint[key])
    policy.eval()
    env = TrackingSim(1, motion, mdp, device, eval_mode=True, reference_reset_action=True,
                      ground_friction_range=None if ground is None else (ground, ground),
                      box_friction_range=None if box is None else (box, box),
                      primitive_soles=settings.get("primitive_soles", False))
    return LiveTracker(playback, task, policy, env, checkpoint)


def main():
    signal.signal(signal.SIGTERM, stop_playback)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmm", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gamepad", action="store_true", help="Read an SDL gamepad (default)")
    mode.add_argument("--scripted", action="store_true", help="Use the given velocity/switches for a repeatable test")
    parser.add_argument("--gamepad-mode", choices=("omnidirectional", "direction-basis", "legacy"),
                        default="omnidirectional")
    parser.add_argument("--gamepad-library")
    parser.add_argument("--gamepad-wait", type=float, default=10.)
    parser.add_argument("--walk-speed", type=float, default=.72)
    parser.add_argument("--run-speed", type=float, help="Full forward stick speed (omnidirectional: 2.2 m/s, other modes: 1.55 m/s)")
    parser.add_argument("--turn-speed", type=float, default=2.8)
    parser.add_argument("--deadzone", type=float, default=.20)
    parser.add_argument("--velocity", type=float, nargs=3, default=(0., 0., 0.))
    parser.add_argument("--switch", type=float, nargs=4, action="append", default=[],
                        metavar=("TIME", "VX", "VY", "WZ"))
    parser.add_argument("--seconds", type=float, default=0., help="Use 0 for continuous interactive playback")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--root-reference", choices=("auto", "local", "world", "body-heading"), default="auto")
    args = parser.parse_args()
    if args.run_speed is None:
        args.run_speed = 2.2 if args.gamepad_mode == "omnidirectional" else 1.55
    limits = (args.seconds, args.walk_speed, args.run_speed, args.turn_speed, args.deadzone, args.gamepad_wait)
    if (not all(math.isfinite(x) for x in (*limits, *args.velocity)) or args.seconds < 0
            or not 0 < args.walk_speed <= args.run_speed or args.turn_speed <= 0
            or not 0 <= args.deadzone < 1 or args.gamepad_wait < 0):
        parser.error("Invalid speed, dead zone, or duration")
    if (args.switch or any(args.velocity)) and not args.scripted:
        parser.error("Use --scripted with velocity commands")
    if args.headless and (not args.scripted or args.seconds < CONTROL_DT):
        parser.error("--headless requires --scripted and a positive duration")
    limit = round(args.seconds / CONTROL_DT) if args.seconds else None
    schedule = {}
    for t, *command in args.switch:
        if not np.isfinite([t, *command]).all() or t <= 0:
            parser.error("Switches must contain a positive time and finite commands")
        tick = round(t / CONTROL_DT)
        if tick == 0 or tick in schedule or (limit is not None and tick >= limit):
            parser.error("Switches must be on distinct steps inside playback")
        schedule[tick] = np.asarray(command)
    try:
        play(args, schedule, limit)
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")


@torch.inference_mode()
def play(args, schedule, limit):
    torch.set_num_threads(2)
    tracker = load_tracker(args.checkpoint, args.lmm, args.device, args.ema,
                           root_reference=getattr(args, "root_reference", "auto"))
    print(f"Root reference mode: {tracker.task.cfg.root_reference}", flush=True)
    env, policy = tracker.env, tracker.policy
    use_direction_basis = not args.scripted and args.gamepad_mode == "direction-basis"
    use_omnidirectional = not args.scripted and args.gamepad_mode == "omnidirectional"
    steering_type = OmnidirectionalSteering if use_omnidirectional else GamepadSteering
    steer = steering_type(args.walk_speed, args.run_speed, args.turn_speed, args.deadzone)
    controls = {
        "omnidirectional": "Left: vx/vy  Right X: yaw rate\nD-pad / LT / RT: turn by angle",
        "direction-basis": "Left: move  Right X: turn/set forward",
        "legacy": "Left: move  Right: aim  LT: hold",
    }[args.gamepad_mode]

    def reset():
        planner = tracker.reset()
        steer.reset(body_yaw(env.state()))
        return planner

    with ExitStack() as stack:
        pad = None
        if not args.scripted:
            print(f"Waiting for an SDL gamepad ({args.gamepad_wait:g} s)...", flush=True)
            pad = Gamepad(args.gamepad_library, wait=args.gamepad_wait)
            stack.callback(pad.close)
            print(f"{pad.name}: {controls}, A walk, Start reset", flush=True)
        viewer = None
        if not args.headless:
            from ..reference_viewer import ReferenceViewer

            viewer = stack.enter_context(ReferenceViewer(None, env.boxes()[0].cpu().numpy(),
                                                         primitive_soles=env.primitive_soles))
        planner = reset()
        history, fallen = None, False
        frame = episode_frame = 0
        command = np.array(args.velocity)
        fields = None
        start = time.monotonic()
        while (limit is None or frame < limit) and (viewer is None or viewer.viewer.is_running()):
            state = env.state()
            if pad:
                axes, _ = pad.read(full_right=True)
                if pad.start_edge():
                    planner = reset()
                    state = env.state()
                    history, fallen, episode_frame = None, False, 0
                walk, orbit, zoom = pad.motion_matching_buttons()
                camera_yaw = 0.
                if viewer is not None:
                    with viewer.viewer.lock():
                        cam = viewer.viewer.cam
                        if orbit and args.gamepad_mode == "legacy":
                            right = np.asarray(axes[2:])
                            length = np.linalg.norm(right)
                            if length > args.deadzone:
                                left, up = right * (min(1., length)**2 / length)
                                cam.azimuth = (cam.azimuth + math.degrees(2 * CONTROL_DT) * left + 180.) % 360. - 180.
                                cam.elevation = float(np.clip(cam.elevation + math.degrees(2 * CONTROL_DT) * up, -72., 0.))
                        cam.distance = float(np.clip(cam.distance + 10 * zoom * CONTROL_DT, .1, 100.))
                        camera_yaw = math.radians(cam.azimuth)
                if use_omnidirectional:
                    robot_yaw = body_yaw(state)
                    dpad, triggers = pad.turning_inputs()
                    command = steer.update_omnidirectional(axes, walk=walk, robot_yaw=robot_yaw,
                                                           dpad=dpad, triggers=triggers)
                elif use_direction_basis:
                    robot_yaw = body_yaw(state)
                    command = steer.update_direction_basis(axes, walk=walk, robot_yaw=robot_yaw)
                else:
                    command = steer.update(axes, walk=walk, orbit=orbit, hold=pad.hold_heading_pressed(),
                                           camera_yaw=camera_yaw)
            else:
                command = schedule.get(frame, command)
            display_frame = episode_frame
            if not fallen:
                reference_command = command
                if use_omnidirectional:
                    reference_command = steer.reference_command(command)
                elif use_direction_basis and isinstance(command, HeadingCommand):
                    _, error = command.resolve(robot_yaw)
                    reference_command = replace(command, heading=tracker.playback.state.yaw + error)
                fields = planner.plan(reference_command)
                obs = env.observe()[:, None]
                history = obs.expand(-1, policy.hist, -1) if history is None else torch.cat((history[:, 1:], obs), 1)
                cmd, window = tracker.command(state, fields)
                transition = tracker.step(policy(history, cmd), window)
                fallen = bool(transition["term"][0])
                episode_frame += 1
            if fallen and viewer is None:
                raise RuntimeError(f"Tracking terminated at {episode_frame * CONTROL_DT:.2f} s")
            if viewer is not None:
                if fallen:
                    display_frame = episode_frame
                    state = unflatten_state(transition["s1"])
                reference_pose = tracker.visible_reference(state, fields, fallen)
                if fallen:
                    status = "Fallen: press Start to reset"
                elif isinstance(command, HeadingCommand):
                    display, error = command.resolve(body_yaw(state))
                    status = (f"vx {display[0]:+.2f}  vy {display[1]:+.2f} m/s"
                              f"   Heading error {math.degrees(error):+.0f} deg")
                else:
                    status = f"vx {command[0]:+.2f}  vy {command[1]:+.2f} m/s   wz {command[2]:+.2f} rad/s"
                if use_direction_basis:
                    status += f"   Forward basis {math.degrees(steer.direction_basis):+.0f} deg"
                viewer.viewer.set_texts((None, None,
                    f"LMM + S17 controller | {tracker.task.cfg.root_reference} reference\n"
                    f"{controls}\nA: walk  Start: reset  Space: pause", status))
                if not viewer.show_reference(state, reference_pose, display_frame * CONTROL_DT):
                    break
            frame += 1
        print(f"Played {frame} frames in {time.monotonic() - start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
