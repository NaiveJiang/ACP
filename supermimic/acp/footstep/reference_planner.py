"""Plan a rolling full-body reference with native Motion Matching or a learned RoBan model."""
import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .motion_matching import Robot, upstream_clip_name, upstream_clips
from .native_search import NativePlayback


MATCH_HZ = 60
POLICY_HZ = 50
REFERENCE_OFFSETS = (1, 5, 10, 20, 30, 50)


@dataclass
class ReferenceWindow:
    current_pose: np.ndarray
    future_pose: np.ndarray
    root_linear_velocity: np.ndarray
    root_angular_velocity: np.ndarray
    joint_velocity: np.ndarray

    def components(self):
        q = self.future_pose
        return (q[:, :3], Rotation.from_quat(q[:, 3:7], scalar_first=True).as_matrix(),
                self.root_linear_velocity, self.root_angular_velocity, q[:, 7:], self.joint_velocity)


class ReferencePlanner:
    def __init__(self, playback, offsets=REFERENCE_OFFSETS):
        self.playback = playback
        self.offsets = np.asarray(offsets)
        if (self.offsets.ndim != 1 or self.offsets.size == 0 or self.offsets.dtype.kind not in "iu"
                or self.offsets[0] < 1 or np.any(self.offsets[1:] <= self.offsets[:-1])):
            raise ValueError("Reference offsets must be positive, strictly increasing integers")
        self.offsets = self.offsets.astype(int)
        self.source_hz = getattr(playback, "fps", MATCH_HZ)
        if int(self.source_hz) != self.source_hz or self.source_hz < POLICY_HZ:
            raise ValueError("Playback must use an integer source rate at least as high as the policy rate")
        self.source_hz = int(self.source_hz)
        robot = playback.robot
        self.columns = np.r_[np.arange(robot.root, robot.root+7), robot.joint_qpos]
        self.frame = 0
        self.tick = 0
        self.previous_command = np.zeros(3)

    def plan(self, command):
        """Advance to this policy sample and forecast using its held body-frame command."""
        command = np.asarray(command, dtype=float)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("Command must contain finite vx, vy, and yaw rate")
        base_tick, fraction = divmod(self.frame*self.source_hz, POLICY_HZ)
        self.playback.advance_reference(self.previous_command, base_tick-self.tick)
        self.tick = base_tick

        # At times between two source ticks, both current-pose brackets still
        # belong to the previous command. Apply the update at the next source tick.
        prefix = []
        if fraction:
            prefix.append(self.playback.forecast_reference(self.previous_command, 0))
            self.playback.advance_reference(self.previous_command, 1)
            self.tick += 1

        # Include the samples bracketing each future pose for centered velocities.
        samples = np.arange(self.offsets[-1]+2)
        source_ticks = (fraction+samples*self.source_hz)/POLICY_HZ
        steps = math.ceil(source_ticks[-1])-len(prefix)
        forecast = self.playback.forecast_reference(command, steps)
        poses = np.concatenate((*prefix, forecast))[:, self.columns]
        self.previous_command = command.copy()
        time = np.arange(len(poses))
        sampled = np.column_stack([np.interp(source_ticks, time, x) for x in poses.T])
        rotation = Slerp(time, Rotation.from_quat(poses[:, 3:7], scalar_first=True))(source_ticks)
        sampled[:, 3:7] = rotation.as_quat(scalar_first=True)

        lo, hi = self.offsets-1, self.offsets+1
        derivative = (sampled[hi]-sampled[lo])*(POLICY_HZ/2.)
        angular = (rotation[hi]*rotation[lo].inv()).as_rotvec()*(POLICY_HZ/2.)
        self.frame += 1
        return ReferenceWindow(sampled[0], sampled[self.offsets], derivative[:, :3], angular, derivative[:, 7:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument("--upstream-roban", type=Path)
    backend.add_argument("--lmm", type=Path, help="Directory containing the RoBan LMM weights")
    parser.add_argument("--s17-search-library", type=Path, nargs=3, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, required=True, help="New directory for the reference and forecast")
    parser.add_argument("--seconds", type=float, default=12.)
    parser.add_argument("--velocity", type=float, nargs=3, default=(0., 0., 0.))
    parser.add_argument("--switch", type=float, nargs=4, action="append", default=[],
                        metavar=("TIME", "VX", "VY", "WZ"))
    args = parser.parse_args()
    if args.upstream_roban and args.s17_search_library is None:
        parser.error("Native Motion Matching requires --s17-search-library")
    if not math.isfinite(args.seconds) or args.seconds < 2./POLICY_HZ or args.out.exists():
        parser.error("Use a positive duration and a new output directory")
    frames = round(args.seconds*POLICY_HZ)
    schedule = {}
    for time, *velocity in args.switch:
        if not np.isfinite([time, *velocity]).all():
            parser.error("Switch times and velocities must be finite")
        step = round(time*POLICY_HZ)
        if not 0 < step < frames or step in schedule:
            parser.error("Each switch must be in a unique frame strictly inside the rollout")
        schedule[step] = velocity

    robot = Robot()
    if args.lmm:
        from .learned_motion import LearnedPlayback
        playback = LearnedPlayback(args.lmm, robot)
    else:
        clips = [(upstream_clip_name(c), c["frames"]) for c in upstream_clips(args.upstream_roban)]
        playback = NativePlayback(*args.s17_search_library, robot=robot, clips=clips)
    planner = ReferencePlanner(playback)
    command = args.velocity
    windows, commands = [], []
    try:
        for step in range(frames):
            command = schedule.get(step, command)
            windows.append(planner.plan(command))
            commands.append(command)
    finally:
        if not args.lmm:
            playback.close()

    output = args.out/"planned"
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"velocity_demo_motion.npz",
                        joint_pos=np.stack([w.current_pose for w in windows]).astype(np.float32),
                        joint_names=np.array(robot.joints), fps=np.float32(POLICY_HZ), robot_id=np.array("roban"))
    np.save(output/"velocity_demo_terrain.npy", np.empty((0, 1, 10), dtype=np.float32))
    reference = [window.components() for window in windows]
    fields = ["root_position", "root_rotation", "root_linear_velocity", "root_angular_velocity",
              "joint_position", "joint_velocity"]
    arrays = {name: np.stack(values).astype(np.float32) for name, values in zip(fields, zip(*reference))}
    np.savez_compressed(args.out/"forecast.npz", offsets=planner.offsets, command=np.asarray(commands), **arrays)
    source = {"lmm": str(args.lmm.resolve())} if args.lmm else {"motion_library": str(args.upstream_roban.resolve())}
    manifest = dict(**source, match_hz=getattr(playback, "fps", MATCH_HZ),
                    policy_hz=POLICY_HZ, offsets=planner.offsets.tolist(), frames=frames, command_frame="body")
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"{frames} full-body references at {POLICY_HZ} Hz, forecast offsets {planner.offsets.tolist()}")
    print(args.out)


if __name__ == "__main__":
    main()
