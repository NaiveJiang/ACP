"""Prepare reproducible command trajectories and held-command LMM forecasts."""

import json
from pathlib import Path
import re

import numpy as np
import torch

from .learned_motion import LearnedPlayback
from .motion_matching import Robot
from .reference_planner import ReferencePlanner
from .lmm_contact import LOOKAHEAD, lmm_identity
from .rollout_controller import digest
from ..robot import JOINT_NAMES, ROBOT_ID


SCHEMA = "lmm_contact_command_scenarios_v1"
DEFAULT_SCENARIOS = {
    "start_stop": [[0., 0., 0., 0.], [1., .65, 0., 0.], [4., 0., 0., 0.]],
    "walk_run": [[0., .6, 0., 0.], [3., 1.5, 0., 0.], [6., .6, 0., 0.]],
    "turn_left": [[0., .65, 0., 0.], [3., .55, 0., .75], [6., .65, 0., 0.]],
    "turn_right": [[0., .65, 0., 0.], [3., .55, 0., -.75], [6., .65, 0., 0.]],
}


def prepare_scenarios(lmm, output, seconds=9., stride=5, schedule_path=None):
    schedules = (DEFAULT_SCENARIOS if schedule_path is None
                 else json.loads(schedule_path.read_text()))
    if not schedules:
        raise ValueError("At least one command scenario is needed")
    scenario_steps = round(seconds * 50)
    # Leave enough measured tail to confirm the last intended planning window.
    steps = scenario_steps + LOOKAHEAD + 1
    output.mkdir(parents=True, exist_ok=False)
    robot = Robot()
    torch.set_num_threads(2)
    manifest = dict(schema=SCHEMA, lmm=lmm_identity(lmm), clips={})
    for name, sequence in schedules.items():
        values = np.asarray(sequence, dtype=float)
        if (not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or values.ndim != 2 or values.shape[1] != 4
                or not np.isfinite(values).all() or values[0, 0] != 0):
            raise ValueError(f"Use [seconds, vx, vy, vyaw] rows starting at zero in {name}")
        changes = np.rint(values[:, 0] * 50).astype(int)
        if np.any(np.diff(changes) <= 0) or changes[-1] >= scenario_steps:
            raise ValueError(f"Scenario transitions must map to ordered 50 Hz frames in {name}")
        clock = np.arange(steps + 1)
        command = values[np.searchsorted(changes, clock, side="right") - 1, 1:]
        anchors = np.unique(np.r_[1, clock[::stride], changes])
        anchors = set(anchors.tolist())
        planner = ReferencePlanner(LearnedPlayback(lmm, robot), offsets=(1,))
        poses, queries, future = [], [], []
        offsets = np.arange(1, LOOKAHEAD + 1)
        for t, control in enumerate(command):
            if t in anchors:
                forecast = planner.plan(control, dense_offsets=offsets)
                future.append(np.concatenate((forecast.current_pose[None], forecast.future_pose)))
                queries.append(t)
            else:
                forecast = planner.plan(control)
            poses.append(forecast.current_pose)
        path = output / "references" / name
        path.mkdir(parents=True)
        motion_path = path / f"{name}_motion.npz"
        query_path = output / f"{name}_queries.npz"
        np.savez_compressed(motion_path, joint_pos=np.asarray(poses, dtype=np.float32),
                            fps=50., joint_names=JOINT_NAMES, robot_id=ROBOT_ID,
                            source_row=clock, upstream_frame=clock)
        np.save(path / f"{name}_terrain.npy", np.empty((0, 1, 10)))
        np.savez_compressed(query_path, command=command.astype(np.float32),
                            frames=np.asarray(queries), forecast=np.asarray(future, dtype=np.float32))
        manifest["clips"][f"{name}/{name}"] = dict(
            case=name, raw_motion=f"lmm-{name}", category=name, mirrored=False,
            reference_start_frame_50hz=0, reference_end_frame_50hz=steps,
            queries=query_path.name, changes=values.tolist(), scenario_steps=scenario_steps,
            motion_sha256=digest(motion_path), queries_sha256=digest(query_path))
        print(f"LMM scenario {name}: {steps} simulated steps, {len(queries)} replan queries", flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def read_scenarios(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or not manifest.get("clips"):
        raise ValueError(f"Expected a generated LMM scenario library at {directory}")
    episodes = {}
    for key, part in manifest["clips"].items():
        queries = directory / part["queries"]
        motion = directory / "references" / f"{key}_motion.npz"
        if ("queries_sha256" in part and digest(queries) != part["queries_sha256"]
                or "motion_sha256" in part and digest(motion) != part["motion_sha256"]):
            raise ValueError(f"Command scenario files changed after generation: {key}")
        episodes[key] = dict(part, queries=str(queries))
    return directory / "references", episodes
