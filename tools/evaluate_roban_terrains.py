"""Closed-loop source-policy terrain evaluation. Run in .venv_roban_collect.

This records source rollouts; ACP export and contact validation are separate.
Courses preserve source geometry and
replace random targets with explicit traversals. Two geometry instances per
terrain/difficulty, five reset yaw trials per instance by default.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
import mujoco

from collect_roban_acp import (
    ROOT, SOURCE, CHECKPOINT, TASK, EpisodeRecorder, digest, save_json,
    terrain_snapshot, ManagerBasedRLAMPEnv, RslRlVecEnvWrapper,
    RecorderTermCfg, TerminationTermCfg, load_env_cfg, load_rl_cfg, load_runner_cls,
)
from mjlab.terrains.terrain_generator import SubTerrainCfg, FlatPatchSamplingCfg
from mjlab.terrains.config import stepping_stones, narrow_beams, random_spread_boxes

LEVELS = (0., .3, .6, 1.)
NAMES = ("flat", "random_rough", "hf_pyramid_slope", "hf_pyramid_slope_inv",
         "hf_pyramid_slope_narrow", "hf_pyramid_slope_inv_narrow", "threshold",
         "stepping_stones", "narrow_beams", "random_spread_boxes")
GEOMETRIES = []
COLLECTION = False


def difficulty_for(name, level):
    if COLLECTION:
        if name == "random_spread_boxes":
            return (0., .1, .2, .3)[level]
        if name == "narrow_beams":
            return (0., .2, .4, .6)[level]
        if "slope" in name:
            return (.3, .5, .75, 1.)[level]
    return LEVELS[level]


@dataclass(kw_only=True)
class CourseCfg(SubTerrainCfg):
    base: SubTerrainCfg
    name: str
    stone_gap_sweep: bool = False

    def function(self, difficulty, spec, rng):
        level = min(3, int(difficulty * 4))
        d = difficulty_for(self.name, level)
        base = deepcopy(self.base)
        base.size, base.flat_patch_sampling = self.size, None
        controlled_gap = None
        if self.name == "stepping_stones" and self.stone_gap_sweep:
            controlled_gap = (.05, .10, .15, .20)[level]
            size = 7.5 / 7 - controlled_gap
            base.stone_size_range = (size, size)
            # Keep eight grid centers; the generator's upper distance is unused.
            base.stone_distance_range = (controlled_gap - .001, controlled_gap)
            base.stone_height_variation = base.stone_size_variation = base.displacement_range = 0.
        result = base.function(d, spec, rng)
        start = result.origin.copy()
        goal = np.array([7.45, 4., 0.])
        if self.name == "stepping_stones":
            # From central platform to the second full stone along a grid row.
            pitch = 7.5 / 7
            start[1] = .25 + 3 * pitch
            goal = np.array([.25 + 6 * pitch, start[1], .2])
        elif self.name == "narrow_beams":
            goal = np.array([7.45, 4., .2])
        elif self.name.endswith("_narrow") and d >= .3:
            goal = np.array([7.65, 4., 0.])
        if self.name == "random_spread_boxes":
            # The source excludes boxes using unrotated bounds; rotated corners
            # can intrude into its central spawn square. Move the start, without
            # deleting obstacles, to a point with a 40 cm obstacle-free disk.
            candidates = np.array([(x, y) for x in np.arange(2.5, 4.51, .05)
                                   for y in np.arange(2.5, 5.51, .05)])
            clearance = np.full(len(candidates), np.inf)
            for tg in result.geometries:
                g = tg.geom
                if g is None or g.type != mujoco.mjtGeom.mjGEOM_BOX or g.pos[2] + g.size[2] <= 1e-6:
                    continue
                yaw = 2 * np.arctan2(g.quat[3], g.quat[0])
                rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                q = np.abs((candidates - np.asarray(g.pos)[:2]) @ rot) - np.asarray(g.size)[:2]
                sdf = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0)
                clearance = np.minimum(clearance, sdf)
            valid = clearance >= .4
            if not valid.any():
                raise ValueError("No collision-free box-course spawn with 0.4 m clearance")
            choices = candidates[valid]
            start[:2] = choices[np.argmin(np.linalg.norm(choices - start[:2], axis=1))]
            goal[1] = start[1]
        result.origin = start
        result.flat_patches = {"target": goal[None]}
        geometry = []
        for g in result.geometries:
            if g.geom is not None:
                geometry.append(dict(type=int(g.geom.type), pos=list(g.geom.pos),
                                     size=list(g.geom.size), quat=list(g.geom.quat)))
        GEOMETRIES.append(dict(terrain=self.name, level=level, difficulty=d,
                               local_start=start, local_goal=goal, geometry=geometry,
                               controlled_gap_m=controlled_gap))
        return result


def assigned_terrain(env, env_ids, **kwargs):
    del kwargs
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    t = env.scene.terrain
    n = len(NAMES)
    for i, name in enumerate(NAMES):
        ids = env_ids[env_ids % n == i]
        cols = [c for c, x in enumerate(t.terrain_type_names) if x == name]
        t.terrain_levels[ids] = (ids // n) % len(LEVELS)
        columns = torch.tensor(cols, device=env.device)
        t.terrain_types[ids] = columns[(ids // (n * len(LEVELS))) % len(cols)]
    t.env_origins[env_ids] = t.terrain_origins[t.terrain_levels[env_ids], t.terrain_types[env_ids]]


def course_complete(env):
    if not hasattr(env, "acp_arrival_step"):
        env.acp_arrival_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    robot = env.scene["robot"].data
    p = robot.data.qpos[:, robot.indexing.free_joint_q_adr[:3]]
    cmd = env.command_manager.get_term("twist")
    goal, start = cmd.pos_command_w[:, :2], cmd.pose_target_start_w[:, :2]
    route, traveled = goal - start, p[:, :2] - start
    length = route.norm(dim=-1).clamp_min(.1)
    along = (route * traveled).sum(-1) / length.square()
    lateral = (route[:, 0] * traveled[:, 1] - route[:, 1] * traveled[:, 0]).abs() / length
    reached = ((p[:, :2] - goal).norm(dim=-1) <= .15) | ((along >= 1) & (lateral <= .2))
    first = reached & (env.acp_arrival_step < 0)
    env.acp_arrival_step[first] = env.episode_length_buf[first]
    return (env.acp_arrival_step >= 0) & (env.episode_length_buf >= env.acp_arrival_step + 20)


def pit_fall(env):
    t = env.scene.terrain
    robot = env.scene["robot"].data
    z = robot.data.qpos[:, robot.indexing.free_joint_q_adr[2]]
    bad = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for i, name in enumerate(t.terrain_type_names):
        if name in ("stepping_stones", "narrow_beams"):
            bad |= (t.terrain_types == i) & (z < .55)
    return bad


class EvalRecorder(EpisodeRecorder):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.completed = Counter()
        self.quota = cfg.params["quota"]

    def record_post_reset(self, env_ids):
        super().record_post_reset(env_ids)
        for e in env_ids.tolist():
            if self.completed[e] >= self.quota:
                self.active.pop(e, None)
            elif e in self.active:
                self.active[e]["meta"]["trial"] = self.completed[e]
                meta=self.active[e]["meta"]
                meta["difficulty"] = difficulty_for(meta["terrain_type"],meta["terrain_row"])

    def finish(self, e, *, incomplete=False):
        if e not in self.active:
            return
        super().finish(e, incomplete=incomplete)
        if not incomplete:
            self.completed[e] += 1


def main():
    global NAMES, COLLECTION
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials-per-instance", type=int, default=5)
    ap.add_argument("--instances", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--max-steps", type=int, default=7500)
    ap.add_argument("--velocity", type=float, default=.45)
    ap.add_argument("--terrains", nargs="+", choices=NAMES, default=list(NAMES))
    ap.add_argument("--stone-gap-sweep", action="store_true", help="Controlled flat-topped stones with actual gaps 5/10/15/20 cm")
    ap.add_argument("--collection",action="store_true",help="Use the verified terrain ranges for reference data collection")
    ap.add_argument("--physics-dt",type=float,default=.005,choices=(.005,.0025))
    args = ap.parse_args()
    NAMES = tuple(args.terrains)
    COLLECTION=args.collection
    import mjlab
    assert Path(mjlab.__file__).is_relative_to(SOURCE), mjlab.__file__
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg, rl = load_env_cfg(TASK, play=True), load_rl_cfg(TASK)
    cfg.seed = args.seed
    cfg.scene.num_envs = len(NAMES) * len(LEVELS) * args.instances
    cfg.episode_length_s = 24.
    cfg.sim.mujoco.timestep, cfg.decimation = args.physics_dt,round(.02/args.physics_dt)
    cfg.curriculum = {}
    cfg.terminations["course_complete"] = TerminationTermCfg(func=course_complete, time_out=True)
    cfg.terminations["pit_fall"] = TerminationTermCfg(func=pit_fall)
    gen = cfg.scene.terrain.terrain_generator
    bases = gen.sub_terrains
    bases.update(stepping_stones=stepping_stones(), narrow_beams=narrow_beams(), random_spread_boxes=random_spread_boxes())
    gen.sub_terrains = {name: CourseCfg(base=bases[name], name=name, stone_gap_sweep=args.stone_gap_sweep,
        flat_patch_sampling={"target": FlatPatchSamplingCfg(num_patches=1)}) for name in NAMES}
    gen.num_rows, gen.num_cols = len(LEVELS), len(NAMES) * args.instances
    gen.curriculum_column_mode, gen.curriculum_difficulty_mode = "proportional", "uniform_bin"
    gen.seed, gen.difficulty_range = args.seed, (0., 1.)
    cfg.scene.terrain.max_init_terrain_level = len(LEVELS) - 1
    cfg.events["randomize_terrain"].func = assigned_terrain
    cfg.events["randomize_terrain"].params = {}
    cfg.scene.entities["robot"].init_state.pos = (0., 0., .81)
    reset = cfg.events["reset_base"].params
    reset["pose_range"].update(x=(0., 0.), y=(0., 0.), pitch=(0., 0.), roll=(0., 0.))
    reset["velocity_range"] = {k: (0., 0.) for k in reset["velocity_range"]}
    cfg.events["reset_robot_joints"].params["position_range"] = (1., 1.)
    cfg.events = {k: v for k, v in cfg.events.items() if v.mode != "startup"}
    cmd = cfg.commands["twist"]
    cmd.spec_velocity_terrain, cmd.spec_velocity_ranges = (), {}
    cmd.resampling_time_range = (100., 100.)
    cmd.rel_standing_envs = cmd.rel_zero_envs = cmd.rel_lateral_envs = cmd.rel_yaw_envs = 0.
    cmd.terrain_turn_fraction = 0.
    cmd.pose_target_random_velocity_terrain = ()
    cmd.pose_target_min_origin_dist = {}
    cmd.pose_target_min_start_dist = 0.
    cmd.pose_reset_yaw_range = (-.15, .15)
    cmd.ranges.lin_vel_x, cmd.ranges.lin_vel_y = (args.velocity, args.velocity), (0., 0.)
    cfg.recorders["acp"] = RecorderTermCfg(func=EvalRecorder, params={"out": str(out / "raw"), "quota": args.trials_per_instance})
    save_json(out / "run.json", dict(args=vars(args), levels=LEVELS, terrains=NAMES, task=TASK,
        checkpoint=str(CHECKPOINT), checkpoint_sha256=digest(CHECKPOINT), evaluator_sha256=digest(__file__),
        source_git_sha=subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip(),
        versions={p: importlib.metadata.version(p) for p in ("torch", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib")},
        note="Source closed-loop policy; nominal dynamics, no pushes, clean reset; fixed routes; requires ACP export and contact validation before training."))
    shutil.copy2(__file__, out / "evaluator.py")
    shutil.copy2(ROOT / "tools/collect_roban_acp.py", out / "collector_dependency.py")
    save_json(out / "environment.json", cfg)
    raw = ManagerBasedRLAMPEnv(cfg=cfg, device="cuda:0")
    env = RslRlVecEnvWrapper(raw, clip_actions=rl.clip_actions)
    runner = load_runner_cls(TASK)(env, asdict(rl), device="cuda:0")
    runner.load(str(CHECKPOINT), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    terrain_snapshot(raw, out)
    save_json(out / "courses.json", GEOMETRIES)
    recorder = raw.acp_recorder
    recorder.ready = True
    obs, _ = env.reset()
    recorder.just_reset.clear()
    start = time.time()
    try:
        with torch.inference_mode():
            for step in range(1, args.max_steps + 1):
                recorder.step_index = step
                obs, _, done, _ = env.step(policy(obs))
                if getattr(policy, "is_recurrent", False):
                    policy.reset(done)
                if step % 100 == 0:
                    print(f"EVAL step={step} elapsed={time.time()-start:.1f}s completed={sum(recorder.completed.values())}/{cfg.scene.num_envs * args.trials_per_instance}", flush=True)
                if all(recorder.completed[e] >= args.trials_per_instance for e in range(raw.num_envs)):
                    break
    finally:
        env.close()
        save_json(out / "run_complete.json", dict(steps=recorder.step_index, elapsed_s=time.time()-start,
            completed=dict(recorder.completed), counts=dict(recorder.counts)))


if __name__ == "__main__":
    main()
