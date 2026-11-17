"""Collect real RoBan rollouts with the source mjlab fork; export separately in ACP.

Run with .venv_roban_collect/bin/python, not the ACP Python environment.
Raw episodes are never automatically promoted to the training dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
import hashlib
import functools
import importlib.metadata
import json
from pathlib import Path
import subprocess
import shutil
import time

import mujoco
import numpy as np
import torch

from mjlab.envs.amp.amp_env import ManagerBasedRLAMPEnv
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data_collect_repo/kuavo_amp_mjlab"
CHECKPOINT = SOURCE / "logs/rsl_rl/robans2_depth_amp_rough/2026-10-18_20-47-13_robans2_student_narrow_turn_10pct-cuda0/model_222600.pt"
TASK = "Mjlab-AMP-Depth-Rough-RobanS2"
CORE = ("flat", "pyramid_stairs", "pyramid_stairs_narrow", "pyramid_inv_stairs", "pyramid_inv_stairs_narrow", "variable_height_stairs_up", "variable_height_stairs_down")
JOINTS = ["waist_yaw_joint"] + [f"leg_{s}{j}_joint" for s in "lr" for j in range(1, 7)] + [f"zarm_{s}{j}_joint" for s in "lr" for j in range(1, 5)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(x):
    if isinstance(x, functools.partial):
        return {"partial": jsonable(x.func), "args": jsonable(x.args), "keywords": jsonable(x.keywords)}
    if is_dataclass(x):
        return {"_class": f"{type(x).__module__}.{type(x).__qualname__}", **{k: jsonable(v) for k, v in vars(x).items()}}
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [jsonable(v) for v in x]
    if isinstance(x, (str, float, int, bool)) or x is None:
        return x
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return x.tolist()
    if callable(x):
        return f"{x.__module__}.{x.__qualname__}"
    return str(x)


def save_json(path, x):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(x), indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def fixed_terrain(env, env_ids, **kwargs):
    """Stable env-to-type assignment, varying instances on reset; no curriculum."""
    del kwargs
    t = env.scene.terrain
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    names = list(dict.fromkeys(t.terrain_type_names))
    for i, name in enumerate(names):
        ids = env_ids[env_ids % len(names) == i]
        columns = torch.tensor([c for c, n in enumerate(t.terrain_type_names) if n == name], device=env.device)
        t.terrain_levels[ids] = torch.randint(t.terrain_origins.shape[0], (len(ids),), device=env.device)
        t.terrain_types[ids] = columns[torch.randint(len(columns), (len(ids),), device=env.device)]
    t.env_origins[env_ids] = t.terrain_origins[t.terrain_levels[env_ids], t.terrain_types[env_ids]]


def collection_complete(env):
    """End after a reached goal and 0.4 s real follow-through, never before 6 s."""
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
    arrived = ((p[:, :2] - goal).norm(dim=-1) <= .3) | ((along >= 1) & (lateral <= .4))
    first = arrived & (env.acp_arrival_step < 0)
    env.acp_arrival_step[first] = env.episode_length_buf[first]
    return (env.acp_arrival_step >= 0) & (env.episode_length_buf >= 300) & (env.episode_length_buf >= env.acp_arrival_step + 20)


class EpisodeRecorder(RecorderTerm):
    """One initial state plus one post-action state per step, including terminal.

    A termination-manager wrapper observes the first done *before* camera reset
    deferral clears the public flags. The callbacks track actual reset env IDs.
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.out = Path(cfg.params["out"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.active = {}
        self.counts = Counter()
        self.serial = 0
        self.just_reset = set()
        self.ready = False
        self.step_index = 0
        self.first_done = {}
        self.robot = env.scene["robot"]
        self.jids = [self.robot.joint_names.index(n) for n in JOINTS]
        old_compute = env.termination_manager.compute

        def compute():
            done = old_compute()
            for e in done.nonzero().flatten().tolist():
                if e not in self.first_done:
                    self.first_done[e] = {
                        "step": int(env.episode_length_buf[e]),
                        "terminated": bool(env.termination_manager.terminated[e]),
                        "time_out": bool(env.termination_manager.time_outs[e]),
                        "terms": [n for n in env.termination_manager.active_terms if bool(env.termination_manager.get_term(n)[e])],
                    }
                else:
                    self.first_done[e]["terminated"] |= bool(env.termination_manager.terminated[e])
            return done

        env.termination_manager.compute = compute
        env.acp_recorder = self

    def snapshot(self):
        d = self.robot.data
        # qpos is current even before forward; no stale terminal xpos/cvel.
        pose = d.data.qpos[:, d.indexing.free_joint_q_adr]
        q = d.joint_pos[:, self.jids]
        qp = torch.cat((pose, q), -1).detach().cpu().numpy().copy()
        qv = d.data.qvel.detach().cpu().numpy().copy()
        cmd = self._env.command_manager.get_term("twist")
        return qp, qv, cmd.vel_command_b.detach().cpu().numpy().copy()

    def record_post_reset(self, env_ids):
        ids = env_ids.tolist()
        if hasattr(self._env, "acp_arrival_step"):
            self._env.acp_arrival_step[env_ids] = -1
        self.just_reset.update(ids)
        qp, qv, cmd = self.snapshot()
        t = self._env.scene.terrain
        twist = self._env.command_manager.get_term("twist")
        for e in ids:
            self.first_done.pop(e, None)
            if not self.ready:
                continue
            self.serial += 1
            row, col = int(t.terrain_levels[e]), int(t.terrain_types[e])
            meta = {"episode_id": f"ep_{self.serial:06d}", "env_id": e, "terrain_type": t.terrain_type_names[col],
                    "terrain_row": row, "terrain_col": col, "terrain_origin": t.env_origins[e].tolist(),
                    "route_start": qp[e, :3].tolist(), "route_goal": twist.pos_command_w[e].tolist(),
                    "start_global_step": self.step_index, "fps": 1 / self._env.step_dt, "root_frame": "base_link"}
            self.active[e] = {"meta": meta, "qpos": [qp[e]], "qvel": [qv[e]], "commands": [cmd[e]], "actions": [], "feet_force": []}

    def append(self, ids):
        if not ids:
            return
        qp, qv, cmd = self.snapshot()
        actions = self._env.action_manager.action.detach().cpu().numpy().copy()
        forces = self._env.avg_feet_force.detach().cpu().numpy().copy()
        for e in ids:
            if e not in self.active:
                continue
            ep = self.active[e]
            ep["qpos"].append(qp[e])
            ep["qvel"].append(qv[e])
            ep["commands"].append(cmd[e])
            ep["actions"].append(actions[e])
            ep["feet_force"].append(forces[e])

    def finish(self, e, *, incomplete=False):
        ep = self.active.pop(e, None)
        if ep is None:
            return
        meta = ep.pop("meta")
        first = self.first_done.get(e)
        meta.update(first_done=first, reset_global_step=self.step_index, incomplete=incomplete)
        qp = np.asarray(ep["qpos"])
        start, goal = np.array(meta["route_start"]), np.array(meta["route_goal"])
        route = goal[:2] - start[:2]
        delta = qp[:, :2] - start[:2]
        along = delta @ route / max(np.dot(route, route), 1e-8)
        lateral = np.abs(delta[:, 0] * route[1] - delta[:, 1] * route[0]) / max(np.linalg.norm(route), 1e-8)
        distance = np.linalg.norm(qp[:, :2] - goal[:2], axis=1)
        meta.update(frames=len(qp), displacement=float(np.linalg.norm(qp[-1, :2]-qp[0, :2])),
                    min_goal_distance=float(distance.min()), max_route_progress=float(along.max()),
                    max_lateral_deviation=float(lateral.max()), finite=bool(np.isfinite(qp).all()))
        # Source course success is stricter than curriculum's progress latch.
        arrived = (distance <= .3) | ((along >= 1) & (lateral <= .4))
        meta["arrived_frame"] = int(np.flatnonzero(arrived)[0]) if arrived.any() else None
        meta["source_success"] = bool(not incomplete and first and not first["terminated"] and meta["finite"]
                                      and len(qp) >= 301 and arrived.any())
        self.qualify(meta, ep)
        category = meta["terrain_type"]
        self.counts[f"{category}/{'incomplete' if incomplete else 'attempts'}"] += 1
        if meta["source_success"]:
            self.counts[f"{category}/source_success"] += 1
        folder = self.out / meta["episode_id"]
        folder.mkdir()
        np.savez_compressed(folder / "states.npz", **{k: np.asarray(v) for k, v in ep.items()},
                            joint_names=np.array(JOINTS), time=np.arange(len(qp)) * self._env.step_dt)
        save_json(folder / "episode.json", meta)
        with (self.out.parent / "episodes.jsonl").open("a") as f:
            f.write(json.dumps(meta) + "\n")
        save_json(self.out.parent / "progress.json", {"step": self.step_index, "counts": dict(self.counts)})

    def qualify(self, meta, episode):
        """Optional task-specific acceptance before writing an immutable episode."""
        pass

    def record_pre_reset(self, env_ids):
        ids = env_ids.tolist()
        self.append(ids)
        for e in ids:
            self.finish(e)

    def record_post_step(self):
        self.append([e for e in self.active if e not in self.just_reset])
        self.just_reset.clear()

    def close(self):
        for e in list(self.active):
            self.finish(e, incomplete=True)


def terrain_snapshot(env, out):
    m, d = env.sim.mj_model, mujoco.MjData(env.sim.mj_model)
    mujoco.mj_forward(m, d)
    robot = env.scene["robot"]
    written = robot.data.data.qpos[:, robot.data.indexing.free_joint_q_adr]
    if not torch.allclose(written, robot.data.root_link_pose_w, atol=1e-5):
        raise ValueError("Free-joint pose is not the base_link world pose")
    mujoco.mj_saveModel(m, str(out / "source_model.mjb"))
    np.savez_compressed(out / "model_parameters.npz", body_mass=env.sim.model.body_mass.cpu().numpy(),
                        body_ipos=env.sim.model.body_ipos.cpu().numpy(),
                        geom_friction=env.sim.model.geom_friction.cpu().numpy(),
                        dof_armature=env.sim.model.dof_armature.cpu().numpy(),
                        default_joint_pos=robot.data.default_joint_pos.cpu().numpy())
    ids = np.asarray(env.scene.terrain.indexing.geom_ids.cpu())
    ids = ids[(m.geom_contype[ids] != 0) | (m.geom_conaffinity[ids] != 0)]
    quat = np.zeros((len(ids), 4))
    for i, g in enumerate(ids):
        mujoco.mju_mat2Quat(quat[i], d.geom_xmat[g])
    np.savez_compressed(out / "terrain_snapshot.npz", geom_id=ids,
                        names=np.array([m.geom(int(g)).name for g in ids]), types=m.geom_type[ids],
                        pos=d.geom_xpos[ids], quat=quat, size=m.geom_size[ids], dataid=m.geom_dataid[ids],
                        hfield_data=m.hfield_data, hfield_adr=m.hfield_adr, hfield_size=m.hfield_size,
                        hfield_nrow=m.hfield_nrow, hfield_ncol=m.hfield_ncol,
                        terrain_origins=env.scene.terrain.terrain_origins.cpu().numpy(),
                        terrain_names=np.array(env.scene.terrain.terrain_type_names))
    save_json(out / "terrain_summary.json", {"colliders": len(ids), "types": dict(Counter(map(int, m.geom_type[ids]))),
                                            "names": env.scene.terrain.terrain_type_names})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    ap.add_argument("--terrains", nargs="+", choices=CORE, default=list(CORE))
    ap.add_argument("--num-envs", type=int, default=28)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--difficulty", type=float, nargs=2, default=(0., .2))
    ap.add_argument("--episode-seconds", type=float, default=12.)
    ap.add_argument("--velocity", type=float, default=.45)
    ap.add_argument("--physics-dt", type=float, default=.005, choices=(.005, .0025), help="Physics step; control sampling remains 50 Hz")
    ap.add_argument("--nominal", action="store_true", help="Disable startup dynamics randomization")
    ap.add_argument("--clean-reset", action="store_true", help="Use nominal joints, zero velocity/pitch and 0.81 m root height instead of the training drop reset")
    args = ap.parse_args()
    import mjlab
    if not Path(mjlab.__file__).is_relative_to(SOURCE):
        raise RuntimeError(f"Wrong mjlab fork: {mjlab.__file__}")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg, rl = load_env_cfg(TASK, play=True), load_rl_cfg(TASK)
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.mujoco.timestep = args.physics_dt
    cfg.decimation = round(.02 / args.physics_dt)
    cfg.episode_length_s = args.episode_seconds
    cfg.terminations["collection_complete"] = TerminationTermCfg(func=collection_complete, time_out=True)
    cfg.curriculum = {}
    gen = cfg.scene.terrain.terrain_generator
    gen.sub_terrains = {k: v for k, v in gen.sub_terrains.items() if k in args.terrains}
    for sub in gen.sub_terrains.values():
        sub.proportion = 1.
    gen.num_rows, gen.num_cols = 2, 2 * len(args.terrains)
    gen.seed, gen.difficulty_range = args.seed, tuple(args.difficulty)
    cfg.scene.terrain.max_init_terrain_level = 1
    cfg.events["randomize_terrain"].func = fixed_terrain
    cfg.events["randomize_terrain"].params = {}
    if args.clean_reset:
        robot = cfg.scene.entities["robot"]
        robot.init_state.pos = (0., 0., .81)
        cfg.events["reset_base"].params["pose_range"]["pitch"] = (0., 0.)
        cfg.events["reset_base"].params["velocity_range"] = {k: (0., 0.) for k in cfg.events["reset_base"].params["velocity_range"]}
        cfg.events["reset_robot_joints"].params["position_range"] = (1., 1.)
    if args.nominal:
        cfg.events = {k: v for k, v in cfg.events.items() if v.mode != "startup"}
    cmd = cfg.commands["twist"]
    cmd.spec_velocity_terrain = tuple(n for n in cmd.spec_velocity_terrain if n in args.terrains)
    cmd.spec_velocity_ranges = {n: v for n, v in cmd.spec_velocity_ranges.items() if n in args.terrains}
    cmd.resampling_time_range = (100., 100.)
    cmd.rel_standing_envs = cmd.rel_zero_envs = cmd.rel_lateral_envs = cmd.rel_yaw_envs = 0.
    cmd.pose_target_random_velocity_terrain = ()
    cmd.ranges.lin_vel_x = (args.velocity, args.velocity)
    cmd.ranges.lin_vel_y = (0., 0.)
    for value in cmd.spec_velocity_ranges.values():
        value.update(lin_vel_x=(args.velocity, args.velocity), lin_vel_y=(0., 0.))
    cfg.recorders["acp"] = RecorderTermCfg(func=EpisodeRecorder, params={"out": str(out / "raw")})
    meta = {"raw_schema_version": 2, "args": vars(args), "task_id": TASK, "source_git_sha": subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip(),
            "collector_sha256": digest(__file__), "checkpoint_sha256": digest(args.checkpoint), "mjlab_file": mjlab.__file__,
            "versions": {p: importlib.metadata.version(p) for p in ("torch", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib")}}
    save_json(out / "run.json", meta)
    shutil.copy2(__file__, out / "collector.py")
    save_json(out / "environment.json", cfg)
    save_json(out / "runner.json", rl)
    raw = ManagerBasedRLAMPEnv(cfg=cfg, device="cuda:0")
    env = RslRlVecEnvWrapper(raw, clip_actions=rl.clip_actions)
    runner = load_runner_cls(TASK)(env, asdict(rl), device="cuda:0")
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    terrain_snapshot(raw, out)
    recorder = raw.acp_recorder
    recorder.ready = True
    obs, _ = env.reset()
    recorder.just_reset.clear()
    start = time.time()
    try:
        with torch.inference_mode():
            for step in range(1, args.steps + 1):
                recorder.step_index = step
                obs, _, done, _ = env.step(policy(obs))
                if getattr(policy, "is_recurrent", False) and callable(getattr(policy, "reset", None)):
                    policy.reset(done)
                if step % 100 == 0:
                    print(f"COLLECT step={step} elapsed={time.time()-start:.1f}s counts={dict(recorder.counts)}", flush=True)
    finally:
        env.close()
        save_json(out / "run_complete.json", {"steps": recorder.step_index, "elapsed_s": time.time()-start, "counts": dict(recorder.counts)})


if __name__ == "__main__":
    main()
