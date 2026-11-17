"""Collect immutable, frame-zero box expert episodes; never publish automatically.

Use the archived roban-skills source with the dedicated collection environment.
Geometry groups and splits are fixed before simulation or quality filtering.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/roban/skills_box_climb_b02c575f/source/roban_skills"
sys.path[:0] = [str(SOURCE / "src"), str(SOURCE / "scripts")]
from verify import read_actor
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.parkour.random_box_tracking import (
    RANDOM_BOX_ABI, RandomTrackingBoxCfg, motion_term, roban_random_box_tracking_env_cfg,
)
from mjlab.tasks.parkour.reference_course import paired_box
from mjlab.utils.torch import configure_torch_backends

JOINTS = ["waist_yaw_joint"] + [f"leg_{s}{j}_joint" for s in "lr" for j in range(1, 7)] + [f"zarm_{s}{j}_joint" for s in "lr" for j in range(1, 5)]
PLANNED = []
CONTACT_TIMECONST = None


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def make_plan(path, seed=20261110):
    rng = np.random.default_rng(seed)
    rows = []
    # Pilot includes all three strata; subsequent groups remain independent.
    for block in range(20):
        for stratum, number in (("regular", 8), ("edge", 2), ("initial", 2)):
            for j in range(number):
                x = rng.uniform(-1, 1, 4)
                if stratum == "regular":
                    x *= (.35, .65, 1.)[block % 3]
                elif stratum == "edge":
                    dims = rng.choice(4, size=2, replace=False)
                    x[dims] = rng.choice([-1., 1.], size=2) * rng.uniform(.85, 1., 2)
                else:
                    x *= .65
                index = sum(r["stratum"] == stratum for r in rows)
                init = np.zeros(24)
                if stratum == "initial":
                    init[:2] = rng.uniform(-.01, .01, 2)
                    init[2] = rng.uniform(-np.deg2rad(2), np.deg2rad(2))
                    init[3:] = rng.uniform(-np.deg2rad(.5), np.deg2rad(.5), 21)
                rows.append(dict(candidate=f"c{len(rows):04d}", group=f"box_v1_{stratum}_{index:04d}",
                                 stratum=stratum, split="holdout" if index % 4 == 3 else "train",
                                 dimension_delta_m=(x[:3] * [.1, .05, .05]).tolist(),
                                 yaw_delta_rad=float(x[3] * np.deg2rad(15)), initial_delta=init.tolist()))
    save(path, dict(seed=seed, source_commit="b02c575f", quotas={"regular": [36, 12], "edge": [12, 4], "initial": [12, 4]}, candidates=rows))


@dataclass
class PlannedBoxCfg(RandomTrackingBoxCfg):
    def function(self, difficulty, spec, rng):
        result = super().function(difficulty, spec, rng)
        if CONTACT_TIMECONST is not None:
            for item in result.geometries:
                if item.geom is not None:
                    item.geom.solref[:] = (CONTACT_TIMECONST, 1.)
        return result

    def sample_box(self, difficulty, rng):
        row = PLANNED.pop(0)
        center, half, yaw = paired_box()
        delta = np.array(row["dimension_delta_m"])
        center[2] += delta[2] / 2
        return center, half + delta / 2, yaw + row["yaw_delta_rad"]


class FirstEpisode(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.armed = False
        self.active = {}
        self.records = []

    def snapshot(self):
        robot = self._env.scene["robot"]
        ids, names = robot.find_joints(JOINTS, preserve_order=True)
        assert list(names) == JOINTS
        data = robot.data
        qp = torch.cat((data.data.qpos[:, data.indexing.free_joint_q_adr], data.joint_pos[:, ids]), -1)
        out = {"qpos": qp, "qvel": data.data.qvel}
        for name in ("feet_contact", "hands_contact", "all_body_contact"):
            sensor = self._env.scene[name]
            for field in ("found", "force", "dist", "pos"):
                value = getattr(sensor.data, field, None)
                if value is not None:
                    out[f"{name}_{field}"] = value
        return {k: v.detach().cpu().numpy().copy() for k, v in out.items()}

    def begin(self, rows, output):
        self.out = output
        snap = self.snapshot()
        term = motion_term(self._env)
        pos, quat, size = term.obstacle()
        for i, row in enumerate(rows):
            self.active[i] = dict(meta={**row, "env": i, "terrain_type": "box_climb",
                                       "terrain_row": i % 4, "terrain_col": i // 4,
                                       "terrain_origin": self._env.scene.env_origins[i].tolist(),
                                       "box_center_local": (pos[i] - self._env.scene.env_origins[i]).tolist(),
                                       "box_full_size_m": (size[i] * 2).tolist(),
                                       "box_yaw_rad": float(2 * torch.atan2(quat[i, 3], quat[i, 0]))},
                                  states={k: [v[i]] for k, v in snap.items()}, actions=[])
        self.armed = True

    def append(self, ids):
        if not ids:
            return
        snap = self.snapshot()
        action = self._env.action_manager.action.cpu().numpy().copy()
        for i in ids:
            ep = self.active[i]
            for k, v in snap.items():
                ep["states"][k].append(v[i])
            ep["actions"].append(action[i])

    def record_pre_reset(self, env_ids):
        if not self.armed:
            return
        ids = [i for i in env_ids.tolist() if i in self.active]
        self.append(ids)  # Capture final qpos BEFORE the environment overwrites it.
        term = motion_term(self._env)
        for i in ids:
            ep = self.active.pop(i)
            failed = bool(self._env.termination_manager.get_term("failure")[i])
            ended = bool(self._env.termination_manager.get_term("reference_end")[i])
            full = int(term.start_frame[i]) == 0
            supported = bool(term.ever_top[i])
            qp = np.array(ep["states"]["qpos"])
            meta = {**ep["meta"], "frames": len(qp), "steps": int(self._env.episode_length_buf[i]),
                    "start_frame": int(term.start_frame[i]), "end_frame": int(term.frame[i]),
                    "failure": failed, "reference_end": ended, "physical_top_support": supported,
                    "source_success": full and ended and supported and not failed and bool(np.isfinite(qp).all()),
                    "mean_body_error_m": float(term.error_sum[i] / term.sample_count[i].clamp_min(1)),
                    "termination_reason": "failure" if failed else "reference_end" if ended else "timeout"}
            folder = self.out / "raw" / meta["candidate"]
            folder.mkdir(parents=True)
            np.savez_compressed(folder / "states.npz", **{k: np.array(v) for k, v in ep["states"].items()},
                                actions=np.array(ep["actions"]), joint_names=np.array(JOINTS),
                                action_joint_names=np.array(JOINTS), time=np.arange(len(qp)) * self._env.step_dt,
                                reference_frame=np.arange(len(qp)),
                                all_body_contact_names=np.array(self._env.scene["all_body_contact"].primary_names))
            save(folder / "episode.json", meta)
            self.records.append(meta)
            print("EPISODE " + json.dumps(meta), flush=True)

    def record_post_step(self):
        if self.armed:
            self.append(list(self.active))


def terrain_snapshot(env, out):
    m = env.sim.mj_model
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mujoco.mj_saveModel(m, str(out / "source_model.mjb"))
    ids = np.array(env.scene.terrain.indexing.geom_ids.cpu())
    ids = ids[(m.geom_contype[ids] != 0) | (m.geom_conaffinity[ids] != 0)]
    quat = np.zeros((len(ids), 4))
    for i, g in enumerate(ids):
        mujoco.mju_mat2Quat(quat[i], d.geom_xmat[g])
    np.savez_compressed(out / "terrain_snapshot.npz", geom_id=ids,
                        names=np.array([m.geom(int(g)).name for g in ids]), types=m.geom_type[ids],
                        pos=d.geom_xpos[ids], quat=quat, size=m.geom_size[ids], dataid=m.geom_dataid[ids],
                        hfield_data=m.hfield_data, hfield_adr=m.hfield_adr, hfield_size=m.hfield_size,
                        hfield_nrow=m.hfield_nrow, hfield_ncol=m.hfield_ncol)
    save(out / "model_metadata.json", dict(nq=m.nq, nv=m.nv, timestep=m.opt.timestep,
                                          joints=[m.joint(i).name for i in range(m.njnt)],
                                          qvel_dof_joint_ids=m.dof_jntid.tolist()))


def main():
    global CONTACT_TIMECONST
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--physics-dt", type=float, default=.002)
    ap.add_argument("--contact-timeconst", type=float, default=None,
                    help="Explicit contact solver profile, applied to both robot and terrain; expert weights stay frozen")
    args = ap.parse_args()
    path = Path(args.plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        make_plan(path)
    plan = json.loads(path.read_text())
    rows = plan["candidates"][args.start:args.start + args.count]
    assert len(rows) == args.count and len(rows) % 4 == 0
    PLANNED.extend(rows)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / "candidate_plan.json", rows)
    torch.set_num_threads(1)
    configure_torch_backends()
    actor, _, checkpoint, _ = read_actor("box_climb", "cuda:0")
    sha = hashlib.sha256((SOURCE / "models/box_climb/model.pt").read_bytes()).hexdigest()
    assert sha == "d4fa9a4729d5b0d374345145e9d953a085f271d3f8840f021348c5f6d1e5a2f5"
    cfg = roban_random_box_tracking_env_cfg(play=True)
    cfg.sim.mujoco.timestep = args.physics_dt
    cfg.decimation = round(.02 / args.physics_dt)
    assert abs(cfg.decimation * args.physics_dt - .02) < 1e-8
    CONTACT_TIMECONST = args.contact_timeconst
    if CONTACT_TIMECONST is not None:
        assert CONTACT_TIMECONST >= 2 * args.physics_dt
        original_spec = cfg.scene.entities["robot"].spec_fn
        def contact_spec():
            spec = original_spec()
            for geom in spec.geoms:
                if geom.contype or geom.conaffinity:
                    geom.solref[:] = (CONTACT_TIMECONST, 1.)
            return spec
        cfg.scene.entities["robot"].spec_fn = contact_spec
    cfg.seed = plan["seed"]
    cfg.scene.num_envs = len(rows)
    gen = cfg.scene.terrain.terrain_generator
    gen.num_rows, gen.num_cols = 4, len(rows) // 4
    old = gen.sub_terrains["random_box"]
    gen.sub_terrains = {"random_box": PlannedBoxCfg(size=old.size, proportion=1.)}
    cfg.commands["base_path"].debug_vis = False
    cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
        name="all_body_contact", primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force", "pos", "dist"),
        reduce="maxforce", num_slots=1))
    cfg.recorders = {"collector": RecorderTermCfg(func=FirstEpisode)}
    started = time.monotonic()
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        assert not PLANNED
        wrapped = RslRlVecEnvWrapper(env)
        term = motion_term(env)
        assert checkpoint["infos"]["random_box_tracking_abi"] == RANDOM_BOX_ABI
        assert checkpoint["infos"]["reference_sha256"] == term.source_sha256
        ids = torch.arange(env.num_envs, device=env.device)
        terrain = env.scene.terrain
        terrain.terrain_levels[:] = ids % 4
        terrain.terrain_types[:] = ids // 4
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        env.reset(seed=plan["seed"], apply_curriculum=False)
        # Physical initial-state changes only; command reference and terrain stay fixed.
        robot = env.scene["robot"]
        delta = torch.tensor([r["initial_delta"] for r in rows], device=env.device)
        pose = robot.data.data.qpos[:, robot.data.indexing.free_joint_q_adr].clone()
        pose[:, :2] += delta[:, :2]
        c, s = torch.cos(delta[:, 2]/2), torch.sin(delta[:, 2]/2)
        w, x, y, z = pose[:, 3:7].unbind(-1)
        pose[:, 3:7] = torch.stack((c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w), -1)
        robot.write_root_link_pose_to_sim(pose)
        jids, _ = robot.find_joints(JOINTS, preserve_order=True)
        robot.write_joint_position_to_sim(robot.data.joint_pos[:, jids] + delta[:, 3:], joint_ids=jids)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.command_manager.compute(dt=0.)
        env._sense(force_camera=True)
        env.observation_manager.reset(ids)
        env.obs_buf = env.observation_manager.compute(update_history=True)
        terrain_snapshot(env, out)
        recorder = env.recorder_manager.get_term("collector")
        recorder.begin(rows, out)
        with torch.inference_mode():
            observation = wrapped.get_observations()
            for step in range(env.max_episode_length):
                assert all(torch.isfinite(v).all() for v in observation.values())
                action = actor(observation)
                assert torch.isfinite(action).all()
                observation, reward, _, _ = wrapped.step(action)
                if (step + 1) % 100 == 0:
                    print(f"PROGRESS {step+1}: {len(recorder.records)}/{len(rows)} done", flush=True)
                if not recorder.active:
                    break
        assert not recorder.active
        save(out / "collection_report.json", dict(source_commit="b02c575f", checkpoint_sha256=sha,
             reference_sha256=term.source_sha256, abi=RANDOM_BOX_ABI, control_dt=env.step_dt,
             physics_dt=args.physics_dt, contact_timeconst_s=CONTACT_TIMECONST,
             seed=plan["seed"], elapsed_s=time.monotonic()-started, episodes=recorder.records,
             success=sum(r["source_success"] for r in recorder.records),
             contact_sampling="At 50 Hz: per-body strongest terrain contact plus net hand/foot forces; exact ACP contacts recomputed offline."))
    finally:
        env.close()


if __name__ == "__main__":
    main()
