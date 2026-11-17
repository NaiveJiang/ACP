"""Generate explicitly synthetic standing fixtures and exercise the selected ACP robot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .robot import (ROBOT_ID, NUM_J, JOINT_NAMES, CONTACT_BODIES,
                    STAND_HEIGHT, default_joint_pos_list, contact_geom_ids, metadata)
from .make_labels import build_model, variant_labels, PER_VARIANT
from .motion import MotionLib
from .kinematics import RobotKinematics
from .mdp import MDP, OBS_DIM, STATE_DIM, XI_DIM, normalize_xi
from .networks import WorldModel, Policy
from .task import ContactCfg, ContactTask, PaperReward, PaperRewardCfg


def make_fixtures(out: Path):
    """Two clips allow both train and validation splits; these are not locomotion demonstrations."""
    out.mkdir(parents=True, exist_ok=True)
    model = build_model(np.zeros((0, 8)))
    data = mujoco.MjData(model)
    data.qpos[:] = np.r_[0., 0., STAND_HEIGHT, 1., 0., 0., 0., default_joint_pos_list()]
    mujoco.mj_kinematics(model, data)
    ground = model.geom("ground").id
    feet = sum(contact_geom_ids(model)[:2], [])
    distance = min(mujoco.mj_geomDistance(model, data, g, ground, 2., np.zeros(6)) for g in feet)
    data.qpos[2] -= distance
    for i in range(2):
        path = out / "synthetic_stand" / f"stand_{i}_motion.npz"
        path.parent.mkdir(exist_ok=True)
        state = np.tile(data.qpos, (151, 1))
        state[:, 0] += i * .1
        np.savez_compressed(path, joint_pos=state, joint_names=np.array(JOINT_NAMES), fps=np.array([50.]),
                            robot_id=np.array(ROBOT_ID), synthetic=np.array(True))
        np.save(str(path).replace("_motion.npz", "_terrain.npy"), np.zeros((0, 1, 10)))
        result = variant_labels((str(path), np.zeros((0, 8)), .005, .1))
        labels = {k: result[k][None] if k in PER_VARIANT else result[k] for k in result}
        labels.update(robot_id=np.array(ROBOT_ID), body_names=np.array(list(CONTACT_BODIES)),
                      link_names=np.array([v[0] for v in CONTACT_BODIES.values()]))
        target = Path(str(out) + "_contacts") / "synthetic_stand" / f"stand_{i}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(f"data/{ROBOT_ID}_smoke"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--skip-sim", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.steps <= 150:
        ap.error("--steps must be between 1 and 150 for these 151-frame fixtures")
    torch.manual_seed(0)
    torch.set_num_threads(4)
    make_fixtures(args.out)
    dev = args.device
    kin = RobotKinematics(dev)
    motion = MotionLib.from_dir(str(args.out), kin, dev)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
    ctc = ContactTask(ContactCfg(), mdp, str(args.out), dev)
    f = motion.clip_start
    v = torch.zeros(len(f), dtype=torch.long, device=dev)
    S = motion.ref_state(f)
    a = torch.zeros(len(f), NUM_J, device=dev)
    xi = torch.zeros(len(f), XI_DIM, device=dev)
    xi[:, 0] = 1.
    cmd = ctc.command(S, f, v, ctc.sample_goal_steps(len(f)), noise=False)
    reward = PaperReward(PaperRewardCfg(), ctc)
    wm = WorldModel(mdp.scan_dim, hidden=(64, 64)).to(dev)
    prediction = wm.step(S, a, normalize_xi(xi), mdp.scan(S, motion.boxes(f, v)))
    loss = -reward(prediction, S, f, v, a, a, a, xi).mean()
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in wm.parameters() if p.grad is not None):
        raise RuntimeError("Non-finite gradients through world model/contact reward")
    pi = Policy(OBS_DIM, ctc.cmd_dim).to(dev)
    actions = pi(mdp.obs(S, a, xi)[:, None].expand(-1, 10, -1), cmd)
    assert actions.shape == (2, NUM_J) and torch.isfinite(actions).all()
    report = {"robot": metadata(), "synthetic_fixture": str(args.out), "state_dim": STATE_DIM,
              "obs_dim": OBS_DIM, "xi_dim": XI_DIM, "command_dim": ctc.cmd_dim,
              "world_model_input": wm.in_dim, "gradient_finite": True, "sim_steps": 0}
    if not args.skip_sim:
        from .env import TrackingSim
        env = TrackingSim(2, motion, mdp, dev, eval_mode=True, reset_noise=False)
        env.reset(torch.arange(2, device=dev), frames=motion.clip_start, variants=v)
        report["actuator_count"] = env.sim.mj_model.nu
        names = [env.sim.mj_model.joint(int(j)).name.split("/")[-1]
                 for j in env.sim.mj_model.actuator_trnid[:, 0]]
        assert names == JOINT_NAMES, names
        terminated, heights = 0, []
        with torch.no_grad():
            for _ in range(args.steps):
                result = env.step(a)
                if not all(torch.isfinite(x).all() for x in env.state()):
                    raise RuntimeError("Non-finite simulator state")
                if not torch.isfinite(result["r"]).all():
                    raise RuntimeError("Non-finite simulator reward")
                terminated += int(result["done"].sum())
                heights.extend(env.state()[0][:, 2].tolist())
        report.update(sim_steps=args.steps, num_envs=2, done_transition_count=terminated,
                      failed_envs=int(env.failed_eval.sum()), controller="zero action / nominal joint targets; no balance policy",
                      root_height_range=[min(heights), max(heights)], simulated_state_finite=True)
    target = args.out / "smoke_report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "robot"}, ensure_ascii=False, indent=2))
    print(f"Report: {target}")


if __name__ == "__main__":
    main()
