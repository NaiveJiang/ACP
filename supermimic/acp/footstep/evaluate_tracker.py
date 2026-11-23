"""Run an oracle-conditioned tracker in mjlab on held-out reference blocks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--max-clips", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    if min(args.num_envs, args.steps) < 1 or args.max_clips < 0:
        p.error("Positive env/step counts and nonnegative max-clips required")
    if os.environ.get("ACP_ROBOT", "roban") != "roban":
        p.error("RoBan task only")
    os.environ["ACP_ROBOT"] = "roban"
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    import numpy as np
    from .reference import FootstepMotion
    from .backend import backend_for_format
    from .env import FootstepTrackingSim
    from ..robot import default_joint_pos_list, metadata
    from ..kinematics import RobotKinematics
    from ..mdp import unflatten_state
    from ...rotations import yaw_mat

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    backend = backend_for_format(ck.get("format"))
    joint=getattr(backend,'joint',False)
    FootstepTask, FootstepMDP, TrackerPolicy = backend.Task, backend.MDP, backend.Policy
    OBS_DIM, ContactRewardCfg = backend.obs_dim, backend.RewardCfg
    if ck["robot"]["fingerprint"] != metadata()["fingerprint"]:
        raise ValueError("Incompatible tracker checkpoint")
    config = ck["config"]
    kin = RobotKinematics(args.device)
    motion = FootstepMotion(ck["split"]["val"], kin, args.device, config["horizons"], args.max_clips,
                            reserve_steps=getattr(backend,"reserve_steps",0))
    task = FootstepTask(motion, kin, ContactRewardCfg(**ck["reward"]))
    mdp = FootstepMDP(kin, motion, torch.tensor(default_joint_pos_list(), device=args.device), task)
    policy = TrackerPolicy(config["hist"], config["policy_hidden"],**(dict(planner_config=config['planner_config']) if joint else {})).to(args.device)
    policy.load_state_dict(ck["pi"])
    policy.eval()
    env = getattr(backend,'Env',FootstepTrackingSim)(args.num_envs, motion, mdp, args.device, eval_mode=True, reset_noise=False)
    history = torch.zeros(args.num_envs, config["hist"], OBS_DIM, device=args.device)
    query = task.sample_queries(args.num_envs)
    reward, heights, positions, contact_matches, episodes = [], [], [], [], []
    velocities, commands, clip_ids = [], [], []
    falls = invalid = 0
    with torch.no_grad():
        for _ in range(args.steps):
            obs = env.observe()
            history = torch.where((env.ep_step == 0)[:, None, None], obs[:, None].expand_as(history),
                                  torch.cat((history[:, 1:], obs[:, None]), 1))
            if joint:query.copy_(task.pack_queries(env.f,query))
            command = task.command(env.state(), env.f, query)
            transition = env.step(policy(history, command))
            S = unflatten_state(transition["s1"])
            f1 = transition["f"]+1
            local_velocity = (yaw_mat(S[1]).transpose(-1, -2)@S[2].unsqueeze(-1)).squeeze(-1)
            velocities.append(torch.cat((local_velocity[:, :2], S[3][:, 2:3]), -1).cpu().numpy())
            # For contact-only controllers, recorded speed is diagnostic only.
            commands.append((command[:,:3] if joint else motion.velocity_command[transition["f"]]).cpu().numpy())
            clip_ids.append(motion.clip_id[transition["f"]].cpu().numpy())
            sole, _, gap = task.foot_geometry(S)
            support = motion.contact[f1]
            positions.extend(torch.linalg.vector_norm(sole-motion.feet[f1, :, :3], dim=-1)[support].tolist())
            contact_matches.append(float(((gap <= task.cfg.contact_distance) == support).float().mean()))
            reward.append(float(transition["r"].mean()))
            heights.extend(S[0][:, 2].tolist())
            falls += int(transition["term"].sum())
            invalid += int((~transition["valid"]).sum())
            ids = transition["done"].nonzero(as_tuple=False).squeeze(-1)
            if len(ids):
                episodes.extend(env.ep_step[ids].tolist())
                # eval_mode uses deterministic physics and explicit resets.
                env.reset(ids)
                query[ids] = task.sample_queries(len(ids))
    mean = lambda values: sum(values)/len(values) if values else None
    actual, desired, visited = np.concatenate(velocities), np.concatenate(commands), np.concatenate(clip_ids)
    def speed_metrics(mask):
        error = actual[mask]-desired[mask]
        return dict(transitions=int(np.sum(mask)), axis_order=["vx", "vy", "vyaw"],
                    command_mean=desired[mask].mean(axis=0).tolist(),
                    measured_mean=actual[mask].mean(axis=0).tolist(),
                    mae=np.abs(error).mean(axis=0).tolist(),
                    rmse=np.sqrt(np.mean(error**2, axis=0)).tolist())
    categories = np.asarray([motion.sources[i]["category"] for i in visited])
    report = dict(format="two_foot_oracle_tracker_eval_v1", controller_format=backend.format,
                  command_dim=backend.command_dim, speed_is_actor_input=(joint or backend.command_dim == 16),
                  checkpoint=str(Path(args.checkpoint).resolve()),
                  iteration=ck["iteration"], args=vars(args), held_out_clips=motion.names,
                  oracle_targets=not joint, completed_episodes=len(episodes), terminations=falls,
                  invalid_transitions=invalid, completed_episode_mean_steps=mean(episodes),
                  mean_reward_per_step=mean(reward), root_height_mean_m=mean(heights), root_height_min_m=min(heights),
                  support_position_mean_m=mean(positions), geometry_contact_match=mean(contact_matches),
                  visited_clips=int(len(np.unique(visited))),
                  speed_tracking=speed_metrics(np.ones(len(visited), dtype=bool)),
                  speed_tracking_per_category={c: speed_metrics(categories == c) for c in sorted(set(categories))},
                  note=("Learned planner closed loop. Foot-position/contact comparisons below are reference-motion diagnostics, not planned-target tracking accuracy."
                        if joint else "Geometric contact proxy; evaluation uses ground-truth reference targets, not a learned planner"))
    if joint:
        report.update(velocity_enters_planner=True,velocity_enters_controller=False,controller_command_dim=13)
        report['reference_support_position_mean_m']=report.pop('support_position_mean_m')
        report['reference_geometry_contact_match']=report.pop('geometry_contact_match')
    (out/"evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
