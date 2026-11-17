"""Collect true body-frame backward, lateral and walking-turn source rollouts."""
from dataclasses import dataclass, asdict
import argparse
import shutil
import time

import numpy as np
import torch
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from collect_roban_acp import *

PROFILES = (
    ("flat_backward", (-.25, 0., 0.)),
    ("flat_left", (0., .22, 0.)),
    ("flat_right", (0., -.22, 0.)),
    ("flat_turn_left", (.30, 0., .20)),
    ("flat_turn_right", (.30, 0., -.20)),
)


class DirectionCommand(UniformVelocityCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        commands = torch.tensor([p[1] for p in PROFILES], device=self.device)
        self.vel_command_b[env_ids] = commands[env_ids % len(PROFILES)]
        self.sampled_vel_command_b[env_ids] = self.vel_command_b[env_ids]
        _,quat=self._written_root_pose(env_ids)
        self.heading_target[env_ids]=self._heading_z(quat)

    def _update_command(self):
        super()._update_command()
        ids=torch.arange(self.num_envs,device=self.device)
        turn=torch.tensor(["turn" in p[0] for p in PROFILES],device=self.device)[ids%len(PROFILES)]
        _,quat=self._written_root_pose(ids)
        err=self.heading_target-self._heading_z(quat)
        err=torch.atan2(torch.sin(err),torch.cos(err))
        self.vel_command_b[~turn,2]=(.8*err[~turn]).clamp(-.3,.3)


@dataclass(kw_only=True)
class DirectionCfg(UniformVelocityCommandCfg):
    def build(self, env):
        return DirectionCommand(self, env)


class DirectionRecorder(EpisodeRecorder):
    def qualify(self, meta, ep):
        name, command = PROFILES[meta["env_id"] % len(PROFILES)]
        q = np.asarray(ep["qpos"])
        qw, qx, qy, qz = q[:, 3:7].T
        yaw = np.unwrap(np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz)))
        v = (q[10:, :2]-q[:-10, :2])/.2
        angle = yaw[:-10]
        vx = np.cos(angle)*v[:, 0]+np.sin(angle)*v[:, 1]
        vy = -np.sin(angle)*v[:, 0]+np.cos(angle)*v[:, 1]
        mx, my = float(vx[25:].mean()), float(vy[25:].mean())
        dyaw = float(yaw[-1]-yaw[0])
        correct = {"flat_backward": mx < -.06 and abs(my)<abs(mx)*.5 and abs(dyaw)<.4,
                   "flat_left": my > .06 and abs(mx)<my*.5 and abs(dyaw)<.4,
                   "flat_right": my < -.06 and abs(mx)<abs(my)*.5 and abs(dyaw)<.4, "flat_turn_left": mx > .15 and dyaw > 1.,
                   "flat_turn_right": mx > .15 and dyaw < -1.}[name]
        first = meta["first_done"]
        meta.update(source_terrain_type="flat", terrain_type=name, requested_twist=command,
                    measured_body_velocity=[mx, my], heading_change_rad=dyaw,
                    source_success=bool(not meta["incomplete"] and first and not first["terminated"]
                                        and meta["finite"] and len(q) >= 590 and correct))


def main():
    global PROFILES
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6500)
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--speed-scale",type=float,default=1.)
    ap.add_argument("--profile-json")
    args = ap.parse_args()
    if args.profile_json:
        import json
        PROFILES=tuple((p[0],tuple(p[1])) for p in json.loads(Path(args.profile_json).read_text()))
    PROFILES=tuple((name,(vx*args.speed_scale,vy*args.speed_scale,w)) for name,(vx,vy,w) in PROFILES)
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    cfg, rl = load_env_cfg(TASK, play=True), load_rl_cfg(TASK)
    cfg.seed, cfg.scene.num_envs, cfg.episode_length_s = args.seed, 20, 12.
    cfg.sim.mujoco.timestep, cfg.decimation = .0025, 8
    cfg.curriculum = {}
    gen = cfg.scene.terrain.terrain_generator
    gen.sub_terrains = {"flat": gen.sub_terrains["flat"]}
    gen.num_rows, gen.num_cols, gen.seed, gen.difficulty_range = 2, 4, args.seed, (0., 0.)
    cfg.events["randomize_terrain"].func = fixed_terrain
    cfg.events["randomize_terrain"].params = {}
    cfg.events = {k:v for k,v in cfg.events.items() if v.mode != "startup"}
    cfg.scene.entities["robot"].init_state.pos = (0., 0., .81)
    cfg.events["reset_base"].params["pose_range"].update(pitch=(0.,0.),roll=(0.,0.),yaw=(-3.14,3.14))
    cfg.events["reset_base"].params["velocity_range"] = {k:(0.,0.) for k in cfg.events["reset_base"].params["velocity_range"]}
    cfg.events["reset_robot_joints"].params["position_range"] = (1.,1.)
    cmd = DirectionCfg(**vars(cfg.commands["twist"]))
    cmd.use_pose_targets = cmd.heading_command = False
    cmd.terrain_turn_fraction = cmd.init_velocity_prob = 0.
    cmd.rel_standing_envs = cmd.rel_zero_envs = cmd.rel_lateral_envs = cmd.rel_yaw_envs = cmd.rel_world_envs = cmd.rel_forward_envs = 0.
    cmd.spec_velocity_terrain, cmd.spec_velocity_ranges = (), {}
    cmd.resampling_time_range = (100.,100.)
    cfg.commands["twist"] = cmd
    cfg.recorders["acp"] = RecorderTermCfg(func=DirectionRecorder, params={"out":str(out/"raw")})
    save_json(out/"run.json",dict(args=vars(args),profiles=PROFILES,checkpoint=str(CHECKPOINT),checkpoint_sha256=digest(CHECKPOINT),raw_schema_version=2))
    save_json(out/"environment.json",cfg);shutil.copy2(__file__,out/"collector.py")
    raw=ManagerBasedRLAMPEnv(cfg=cfg,device="cuda:0");env=RslRlVecEnvWrapper(raw,clip_actions=rl.clip_actions)
    runner=load_runner_cls(TASK)(env,asdict(rl),device="cuda:0")
    runner.load(str(CHECKPOINT),load_cfg={"actor":True},strict=True,map_location="cuda:0")
    policy=runner.get_inference_policy(device="cuda:0");terrain_snapshot(raw,out)
    rec=raw.acp_recorder;rec.ready=True;obs,_=env.reset();rec.just_reset.clear();start=time.time()
    try:
        with torch.inference_mode():
            for step in range(1,args.steps+1):
                rec.step_index=step;obs,_,done,_=env.step(policy(obs));policy.reset(done)
                if step%200==0:print(f"DIRECTIONS {step} {time.time()-start:.1f}s {dict(rec.counts)}",flush=True)
    finally:
        env.close();save_json(out/"run_complete.json",dict(steps=rec.step_index,counts=dict(rec.counts)))


if __name__=="__main__":main()
