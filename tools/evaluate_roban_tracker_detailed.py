"""Instrument the existing reference/planner evaluator without changing the trainer.

Adds per-clip tracking/contact metrics, physics-overflow checks and optional
training-range domain randomization. Finished worlds are held above terrain
only on subsequent steps, after their terminal transition has been scored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

os.environ.setdefault("ACP_ROBOT", "roban")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--planner", type=Path)
    ap.add_argument("--eta-obs", type=float, default=1.)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", nargs="*", choices=["obs", "command", "action"], default=[])
    ap.add_argument("--goal-s", type=float, default=1.)
    ap.add_argument("--domain-randomization", action="store_true")
    ap.add_argument("--friction", type=float)
    ap.add_argument("--record", type=Path)
    ap.add_argument("--clip", help="Evaluate only this motion name, e.g. box_climb/clean_b02_48_c0074")
    ap.add_argument("--calibrate-contact", action="store_true")
    args = ap.parse_args()
    if not 0. <= args.eta_obs <= 1.:
        ap.error("--eta-obs must be between 0 and 1")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    import warp as wp
    import supermimic.acp.env as env_module
    import supermimic.acp.task as task_module
    from supermimic.acp.robot import NUM_J, COM_BODY, CONTACT_BODIES
    from supermimic.rotations import rot_angle_sq

    if args.clip:
        from supermimic.acp.motion import MotionLib

        def load_selected_clip(path, kin, device, pattern=None):
            file = Path(path) / (args.clip + "_motion.npz")
            if not file.is_file():
                raise FileNotFoundError(file)
            return MotionLib([str(file)], kin, device, root=path)

        MotionLib.from_dir = staticmethod(load_selected_clip)

    started = time.time()
    instances, distances, placements = [], [], []
    collecting = True
    original_sim = env_module.TrackingSim
    original_distance = task_module.ContactTask.body_distance
    original_placement = task_module.placement_error

    class InstrumentedSim(original_sim):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            instances.append(self)
            self.samples = []
            self.active_overflow = torch.zeros(self.N, dtype=torch.int32, device=self.device)
            self.inactive_overflow = torch.zeros_like(self.active_overflow)
            if args.domain_randomization:
                self.reset_noise = True
                self.xi[:, 0].uniform_(*env_module.FRICTION)
                self.xi[:, 1:1+NUM_J].uniform_(-env_module.JOINT_OFFSET, env_module.JOINT_OFFSET)
                com = torch.stack([torch.empty(self.N, device=self.device).uniform_(lo, hi)
                                   for lo, hi in env_module.COM_RANGE], -1)
                self.xi[:, 1+NUM_J:] = com
                torso = self.robot.indexing.body_ids[self.robot.find_bodies(COM_BODY)[0][0]]
                self.sim.model.body_ipos[:, torso] = self.sim.get_default_field("body_ipos")[torso] + com
                self.sim.model.geom_friction[:, self._terrain_geoms, 0] = env_module.FRICTION[0]
                self.sim.model.geom_friction[:, self.robot.indexing.geom_ids, 0] = self.xi[:, :1]
            if args.friction is not None:
                self.xi[:, 0] = args.friction
                self.sim.model.geom_friction[:, self._terrain_geoms, 0] = args.friction
                self.sim.model.geom_friction[:, self.robot.indexing.geom_ids, 0] = args.friction

        def step(self, action):
            alive = ~self.done_eval.clone()
            inactive = (~alive).nonzero(as_tuple=False).flatten()
            if len(inactive):
                pose = torch.zeros(len(inactive), 7, device=self.device)
                pose[:, 2], pose[:, 3] = 20., 1.
                self.robot.write_root_link_pose_to_sim(pose, env_ids=inactive)
                self.robot.write_root_link_velocity_to_sim(torch.zeros(len(inactive), 6, device=self.device), env_ids=inactive)
                self.robot.write_joint_position_to_sim(self.default_q.expand(len(inactive), -1), env_ids=inactive)
                self.robot.write_joint_velocity_to_sim(torch.zeros(len(inactive), NUM_J, device=self.device), env_ids=inactive)
                action = action.clone()
                action[inactive] = 0.
            overflow = wp.to_torch(self.sim.wp_data.overflow)
            overflow.zero_()
            out = super().step(action)
            self.active_overflow |= torch.where(alive, overflow, 0)
            self.inactive_overflow |= torch.where(alive, 0, overflow)
            p, R, v, w, q, qd = self.state()
            packed = torch.cat((p, R.flatten(-2), v, w, q, qd), -1)
            self.samples.append((packed, out["f"].clone(), alive, out["r"].clone(), action.clone()))
            if len(self.samples) % 250 == 0:
                print(f"progress control_step={len(self.samples)} active={int((~self.done_eval).sum())}", flush=True)
            return out

    def distance_hook(self, *a, **kw):
        result = original_distance(self, *a, **kw)
        if collecting:
            distances.append(result.detach().clone())
        return result

    def placement_hook(*a, **kw):
        result = original_placement(*a, **kw)
        if collecting:
            placements.append(result.detach().clone())
        return result

    env_module.TrackingSim = InstrumentedSim
    task_module.ContactTask.body_distance = distance_hook
    task_module.placement_error = placement_hook
    raw_path = args.out.with_name(args.out.stem + "_raw.json")
    sys.argv = ["supermimic.acp.evaluate", "--ckpt", str(args.ckpt.resolve()), "--out", str(raw_path),
                "--seed", str(args.seed), "--goal_s", str(args.goal_s)]
    if args.planner:
        sys.argv += ["--planner", str(args.planner.resolve()), "--eta_obs", str(args.eta_obs)]
    if args.noise:
        sys.argv += ["--train_noise", *args.noise]
    if args.record:
        sys.argv += ["--record", str(args.record)]
    ns = runpy.run_module("supermimic.acp.evaluate", run_name="__main__")
    collecting = False
    env, ctc, motion = instances[0], ns["ctc"], ns["motion"]
    assert len(distances) == len(placements) == len(env.samples)
    packed, before_f, alive, reward, actions = [torch.stack([sample[i] for sample in env.samples]) for i in range(5)]
    del env.samples
    T, C = alive.shape
    after_f = before_f + 1
    touch = (torch.stack(distances) <= ctc.cfg.touch_dist) & alive[..., None]
    label = ctc.labels[after_f, env.v[None, :]] & alive[..., None]
    pe = torch.stack(placements)
    previous_touch = torch.cat((torch.ones_like(touch[:1]), touch[:-1]), 0)
    onsets = touch & ~previous_touch
    valid_onsets = onsets & pe.isfinite()
    pe = torch.where(valid_onsets, pe, torch.nan)
    foot_ids = [list(CONTACT_BODIES).index(n) for n in ("left_foot", "right_foot")]
    link_ids = ctc.link_idx[foot_ids]
    valid_indices = alive.flatten().nonzero(as_tuple=False).flatten()
    metric_names = ["root_xy_error_m", "root_z_abs_error_m", "root_orientation_error_rad", "root_yaw_error_rad",
                    "root_linear_velocity_error_m_s", "joint_angle_rmse_rad",
                    "left_foot_link_world_error_m", "right_foot_link_world_error_m",
                    "left_foot_link_local_error_m", "right_foot_link_local_error_m",
                    "left_foot_orientation_error_rad", "right_foot_orientation_error_rad",
                    "left_foot_link_horizontal_speed_m_s", "right_foot_link_horizontal_speed_m_s"]
    values = torch.full((T*C, len(metric_names)), torch.nan, device=env.device)
    with torch.no_grad():
        for indices in valid_indices.split(512):
            x = packed.reshape(T*C, -1)[indices]
            p, R, v, w, q, qd = x[:, :3], x[:, 3:12].view(-1, 3, 3), x[:, 12:15], x[:, 15:18], x[:, 18:18+NUM_J], x[:, 18+NUM_J:]
            f = after_f.flatten()[indices]
            rp, rR = motion.root_p[f], motion.root_R[f]
            bp, bR, bv, _ = env.mdp.kin.forward(p, R, v, w, q, qd)
            feet, rfeet = bp[:, link_ids], motion.body_p[f][:, link_ids]
            yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0]) - torch.atan2(rR[:, 1, 0], rR[:, 0, 0])
            foot_world = (feet-rfeet).norm(dim=-1)
            foot_local = (task_module.to_base(feet,p,R)-task_module.to_base(rfeet,rp,rR)).norm(dim=-1)
            foot_angle = rot_angle_sq(bR[:, link_ids], motion.body_R[f][:, link_ids]).clamp_min(0).sqrt()
            speed = bv[:, link_ids, :2].norm(dim=-1)
            base = torch.stack(((p[:,:2]-rp[:,:2]).norm(dim=-1), (p[:,2]-rp[:,2]).abs(),
                                rot_angle_sq(R,rR).clamp_min(0).sqrt(), torch.atan2(torch.sin(yaw),torch.cos(yaw)).abs(),
                                (v-motion.root_v[f]).norm(dim=-1), (q-motion.q[f]).square().mean(-1).sqrt()), -1)
            values[indices] = torch.cat((base, foot_world, foot_local, foot_angle, speed), -1)
    values = values.view(T,C,-1).cpu().numpy()
    active_np = alive.cpu().numpy()
    states_finite = torch.isfinite(packed).all(-1).cpu().numpy()
    reward_np = reward.cpu().numpy()
    touches, labels, places = touch.cpu().numpy(), label.cpu().numpy(), pe.cpu().numpy()
    onset_np, matched_np = onsets.cpu().numpy(), valid_onsets.cpu().numpy()
    action_np = actions.cpu().numpy()
    flags, inactive_flags = env.active_overflow.cpu().tolist(), env.inactive_overflow.cpu().tolist()
    failed = env.failed_eval.cpu().tolist()
    lens = motion.clip_len.cpu().numpy()-1
    endpoint = packed.cpu().numpy()
    if args.record:
        # Preserve exact terminal states, including the final world's last
        # transition, which the original pre-step recorder cannot capture.
        from supermimic.acp import heightfield
        from supermimic.rotations import mat_to_quat
        with np.load(args.record) as source:
            trajectory = {key:source[key] for key in source.files}
        trajectory.update(post_root_p=endpoint[...,:3],
                          post_root_quat=mat_to_quat(packed[...,3:12].reshape(T,C,3,3)).cpu().numpy(),
                          post_q=endpoint[...,18:18+NUM_J], post_f=after_f.cpu().numpy(),
                          heightfield_paths=np.array([asset["path"] for asset in heightfield.ASSETS]),
                          checkpoint=np.array(str(args.ckpt.resolve())),
                          planner_checkpoint=np.array(str(args.planner.resolve()) if args.planner else ""),
                          eta_obs=np.array(args.eta_obs if args.planner else 0.),
                          noise=np.array(args.noise,dtype=str), seed=np.array(args.seed))
        np.savez_compressed(args.record, **trajectory)
    ref_z = motion.root_p[after_f,2].cpu().numpy()
    ref_tilt = motion.root_R[after_f,2,2].cpu().numpy()
    def stats(x):
        x = np.asarray(x); x = x[np.isfinite(x)]
        if not len(x):
            return {"count":0,"mean":None,"rmse":None,"p50":None,"p95":None,"max":None}
        return {"count":int(len(x)),"mean":float(x.mean()),"rmse":float(np.sqrt(np.mean(x*x))),
                "p50":float(np.quantile(x,.5)),"p95":float(np.quantile(x,.95)),"max":float(x.max())}
    episodes = []
    for c,name in enumerate(motion.names):
        mask = active_np[:,c]; n = int(mask.sum()); last = n-1
        finite = bool(states_finite[mask,c].all() and np.isfinite(values[mask,c]).all())
        tracking = {key:stats(values[mask,c,i]) for i,key in enumerate(metric_names)}
        contacts = {}
        for b,body in enumerate(CONTACT_BODIES):
            tp = int((touches[:,c,b]&labels[:,c,b]).sum()); nt=int(touches[:,c,b].sum()); nl=int(labels[:,c,b].sum())
            contacts[body] = {"true_positive_frames":tp,"touch_frames":nt,"label_frames":nl,
                              "precision":tp/nt if nt else None,"recall":tp/nl if nl else None,
                              "f1":2*tp/(nt+nl) if nt+nl else None,
                              "onsets":int(onset_np[:,c,b].sum()),"matched_onsets":int(matched_np[:,c,b].sum()),
                              "placement_error_m":stats(places[:,c,b]),
                              "placement_within_0.1":float((places[matched_np[:,c,b],c,b]<=.1).mean()) if matched_np[:,c,b].any() else None}
        for side,b,vi in [("left",foot_ids[0],12),("right",foot_ids[1],13)]:
            stance = touches[:,c,b]&labels[:,c,b]
            tracking[side+"_support_link_horizontal_speed_m_s"] = stats(values[stance,c,vi])
        action_delta = np.sqrt(np.mean(np.diff(action_np[:n,c],axis=0)**2,axis=-1))
        row = {"clip":name,"category":name.split('/')[0],"failed":bool(failed[c]),
               "completed":bool(not failed[c] and n==lens[c]),"steps":n,"reference_steps":int(lens[c]),
               "duration_s":n*.02,"reference_progress":n/int(lens[c]),
               "normalized_return":float(reward_np[mask,c].sum()/(lens[c]*.1)),
               "active_physics_overflow_flags":int(flags[c]),"inactive_physics_overflow_flags":int(inactive_flags[c]),
               "finite_active_states_and_errors":finite,"valid_physics":bool(not flags[c] and finite),
               "failure_height":bool(failed[c] and abs(endpoint[last,c,2]-ref_z[last,c])>.25),
               "failure_tilt":bool(failed[c] and abs(endpoint[last,c,11]-ref_tilt[last,c])>.8),
               "tracking":tracking,"contacts":contacts,"action_increment_rmse":stats(action_delta),
               "root_xy_error_at_end_m":float(values[last,c,0]),
               "friction":float(env.xi[c,0]),"joint_offsets":env.xi[c,1:1+NUM_J].cpu().tolist(),
               "com_offset":env.xi[c,1+NUM_J:].cpu().tolist()}
        episodes.append(row)
    result = {"protocol":{"seed":args.seed,"noise":args.noise,"goal_s":args.goal_s,
                          "requested_clip":args.clip,
                          "domain_randomization":args.domain_randomization,"friction_override":args.friction,
                          "finished_worlds_held_above_terrain":True,
                          "reference_commands":not args.planner or args.eta_obs == 0.,"first_variant_only":True,
                          "planner_checkpoint":str(args.planner.resolve()) if args.planner else None,
                          "planner_checkpoint_sha256":hashlib.sha256(args.planner.read_bytes()).hexdigest() if args.planner else None,
                          "eta_obs":args.eta_obs if args.planner else 0.,
                          "command_source":"reference" if not args.planner or args.eta_obs == 0. else "planner" if args.eta_obs == 1. else "mixed",
                          "metrics_after_transition_include_failure_step":True,
                          "placement_targets":"selected reference/planner targets within the last five command frames; before tracker command noise, includes planner scan noise when enabled",
                          "foot_position_metric":"foot link origins, not sole center or nearest contact point",
                          "support_speed_metric":"foot link horizontal speed during actual+reference contact; rolling foot motion is included"},
              "checkpoint":str(args.ckpt.resolve()),"checkpoint_sha256":hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
              "original_evaluation":ns["res"],"episodes":episodes,"elapsed_seconds":time.time()-started}
    if args.calibrate_contact:
        calibration = []
        with torch.no_grad():
            for c,name in enumerate(motion.names):
                path=Path(ctc.cfg.contacts_dir or (str(ns['motion_dir']).rstrip('/')+'_contacts'))/(name+'.npz')
                lo=int(motion.clip_start[c]); hi=lo+int(motion.clip_len[c])
                with np.load(path) as labels_file:
                    for b in foot_ids:
                        frames=torch.unique(ctc.next_onset[lo:hi,0,b]); frames=frames[frames>=0]
                        if not len(frames): continue
                        S=motion.ref_state(frames)
                        cp=ctc.contact_point(S,motion.boxes(frames,torch.zeros_like(frames)))[:,b]
                        expected=torch.tensor(labels_file['contact_pos_w'][0,frames.cpu().numpy()-lo,b],device=env.device)
                        error=(cp-expected).norm(dim=-1).cpu().numpy()
                        calibration.append({"clip":name,"body":list(CONTACT_BODIES)[b],"reference_pose_contact_point_disagreement_m":stats(error)})
        result["contact_geometry_calibration"] = calibration
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"out":str(args.out),"completed":sum(x['completed'] for x in episodes),
                      "invalid_physics_episodes":sum(not x['valid_physics'] for x in episodes),
                      "seconds":time.time()-started}),flush=True)


if __name__ == "__main__":
    main()
