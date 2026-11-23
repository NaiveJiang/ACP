"""Matched physical playbacks, posture/jitter diagnostics and rendered artifacts.

Inference only; never modifies checkpoints, rewards or training jobs.
"""
import argparse
import copy
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--category', default='forward')
    p.add_argument('--case', default='')
    p.add_argument('--seconds', type=float, default=6.)
    p.add_argument('--phase', type=float, default=.25)
    p.add_argument('--query-dt', type=float, default=.3)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--video', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    os.environ['ACP_ROBOT'] = 'roban'
    os.environ['MUJOCO_GL'] = 'egl'
    import numpy as np
    import torch
    import mujoco
    from PIL import Image, ImageDraw
    import imageio_ffmpeg
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from supermimic.acp.footstep.backend import backend_for_format
    from supermimic.acp.footstep.reference import FootstepMotion
    from supermimic.acp.footstep.env import FootstepTrackingSim
    from supermimic.acp.kinematics import RobotKinematics
    from supermimic.acp.robot import JOINT_NAMES, default_joint_pos_list, action_scale
    from supermimic.acp.mdp import unflatten_state

    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    checkpoints = [torch.load(c, map_location='cpu', weights_only=True) for c in args.checkpoints]
    candidates = [c for c in checkpoints[0]['split']['val'] if c['category'] == args.category
                  and (not args.case or args.case in c['case'])]
    if not candidates:
        raise ValueError('No matching validation reference')
    selected = candidates[0]
    identity = lambda c: (c['source'], c['case'], c['start'], c['end'])
    if not all(any(identity(c) == identity(selected) for c in ck['split']['val']) for ck in checkpoints):
        raise ValueError('Selected reference is not held out in every checkpoint')
    groups = {'upper': [0]+list(range(13,21)), 'legs': list(range(1,13))}
    results, all_records, rows = [], [], []
    initial = None
    for number, (path, ck) in enumerate(zip(args.checkpoints, checkpoints)):
        torch.manual_seed(123)
        backend = backend_for_format(ck['format'])
        joint=getattr(backend,'joint',False)
        config = ck['config']
        h = next(i for i,t in enumerate(config['horizons']) if abs(t-args.query_dt)<1e-6)
        kin = RobotKinematics(args.device)
        motion = FootstepMotion([selected], kin, args.device, config['horizons'])
        task = backend.Task(motion, kin, backend.RewardCfg(**ck['reward']))
        mdp = backend.MDP(kin, motion, torch.tensor(default_joint_pos_list(),device=args.device), task)
        pi = backend.Policy(config['hist'],config['policy_hidden'],**(dict(planner_config=config['planner_config']) if joint else {})).to(args.device).eval()
        pi.load_state_dict(ck['pi'])
        env = getattr(backend,'Env',FootstepTrackingSim)(1,motion,mdp,args.device,eval_mode=True,reset_noise=False)
        start = motion.clip_start[0]+int(args.phase*(int(motion.clip_len[0])-1))
        env.reset(torch.tensor([0],device=args.device),frames=start.reshape(1)); env.refresh()
        first = np.r_[env.sim.data.qpos[0].cpu().numpy(),env.sim.data.qvel[0].cpu().numpy()]
        if initial is None:
            initial = first
        np.testing.assert_allclose(first,initial,atol=1e-7,rtol=0)
        query = torch.tensor([h],device=args.device)
        history = None
        maximum = min(round(args.seconds*50),int(motion.clip_end[0]-start))
        sample_steps = set(np.linspace(0,maximum-1,6).astype(int))
        records = {k: [] for k in ('q','qd','q_ref','qd_ref','a','command','contact','contact_ref','reward','qpos','torque')}
        contributions = []
        frames = []
        writer = renderer = None
        model = copy.copy(env.sim.mj_model)
        dofs = [int(model.joint('robot/'+n).dofadr[0]) for n in JOINT_NAMES]
        data = mujoco.MjData(model)
        model.vis.global_.offwidth = 640; model.vis.global_.offheight = 480
        model.vis.headlight.ambient[:] = .45
        camera = mujoco.MjvCamera()
        camera.distance = 2.2; camera.azimuth = 135; camera.elevation = -12
        if args.video:
            renderer = mujoco.Renderer(model,height=480,width=640)
            writer = imageio_ffmpeg.write_frames(str(out/f'play_{number}.mp4'),(640,480),fps=50,
                                                 codec='libx264',macro_block_size=2,output_params=['-crf','22'])
            writer.send(None)
        previous = torch.zeros(1,21,device=args.device)
        termination = False
        try:
            with torch.no_grad():
                for step in range(maximum):
                    obs = env.observe()
                    history = (obs[:,None].expand(-1,config['hist'],-1).clone() if history is None
                               else torch.cat((history[:,1:],obs[:,None]),1))
                    if joint:query.copy_(task.pack_queries(env.f,query))
                    command = task.command(env.state(),env.f,query)
                    if joint:
                        command=pi.contact_plan(history,command)
                        action=pi.controller_action(history,command)
                    else:action = pi(history,command)
                    tr = env.step(action)
                    S = unflatten_state(tr['s1']); f = tr['f']+1
                    sole,_,gap = task.foot_geometry(S)
                    for key,value in (
                        ('q',S[4][0]),('qd',S[5][0]),('q_ref',motion.q[f][0]),('qd_ref',motion.qd[f][0]),
                        ('a',action[0]),('command',command[0]),('contact',(gap[0]<=task.cfg.contact_distance)),
                        ('contact_ref',motion.contact[f][0]),('reward',tr['r'][0]),
                        ('qpos',env.sim.data.qpos[0]),('torque',env.sim.data.qfrc_actuator[0,dofs])):
                        records[key].append(value.cpu().numpy().copy())
                    if backend.command_dim == 13:
                        c = task.cfg
                        term = task.reward_terms(S,f)
                        upper_error = (S[4][:,mdp.upper_indices]-motion.q[f][:,mdp.upper_indices]).square().mean(-1)
                        contributions.append(dict(contact=float(c.contact_weight*term['contact'][0]),
                            position=float(c.position_weight*term['position'][0]),yaw=float(c.yaw_weight*term['yaw'][0]),
                            upper_pose=float(c.upper_pose_weight*torch.exp(-upper_error/c.upper_pose_std**2)[0]),
                            action_rate=float(c.action_rate_weight*(action-previous).square().sum()),
                            joint_speed=float(c.joint_speed_weight*S[5].square().sum())))
                        if hasattr(mdp, 'regularization_terms'):
                            reg = mdp.regularization_terms(S, f, action, previous)
                            contributions[-1]['upper_pose'] = float(-c.upper_pose_weight*reg['upper_pose'][0])
                            contributions[-1]['upper_velocity'] = float(-c.upper_velocity_weight*reg['upper_velocity'][0])
                    previous = action
                    if renderer:
                        data.qpos[:] = records['qpos'][-1]
                        data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
                        mujoco.mj_forward(model,data)
                        camera.lookat[:] = S[0][0].cpu().numpy()+np.array([0,0,-.1])
                        renderer.update_scene(data,camera=camera)
                        rgb = renderer.render().copy()
                        writer.send(rgb)
                        if step in sample_steps:
                            image = Image.fromarray(rgb).resize((320,240))
                            draw = ImageDraw.Draw(image)
                            draw.rectangle((0,0,320,28),fill='white')
                            draw.text((5,5),f"{backend.command_dim}D  t={step*.02:.2f}s",fill='black')
                            frames.append(image)
                    if bool(tr['done'][0]):
                        termination = bool(tr['term'][0]); break
        finally:
            if writer: writer.close()
            if renderer: renderer.close()
        record = {k:np.asarray(v) for k,v in records.items()}
        np.savez_compressed(out/f'rollout_{number}.npz',**record,joint_names=np.asarray(JOINT_NAMES))
        # Omit the first 0.5 s from signal statistics to avoid reset transients.
        cut = min(25,len(record['q'])//4)
        stats = dict(checkpoint=str(Path(path).resolve()),format=ck['format'],steps=len(record['q']),
                     terminated=termination,mean_reward=float(record['reward'].mean()),groups={})
        delta = np.diff(record['a'],axis=0)
        binary = record['command'][:,backend.contact_offset:backend.contact_offset+2]
        events = (np.diff(binary,axis=0)!=0).any(axis=1)
        stats['action_jump_goal_contact_changed'] = float(np.abs(delta[events]).mean()) if events.any() else None
        stats['action_jump_goal_contact_unchanged'] = float(np.abs(delta[~events]).mean()) if (~events).any() else None
        for name,indices in groups.items():
            q = record['q'][cut:,indices]; qd = record['qd'][cut:,indices]
            qr = record['q_ref'][cut:,indices]; qdr = record['qd_ref'][cut:,indices]
            def high_frequency(values):
                power = np.abs(np.fft.rfft(values-values.mean(0),axis=0))**2
                frequencies = np.fft.rfftfreq(len(values),d=.02)
                return float(power[frequencies>=8].sum()/max(power[frequencies>=.5].sum(),1e-12))
            stats['groups'][name] = dict(pose_rmse_deg=float(np.rad2deg(np.sqrt(np.mean((q-qr)**2)))),
                joint_speed_rms=float(np.sqrt(np.mean(qd**2))),reference_joint_speed_rms=float(np.sqrt(np.mean(qdr**2))),
                normalized_action_step_rms=float(np.sqrt(np.mean(delta[cut:,indices]**2))),
                joint_target_step_rms_deg=float(np.rad2deg(np.sqrt(np.mean((delta[cut:,indices]*np.array([action_scale(JOINT_NAMES[i]) for i in indices]))**2)))),
                high_frequency_qd_fraction=high_frequency(qd),reference_high_frequency_qd_fraction=high_frequency(qdr),
                action_saturation_fraction=float(np.mean(np.abs(record['a'][cut:,indices])>4.9)))
        stats['joint_pose_rmse_deg'] = dict(zip(JOINT_NAMES,np.rad2deg(np.sqrt(np.mean((record['q'][cut:]-record['q_ref'][cut:])**2,axis=0))).tolist()))
        if contributions:
            stats['reward_contribution_mean_before_dt'] = {k:float(np.mean([x[k] for x in contributions])) for k in contributions[0]}
        results.append(stats); all_records.append(record); rows.append(frames)
        print(json.dumps(stats),flush=True)
    report = dict(args=vars(args),reference=selected,identical_initial_states=True,results=results)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    if args.video:
        sheet = Image.new('RGB',(6*320,len(rows)*240),'white')
        for row,frames in enumerate(rows):
            for col,frame in enumerate(frames):sheet.paste(frame,(col*320,row*240))
        sheet.save(out/'contact_sheet.png')
    figure,axes=plt.subplots(3,2,figsize=(14,9),sharex=True)
    for i,(stats,r) in enumerate(zip(results,all_records)):
        t=np.arange(len(r['q']))*.02; label=f"{len(r['command'][0])}D"
        for col,group in enumerate(('upper','legs')):
            ids=groups[group]
            axes[0,col].plot(t,np.rad2deg(np.sqrt(np.mean((r['q'][:,ids]-r['q_ref'][:,ids])**2,axis=1))),label=label)
            axes[1,col].plot(t,np.sqrt(np.mean(r['qd'][:,ids]**2,axis=1)),label=label)
            axes[2,col].plot(t[1:],np.sqrt(np.mean(np.diff(r['a'],axis=0)[:,ids]**2,axis=1)),label=label)
            axes[0,col].set_title(group)
    for row,title in enumerate(('Pose error (deg)','Joint speed RMS (rad/s)','Action step RMS')):
        for col in range(2):axes[row,col].set_ylabel(title);axes[row,col].legend();axes[row,col].grid(alpha=.3)
    figure.tight_layout();figure.savefig(out/'signals.png',dpi=160);plt.close(figure)


if __name__=='__main__':main()
