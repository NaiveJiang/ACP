"""Keyboard goal control of a real planner--tracker loop in a connected 20-zone playground.

W/S: move goal forward/back; A/D: move goal left/right (robot heading by default).
Q/E: goal heading; M: world/robot axes; X: goal at current position.
Tab / Shift-Tab or the right panel: select terrain; G: goal at its exit;
N: goal at its entrance; R: reset robot at selected entrance; Home: flat start.
Space: pause; F: follow camera; O: overview; P: contact markers; Esc: quit.
H: planner terrain scan (cyan); J: tracker terrain scan (orange); both on by default.
Click the minimap to set a world goal. Mouse drag/scroll rotates/pans/zooms the camera.

This is reference-free, nominal inference. Stage 2 checkpoints are tested zero-shot;
they are not automatically trained for arbitrary navigation by this application.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from . import networks
from .checkpoint import load_checkpoint
from .mdp import MDP, OBS_DIM, REF_ORI_DIMS, XI_DIM
from .motion import CONTROL_DT
from .networks import Policy
from .planner import ContactPlanner, PlannerCfg, PlannerScan, plan
from .playground import Playground
from .robot import (JOINT_NAMES, NUM_J, NB, ROOT_BODY, ROBOT_ID, STAND_HEIGHT, action_scale,
                    default_joint_pos_list, pd_params)
from .task import ContactCfg, contact_command, from_base, goal_command
from .terrain import grid_offsets, height_scan

ROOT = Path(__file__).resolve().parents[2]
CONTACT_COLORS = [(1,.2,.2,1),(.2,.4,1,1),(1,.6,.1,1),(1,.6,.1,1),(.2,1,.4,1),
                  (.2,1,.4,1),(.8,.3,1,1),(.8,.3,1,1),(1,1,.2,1)]
PLANNER_SCAN_COLOR = (.05,.85,1.,1.)
TRACKER_SCAN_COLOR = (1.,.55,.08,1.)


def resolve_checkpoint(value: str) -> Path:
    if value == 'latest':
        files = [p for pattern in ('acp_roban_stage2*', 'acp_roban_stage3*')
                 for run in (ROOT/'logs').glob(pattern) for p in run.glob('ckpt_*.pt')]
        if not files:raise FileNotFoundError('No Stage 2/3 checkpoint; pass --ckpt /path/to/checkpoint.pt')
        return max(files, key=lambda p:p.stat().st_mtime).resolve()
    path=Path(value).expanduser().resolve()
    if path.is_dir():
        files=list(path.glob('ckpt_*.pt'))
        path=max(files,key=lambda p:p.stat().st_mtime) if files else path/'last.pt'
    if not path.is_file():raise FileNotFoundError(path)
    return path


class GoalCursor:
    def __init__(self, arena: Playground):
        self.arena=arena
        self.position=np.zeros(3,dtype=float)
        self.yaw=0.
        self.robot_axes=True
        self.selected=0

    def set(self, xy, yaw=None):
        xmin,xmax,ymin,ymax=self.arena.bounds
        self.position[:2]=np.clip(xy,[xmin+.3,ymin+.3],[xmax-.3,ymax-.3])
        self.position[2]=float(self.arena.height(self.position[:2]))+STAND_HEIGHT
        if yaw is not None:self.yaw=math.atan2(math.sin(yaw),math.cos(yaw))

    def move(self, keys: set[str], dt: float, robot_yaw: float, speed: float):
        dx=float('W' in keys)-float('S' in keys)
        dy=float('A' in keys)-float('D' in keys)
        if dx or dy:
            angle=robot_yaw if self.robot_axes else 0.
            direction=np.array([math.cos(angle)*dx-math.sin(angle)*dy,math.sin(angle)*dx+math.cos(angle)*dy])
            self.set(self.position[:2]+direction/max(1.,np.linalg.norm(direction))*speed*dt)
        self.yaw += (float('Q' in keys)-float('E' in keys))*1.2*dt
        self.yaw=math.atan2(math.sin(self.yaw),math.cos(self.yaw))


class ClosedLoop:
    def __init__(self, checkpoint: Path, arena: Playground, device='cuda:0', record: Path | None=None):
        if ROBOT_ID!='roban':raise ValueError('Use --robot roban for this playground')
        self.device=torch.device(device)
        ck=load_checkpoint(checkpoint,map_location='cpu',weights_only=False)
        if not {'pi','planner','planner_cfg','ctc_cfg'} <= ck.keys():
            raise ValueError('A Stage 2/3 checkpoint with BOTH tracker and planner is required; Stage 1 alone is insufficient')
        for name in ('pi','planner'):
            if not all(torch.isfinite(v).all() for v in ck[name].values()):
                raise ValueError(f'{name} has nonfinite weights in {checkpoint}')
        ia=ck.get('init_args',ck['args'])
        self.hist_length=ia.get('hist',10)
        self.ccfg=ContactCfg(**ck['ctc_cfg']);self.pcfg=PlannerCfg(**ck['planner_cfg'])
        if self.hist_length<self.pcfg.hist:raise ValueError('Tracker history is shorter than planner history')
        self.offsets=grid_offsets(self.ccfg.scan_x,self.ccfg.scan_y,device=device)
        self.command_dim=NB*3+3+(0 if self.ccfg.blind else len(self.offsets))
        self.pi=Policy(OBS_DIM,self.command_dim,hist=self.hist_length).to(device).eval()
        self.pi.load_state_dict(ck['pi'])
        self.planner=ContactPlanner(OBS_DIM-REF_ORI_DIMS,self.pcfg).to(device).eval()
        self.planner.load_state_dict(ck['planner'])
        # Observe the actual network input without modifying the shared planner path.
        self._scan_hook=self.planner.register_forward_pre_hook(self._capture_planner_scan)
        networks.AMP_DTYPE=torch.bfloat16 if self.device.type=='cuda' and ck['args'].get('amp')=='bf16' else None
        self.scan=PlannerScan(self.pcfg,device)
        # Stage 3 saved goal radius takes precedence; Stage 2 keeps its trained command radius.
        self.goal_radius=ck.get('goal_cfg',{}).get('radius',self.ccfg.goal_radius)
        self.checkpoint=checkpoint;self.iteration=int(ck['iteration'])
        self.stage=3 if 'goal_cfg' in ck else 2
        self.arena=arena;self.model=arena.build_model();self.data=mujoco.MjData(self.model)
        # Episode resets belong to the user, including after numerical trouble.
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_AUTORESET
        self._backup=np.empty(mujoco.mj_stateSize(self.model,mujoco.mjtState.mjSTATE_INTEGRATION))
        self.joint_ids=np.array([self.model.joint(n).id for n in JOINT_NAMES])
        self.qadr=self.model.jnt_qposadr[self.joint_ids]
        self.vadr=self.model.jnt_dofadr[self.joint_ids]
        self.root_id=self.model.body(ROOT_BODY).id
        self.default=np.asarray(default_joint_pos_list())
        self.alpha=np.asarray([action_scale(n) for n in JOINT_NAMES])
        self.kp,self.kd,self.limits,_=map(np.asarray,pd_params())
        self.obs_context=SimpleNamespace(default_q=torch.tensor(self.default,dtype=torch.float32,device=device))
        self.xi=torch.zeros(1,XI_DIM,device=device);self.xi[:,0]=1.
        self.cursor=GoalCursor(arena)
        self.record=record.open('w') if record else None
        self.total_steps=0;self.resets=0;self.numerical_warnings=0;self.latencies=[]
        self.last_command=None
        self.reset(0)
        self._event('loaded',checkpoint=str(checkpoint),iteration=self.iteration,stage=self.stage)

    def _event(self, kind, **values):
        if self.record:
            self.record.write(json.dumps(dict(event=kind,step=self.total_steps,**values))+'\n')
            self.record.flush()

    @property
    def yaw(self):
        R=self.data.xmat[self.root_id].reshape(3,3)
        return math.atan2(R[1,0],R[0,0])

    def reset(self, index):
        self.cursor.selected=index
        start=self.arena.stations[index].start
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[:3]=[start[0],start[1],float(self.arena.height(start[:2]))+STAND_HEIGHT]
        self.data.qpos[3:7]=[1,0,0,0]
        self.data.qpos[self.qadr]=self.default
        for _ in range(4):
            mujoco.mj_forward(self.model,self.data)
            depth=min([0.]+[c.dist for c in self.data.contact])
            if depth>=-.001:break
            self.data.qpos[2]+=-depth+.002
        mujoco.mj_forward(self.model,self.data)
        self.previous=torch.zeros(1,NUM_J,device=self.device)
        self.history=None;self.targets=np.zeros((9,3));self.valid=np.zeros(9,dtype=bool)
        self.last_command=None
        self.last_action=np.zeros(NUM_J);self.last_torque=np.zeros(NUM_J)
        self.warnings=[];self.resets+=1
        self.root_clearance=float(self.data.qpos[2]-self.arena.height(self.data.qpos[:2]))
        self.cursor.set(self.arena.stations[index].goal[:2],yaw=0.)
        # Show the initial scan while paused, without taking a policy or physics step.
        S=self.state();boxes=self.arena.tensor(self.device,self.data.qpos[:2])
        self._planner_input_points=self.scan.points(self.scan.heights(S,boxes))
        self.update_scan_display(S,None if self.ccfg.blind else height_scan(S[0],S[1],boxes,self.offsets))
        self.scan_preview=True
        self._event('reset',station=index+1)

    def set_goal(self, xy, face=True):
        delta=np.asarray(xy)-self.data.qpos[:2]
        yaw=math.atan2(delta[1],delta[0]) if face and np.linalg.norm(delta)>.05 else self.cursor.yaw
        self.cursor.set(xy,yaw)
        self._event('goal',goal=self.cursor.position.tolist(),yaw=self.cursor.yaw)

    def state(self):
        R=self.data.xmat[self.root_id].reshape(3,3).copy()
        arrays=(self.data.qpos[:3],R,self.data.qvel[:3],R@self.data.qvel[3:6],
                self.data.qpos[self.qadr],self.data.qvel[self.vadr])
        return tuple(torch.as_tensor(np.array(x),dtype=torch.float32,device=self.device).unsqueeze(0) for x in arrays)

    def _capture_planner_scan(self, module, inputs):
        self._planner_input_points=inputs[2].detach()

    @torch.inference_mode()
    def update_scan_display(self, S, tracker_heights):
        """Cache world points in the input state's frame, before the physics step.

        Planner inputs store terrain minus root Z; tracker commands store the opposite.
        Keep the cached positions unshifted: marker clearance is a rendering detail.
        """
        self.planner_scan_points=from_base(self._planner_input_points,S[0],S[1])[0].cpu().numpy().copy()
        if tracker_heights is None:
            self.tracker_scan_points=np.empty((0,3))
        else:
            pts=torch.cat((self.offsets[None],-tracker_heights.unsqueeze(-1)),dim=-1)
            self.tracker_scan_points=from_base(pts,S[0],S[1])[0].cpu().numpy().copy()
        self.scan_time=float(self.data.time)
        self.scan_preview=False

    def update_diagnostics(self, messages=()):
        """Display-only indicators. None of these can stop or reset the rollout."""
        self.warnings=list(messages)
        self.root_clearance=float(self.data.qpos[2]-self.arena.height(self.data.qpos[:2]))
        if self.root_clearance<.20:
            self.warnings.append(f'Low root clearance: {self.root_clearance:.3f} m')
        upright=float(self.data.xmat[self.root_id].reshape(3,3)[2,2])
        if upright<.2:
            self.warnings.append(f'Torso tilt: {math.degrees(math.acos(np.clip(upright,-1.,1.))):.1f} deg')
        xmin,xmax,ymin,ymax=self.arena.bounds
        if not (xmin+.1<self.data.qpos[0]<xmax-.1 and ymin+.1<self.data.qpos[1]<ymax-.1):
            self.warnings.append('Outside playground')

    @torch.inference_mode()
    def step(self):
        started=time.perf_counter()
        messages=[]
        S=self.state()
        obs=MDP.obs(self.obs_context,S,self.previous,self.xi)
        self.history=obs[:,None].repeat(1,self.hist_length,1) if self.history is None else torch.cat((self.history[:,1:],obs[:,None]),1)
        boxes=self.arena.tensor(self.device,self.data.qpos[:2])
        gp=torch.tensor(self.cursor.position,dtype=torch.float32,device=self.device)[None]
        gy=torch.tensor([self.cursor.yaw],dtype=torch.float32,device=self.device)
        goal=goal_command(S,gp,gy,self.goal_radius)
        targets,valid=plan(self.planner,self.scan,self.pcfg,None,self.history,S,None,None,None,
                           noise=False,goal=goal,boxes=boxes)
        command=contact_command(self.ccfg,S,targets,valid,goal,boxes,self.offsets)
        self.update_scan_display(S,None if self.ccfg.blind else command[:,NB*3+3:])
        raw_action=self.pi(self.history,command)
        if not bool(torch.isfinite(raw_action).all()):
            # An invalid output is not an episode boundary. Keep the last finite command
            # for this tick and retry inference next tick; R remains fully responsive.
            action=self.previous.clone()
            messages.append('Nonfinite action: using previous action')
            self.numerical_warnings+=1
        else:
            action=raw_action.clamp(-20,20)
        self.last_command=command.cpu().numpy()[0]
        self.targets,self.valid=targets.cpu().numpy()[0],valid.cpu().numpy()[0]
        self.valid &= np.isfinite(self.targets).all(axis=1)
        self.last_action=action.cpu().numpy()[0]
        target=self.default+self.alpha*self.last_action
        mujoco.mj_getState(self.model,self.data,self._backup,mujoco.mjtState.mjSTATE_INTEGRATION)
        finite=True
        for _ in range(4):
            vel=self.data.qvel[self.vadr]
            self.last_torque=np.clip(self.kp*(target-self.data.qpos[self.qadr])-self.kd*vel-.2*np.tanh(vel/.1),-self.limits,self.limits)
            self.data.ctrl[:]=self.last_torque
            mujoco.mj_step(self.model,self.data)
            finite=all(np.isfinite(x).all() for x in (self.data.qpos,self.data.qvel,self.data.qacc))
            if not finite:break
        if finite:
            mujoco.mj_forward(self.model,self.data)
            finite=all(np.isfinite(x).all() for x in (self.data.qpos,self.data.qvel,self.data.qacc,self.data.xmat))
        if not finite:
            # Roll back only this unusable numerical transition, not the episode.
            # There is no latched stop, automatic spawn reset, or curriculum change.
            mujoco.mj_setState(self.model,self.data,self._backup,mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(self.model,self.data)
            messages.append('Invalid numerical step discarded; R to reset')
            self.numerical_warnings+=1
        else:
            self.previous=action
        self.total_steps+=1
        self.update_diagnostics(messages)
        self.latencies.append(time.perf_counter()-started)
        if len(self.latencies)>1000:self.latencies=self.latencies[-1000:]
        if self.record:
            self.record.write(json.dumps(dict(event='step',step=self.total_steps,sim_time=float(self.data.time),
                qpos=self.data.qpos.tolist(),goal=self.cursor.position.tolist(),goal_yaw=self.cursor.yaw,
                targets=self.targets.tolist(),valid=self.valid.tolist(),action=self.last_action.tolist(),
                warnings=self.warnings,root_clearance=self.root_clearance))+'\n')

    def summary(self):
        return dict(checkpoint=str(self.checkpoint),iteration=self.iteration,stage=self.stage,stations=len(self.arena.stations),
            steps=self.total_steps,sim_seconds=self.total_steps*CONTROL_DT,resets=self.resets,
            root_position=self.data.qpos[:3].tolist(),goal=self.cursor.position.tolist(),warnings=self.warnings,
            root_clearance=self.root_clearance,automatic_termination=False,numerical_warnings=self.numerical_warnings,
            mean_control_ms=1000*float(np.mean(self.latencies)) if self.latencies else None,
            command_dim=self.command_dim,planner_points=self.scan.num_z,tracker_points=len(self.tracker_scan_points),reference_motion_used=False,
            physics_backend='MuJoCo CPU float64; same nominal ACP PD, timestep and solver settings',
            nominal_friction=1.,action_limit=20.,physics_dt=self.model.opt.timestep,control_dt=CONTROL_DT)

    def close(self):
        self._scan_hook.remove()
        if self.record:self.record.close()


def add_markers(scene, loop, contacts=True, planner_scan=True, tracker_scan=True):
    def geom(kind,size,pos,rgba,label=''):
        if scene.ngeom>=scene.maxgeom:return None
        g=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g,kind,np.asarray(size,dtype=float),np.asarray(pos,dtype=float),np.eye(3).ravel(),np.asarray(rgba,dtype=np.float32))
        g.label=label;scene.ngeom+=1
        return g
    for i,s in enumerate(loop.arena.stations):
        geom(mujoco.mjtGeom.mjGEOM_SPHERE,[.015,0,0],s.start+[0,0,.55],s.color,f'{i+1:02d} {s.name}')
    surface=loop.cursor.position.copy();surface[2]-=STAND_HEIGHT
    geom(mujoco.mjtGeom.mjGEOM_SPHERE,[.11,0,0],surface+[0,0,.12],[.15,1.,.45,1],'GOAL')
    arrow=geom(mujoco.mjtGeom.mjGEOM_ARROW,[.03,0,0],surface,[.15,1.,.45,1])
    if arrow is not None:
        end=surface+np.array([.5*math.cos(loop.cursor.yaw),.5*math.sin(loop.cursor.yaw),.04])
        mujoco.mjv_connector(arrow,mujoco.mjtGeom.mjGEOM_ARROW,.025,surface+[0,0,.04],end)
    # Larger orange discs keep both layers visible where their XY grids overlap.
    # The stored input points remain exact; only the drawn markers are lifted slightly.
    if tracker_scan:
        for point in loop.tracker_scan_points:
            if np.isfinite(point).all():
                geom(mujoco.mjtGeom.mjGEOM_CYLINDER,[.027,.003,0],point+[0,0,.006],TRACKER_SCAN_COLOR)
    if planner_scan:
        for point in loop.planner_scan_points:
            if np.isfinite(point).all():
                geom(mujoco.mjtGeom.mjGEOM_SPHERE,[.014,0,0],point+[0,0,.016],PLANNER_SCAN_COLOR)
    if contacts:
        for i in np.flatnonzero(loop.valid):
            geom(mujoco.mjtGeom.mjGEOM_SPHERE,[.035,0,0],loop.targets[i]+[0,0,.025],CONTACT_COLORS[i])


def camera_for(loop, overview=False):
    camera=mujoco.MjvCamera()
    if overview:
        camera.lookat[:]=[17,11,.1];camera.distance=50;camera.elevation=-70;camera.azimuth=105
    else:
        camera.lookat[:]=loop.data.qpos[:3];camera.distance=4.5;camera.elevation=-22;camera.azimuth=135
    return camera


class Window:
    def __init__(self,loop,args):
        import glfw
        self.glfw=glfw;self.loop=loop;self.args=args
        if not glfw.init():raise RuntimeError('Cannot open display. Run from a graphical terminal, or use --headless --seconds 5')
        glfw.window_hint(glfw.VISIBLE,not args.hidden)
        self.window=glfw.create_window(args.width,args.height,'RoBan | WASD Goal Playground',None,None)
        if not self.window:
            glfw.terminate();raise RuntimeError('GLFW window creation failed')
        glfw.make_context_current(self.window);glfw.swap_interval(1)
        self.context=mujoco.MjrContext(loop.model,mujoco.mjtFontScale.mjFONTSCALE_100)
        self.scene=mujoco.MjvScene(loop.model,maxgeom=4000);self.option=mujoco.MjvOption()
        self.camera=camera_for(loop,args.overview)
        self.overview=args.overview;self.follow=not args.overview;self.contacts=True
        self.planner_scan=True;self.tracker_scan=True
        self.paused=not args.autostart;self.keys=set();self.mouse=None;self.width=args.width;self.height=args.height
        self.station_rects=[];self.map_rect=None
        glfw.set_key_callback(self.window,self.on_key)
        glfw.set_window_focus_callback(self.window,lambda w,focused:self.keys.clear() if not focused else None)
        glfw.set_mouse_button_callback(self.window,self.on_button)
        glfw.set_cursor_pos_callback(self.window,self.on_mouse)
        glfw.set_scroll_callback(self.window,lambda w,x,y:mujoco.mjv_moveCamera(loop.model,mujoco.mjtMouse.mjMOUSE_ZOOM,0,-.05*y,self.camera))

    def on_key(self,window,key,scancode,action,mods):
        g=self.glfw
        name=chr(key) if 65<=key<=90 else ''
        if action==g.RELEASE:self.keys.discard(name);return
        if action==g.PRESS:self.keys.add(name)
        if action!=g.PRESS:return
        if key==g.KEY_ESCAPE:g.set_window_should_close(window,True)
        elif key==g.KEY_SPACE:self.paused=not self.paused;self.keys.clear()
        elif key==g.KEY_TAB:self.loop.cursor.selected=(self.loop.cursor.selected+(-1 if mods&g.MOD_SHIFT else 1))%len(self.loop.arena.stations)
        elif g.KEY_1<=key<=g.KEY_9:self.loop.cursor.selected=key-g.KEY_1
        elif key==g.KEY_0:self.loop.cursor.selected=9
        elif key==g.KEY_M:self.loop.cursor.robot_axes=not self.loop.cursor.robot_axes
        elif key==g.KEY_P:self.contacts=not self.contacts
        elif key==g.KEY_H:self.planner_scan=not self.planner_scan
        elif key==g.KEY_J:self.tracker_scan=not self.tracker_scan
        elif key==g.KEY_F:self.follow=True;self.overview=False;self.camera=camera_for(self.loop)
        elif key==g.KEY_O:self.overview=not self.overview;self.follow=not self.overview;self.camera=camera_for(self.loop,self.overview)
        elif key in (g.KEY_R,g.KEY_HOME):
            self.loop.reset(0 if key==g.KEY_HOME else self.loop.cursor.selected);self.keys.clear()
            if self.follow:self.camera=camera_for(self.loop)
        elif key==g.KEY_X:self.loop.cursor.set(self.loop.data.qpos[:2],self.loop.yaw)
        elif key in (g.KEY_G,g.KEY_N):
            station=self.loop.arena.stations[self.loop.cursor.selected]
            self.loop.set_goal((station.goal if key==g.KEY_G else station.start)[:2])

    def mouse_pixel(self):
        x,y=self.glfw.get_cursor_pos(self.window);w,h=self.glfw.get_window_size(self.window)
        return x*self.width/max(1,w),y*self.height/max(1,h)

    def on_button(self,window,button,action,mods):
        g=self.glfw;x,y=self.mouse_pixel()
        if action==g.RELEASE:self.mouse=None;return
        if button==g.MOUSE_BUTTON_LEFT:
            for i,(rx,ry,rw,rh) in enumerate(self.station_rects):
                if rx<=x<=rx+rw and ry<=y<=ry+rh:self.loop.cursor.selected=i;return
            if self.map_rect:
                rx,ry,rw,rh=self.map_rect
                if rx<=x<=rx+rw and ry<=y<=ry+rh:
                    xmin,xmax,ymin,ymax=self.loop.arena.bounds
                    self.loop.set_goal([xmin+(x-rx)/rw*(xmax-xmin),ymax-(y-ry)/rh*(ymax-ymin)]);return
        if x<self.width-350:self.mouse=(x,y,button)

    def on_mouse(self,window,x,y):
        if self.mouse is None:return
        px,py,button=self.mouse;x,y=self.mouse_pixel()
        move=mujoco.mjtMouse.mjMOUSE_ROTATE_H if button==self.glfw.MOUSE_BUTTON_LEFT else mujoco.mjtMouse.mjMOUSE_MOVE_H
        if button!=self.glfw.MOUSE_BUTTON_LEFT:self.follow=False
        mujoco.mjv_moveCamera(self.loop.model,move,(x-px)/self.height,(y-py)/self.height,self.camera)
        self.mouse=(x,y,button)

    def rect(self,x,y,w,h,color):
        mujoco.mjr_rectangle(mujoco.MjrRect(int(x),int(self.height-y-h),int(w),int(h)),*color)

    def label(self,x,y,w,h,text):
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,
            mujoco.MjrRect(int(x),int(self.height-y-h),int(w),int(h)),text,'',self.context)

    def draw(self):
        self.width,self.height=self.glfw.get_framebuffer_size(self.window)
        if self.width<1 or self.height<1:return
        if self.follow:self.camera.lookat[:]=self.loop.data.qpos[:3]
        viewport=mujoco.MjrRect(0,0,max(1,self.width-350),self.height)
        mujoco.mjv_updateScene(self.loop.model,self.loop.data,self.option,None,self.camera,mujoco.mjtCatBit.mjCAT_ALL,self.scene)
        add_markers(self.scene,self.loop,self.contacts,self.planner_scan,self.tracker_scan)
        mujoco.mjr_render(viewport,self.scene,self.context)
        goal=self.loop.cursor.position;p=self.loop.data.qpos[:3]
        distance=np.linalg.norm(goal[:2]-p[:2])
        state='PAUSED - Space to run' if self.paused else 'RUNNING - manual reset'
        hud=f'{state}\nStage {self.loop.stage} | checkpoint {self.loop.iteration:,} | nominal\nGoal XY ({goal[0]:.2f}, {goal[1]:.2f}) | distance {distance:.2f} m\nWASD axes: {"ROBOT" if self.loop.cursor.robot_axes else "WORLD"} | valid contacts {int(self.loop.valid.sum())}/9'
        hud+=f'\nH planner scan (cyan): {"ON" if self.planner_scan else "OFF"} {len(self.loop.planner_scan_points)} pts'
        hud+=f' | J tracker (orange): {"ON" if self.tracker_scan else "OFF"} {len(self.loop.tracker_scan_points)} pts'
        hud+=f'\nScan: {"reset preview" if self.loop.scan_preview else "last policy input"} | t={self.loop.scan_time:.2f}s'
        for warning in self.loop.warnings:hud+='\n'+warning
        self.label(12,10,viewport.width-24,24*(7+len(self.loop.warnings)),hud)
        self.label(12,self.height-196,viewport.width-24,192,
            'W/S forward/back   A/D left/right   Q/E goal heading\nM world/robot axes   X goal here   Space pause\nTab select terrain   G goal at exit   N goal at entrance\nR reset at selected entrance   Home reset at flat lane\nH planner scan   J tracker scan   P contact points\nO overview   F follow   Esc quit\nMouse drag/scroll: camera   Click minimap: set goal\nStage 2: zero-shot goal control, no global path planning')
        left=self.width-350
        self.rect(left,0,350,self.height,(.055,.075,.105,1))
        self.label(left+10,8,330,38,'ROBAN / TERRAIN LAB')
        self.label(left+10,42,330,54,'Select a lane, then G to target it.\nR resets robot; selection does not.')
        row_h=min(25,max(18,(self.height-370)//20))
        self.station_rects=[]
        for i,s in enumerate(self.loop.arena.stations):
            y=106+i*row_h;self.station_rects.append((left+8,y,334,row_h))
            if i==self.loop.cursor.selected:self.rect(left+8,y,334,row_h,(.17,.32,.36,1))
            self.label(left+14,y-2,320,row_h+3,f'{i+1:02d}  {s.name}')
        self.label(left+10,110+20*row_h,330,48,'MINIMAP | click to set goal\nWhite robot / green goal')
        mx,my,mw,mh=left+18,self.height-205,314,185
        self.map_rect=(mx,my,mw,mh)
        self.rect(mx,my,mw,mh,(.11,.15,.18,1))
        xmin,xmax,ymin,ymax=self.loop.arena.bounds
        def mapxy(x,y):return mx+(x-xmin)/(xmax-xmin)*mw,my+(ymax-y)/(ymax-ymin)*mh
        for i,s in enumerate(self.loop.arena.stations):
            x,y=mapxy(s.origin[0],s.origin[1]+1.7)
            w,h=7.5/(xmax-xmin)*mw,3.4/(ymax-ymin)*mh
            color=s.color if i==self.loop.cursor.selected else tuple(c*.55 for c in s.color[:3])+(1.,)
            self.rect(x,y,w,h,color)
        for position,color,size in [(p,(1,1,1,1),6),(goal,(.1,1,.4,1),7)]:
            x,y=mapxy(*position[:2]);self.rect(x-size/2,y-size/2,size,size,color)

    def screenshot(self,path):
        from PIL import Image
        rgb=np.empty((self.height,self.width,3),dtype=np.uint8)
        mujoco.mjr_readPixels(rgb,None,mujoco.MjrRect(0,0,self.width,self.height),self.context)
        path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(np.flipud(rgb)).save(path)

    def run(self):
        g=self.glfw;start=last=time.perf_counter();acc=0.;saved=False;frames=0
        try:
            while not g.window_should_close(self.window):
                now=time.perf_counter();dt=min(now-last,.1);last=now
                if self.args.seconds and now-start>=self.args.seconds:break
                g.poll_events()
                self.loop.cursor.move(self.keys,dt,self.loop.yaw,self.args.goal_speed)
                if not self.paused:
                    acc=min(acc+dt*self.args.speed,.10)
                    while acc>=CONTROL_DT:
                        self.loop.step();acc-=CONTROL_DT
                else:acc=0.
                self.draw()
                frames+=1
                if self.args.screenshot and not saved and frames>=2 and (self.paused or self.loop.total_steps>=10):
                    self.screenshot(self.args.screenshot);saved=True
                g.swap_buffers(self.window)
                time.sleep(.001)
        finally:
            self.context.free();g.destroy_window(self.window);g.terminate()


def headless(loop,args):
    renderer=None
    if args.screenshot:
        from PIL import Image
        loop.model.vis.global_.offwidth=args.width;loop.model.vis.global_.offheight=args.height
        renderer=mujoco.Renderer(loop.model,args.height,args.width)
        renderer.update_scene(loop.data,camera_for(loop,args.overview));add_markers(renderer.scene,loop)
        args.screenshot.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(renderer.render()).save(args.screenshot)
    try:
        for _ in range(math.ceil((args.seconds or 5.)/CONTROL_DT)):
            loop.step()
    finally:
        if renderer:renderer.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt',default='latest',help='Stage 2/3 checkpoint, run directory, or latest')
    ap.add_argument('--device',default='auto',help='auto uses CPU for low-latency single-robot inference; also cpu or cuda:0')
    ap.add_argument('--station',type=int,default=1,help='Initial terrain 1..20')
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--goal-speed',type=float,default=.8,help='WASD goal-cursor speed in m/s')
    ap.add_argument('--speed',type=float,default=1.,help='Simulation/wall time ratio')
    ap.add_argument('--width',type=int,default=1440);ap.add_argument('--height',type=int,default=900)
    ap.add_argument('--autostart',action='store_true',help='Start unpaused (default: press Space)')
    ap.add_argument('--overview',action='store_true')
    ap.add_argument('--headless',action='store_true',help='Run a fixed goal without a window')
    ap.add_argument('--hidden',action='store_true',help=argparse.SUPPRESS)
    ap.add_argument('--seconds',type=float,help='Stop after this many wall seconds (headless: simulated seconds)')
    ap.add_argument('--screenshot',type=Path)
    ap.add_argument('--record',type=Path,help='Optional JSONL state, goal and contact recording')
    ap.add_argument('--out',type=Path,default=ROOT/'artifacts/roban/playground')
    args=ap.parse_args()
    if not 1<=args.station<=20:ap.error('--station must be 1..20')
    if args.goal_speed<=0 or args.speed<=0:ap.error('Speeds must be positive')
    if args.width<900 or args.height<720:ap.error('Window size must be at least 900 x 720')
    if args.seconds is not None and args.seconds<=0:ap.error('--seconds must be positive')
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    if args.device=='auto':args.device='cpu'
    checkpoint=resolve_checkpoint(args.ckpt)
    arena=Playground(args.out,args.seed)
    if args.record:args.record.parent.mkdir(parents=True,exist_ok=True)
    loop=ClosedLoop(checkpoint,arena,args.device,args.record)
    if args.station!=1:loop.reset(args.station-1)
    print(f'Loaded Stage {loop.stage} checkpoint {loop.iteration}: {checkpoint}',flush=True)
    print(f'{len(arena.stations)} terrain zones; device={args.device}; command={loop.command_dim}; planner points={loop.scan.num_z}',flush=True)
    print('Space: run/pause | WASD: goal | Tab: select | G: goal at exit | R: reset | O: overview',flush=True)
    print('H: planner scan (cyan) | J: tracker scan (orange) | P: contact targets; all visible by default',flush=True)
    report=dict(started_at=datetime.now().astimezone().isoformat(),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    try:
        if args.headless:headless(loop,args)
        else:Window(loop,args).run()
    finally:
        report.update(loop.summary());loop.close()
        output=args.out/f'session_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.json'
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        print(f'Session: {output}',flush=True)


if __name__=='__main__':main()
