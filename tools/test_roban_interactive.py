"""Integration checks: actual physics/scans, checkpoint inference, controls, and shared command compatibility.

ACP_ROBOT=roban PYTHONPATH=. /home/liujunjie/InstinctMJ/.venv/bin/python tools/test_roban_interactive.py
"""
from pathlib import Path
from types import SimpleNamespace
import json
import math
import os
import unittest
from unittest.mock import patch

os.environ.setdefault('ACP_ROBOT','roban')
import mujoco
import numpy as np
import torch

from supermimic.acp.interactive import ClosedLoop, GoalCursor, resolve_checkpoint
from supermimic.acp.playground import Playground
from supermimic.acp.planner import plan
from supermimic.acp.roban import RobanPdActuator, PARAMS
from supermimic.acp.robot import JOINT_NAMES, STAND_HEIGHT
from supermimic.acp.task import ContactTask, contact_command, to_base
from supermimic.acp.terrain import box_heights, height_scan

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/roban/playground/validation'


class InteractiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(42)
        cls.arena=Playground(OUT,seed=0)
        cls.loop=ClosedLoop(resolve_checkpoint(os.environ.get('ACP_INTERACTIVE_CKPT','latest')),cls.arena,'cpu')
        cls.evidence={}

    @classmethod
    def tearDownClass(cls):
        cls.loop.close()
        (OUT/'integration_results.json').write_text(json.dumps(cls.evidence,indent=2)+'\n')

    def test_terrain_scans_match_mujoco_rays(self):
        rng=np.random.default_rng(42)
        xy=rng.uniform([-1.5,-2.5],[35.5,24.5],(700,2))
        # Include every ramp/rough interior, every box top and every station start/exit.
        extra=[]
        for s in self.arena.stations:
            extra.extend([s.start[:2],s.goal[:2]])
        for i in [5,6,7,8,9]:
            ox,oy=self.arena.stations[i].origin
            extra.extend(rng.uniform([ox+1.1,oy-1.1],[ox+5.9,oy+1.1],(80,2)))
        xy=np.concatenate([xy,np.asarray(extra),self.arena.boxes[:,:2]])
        heights=box_heights(torch.tensor(xy,dtype=torch.float32)[None],self.arena.tensor('cpu'))[0].numpy()
        np.testing.assert_allclose(self.arena.height(xy),heights,atol=1e-5)
        gid=np.empty(1,dtype=np.int32);group=np.array([0,0,1,0,0,0],dtype=np.uint8)
        actual=[]
        for x,y in xy:
            dist=mujoco.mj_ray(self.loop.model,self.loop.data,np.array([x,y,3.]),np.array([0.,0.,-1.]),group,1,-1,gid)
            self.assertGreaterEqual(dist,0.)
            actual.append(3.-dist)
        error=np.abs(np.asarray(actual)-heights)
        self.assertLess(float(error.max()),2e-5)
        # Local culling must not remove anything visible to the full scan.
        for s in self.arena.stations:
            center=s.start[:2]+[2.8,0.]
            pts=rng.uniform(center-[.95,.95],center+[.95,.95],(120,2))
            full=box_heights(torch.tensor(pts,dtype=torch.float32)[None],self.arena.tensor('cpu'))
            local=box_heights(torch.tensor(pts,dtype=torch.float32)[None],self.arena.tensor('cpu',center))
            torch.testing.assert_close(local,full)
        self.evidence['terrain']=dict(ray_samples=len(xy),max_height_error_m=float(error.max()),local_scan_parity=True)

    def test_controller_matches_training_and_limits(self):
        loop=self.loop;rng=np.random.default_rng(1)
        q=rng.normal(0,.7,(128,21));qd=rng.normal(0,7,(128,21));target=rng.normal(0,2,(128,21))
        tensor=lambda x:torch.tensor(x,dtype=torch.float64)
        cmd=SimpleNamespace(pos=tensor(q),vel=tensor(qd),position_target=tensor(target),
            velocity_target=torch.zeros(128,21,dtype=torch.float64),effort_target=torch.zeros(128,21,dtype=torch.float64))
        params=dict(stiffness=tensor(loop.kp),damping=tensor(loop.kd),force_limit=tensor(loop.limits))
        actual=RobanPdActuator.control_law(params,cmd).numpy()
        expected=np.clip(loop.kp*(target-q)-loop.kd*qd-.2*np.tanh(qd/.1),-loop.limits,loop.limits)
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        np.testing.assert_allclose(loop.model.actuator_forcerange,np.stack([-loop.limits,loop.limits],axis=1))
        np.testing.assert_allclose(loop.model.dof_armature[loop.vadr],[PARAMS[n][4] for n in JOINT_NAMES])
        self.assertEqual(loop.model.opt.timestep,.005)
        self.assertEqual(loop.model.opt.solver,mujoco.mjtSolver.mjSOL_NEWTON)
        self.evidence['controller']=dict(random_states=128,training_torque_parity=True,effort_limits_equal=True)

    def test_keyboard_goal_world_and_robot_axes(self):
        cursor=GoalCursor(self.arena);cursor.set([.25,0],yaw=0.)
        cursor.move({'W'},1.,math.pi/2,.8)
        np.testing.assert_allclose(cursor.position[:2],[.25,.8],atol=1e-6)
        cursor.move({'A'},1.,math.pi/2,.8)
        np.testing.assert_allclose(cursor.position[:2],[-.55,.8],atol=1e-6)
        cursor.robot_axes=False;old=cursor.position.copy()
        cursor.move({'W','D'},1.,math.pi/2,.8)
        self.assertAlmostEqual(np.linalg.norm(cursor.position[:2]-old[:2]),.8)
        cursor.move({'Q'},.5,0.,.8);self.assertAlmostEqual(cursor.yaw,.6)
        cursor.set(self.arena.boxes[0,:2]);self.assertAlmostEqual(cursor.position[2],float(self.arena.height(cursor.position[:2]))+STAND_HEIGHT)
        cursor.set([1e5,-1e5]);self.assertLess(cursor.position[0],self.arena.bounds[1]);self.assertGreater(cursor.position[1],self.arena.bounds[2])
        self.evidence['keyboard']=dict(robot_axes=True,world_axes=True,diagonal_speed_normalized=True,height_snapped=True)

    def test_shared_inference_without_reference(self):
        loop=self.loop;loop.reset(0);loop.step()
        self.assertEqual(loop.last_command.shape,(93,))
        self.assertEqual(loop.scan.num_z,187)
        self.assertTrue(np.isfinite(loop.last_command).all())
        S=loop.state();boxes=self.arena.tensor('cpu',loop.data.qpos[:2])
        zeros=torch.zeros(1,dtype=torch.long);goal=torch.tensor([[.7,-.2,.3]])
        proxy=SimpleNamespace(motion=SimpleNamespace(boxes=lambda f,v:boxes))
        a,av=plan(loop.planner,loop.scan,loop.pcfg,proxy,loop.history,S,zeros,zeros,None,noise=False,goal=goal)
        b,bv=plan(loop.planner,loop.scan,loop.pcfg,None,loop.history,S,None,None,None,noise=False,goal=goal,boxes=boxes)
        torch.testing.assert_close(a,b);torch.testing.assert_close(av,bv)
        proxy.cfg=loop.ccfg;proxy.scan_offsets=loop.offsets
        old=ContactTask.command(proxy,S,zeros,zeros,None,False,targets_w=a,valid=av,goal=goal)
        new=contact_command(loop.ccfg,S,a,av,goal,boxes,loop.offsets)
        torch.testing.assert_close(new,old)
        torch.testing.assert_close(new[:,27:30],goal)
        torch.testing.assert_close(new[:,30:],height_scan(S[0],S[1],boxes,loop.offsets))
        self.evidence['interface']=dict(command_dim=93,planner_points=187,explicit_goal_plan_parity=True,reference_required=False)

    def test_display_scans_match_network_inputs_and_terrain(self):
        loop=self.loop;captured={};max_error=0.
        def planner_input(module,inputs):captured['planner']=inputs[2].detach().clone()
        def tracker_input(module,inputs):captured['tracker']=inputs[1].detach().clone()
        handles=[loop.planner.register_forward_pre_hook(planner_input),loop.pi.register_forward_pre_hook(tracker_input)]
        try:
            for station in (0,2,5,12):
                for yaw in (0.,.7):
                    loop.reset(station)
                    self.assertTrue(loop.scan_preview)
                    self.assertEqual(loop.planner_scan_points.shape,(187,3))
                    self.assertEqual(loop.tracker_scan_points.shape,(63,3))
                    # Across a stair edge / box edge, or on a slope, with a rotated root.
                    xy=np.asarray(self.arena.stations[station].origin)+[2.31,.07]
                    loop.data.qpos[:3]=[*xy,float(self.arena.height(xy))+STAND_HEIGHT]
                    loop.data.qpos[3:7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
                    mujoco.mj_forward(loop.model,loop.data)
                    S=loop.state();before=loop.data.time
                    loop.step()
                    self.assertFalse(loop.scan_preview)
                    self.assertEqual(loop.scan_time,before)
                    planner=to_base(torch.tensor(loop.planner_scan_points)[None],S[0],S[1])
                    tracker=to_base(torch.tensor(loop.tracker_scan_points)[None],S[0],S[1])
                    torch.testing.assert_close(planner,captured['planner'],atol=3e-6,rtol=1e-5)
                    torch.testing.assert_close(tracker[0,:,:2],loop.offsets,atol=3e-6,rtol=1e-5)
                    torch.testing.assert_close(-tracker[...,2],captured['tracker'][:,30:],atol=3e-6,rtol=1e-5)
                    points=np.concatenate((loop.planner_scan_points,loop.tracker_scan_points))
                    error=float(np.max(np.abs(points[:,2]-self.arena.height(points[:,:2]))))
                    max_error=max(max_error,error)
                    self.assertLess(error,3e-6)
            self.assertGreater(np.ptp(loop.planner_scan_points[:,2]),.4)
            self.assertGreater(np.ptp(loop.tracker_scan_points[:,2]),.4)
        finally:
            for handle in handles:handle.remove()
        self.evidence['scan_display']=dict(planner_points=187,tracker_points=63,poses=8,
            matches_actual_network_inputs=True,max_terrain_error_m=max_error,reset_preview=True)

    def test_all_station_resets_and_closed_loop_flat(self):
        loop=self.loop
        for i in range(20):
            loop.reset(i)
            for _ in range(5):loop.step()
            self.assertTrue(np.isfinite(loop.data.qpos).all())
            self.assertTrue(np.isfinite(loop.last_action).all())
            self.assertLessEqual(abs(loop.last_action).max(),20.)
            self.assertAlmostEqual(loop.data.time,.1)
            self.assertFalse(loop.warnings,f'{i+1}: {loop.warnings}')
        loop.reset(0);start=loop.data.qpos.copy()
        for _ in range(100):loop.step()
        self.assertAlmostEqual(loop.data.time,2.)
        self.assertFalse(loop.warnings)
        distance=float(np.linalg.norm(loop.data.qpos[:2]-start[:2]))
        self.assertGreater(distance,.2)
        self.evidence['rollout']=dict(station_resets_tested=20,steps_per_station=5,flat_seconds=2.,flat_distance_m=distance,
                                      checkpoint=str(loop.checkpoint),finite=True,mean_control_ms=loop.summary()['mean_control_ms'])

    def test_diagnostics_never_stop_and_manual_reset_recovers(self):
        loop=self.loop;loop.reset(0)
        resets=loop.resets
        # Deliberately put the robot outside the arena, low and on its side.
        loop.data.qpos[:3]=[self.arena.bounds[1]+1.,0.,.15]
        loop.data.qpos[3:7]=[math.sqrt(.5),0.,math.sqrt(.5),0.]
        mujoco.mj_forward(loop.model,loop.data)
        loop.update_diagnostics()
        self.assertTrue(any('Low root' in w for w in loop.warnings))
        self.assertTrue(any('Torso tilt' in w for w in loop.warnings))
        self.assertIn('Outside playground',loop.warnings)
        for _ in range(10):loop.step()
        self.assertAlmostEqual(loop.data.time,.2)
        self.assertEqual(loop.resets,resets)
        self.assertFalse(loop.summary()['automatic_termination'])
        self.assertTrue(np.isfinite(loop.data.qpos).all())
        loop.reset(0)
        self.assertEqual(loop.resets,resets+1)
        self.assertEqual(loop.data.time,0.)
        self.assertEqual(loop.warnings,[])
        self.assertIsNone(loop.history)
        self.assertTrue(bool((loop.previous==0).all()))
        loop.step()
        self.assertAlmostEqual(loop.data.time,.02)
        self.evidence['manual_reset']=dict(low_root_stops=False,tilt_stops=False,bounds_stop=False,
                                            automatic_resets=False,reset_clears_history=True)

    def test_nonfinite_output_or_step_does_not_latch_stop(self):
        loop=self.loop;loop.reset(0);loop.step()
        resets=loop.resets;previous=loop.previous.clone();before=loop.data.time
        with patch.object(loop.pi,'forward',return_value=torch.full_like(previous,float('nan'))):
            loop.step()
        self.assertAlmostEqual(loop.data.time,before+.02)
        np.testing.assert_array_equal(loop.last_action,previous.numpy()[0])
        self.assertTrue(any('Nonfinite action' in w for w in loop.warnings))
        before=loop.data.time;qpos=loop.data.qpos.copy();qvel=loop.data.qvel.copy()
        def invalid_step(model,data):
            data.qpos[0]=np.nan
            data.time+=model.opt.timestep
        with patch('mujoco.mj_step',side_effect=invalid_step):loop.step()
        self.assertEqual(loop.data.time,before)
        np.testing.assert_array_equal(loop.data.qpos,qpos)
        np.testing.assert_array_equal(loop.data.qvel,qvel)
        self.assertTrue(any('Invalid numerical step' in w for w in loop.warnings))
        loop.step()
        self.assertAlmostEqual(loop.data.time,before+.02)
        self.assertEqual(loop.resets,resets)
        self.assertFalse(loop.warnings)
        self.assertTrue(loop.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_AUTORESET)
        self.evidence['numerics']=dict(nonfinite_action_retried=True,invalid_step_discarded=True,next_step_continues=True)


if __name__=='__main__':unittest.main(verbosity=2)
