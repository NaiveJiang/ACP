"""Cross-leg forces, differentiable separation, primitive masks and old-format safety."""
import os
os.environ['ACP_ROBOT']='roban'

import unittest
import numpy as np
import mujoco
import torch

from tools import test_footstep_tracker as fixture
from supermimic.acp.footstep.leg_collision import (collision_spec,segment_distance,LegCollisionTask,
                                                 LegCollisionMDP,LegCollisionSim)
from supermimic.acp.footstep.backend import backend_for_format,LEG_COLLISION_FORMAT,GAIT_STYLE_FORMAT
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.robot import get_spec,default_joint_pos_list,JOINT_NAMES


class LegCollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.kin=RobotKinematics('cpu')

    def setUp(self):
        fixture.TrackerTests.setUp(self)
        self.task=LegCollisionTask(self.motion,self.kin)
        self.mdp=LegCollisionMDP(self.kin,self.motion,torch.tensor(default_joint_pos_list()),self.task)

    def tearDown(self):fixture.TrackerTests.tearDown(self)

    def test_segment_distance_parallel_crossing_and_point_degeneracy(self):
        a=torch.tensor([[-1.,0.,0.],[-1.,0.,0.],[0.,0.,0.]])
        b=torch.tensor([[1.,0.,0.],[1.,0.,0.],[0.,0.,0.]])
        c=torch.tensor([[-1.,.2,0.],[0.,-1.,0.],[0.,2.,0.]])
        d=torch.tensor([[1.,.2,0.],[0.,1.,0.],[0.,2.,0.]])
        a.requires_grad_()
        value=segment_distance(a,b,c,d)
        torch.testing.assert_close(value,torch.tensor([.2,0.,2.]),atol=2e-6,rtol=0)
        self.assertTrue(torch.isfinite(torch.autograd.grad(value.sum(),a)[0]).all())

    def test_collision_masks_keep_visuals_terrain_and_same_leg_behavior(self):
        old=get_spec().compile();new=collision_spec().compile()
        ids={s:[] for s in ('l','r')}
        for g in range(new.ngeom):
            body=new.body(int(new.geom_bodyid[g])).name
            side='l' if body.startswith('leg_l') else 'r' if body.startswith('leg_r') else None
            active=bool(old.geom_contype[g] or old.geom_conaffinity[g])
            if side and active:
                ids[side].append(g)
                self.assertEqual(int(new.geom_contype[g]),2 if side=='l' else 4)
                self.assertTrue(int(new.geom_conaffinity[g]) & 1)
            else:
                self.assertEqual(new.geom_contype[g],old.geom_contype[g])
                self.assertEqual(new.geom_conaffinity[g],old.geom_conaffinity[g])
        for side in ('l','r'):
            a,b=ids[side][:2]
            self.assertFalse(int(new.geom_contype[a]) & int(new.geom_conaffinity[b]))
        a,b=ids['l'][0],ids['r'][0]
        self.assertTrue(int(new.geom_contype[a]) & int(new.geom_conaffinity[b]))
        # The original robot definition remains terrain-only.
        self.assertTrue((get_spec().compile().geom_contype == old.geom_contype).all())

    def test_forced_crossing_creates_real_contacts_and_forces(self):
        counts=[]
        for spec in (get_spec(),collision_spec()):
            m=spec.compile();d=mujoco.MjData(m)
            d.qpos[:7]=[0,0,.81,1,0,0,0]
            q=default_joint_pos_list();q[2]=-.7;q[8]=0.
            for j,name in enumerate(JOINT_NAMES):d.qpos[m.joint(name).qposadr[0]]=q[j]
            mujoco.mj_forward(m,d)
            counts.append(d.ncon)
            if d.ncon:
                forces=[]
                for i in range(d.ncon):
                    force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force);forces.append(force[0])
                self.assertGreater(max(forces),0.)
        self.assertEqual(counts[0],0)
        self.assertGreater(counts[1],0)

    def test_standing_clear_and_closing_legs_have_restoring_gradient(self):
        f=torch.tensor([0]);S=list(self.motion.ref_state(f))
        self.assertEqual(float(self.mdp.collision_cost(S)),0.)
        S[4]=S[4].clone();S[4][0,2]=-.4;S[4].requires_grad_()
        cost=self.mdp.collision_cost(S)
        grad=torch.autograd.grad(cost.sum(),S[4])[0]
        self.assertGreater(float(cost.detach()),0.)
        self.assertLess(float(grad[0,2]),0.)
        self.assertTrue(torch.isfinite(grad).all())
        moved=list(S);moved[0]=S[0]+torch.tensor([[2.,3.,1.]])
        torch.testing.assert_close(self.mdp.collision_cost(moved),cost,atol=1e-4,rtol=1e-4)

    def test_compiled_reward_and_format_selects_physics(self):
        f=torch.tensor([0]);S=list(self.motion.ref_state(f));a=torch.zeros(1,21)
        S[4]=S[4].clone();S[4][0,2]=-.4;S[4].requires_grad_()
        compiled=torch.compile(self.mdp.reward,backend='eager',fullgraph=True)
        torch.testing.assert_close(compiled(S,f,a,a),self.mdp.reward(S,f,a,a))
        grad=torch.autograd.grad(compiled(S,f,a,a).sum(),S[4])[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertIs(backend_for_format(LEG_COLLISION_FORMAT).Env,LegCollisionSim)
        self.assertFalse(hasattr(backend_for_format(GAIT_STYLE_FORMAT),'Env'))


if __name__=='__main__':unittest.main()
