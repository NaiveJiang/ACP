"""Leg style, support-only tilt, transform invariance and legacy policy compatibility."""
import os
os.environ['ACP_ROBOT']='roban'

from dataclasses import asdict
import unittest
import torch

from tools import test_footstep_tracker as fixture
from supermimic.acp.footstep.gait_style import GaitStyleTask,GaitStyleMDP,GaitStylePolicy,GaitStyleCfg
from supermimic.acp.footstep.upper_supervised import UpperSupervisedMDP,UpperSupervisedPolicy
from supermimic.acp.footstep.backend import backend_for_format,GAIT_STYLE_FORMAT
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.robot import default_joint_pos_list
from supermimic.acp.mdp import XI_DIM
from supermimic.rotations import quat_to_mat,mat_to_quat


class GaitStyleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.kin=RobotKinematics('cpu')

    def setUp(self):
        fixture.TrackerTests.setUp(self)
        self.task=GaitStyleTask(self.motion,self.kin)
        self.mdp=GaitStyleMDP(self.kin,self.motion,torch.tensor(default_joint_pos_list()),self.task)

    def tearDown(self):
        fixture.TrackerTests.tearDown(self)

    def test_reference_motion_has_zero_gait_cost_and_large_errors_keep_gradient(self):
        f=torch.tensor([0]);S=list(self.motion.ref_state(f))
        for cost in self.mdp.gait_terms(S,f).values():
            torch.testing.assert_close(cost,torch.zeros_like(cost),atol=1e-8,rtol=0)
        S[4]=S[4].clone().requires_grad_()
        moved=list(S);moved[4]=S[4]+torch.tensor([[0.]+[1.4]*12+[0.]*8])
        cost=self.mdp.gait_terms(moved,f)['leg_pose']
        grad=torch.autograd.grad(cost.sum(),S[4])[0]
        self.assertTrue((grad[:,1:13] > .1).all())
        self.assertEqual(float(grad[:,self.mdp.upper_indices].abs().sum()),0.)
        moved[5]=S[5]+torch.tensor([[0.]+[2.]*12+[0.]*8])
        self.assertGreater(float(self.mdp.gait_terms(moved,f)['leg_velocity']),0.)

    def test_pose_supervision_adds_a_cost_even_when_feet_already_match(self):
        f=torch.tensor([0]);S=self.motion.ref_state(f);a=torch.zeros(1,21)
        old=UpperSupervisedMDP(self.kin,self.motion,torch.tensor(default_joint_pos_list()),self.task)
        before=self.mdp.reward(S,f,a,a)
        torch.testing.assert_close(before,old.reward(S,f,a,a),atol=1e-7,rtol=0)
        # Isolate the new label: measured reference feet and actual FK stay fixed.
        self.motion.q[0,3]+=.3
        self.assertLess(float(self.mdp.reward(S,f,a,a)),float(old.reward(S,f,a,a)))

    def test_swing_foot_tilt_is_not_penalized_as_support(self):
        f=torch.tensor([0]);S=self.motion.ref_state(f)
        p,R,gap=self.task.foot_geometry(S);R=R.clone()
        angle=torch.tensor(.4);c,s=angle.cos(),angle.sin()
        Rx=torch.tensor([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
        R[:,0]=R[:,0]@Rx
        self.task.foot_geometry=lambda state:(p,R,gap)
        self.motion.contact[0,0]=False
        torch.testing.assert_close(self.mdp.gait_terms(S,f)['stance_tilt'],torch.zeros(1),atol=1e-8,rtol=0)
        self.motion.contact[0,0]=True
        self.assertGreater(float(self.mdp.gait_terms(S,f)['stance_tilt']),.1)

    def test_world_translation_yaw_and_velocity_commands_do_not_lock_motion(self):
        f=torch.tensor([0]);query=torch.tensor([0]);S=self.motion.ref_state(f);a=torch.zeros(1,21)
        goal=self.task.command(S,f,query);reward=self.mdp.reward(S,f,a,a)
        c,s=torch.tensor(.8).cos(),torch.tensor(.8).sin()
        H=torch.tensor([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        t=torch.tensor([3.,-2.,0.])
        moved=(S[0]@H.T+t,H@S[1],S[2]@H.T,S[3]@H.T,S[4],S[5])
        self.motion.root_p=self.motion.root_p@H.T+t
        self.motion.root_R=H@self.motion.root_R
        self.motion.feet[:,:,:3]=self.motion.feet[:,:,:3]@H.T+t
        self.motion.feet[:,:,3:7]=mat_to_quat(H@quat_to_mat(self.motion.feet[:,:,3:7]))
        self.motion.velocity_command.fill_(999.)
        self.motion.root_v.fill_(999.);self.motion.root_w.fill_(999.)
        torch.testing.assert_close(self.task.command(moved,f,query),goal,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(self.mdp.reward(moved,f,a,a),reward,atol=1e-6,rtol=1e-6)

    def test_knee_hyperextension_has_restoring_gradient(self):
        f=torch.tensor([0]);S=list(self.motion.ref_state(f))
        S[4]=S[4].clone();S[4][:,[4,10]]=-.05;S[4].requires_grad_()
        cost=self.mdp.gait_terms(S,f)['knee']
        grad=torch.autograd.grad(cost.sum(),S[4])[0]
        self.assertTrue((grad[:,[4,10]] < 0).all())
        self.assertTrue((grad[:,[0,1,2,3,5,6,7,8,9,11,12]] == 0).all())

    def test_critic_dimensions_v4_policy_copy_and_compiled_reward(self):
        f=torch.tensor([0]);query=torch.tensor([0]);S=list(self.motion.ref_state(f))
        value=self.mdp.critic_input(S,f,query,torch.zeros(1,XI_DIM))
        self.assertEqual(value.shape,(1,self.mdp.critic_input_dim))
        old=UpperSupervisedPolicy(hidden=[32]);pi=GaitStylePolicy(hidden=[32])
        pi.load_state_dict(old.state_dict())
        h,g=torch.randn(2,5,69),torch.randn(2,13)
        torch.testing.assert_close(pi(h,g),old(h,g),rtol=0,atol=0)
        restored=backend_for_format(GAIT_STYLE_FORMAT).RewardCfg(**asdict(self.task.cfg))
        self.assertEqual(restored,self.task.cfg)
        S[4]=S[4].clone().requires_grad_();a=torch.zeros(1,21)
        reward=self.mdp.reward(S,f,a,a)
        compiled=torch.compile(self.mdp.reward,backend='eager',fullgraph=True)
        torch.testing.assert_close(compiled(S,f,a,a),reward)
        self.assertTrue(torch.isfinite(torch.autograd.grad(compiled(S,f,a,a).sum(),S[4])[0]).all())


if __name__=='__main__':unittest.main()
