"""Physical semantics: planar support, real future deadlines, slip and reset-safe labels."""
import os
os.environ['ACP_ROBOT']='roban'

import unittest
import torch

from tools import test_footstep_tracker as fixture
from supermimic.acp.footstep.consistent_joint import ConsistentJointTask,ConsistentJointMDP,ConsistentJointPolicy,HindsightSamples
from supermimic.acp.footstep.leg_collision import LegCollisionPolicy
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.robot import default_joint_pos_list
from supermimic.acp.mdp import XI_DIM,flatten_state
from supermimic.acp.buffer import ReplayBuffer
from supermimic.acp.task import to_base


class ConsistentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.kin=RobotKinematics('cpu')

    def setUp(self):
        fixture.TrackerTests.setUp(self)
        self.task=ConsistentJointTask(self.motion,self.kin)
        self.mdp=ConsistentJointMDP(self.kin,self.motion,torch.tensor(default_joint_pos_list()),self.task)
        self.policy=ConsistentJointPolicy(hidden=[32],planner_config=dict(hist=5,hidden=[16]))

    def tearDown(self):fixture.TrackerTests.tearDown(self)

    def stance(self):
        S=list(self.motion.ref_state(torch.tensor([0])))
        S[0]=S[0].clone();S[2]=torch.zeros_like(S[2]);S[3]=torch.zeros_like(S[3]);S[5]=torch.zeros_like(S[5])
        gap=self.task.ground.foot_state(S)[2]
        S[0][:,2]-=gap.min(-1).values
        return S

    def test_projected_support_is_on_plane_and_raw_height_cannot_encode_velocity(self):
        S=self.stance();obs=self.mdp.obs(S,torch.zeros(1,21),torch.zeros(1,XI_DIM))
        hist=obs[:,None].expand(-1,5,-1);cmd=torch.tensor([[.2,0.,0.,.02]])
        with torch.no_grad():self.policy.planner.contact.bias.fill_(10.)
        before=self.policy.contact_plan(hist,cmd)
        with torch.no_grad():self.policy.planner.position.bias[2]=100.;self.policy.planner.position.bias[5]=-100.
        after=self.policy.contact_plan(hist,cmd)
        torch.testing.assert_close(after,before)
        torch.testing.assert_close(after[:,[5,8]]+S[0][:,2,None],self.policy.ground.flat_sole_clearance.expand(1,2),atol=1e-6,rtol=0)
        after.sum().backward()
        self.assertEqual(float(self.policy.planner.position.bias.grad[[2,5]].abs().sum()),0.)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.policy.parameters()))

    def test_kinematic_contact_point_slip_and_reference_swing_are_penalized(self):
        S=self.stance();stand=self.mdp.physical_gait_terms(S,torch.tensor([0]))
        self.assertEqual(float(stand['slip_cost']),0.)
        moving=list(S);moving[2]=torch.tensor([[.2,0.,0.]])
        self.assertGreater(float(self.mdp.physical_gait_terms(moving,torch.tensor([0]))['slip_cost']),.1)
        swing=self.mdp.physical_gait_terms(S,torch.tensor([3]))
        self.assertGreater(float(swing['phase_contact_cost']),float(stand['phase_contact_cost'])+.1)
        self.motion.root_p[3,2]+=.2
        self.assertGreater(float(self.mdp.physical_gait_terms(S,torch.tensor([3]))['swing_clearance_cost']),0.)

    def test_future_pair_uses_its_deadline_and_does_not_clamp_long_query_to_last_state(self):
        H=3;S=self.stance()
        states=tuple(torch.stack([x.clone() for _ in range(H+1)]) for x in S)
        states=list(states);states[0]=states[0].clone();states[0][:,0,0]+=torch.arange(H+1)*.01
        flat=tuple(x.flatten(0,1) for x in states)
        feet,R,gap,_=self.task.ground.foot_state(flat)
        feet=feet.reshape(H+1,1,2,3);R=R.reshape(H+1,1,2,3,3);gap=gap.reshape(H+1,1,2)
        plans=torch.zeros(H,1,15);plans[:,:,0]=.06;plans[:,:,1:3]=1.
        plans[:,:,9:13:2]=R[-1,:, :,1,0]/(R[-1,:,:,0,0].square()+R[-1,:,:,1,0].square()).sqrt()
        plans[:,:,10:13:2]=R[-1,:,:,0,0]/(R[-1,:,:,0,0].square()+R[-1,:,:,1,0].square()).sqrt()
        plans[:,:,13:15]=torch.sigmoid((self.task.cfg.contact_distance-gap[-1])/self.task.cfg.contact_temperature)
        for t in range(H):plans[t,:,3:9]=to_base(feet[-1],states[0][t],states[1][t]).flatten(1)
        query=self.task.pack_queries(torch.tensor([0]),torch.tensor([1]))
        self.task.cfg.plan_ground_weight=0.
        loss=lambda plan:self.task.trajectory_supervision_loss(tuple(states),query,plan,torch.ones(1,dtype=torch.bool),torch.ones(H,1),torch.ones(1))
        good=loss(plans)
        changed=plans.clone();changed[1:,:,3:9]=999.
        torch.testing.assert_close(loss(changed),good)
        bad=plans.clone();bad[0,0,3]+=.2;bad.requires_grad_()
        value=loss(bad);self.assertGreater(float(value.detach()),float(good)+.1)
        gradient=torch.autograd.grad(value,bad)[0]
        self.assertGreater(float(gradient[0,0,3]),0.)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_hindsight_labels_use_future_real_state_and_exclude_reset_windows(self):
        buffer=ReplayBuffer(8,1,'cpu',extra={'o':(69,torch.float32)})
        S=self.stance();previous=torch.zeros(1,21);xi=torch.zeros(1,XI_DIM)
        for t in range(6):
            current=[x.clone() for x in S];future=[x.clone() for x in S]
            current[0][:,0]+=.01*t;future[0][:,0]+=.01*(t+1)
            f=torch.tensor([t]);query=self.task.pack_queries(f,torch.tensor([1]))
            buffer.add(dict(s=flatten_state(current),s1=flatten_state(future),a=previous,a_prev=previous,xi=xi,
                            o=self.mdp.obs(current,previous,xi),r=torch.zeros(1),f=f,v=torch.zeros(1,dtype=torch.long),
                            goal_steps=query,goal=torch.zeros(1,4),ep_step=torch.tensor([t],dtype=torch.int32),
                            done=torch.zeros(1,dtype=torch.bool),valid=torch.ones(1,dtype=torch.bool)))
        sampler=HindsightSamples(self.task,5);batch=sampler.sample(buffer,128)
        valid=batch['ft_valid'];self.assertTrue(valid.any())
        feet=self.task.ground.foot_state(S)[0]
        expected=to_base(feet,S[0],S[1]);expected[:,:,0]+=.03
        torch.testing.assert_close(batch['ft_position'][valid],expected.expand(int(valid.sum()),-1,-1),atol=1e-6,rtol=0)
        buffer.data['done'].fill_(True)
        invalid=sampler.sample(buffer,32)
        self.assertFalse(invalid['ft_valid'].any())
        self.assertEqual(float(self.task.hindsight_loss(self.policy,invalid).detach()),0.)

    def test_clean_controller_copy_and_geometry_buffers_survive_device_transfer(self):
        old=LegCollisionPolicy(hidden=[32]);self.policy.init_controller(old.state_dict())
        h,g=torch.randn(2,5,69),torch.randn(2,13)
        torch.testing.assert_close(self.policy.controller_action(h,g),old(h,g),rtol=0,atol=0)
        if torch.cuda.is_available():
            self.policy.cuda()
            self.assertEqual(self.policy.ground.kin.axes.device,self.policy.ground.axes.device)
            self.assertEqual(self.policy.ground.kin.axes.data_ptr(),self.policy.ground.axes.data_ptr())


if __name__=='__main__':unittest.main()
