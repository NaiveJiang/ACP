"""Planner/controller gradients, held commands, offline labels and deployment contract."""
import os
os.environ['ACP_ROBOT']='roban'

from pathlib import Path
import unittest
import numpy as np
import torch

from tools import test_footstep_tracker as fixture
from supermimic.acp.footstep.joint import JointPolicy,JointTask,JointMDP,PlannerSamples,decode_plan
from supermimic.acp.footstep.leg_collision import LegCollisionPolicy
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.mdp import XI_DIM,flatten_state
from supermimic.acp.robot import default_joint_pos_list


class JointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.kin=RobotKinematics('cpu')

    def setUp(self):
        fixture.TrackerTests.setUp(self)
        self.task=JointTask(self.motion,self.kin)
        self.mdp=JointMDP(self.kin,self.motion,torch.tensor(default_joint_pos_list()),self.task)
        self.policy=JointPolicy(hidden=[32],planner_config=dict(hist=5,hidden=[16]))

    def tearDown(self):fixture.TrackerTests.tearDown(self)

    def sampler(self):
        folder=Path(self.temp.name)/'bc_cache';folder.mkdir()
        n=len(self.motion.q);future=np.arange(n)[:,None]+np.array([1,3])[None]
        np.save(folder/'valid.npy',future<n);np.save(folder/'future_index.npy',future)
        self.motion.sources[0].update(cache=str(folder),category='walk')
        return PlannerSamples(self.motion,self.task,5)

    def test_decoder_is_binary_13d_with_contact_gradient_and_unit_heading(self):
        logits=torch.tensor([[1.,-1.]],requires_grad=True)
        pred=dict(contact_logits=logits,position=torch.ones(1,2,3),heading=torch.zeros(1,2,2))
        decoded=decode_plan(pred,torch.tensor([[.3]]))
        self.assertEqual(decoded.shape,(1,13))
        torch.testing.assert_close(decoded[:,1:3],torch.tensor([[1.,0.]]))
        torch.testing.assert_close(decoded[:,6:9],torch.zeros(1,3))
        torch.testing.assert_close(decoded[:,9:11],torch.tensor([[0.,1.]]))
        decoded.sum().backward();self.assertTrue((logits.grad>0).all())

    def test_deployed_controller_only_sees_plan_and_both_networks_get_gradients(self):
        with torch.no_grad():self.policy.planner.contact.bias.fill_(2.)
        hist=torch.randn(3,5,69);cmd=torch.tensor([[.2,0.,0.,.3]]).expand(3,-1)
        plan=self.policy.contact_plan(hist,cmd)
        action=self.policy(hist,cmd)
        torch.testing.assert_close(action,self.policy.controller_action(hist,plan))
        action.sum().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.policy.planner.parameters() if p.grad is not None),0.)
        self.assertGreater(float(self.policy.dec[-1].weight.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in self.policy.teacher.parameters()))

    def test_controller_initialization_and_teacher_are_exact_and_frozen(self):
        old=LegCollisionPolicy(hidden=[32])
        self.policy.init_controller(old.state_dict())
        h,g=torch.randn(2,5,69),torch.randn(2,13)
        torch.testing.assert_close(self.policy.controller_action(h,g),old(h,g),rtol=0,atol=0)
        torch.testing.assert_close(self.policy.teacher(h,g),old(h,g),rtol=0,atol=0)
        self.assertTrue(all(not p.requires_grad for p in self.policy.teacher.parameters()))

    def test_imagination_does_not_look_up_future_velocity_commands(self):
        self.motion.velocity_command[2:]=torch.tensor([-.4,0.,0.])
        task=JointTask(self.motion,self.kin)
        f=torch.tensor([0]);query=task.pack_queries(f,torch.tensor([0]));S=self.motion.ref_state(f)
        torch.testing.assert_close(task.command(S,torch.tensor([4]),query)[:,:3],torch.tensor([[.2,0.,0.]]))
        self.assertFalse(bool(task.eligible(f,query,2)))
        new=task.pack_queries(torch.tensor([2]),torch.tensor([0]))
        torch.testing.assert_close(task.command(S,torch.tensor([4]),new)[:,:3],torch.tensor([[-.4,0.,0.]]))
        mdp=JointMDP(self.kin,self.motion,torch.tensor(default_joint_pos_list()),task);a=torch.zeros(1,21)
        before=mdp.reward_with_query(S,torch.tensor([3]),a,a,query)
        self.motion.velocity_command[3]=999.
        torch.testing.assert_close(mdp.reward_with_query(S,torch.tensor([3]),a,a,query),before)

    def test_velocity_reward_replaces_world_foot_path_and_critic_dimensions_match(self):
        f=torch.tensor([0]);S=self.motion.ref_state(f);a=torch.zeros(1,21)
        query=self.task.pack_queries(f,torch.tensor([0]))
        before=self.mdp.reward_with_query(S,f,a,a,query)
        self.motion.feet[:,:,:3]+=100.
        torch.testing.assert_close(self.mdp.reward_with_query(S,f,a,a,query),before)
        wrong=list(S);wrong[2]=-S[2]
        self.assertLess(float(self.mdp.reward_with_query(wrong,f,a,a,query)),float(before))
        self.assertEqual(self.mdp.critic_input(S,f,query,torch.zeros(1,XI_DIM)).shape,(1,self.mdp.critic_input_dim))

    def test_offline_samples_do_not_borrow_history_and_supervision_compiles(self):
        self.motion.q[:,0]=torch.arange(len(self.motion.q))*.01
        sampler=self.sampler();bc=sampler.sample(32)
        self.assertEqual(bc['bc_history'].shape,(32,5,48))
        torch.testing.assert_close(bc['bc_history'][:,1:,6]-bc['bc_history'][:,:-1,6],torch.full((32,4),.01),atol=1e-7,rtol=0)
        self.assertTrue((bc['bc_history'][:,0,6]>=0).all())
        frames=torch.zeros(32,dtype=torch.long)
        batch=dict(bc,s=flatten_state(self.motion.ref_state(frames)),f=frames,
                   hist=torch.zeros(32,5,69),goal_steps=self.task.pack_queries(frames,frames),
                   xi=torch.zeros(32,XI_DIM),usable=torch.ones(32,dtype=torch.bool))
        expected=self.task.joint_supervision_loss(self.policy,batch)
        compiled=torch.compile(self.task.joint_supervision_loss,backend='eager',fullgraph=True)
        torch.testing.assert_close(compiled(self.policy,batch),expected)
        expected.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.policy.parameters()))


if __name__=='__main__':unittest.main()
