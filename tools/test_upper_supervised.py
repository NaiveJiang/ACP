"""Direct upper supervision must survive saturation and leave the leg head alone."""
import os
os.environ['ACP_ROBOT'] = 'roban'

import io
import unittest
import torch

from tools import test_footstep_tracker as fixture
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.mdp import XI_DIM, flatten_state
from supermimic.acp.footstep.stable_contact import StableContactPolicy
from supermimic.acp.footstep.upper_supervised import UpperSupervisedPolicy, UpperSupervisedTask
from supermimic.acp.footstep.backend import backend_for_format, UPPER_SUPERVISED_FORMAT


class UpperSupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.kin = RobotKinematics('cpu')

    def setUp(self):
        fixture.TrackerTests.setUp(self)
        self.task = UpperSupervisedTask(self.motion, self.kin)
        self.policy = UpperSupervisedPolicy(hidden=[32])
        f = torch.tensor([0, 1])
        self.batch = dict(s=flatten_state(self.motion.ref_state(f)), f=f,
                          hist=torch.zeros(2,5,69), goal_steps=torch.zeros(2,dtype=torch.long),
                          xi=torch.zeros(2,XI_DIM), usable=torch.ones(2,dtype=torch.bool))

    def tearDown(self):
        fixture.TrackerTests.tearDown(self)

    def test_v3_initialization_preserves_exact_leg_mapping(self):
        old = StableContactPolicy(hidden=[32])
        with torch.no_grad():old.dec[-1].weight.normal_(0,.1)
        self.policy.init_from_v3(old.state_dict())
        h, g = torch.randn(3,5,69), torch.randn(3,13)
        torch.testing.assert_close(self.policy(h,g)[:,1:13],old(h,g)[:,1:13],rtol=0,atol=0)

    def test_wrong_saturated_upper_joint_has_gradient_without_world_model(self):
        self.motion.q[1:3,15] = -.2
        with torch.no_grad():self.policy.upper_dec[-1].bias[3] = 100.
        loss = self.task.supervision_loss(self.policy,self.batch)
        loss.backward()
        self.assertGreater(float(self.policy.upper_dec[-1].bias.grad[3]),0.)
        self.assertTrue(all(p.grad is None for p in self.policy.enc.parameters()))
        self.assertTrue(all(p.grad is None for p in self.policy.dec.parameters()))
        for p in self.policy.parameters():
            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())
        h,g = torch.randn(2,5,69),torch.randn(2,13)
        before = self.policy(h,g)[:,1:13].detach().clone()
        torch.optim.SGD(self.policy.parameters(),lr=.01).step()
        torch.testing.assert_close(self.policy(h,g)[:,1:13],before,rtol=0,atol=0)

    def test_next_frame_targets_follow_physical_mapping_and_encoder_offset(self):
        self.batch['xi'][:,1:22] = .005
        f = self.batch['f'];ids = self.task.upper_indices
        target = self.task.upper_target(self.policy,f,self.batch['xi'])
        reconstructed = self.task.default_q[ids]+self.batch['xi'][:,1:22][:,ids]+self.task.alpha[ids]*target
        expected = self.motion.q[f+1][:,ids]+self.task.damping_lead[ids]*self.motion.qd[f+1][:,ids]
        torch.testing.assert_close(reconstructed,expected,rtol=0,atol=1e-7)
        earlier = target.clone()
        self.motion.q[f+1,16] -= .1
        after = self.task.upper_target(self.policy,f,self.batch['xi'])
        self.assertTrue((after[:,4] < earlier[:,4]).all())

    def test_masked_nonfinite_rows_are_ignored_and_all_invalid_has_zero_gradient(self):
        self.batch['usable'][1] = False
        good = self.task.supervision_loss(self.policy,self.batch)
        for key in ('s','hist','xi'):self.batch[key][1] = float('nan')
        torch.testing.assert_close(self.task.supervision_loss(self.policy,self.batch),good)
        self.batch['usable'].zero_()
        loss = self.task.supervision_loss(self.policy,self.batch)
        self.assertEqual(float(loss.detach()),0.)
        loss.backward()
        for p in self.policy.parameters():
            if p.grad is not None:self.assertTrue((p.grad == 0).all())

    def test_13d_interface_compiled_loss_and_checkpoint_roundtrip(self):
        loss = self.task.supervision_loss(self.policy,self.batch)
        compiled = torch.compile(self.task.supervision_loss,backend='eager',fullgraph=True)
        torch.testing.assert_close(compiled(self.policy,self.batch),loss)
        storage = io.BytesIO()
        torch.save(dict(format=UPPER_SUPERVISED_FORMAT,pi=self.policy.state_dict()),storage)
        storage.seek(0);ck=torch.load(storage,weights_only=True)
        pi = backend_for_format(ck['format']).Policy(hidden=[32])
        pi.load_state_dict(ck['pi'])
        h,g=torch.randn(2,5,69),torch.randn(2,13)
        torch.testing.assert_close(self.policy(h,g),pi(h,g))
        a = pi(h,g)
        self.assertTrue(((a >= pi.action_lo) & (a <= pi.action_hi)).all())


if __name__ == '__main__':
    unittest.main()
