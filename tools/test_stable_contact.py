"""Regression checks for the bad-posture gradient and physical target bounds."""
import os
os.environ['ACP_ROBOT'] = 'roban'

import io
import unittest

import torch

from tools import test_footstep_tracker as fixture
from supermimic.acp.footstep.stable_contact import StableContactPolicy, StableContactTask, StableContactMDP
from supermimic.acp.footstep.backend import backend_for_format, STABLE_CONTACT_FORMAT
from supermimic.acp.robot import default_joint_pos_list, JOINT_NAMES, action_scale
from supermimic.acp.mdp import XI_DIM
from supermimic.acp.kinematics import RobotKinematics


class StableContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.kin = RobotKinematics('cpu')

    def setUp(self):
        fixture.TrackerTests.setUp(self)
        self.task = StableContactTask(self.motion, self.kin)
        self.mdp = StableContactMDP(self.kin, self.motion, torch.tensor(default_joint_pos_list()), self.task)

    def tearDown(self):
        fixture.TrackerTests.tearDown(self)

    def test_extreme_policy_and_noisy_exploration_keep_pd_targets_inside_soft_limits(self):
        policy = StableContactPolicy(hidden=[32])
        with torch.no_grad():
            policy.dec[-1].weight.zero_()
            for sign in (-1, 1):
                policy.dec[-1].bias.fill_(sign*1000)
                a = policy(torch.randn(4,5,69), torch.randn(4,13))
                for offset in (-.01, .01):
                    for action in (a, policy.clamp_action(a+torch.randn_like(a)*100)):
                        target = torch.tensor(default_joint_pos_list())+offset+torch.tensor([action_scale(n) for n in JOINT_NAMES])*action
                        self.assertTrue((target >= self.mdp.soft_lo-1e-6).all())
                        self.assertTrue((target <= self.mdp.soft_hi+1e-6).all())
            policy.dec[-1].bias.zero_()
            torch.testing.assert_close(policy(torch.zeros(1,5,69),torch.zeros(1,13)),torch.zeros(1,21))

    def test_large_upper_pose_error_has_restoring_gradient_and_preserves_reference_motion(self):
        f = torch.tensor([0]); S = list(self.motion.ref_state(f))
        S[4] = S[4].clone().requires_grad_()
        a = torch.zeros(1,21)
        ref = self.mdp.regularization_terms(S,f,a,a)
        self.assertEqual(float(ref['upper_pose'].detach()), 0.)
        bad = list(S); bad[4] = S[4]+torch.tensor([[1.2]+[0.]*12+[.9]*8])
        cost = self.mdp.regularization_terms(bad,f,a,a)['upper_pose']
        grad = torch.autograd.grad(cost.sum(),S[4])[0]
        self.assertGreater(float(cost.detach()),.5)
        self.assertTrue((grad[:,self.mdp.upper_indices] > .1).all())
        self.assertTrue((grad[:,1:13] == 0).all())
        moving = list(S); moving[5] = S[5].clone(); moving[5][:,13] += 2.
        self.assertGreater(float(self.mdp.regularization_terms(moving,f,a,a)['upper_velocity']),0.)

    def test_no_desired_velocity_input_and_privileged_critic_dimensions(self):
        f = torch.tensor([0]); query = torch.tensor([0]); S = self.motion.ref_state(f)
        before = self.task.command(S,f,query)
        reward = self.mdp.reward(S,f,torch.zeros(1,21),torch.zeros(1,21))
        self.motion.velocity_command.fill_(123.)
        torch.testing.assert_close(self.task.command(S,f,query),before)
        torch.testing.assert_close(self.mdp.reward(S,f,torch.zeros(1,21),torch.zeros(1,21)),reward)
        self.assertEqual(before.shape,(1,13))
        value = self.mdp.critic_input(S,f,query,torch.zeros(1,XI_DIM))
        self.assertEqual(value.shape,(1,self.mdp.critic_input_dim))

    def test_smoothness_cost_and_compiled_reward_gradients(self):
        f = torch.tensor([0]); S = self.motion.ref_state(f)
        prev = torch.zeros(1,21)
        a = torch.full((1,21),.3,requires_grad=True)
        low = self.mdp.reward(S,f,prev,prev)
        high = self.mdp.reward(S,f,a,prev)
        self.assertGreater(float(low.detach()),float(high.detach()))
        compiled = torch.compile(self.mdp.reward,backend='eager',fullgraph=True)
        torch.testing.assert_close(compiled(S,f,a,prev),high)
        grad = torch.autograd.grad(compiled(S,f,a,prev).sum(),a)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue((grad < 0).all())

    def test_checkpoint_dispatch_restores_action_bounds(self):
        policy = StableContactPolicy(hidden=[32])
        data = io.BytesIO()
        torch.save(dict(format=STABLE_CONTACT_FORMAT,pi=policy.state_dict()),data)
        data.seek(0); ck = torch.load(data,weights_only=True)
        restored = backend_for_format(ck['format']).Policy(hidden=[32])
        restored.load_state_dict(ck['pi'])
        h,g = torch.randn(2,5,69),torch.randn(2,13)
        torch.testing.assert_close(restored(h,g),policy(h,g))


if __name__ == '__main__':
    unittest.main()
