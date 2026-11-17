"""The contact controller must not depend on commanded/reference body velocity."""
import os
os.environ["ACP_ROBOT"] = "roban"

import unittest
import torch

from tools import test_footstep_tracker as fixture
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.robot import default_joint_pos_list, NUM_J
from supermimic.acp.mdp import XI_DIM
from supermimic.acp.footstep.task import FootstepTask
from supermimic.acp.footstep.backend import backend_for_format, VELOCITY_FORMAT, CONTACT_FORMAT
from supermimic.acp.footstep.contact_only.task import ContactOnlyTask, ContactOnlyMDP, ContactOnlyPolicy


class ContactOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.kin = RobotKinematics('cpu')

    def setUp(self):
        fixture.TrackerTests.setUp(self)
        self.task = ContactOnlyTask(self.motion, self.kin)
        self.mdp = ContactOnlyMDP(self.kin, self.motion, torch.tensor(default_joint_pos_list()), self.task)

    def tearDown(self):
        fixture.TrackerTests.tearDown(self)

    def test_13d_command_matches_contact_part_of_legacy(self):
        f = torch.tensor([0, 0]); query = torch.tensor([0, 1]); S = self.motion.ref_state(f)
        expected = FootstepTask(self.motion, self.kin).command(S, f, query)[:, 3:]
        actual = self.task.command(S, f, query)
        self.assertEqual(actual.shape, (2, 13))
        torch.testing.assert_close(actual, expected)
        self.assertEqual(float(actual[1, 1]), 0.)
        self.assertTrue((actual[1, 3:6] == 0).all())

    def test_speed_and_reference_trajectory_do_not_affect_reward_or_command(self):
        f = torch.tensor([0, 1]); query = torch.tensor([0, 1]); S = self.motion.ref_state(f)
        zero = torch.zeros(2, NUM_J)
        xi = torch.zeros(2, XI_DIM)
        before = self.task.command(S, f, query)
        reward = self.mdp.reward(S, f, zero, zero)
        critic = self.mdp.critic_input(S, f, query, xi)
        self.motion.velocity_command[:] = float('nan')
        self.motion.root_v += 100
        self.motion.root_w += 100
        self.motion.body_v += 100
        self.motion.body_w += 100
        self.motion.root_p[:, :2] += 100
        self.motion.q[:, 1:13] += 100
        torch.testing.assert_close(self.task.command(S, f, query), before)
        torch.testing.assert_close(self.mdp.reward(S, f, zero, zero), reward)
        torch.testing.assert_close(self.mdp.critic_input(S, f, query, xi), critic)
        self.assertEqual(critic.shape[-1], self.mdp.critic_input_dim)

    def test_position_target_changes_reward_and_gradients_are_finite(self):
        f = torch.tensor([0]); S = list(self.motion.ref_state(f))
        S[0] = S[0].clone().requires_grad_()
        zero = torch.zeros(1, NUM_J)
        first = self.mdp.reward(S, f, zero, zero)
        self.motion.feet[0, :, 0] += .25
        second = self.mdp.reward(S, f, zero, zero)
        self.assertLess(float(second.detach()), float(first.detach()))
        second.sum().backward()
        self.assertTrue(torch.isfinite(S[0].grad).all())

    def test_policy_only_accepts_13d_and_v1_dispatch_is_unchanged(self):
        pi = ContactOnlyPolicy(hist=2, hidden=[32])
        self.assertEqual(pi.enc[0].in_features, 13)
        action = pi(torch.zeros(3, 2, 69), torch.zeros(3, 13))
        self.assertEqual(action.shape, (3, 21))
        with self.assertRaises(RuntimeError):
            pi(torch.zeros(3, 2, 69), torch.zeros(3, 16))
        self.assertIs(backend_for_format(VELOCITY_FORMAT).Task, FootstepTask)
        self.assertIs(backend_for_format(CONTACT_FORMAT).Task, ContactOnlyTask)

    def test_termination_does_not_depend_on_reference_root_pose(self):
        f = torch.tensor([0]); S = self.motion.ref_state(f)
        expected = self.mdp.soft_termination(S, f)
        self.motion.root_p[:, 2] += 100
        self.motion.root_R[:, 2, 2] = -1
        torch.testing.assert_close(self.mdp.soft_termination(S, f), expected)
        self.assertFalse(self.mdp.termination(S, f).any())


if __name__ == '__main__':
    unittest.main()
