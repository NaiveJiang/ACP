import os
import unittest

import torch

os.environ.setdefault("ACP_ROBOT", "roban")

from supermimic.acp.footstep.next_contact import (
    NextContactPlanner, hindsight, pack, touchdown_indices, world_plan,
)


class NextContactTest(unittest.TestCase):
    def test_onset_is_future_and_sustained(self):
        contact = torch.zeros(100, 1, 2, dtype=torch.bool)
        contact[:, 0, 1] = True
        contact[5:7, 0, 0] = True
        contact[20:45, 0, 0] = True
        contact[70:, 0, 0] = True
        index = touchdown_indices(contact)
        self.assertEqual(index[0, 0].tolist(), [20, 100])
        self.assertEqual(index[20, 0].tolist(), [70, 100])

    def test_hindsight_masks_censored_feet(self):
        n = 100
        contact = torch.zeros(n, 1, 2, dtype=torch.bool)
        contact[20:, 0, 0] = True
        position = torch.zeros(n, 1, 2, 3)
        position[20, 0, 0, 0] = .2
        roots = torch.zeros(n, 1, 3)
        rotation = torch.eye(3).expand(n, 1, 3, 3)
        velocity = torch.zeros(n, 1, 3)
        velocity[40:, 0, 0] = .2
        batch = hindsight(position, torch.zeros(n, 1, 2), roots, rotation, contact, velocity,
                          torch.zeros(n, 1, 2, 48), torch.ones(1, dtype=torch.bool))
        self.assertEqual(batch["known"][0].tolist(), [True, False])
        self.assertEqual(batch["valid"][0].tolist(), [True, False])
        self.assertAlmostEqual(float(batch["position"][0, 0, 0]), .2)

        velocity.zero_()
        batch = hindsight(position, torch.zeros(n, 1, 2), roots, rotation, contact, velocity,
                          torch.zeros(n, 1, 2, 48), torch.ones(1, dtype=torch.bool))
        self.assertEqual(batch["known"][0].tolist(), [True, True])
        self.assertEqual(batch["valid"][0].tolist(), [True, False])

    def test_planner_and_controller_command_shapes(self):
        history = torch.zeros(3, 2, 48)
        velocity = torch.zeros(3, 3)
        valid = torch.tensor([[True, False]]).expand(3, -1)
        position = torch.tensor([[[.2, .1, -.77], [-.1, -.1, -.8]]]).expand(3, -1, -1)
        heading = torch.tensor([[[0., 1.], [0., 1.]]]).expand(3, -1, -1)
        planner = NextContactPlanner(2)
        loss = planner.loss(dict(history=history, velocity=velocity, known=torch.ones_like(valid),
                                 valid=valid, position=position, heading=heading))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(planner.plan(history, velocity).shape, (3, 12))
        state = torch.ones(3, 3), torch.eye(3).expand(3, -1, -1)
        command = pack(valid, position, heading)
        present, destination, yaw = world_plan(command, state)
        torch.testing.assert_close(present.bool(), valid)
        torch.testing.assert_close(destination[:, 0], position[:, 0]+1)
        self.assertEqual(yaw.shape, (3, 2))
