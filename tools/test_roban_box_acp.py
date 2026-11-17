"""Regression checks for ignored side targets and articulated knee-cap contacts."""
import os
os.environ["ACP_ROBOT"] = "roban"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import torch

from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.planner import PlannerBuffer, PlannerCfg, PlannerScan, height_target_supervision, planner_loss
from supermimic.acp.robot import JOINT_NAMES, contact_geom_ids, default_joint_pos_list, get_spec
from supermimic.acp.task import BodySpheres, ContactTask


class BoxCollectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_side_contact_is_ignored_not_negative_or_moved(self):
        scan = PlannerScan(PlannerCfg(), "cpu")
        self.assertEqual(scan.num_z, 187)
        boxes = torch.tensor([[[0., 0., .6, 1., 0., .4, .3, .3, 0.]]])
        S = (torch.zeros(1, 3), torch.eye(3)[None])
        targets = torch.zeros(1, 9, 3)
        targets[0, :4] = torch.tensor([[.1, 0, .6], [.4, 0, .3], [.7, 0, 0], [.4005, 0, .6]])
        saved = targets.clone()
        occ = torch.zeros(1, 9, dtype=torch.bool)
        occ[0, :4] = True
        mask = height_target_supervision(targets, occ, S, boxes)
        self.assertEqual(mask[0, :4].tolist(), [True, False, True, True])
        self.assertTrue(mask[0, 4:].all())  # real negative examples remain supervised
        self.assertTrue(torch.equal(targets, saved))
        self.assertTrue(occ[0, 1])  # ignored side contact is still a true contact
        points = scan.points(scan.heights(S, boxes))
        self.assertEqual(points.shape, (1, 187, 3))
        logits = torch.randn(1, 9, 187, requires_grad=True)
        occupancy = torch.randn(1, 9, requires_grad=True)
        losses = planner_loss(logits.log_softmax(-1), occupancy, points, targets, occ, .1, mask)
        sum(losses).backward()
        self.assertTrue(torch.equal(logits.grad[:, 1], torch.zeros_like(logits.grad[:, 1])))
        self.assertEqual(float(occupancy.grad[0, 1]), 0.)
        self.assertGreater(float(occupancy.grad[0, 4]), 0.)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_all_ignored_batch_and_buffer_mask(self):
        logit = torch.randn(2, 9, 187, requires_grad=True)
        occ_logit = torch.randn(2, 9, requires_grad=True)
        mask = torch.zeros(2, 9, dtype=torch.bool)
        target = torch.rand(2, 9, 3)
        occ = torch.ones_like(mask)
        losses = planner_loss(logit.log_softmax(-1), occ_logit, torch.rand(2, 187, 3), target, occ, .1, mask)
        self.assertEqual(sum(float(x.detach()) for x in losses), 0.)
        sum(losses).backward()
        self.assertTrue((occ_logit.grad == 0).all())
        buffer = PlannerBuffer(4, 5, 69, 187, "cpu")
        data = {k: v[:2].clone() for k, v in buffer.data.items()}
        data["supervision"][:] = mask
        buffer.add(data)
        self.assertFalse(buffer.sample(3)["supervision"].any())
        del data["supervision"]
        buffer.add(data)
        self.assertTrue(buffer.data["supervision"][2:].all())

    def test_knee_cap_tracks_its_parent_not_the_shin(self):
        m = get_spec().compile()
        legacy, extended = contact_geom_ids(m), contact_geom_ids(m, "roban_knee_caps_v1")
        for b in range(9):
            self.assertEqual(len(extended[b])-len(legacy[b]), int(b in (2, 3)))
        self.assertEqual(len({g for group in extended for g in group}), sum(map(len, extended)))
        self.assertNotIn(m.geom("leg_l2_link_collision1").id, extended[2])
        kin = RobotKinematics("cpu")
        sp = BodySpheres("roban_knee_caps_v1")
        task = SimpleNamespace(cfg=SimpleNamespace(contact_profile="roban_knee_caps_v1"),
            mdp=SimpleNamespace(kin=kin), sph_frame_idx=torch.tensor([kin.body_names.index(n) for n in sp.frames]),
            sph_pts=sp.pts)
        q = torch.tensor(default_joint_pos_list())[None].repeat(2, 1)
        q[1, JOINT_NAMES.index("leg_l4_joint")] += .9
        S = (torch.tensor([[.1, .2, .9]]).repeat(2, 1), torch.eye(3)[None].repeat(2, 1, 1),
             torch.zeros(2, 3), torch.zeros(2, 3), q, torch.zeros_like(q))
        centers = ContactTask._sphere_centers(task, None, S)
        cap = [i for i, n in enumerate(sp.frames) if n == "leg_l3_link"]
        shin = [i for i, n in enumerate(sp.frames) if n == "leg_l4_link"]
        torch.testing.assert_close(centers[0, cap], centers[1, cap], atol=1e-6, rtol=0)
        self.assertGreater(float((centers[0, shin]-centers[1, shin]).norm()), .1)
        d = mujoco.MjData(m)
        for i in range(2):
            d.qpos[:] = np.r_[S[0][i], [1., 0, 0, 0], q[i]]
            mujoco.mj_kinematics(m, d)
            expected = np.array([d.xpos[m.body(n).id] + d.xmat[m.body(n).id].reshape(3, 3) @ p.numpy()
                                 for n, p in zip(sp.frames, sp.pts)])
            np.testing.assert_allclose(centers[i].numpy(), expected, atol=3e-7)


if __name__ == "__main__":
    unittest.main()
