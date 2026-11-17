"""Independent planner tests; needs only NumPy and PyTorch."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from supermimic.acp.footstep.data import ContactDataset, JOINT_NAMES, observations, split_clips
from supermimic.acp.footstep.model import FootstepPlanner, planner_loss


class FootstepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        n, h = 30, 2
        self.clips = []
        for k in range(3):
            folder = self.root / str(k)
            folder.mkdir()
            index = np.arange(n)[:, None] + np.array([2, 5])[None, :]
            data = dict(state=np.repeat(np.arange(n, dtype=np.float32)[:, None], 48, axis=1),
                        velocity=np.ones((n, 3), np.float32), dt=np.tile([.04, .1], (n, 1)).astype(np.float32),
                        contact=np.ones((n, h, 2), np.float32), position=np.zeros((n, h, 2, 3), np.float32),
                        heading=np.tile([0., 1.], (n, h, 2, 1)).astype(np.float32),
                        valid=index < n, future_index=np.where(index < n, index, -1))
            for name, arr in data.items():
                np.save(folder/(name+".npy"), arr)
            self.clips.append(dict(source="synthetic", category="forward", case=str(k), frames=n,
                                   cache=str(folder), horizons=[.04, .1], fingerprint=str(k), fps=50.))

    def tearDown(self):
        self.temp.cleanup()

    def test_named_joint_mapping_and_body_frame_gravity(self):
        names = list(reversed(JOINT_NAMES))
        meta = dict(joints=[dict(name=n, qpos_adr=7+i, dof_adr=6+i) for i, n in enumerate(names)])
        qpos, qvel = np.zeros((2, 30)), np.zeros((2, 29))
        qpos[:, 3] = 1
        qpos[:, 7:28] = np.arange(21)
        qvel[:, 6:27] = np.arange(21)+100
        qvel[:, 3:6] = [1, 2, 3]
        result = observations(qpos, qvel, meta)
        np.testing.assert_array_equal(result[0, :6], [0, 0, -1, 1, 2, 3])
        np.testing.assert_array_equal(result[0, 6:27], np.arange(21)[::-1])
        np.testing.assert_array_equal(result[0, 27:], (np.arange(21)+100)[::-1])
        qpos[:, 3:7] = [np.sqrt(.5), np.sqrt(.5), 0, 0]
        np.testing.assert_allclose(observations(qpos, qvel, meta)[0, :3], [0, -1, 0], atol=1e-6)

    def test_whole_clip_split_and_determinism(self):
        a = split_clips(self.clips, seed=7)
        self.assertEqual(a, split_clips(self.clips, seed=7))
        self.assertFalse({c['case'] for c in a['train']} & {c['case'] for c in a['val']})

    def test_temporal_split_cannot_borrow_history_or_targets(self):
        split = split_clips(self.clips[:1], val_fraction=.3)
        for part in ('train', 'val'):
            ds = ContactDataset(split[part], hist=5)
            for index in range(len(ds)):
                interval, frame, horizon = ds.locate(index)
                self.assertGreaterEqual(frame-4, interval['start'])
                self.assertLess(frame+[2,5][horizon], interval['end'])
                sample = ds[index]
                np.testing.assert_array_equal(sample['state_history'][:, 0], np.arange(frame-4,frame+1))
                self.assertEqual(sample['target_heading'].shape, (2, 2))

    def test_invalid_pair_excluded_and_balancing_equalizes_horizons(self):
        path = Path(self.clips[0]['cache'])/'valid.npy'
        valid = np.load(path)
        valid[8, 0] = False
        np.save(path, valid)
        ds = ContactDataset([dict(self.clips[0], start=0, end=30)], hist=5)
        pairs = ds.entries[0][1]
        self.assertFalse(((pairs[:,0] == 8)&(pairs[:,1] == 0)).any())
        weights = ds.balanced_weights()
        self.assertAlmostEqual(weights[pairs[:,1] == 0].sum(), weights[pairs[:,1] == 1].sum())
        with self.assertRaises(IndexError):
            ds[len(ds)]

    def test_loss_uses_true_support_mask_and_decoder_is_16d(self):
        torch.manual_seed(3)
        model = FootstepPlanner(hidden=(32, 16))
        state, vel, dt = torch.randn(4,5,48), torch.randn(4,3), torch.full((4,1), .3)
        pred = model(state, vel, dt)
        mask = torch.tensor([[1.,0.]]).expand(4,-1)
        batch = dict(support_mask=mask, target_contact=mask,
                     target_position=torch.zeros(4,2,3), target_heading=torch.tensor([0.,1.]).expand(4,2,2))
        first = planner_loss(pred, batch)
        batch['target_position'][:,1] = 999
        batch['target_heading'] = batch['target_heading'].clone()
        batch['target_heading'][:,1] = -999
        second = planner_loss(pred, batch)
        torch.testing.assert_close(first['total'], second['total'])
        first['total'].backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        command = model.command(state, vel, dt)
        self.assertEqual(command.shape, (4,16))
        torch.testing.assert_close(command[:,:3], vel)
        torch.testing.assert_close(command[:,3:4], dt)

    def test_airborne_targets_still_train_contacts(self):
        pred = dict(contact_logits=torch.zeros(3,2,requires_grad=True), position=torch.randn(3,2,3,requires_grad=True),
                    heading=torch.randn(3,2,2,requires_grad=True))
        batch = dict(support_mask=torch.zeros(3,2), target_contact=torch.zeros(3,2),
                     target_position=torch.zeros(3,2,3), target_heading=torch.zeros(3,2,2))
        loss = planner_loss(pred, batch)
        self.assertEqual(float(loss['position'].detach()), 0.)
        self.assertEqual(float(loss['heading'].detach()), 0.)
        loss['total'].backward()
        self.assertTrue((pred['contact_logits'].grad > 0).all())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
