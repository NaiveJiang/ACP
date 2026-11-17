"""Regression coverage for simulator failures poisoning global normalization."""
import os
os.environ["ACP_ROBOT"] = "roban"
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from supermimic.acp.buffer import ReplayBuffer
from supermimic.acp.networks import RunningNorm
from supermimic.acp.sim_numerics import finite_step_state, PhysicalStateGuard, TransitionGuard


class SimNumericsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_running_norm_matches_population_and_rejects_bad_updates_atomically(self):
        norm = RunningNorm(3)
        a = torch.tensor([[1., 2., 3.], [3., 8., 6.]])
        b = torch.tensor([[2., 3., 7.]])
        norm.update(a); norm.update(b)
        torch.testing.assert_close(norm.mean, torch.cat((a, b)).mean(0))
        torch.testing.assert_close(norm.var, torch.cat((a, b)).var(0, unbiased=False))
        saved = {k: v.clone() for k, v in norm.state_dict().items()}
        for bad in (float('nan'), float('inf'), 1e30):
            with self.subTest(bad=bad), self.assertRaises(FloatingPointError):
                norm.update(torch.full((2, 3), bad))
            for k, v in norm.state_dict().items():
                torch.testing.assert_close(v, saved[k])
        norm.update(torch.empty(0, 3))
        for k, v in norm.state_dict().items():
            torch.testing.assert_close(v, saved[k])

    def test_invalid_state_preserves_good_rows_and_does_not_modify_raw_evidence(self):
        old = (torch.zeros(3, 3), torch.eye(3).expand(3, 3, 3).clone())
        current = tuple(x.clone() for x in old)
        current[0][0, 0] = 2.
        current[0][1, 0] = float('nan')
        current[1][2, 0, 0] = float('inf')
        safe, valid = finite_step_state(old, current)
        self.assertEqual(valid.tolist(), [True, False, False])
        self.assertTrue(all(x.isfinite().all() for x in safe))
        self.assertEqual(safe[0][0, 0], 2.)
        self.assertTrue(current[0][1, 0].isnan())
        self.assertTrue(current[1][2, 0, 0].isinf())

    def test_gross_outliers_are_isolated_but_falls_and_motor_overspeed_are_retained(self):
        n, j = 8, 2
        old = (torch.zeros(n, 3), torch.eye(3).expand(n, 3, 3).clone(), torch.zeros(n, 3),
               torch.zeros(n, 3), torch.zeros(n, j), torch.zeros(n, j))
        old[4][1, 0], old[4][3, 0] = 1.2, -2.1
        cur = tuple(x.clone() for x in old)
        cur[0][0, 2], cur[2][0, 2], cur[5][0, 0] = -.6, -5., 20.  # ordinary fall / 2x motor speed
        cur[4][0, 1] = 2.1  # small joint-limit violation also remains valid
        cur[4][1, 0] = 3.1  # position beyond the range plus 1 rad, without a 4 rad jump
        cur[5][2, 0] = 101.
        cur[4][3, 0] = 2.1  # a 4.2 rad jump within the loose absolute range
        cur[2][4, 0] = 51.
        cur[3][5, 0] = 201.
        cur[0][6, 0] = 1.01
        cur[4][7, 0] = float('nan')
        guard = PhysicalStateGuard(-2 * torch.ones(j), 2 * torch.ones(j), [10., 10.], .02)
        safe, valid, reason = guard(old, cur)
        self.assertEqual(valid.tolist(), [True] + [False] * 7)
        self.assertEqual(reason.tolist(), [0, 2, 4, 8, 16, 32, 64, 1])
        for original, raw, sanitized in zip(old, cur, safe):
            torch.testing.assert_close(sanitized[0], raw[0])
            torch.testing.assert_close(sanitized[1:], original[1:])
        self.assertEqual(cur[5][2, 0], 101.)
        self.assertTrue(cur[4][7, 0].isnan())

    def test_finite_outlier_diagnostics_record_actual_reason_and_raw_values(self):
        motion = SimpleNamespace(clip_id=torch.tensor([0, 1]), names=['flat/a', 'slope/b'])
        out = {'valid': torch.tensor([True, False]), 'invalid_reason': torch.tensor([0, 4]),
               'f': torch.tensor([0, 1]), '_invalid_qvel': torch.tensor([[1e6]])}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'numerics').mkdir()
            prior = Path(tmp)/'numerics/first_bad_sim_transition.pt'
            torch.save({'earlier_nan': True}, prior)
            TransitionGuard(tmp).observe(out, motion, 10, 22125)
            self.assertEqual(torch.load(prior, weights_only=True), {'earlier_nan': True})
            evidence = torch.load(Path(tmp)/'numerics/first_bad_finite_sim_transition.pt', weights_only=True)
            self.assertEqual(evidence['event']['reasons'], [['joint_velocity_outlier']])
            self.assertEqual(evidence['raw']['_invalid_qvel'].item(), 1e6)

    def test_replay_masks_invalid_terminal_but_keeps_normal_terminal(self):
        for numerical in (False, True):
            with self.subTest(numerical=numerical):
                buf = ReplayBuffer(4, 1, 'cpu')
                for t in range(4):
                    out = {k: torch.zeros_like(v[0]) for k, v in buf.data.items()}
                    out['valid'][:] = not (numerical and t == 1)
                    out['done'][:] = t == 1
                    out['f'][:] = t
                    out['ep_step'][:] = t % 2
                    buf.add(out)
                batch = buf.sample_windows(8, 4)
                expected = [1., 0., 0., 0.] if numerical else [1., 1., 0., 0.]
                torch.testing.assert_close(batch['mask'], torch.tensor(expected).expand(8, 4))
                start, history = buf.sample_starts(128, 2)
                if numerical:
                    self.assertTrue((~start['valid'][start['f'] == 1]).all())

    def test_guard_records_raw_failure_and_stops_repeated_failures(self):
        motion = SimpleNamespace(clip_id=torch.tensor([0, 1]), names=['flat/a', 'slope/b'])
        out = {'valid': torch.tensor([True, False]), 'f': torch.tensor([0, 1]),
               '_invalid_qpos': torch.tensor([[float('nan')]])}
        with tempfile.TemporaryDirectory() as tmp:
            guard = TransitionGuard(tmp, max_consecutive=2)
            guard.observe(out, motion, 10, 4001)
            evidence = torch.load(Path(tmp)/'numerics/first_bad_sim_transition.pt', weights_only=True)
            self.assertEqual(evidence['event']['clips'], ['slope/b'])
            self.assertTrue(evidence['raw']['_invalid_qpos'].isnan().all())
            guard.observe({**out, 'valid': torch.ones(2, dtype=torch.bool)}, motion, 12, 4001)
            self.assertEqual(guard.consecutive, 0)
            guard.observe(out, motion, 14, 4002)
            with self.assertRaisesRegex(FloatingPointError, 'Persistent simulator'):
                guard.observe(out, motion, 16, 4002)

    def test_planner_rejects_an_episode_with_an_invalid_transition(self):
        from supermimic.acp.planner import label_episodes
        buf = ReplayBuffer(4, 1, 'cpu')
        buf.data['valid'][:] = True
        buf.data['valid'][1] = False
        samples, count = label_episodes(buf, torch.tensor([0]), torch.tensor([3]), torch.tensor([4]),
            SimpleNamespace(motion=None), None, None, success=torch.tensor([True]))
        self.assertIsNone(samples)
        self.assertEqual(count, 0)

    @unittest.skipUnless(torch.cuda.is_available() and Path('data/roban_motion/flat').is_dir(),
                         'Requires the RoBan data and a CUDA simulator')
    def test_real_simulator_resets_injected_nan_world_and_preserves_norms(self):
        from supermimic.acp.env import TrackingSim
        from supermimic.acp.kinematics import RobotKinematics
        from supermimic.acp.mdp import MDP, unflatten_state
        from supermimic.acp.motion import MotionLib
        from supermimic.acp.networks import WorldModel, WM_OUT
        from supermimic.acp.robot import NUM_J, default_joint_pos_list
        kin = RobotKinematics('cuda')
        motion = MotionLib.from_dir('data/roban_motion', kin, 'cuda', pattern='flat/*_motion.npz')
        mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device='cuda'))
        env = TrackingSim(8, motion, mdp, 'cuda')
        env.observe()
        original_state, calls = env.state, 0

        def injected_state():
            nonlocal calls
            calls += 1
            if calls == 2:  # state after the four real physics substeps
                env.sim.data.qpos[0, 0] = float('nan')
                env.sim.data.qvel[0, 0] = float('nan')
            return original_state()

        env.state = injected_state
        out = env.step(torch.zeros(8, NUM_J, device='cuda'))
        env.state = original_state
        self.assertFalse(out['valid'][0])
        self.assertTrue(out['valid'][1:].all())
        self.assertTrue(out['done'][0] and out['term'][0])
        self.assertEqual(out['r'][0], 0.)
        self.assertEqual(env.ep_step[0], 0)
        self.assertTrue(out['_invalid_qpos'].isnan().any())
        self.assertTrue(all(x.isfinite().all() for x in env.state()))
        self.assertTrue(env.observe().isfinite().all())
        norm = RunningNorm(WM_OUT).to('cuda')
        target = WorldModel.targets(unflatten_state(out['s']), unflatten_state(out['s1']))
        norm.update(target[out['valid']])
        self.assertEqual(norm.count, 7.)
        self.assertTrue(norm.mean.isfinite().all() and norm.var.isfinite().all())
        out['goal_steps'] = torch.ones(8, dtype=torch.long, device='cuda')
        buf = ReplayBuffer(2, 8, 'cuda')
        buf.add(out)
        batch = buf.sample_windows(128, 1)
        self.assertTrue((~batch['valid']).any())
        self.assertTrue((batch['mask'][~batch['valid']] == 0).all())

    @unittest.skipUnless(torch.cuda.is_available() and Path('data/roban_motion/flat').is_dir(),
                         'Requires the RoBan data and a CUDA simulator')
    def test_real_simulator_separates_finite_outlier_fall_and_timeout(self):
        from supermimic.acp.env import TrackingSim
        from supermimic.acp.kinematics import RobotKinematics
        from supermimic.acp.mdp import MDP, unflatten_state
        from supermimic.acp.motion import MotionLib
        from supermimic.acp.networks import WorldModel, WM_OUT
        from supermimic.acp.robot import NUM_J, default_joint_pos_list
        torch.manual_seed(20261112)
        kin = RobotKinematics('cuda')
        motion = MotionLib.from_dir('data/roban_motion', kin, 'cuda', pattern='flat/*_motion.npz')
        mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device='cuda'))
        env = TrackingSim(8, motion, mdp, 'cuda', reset_noise=False)
        env.ep_step[2] = env.max_episode_steps - 1
        env.start_bin[:] = 0
        env.start_bin[0] = motion.num_bins - 1
        env.bin_G[-1] = 123.
        env.observe()
        original_state, calls = env.state, 0

        def injected_state():
            nonlocal calls
            calls += 1
            if calls == 2:
                idx = env.robot.indexing
                env.sim.data.qpos[0, idx.joint_q_adr[2]] = 100.
                env.sim.data.qvel[0, idx.joint_v_adr[2]] = 1e6
                env.sim.data.qpos[1, idx.free_joint_q_adr[2]] -= .6  # ordinary height termination
            return original_state()

        env.state = injected_state
        out = env.step(torch.zeros(8, NUM_J, device='cuda'))
        env.state = original_state
        self.assertEqual(out['valid'].tolist(), [False] + [True] * 7)
        self.assertTrue(out['term'][0] and out['term'][1])
        self.assertTrue(out['done'][:3].all())
        self.assertFalse(out['term'][2])  # timeout, not task failure
        self.assertEqual(out['invalid_reason'][0].item(), 2 | 4 | 8)
        self.assertEqual(out['r'][0], 0.)
        self.assertGreater(out['r'][1], 0.)
        self.assertTrue((env.ep_step[:3] == 0).all())
        self.assertEqual(env.bin_G[-1], 123.)  # numerical failures cannot change reset difficulty
        self.assertEqual(out['_invalid_state'][5][0, 2], 1e6)
        self.assertTrue(all(x.isfinite().all() for x in env.state()))
        target = WorldModel.targets(unflatten_state(out['s']), unflatten_state(out['s1']))
        norm = RunningNorm(WM_OUT).to('cuda')
        norm.update(target[out['valid']])
        self.assertEqual(norm.count, 7.)  # includes ordinary terminal and timeout transitions
        torch.testing.assert_close(norm.mean, target[1:].mean(0))
        torch.testing.assert_close(norm.var, target[1:].var(0, unbiased=False))
        out['goal_steps'] = torch.ones(8, dtype=torch.long, device='cuda')
        buf = ReplayBuffer(2, 8, 'cuda')
        buf.add(out)
        batch = buf.sample_windows(512, 1)
        self.assertTrue((batch['mask'][~batch['valid']] == 0).all())
        normal_terminal = batch['done'] & batch['valid']
        self.assertTrue(normal_terminal.any())
        self.assertTrue((batch['mask'][normal_terminal] == 1).all())
        with tempfile.TemporaryDirectory() as tmp:
            TransitionGuard(tmp).observe(out, motion, 8, 1)
            evidence = torch.load(Path(tmp)/'numerics/first_bad_sim_transition.pt', weights_only=True)
            self.assertIn('joint_velocity_outlier', evidence['event']['reasons'][0])
            self.assertEqual(evidence['raw']['_invalid_state'][5][0, 2], 1e6)


if __name__ == '__main__':
    unittest.main()
