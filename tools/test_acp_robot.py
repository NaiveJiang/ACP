"""Run in separate processes with ACP_ROBOT=g1 and ACP_ROBOT=roban."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from supermimic.acp import kinematics
from supermimic.acp.robot import (ROBOT_ID, NUM_J, JOINT_NAMES, CONTACT_BODIES, ROOT_BODY,
    get_spec, contact_geom_ids, default_joint_pos_list, pd_params, metadata)
from supermimic.acp.task import BodySpheres
from supermimic.acp.make_labels import build_model, surface_points
from supermimic.acp.checkpoint import save_checkpoint, load_checkpoint
from supermimic.acp.motion import load_php_npz
from supermimic.acp.render import scene_xml
from supermimic.rotations import quat_to_mat


class RobotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_fk_positions_rotations_velocities_and_fixed_frames(self):
        # Nonzero joint anchors and a rotated fixed parent exercise more than this asset's zero-anchor hinges.
        spec = get_spec()
        spec.joint(JOINT_NAMES[1]).pos[:] = (.01, -.02, .015)
        model = spec.compile()
        with patch.object(kinematics, "get_spec", return_value=spec):
            kin = kinematics.RobotKinematics("cpu")
        rng = np.random.default_rng(2)
        for _ in range(8):
            data = mujoco.MjData(model)
            quat = rng.normal(size=4); quat /= np.linalg.norm(quat)
            q = np.array(default_joint_pos_list()) + rng.uniform(-.15, .15, NUM_J)
            data.qpos[:] = np.r_[rng.normal(size=3), quat, q]
            data.qvel[:] = rng.normal(size=model.nv)
            mujoco.mj_forward(model, data)
            R = quat_to_mat(torch.tensor(quat, dtype=torch.float32))
            state = [torch.tensor(data.qpos[:3]).float(), R, torch.tensor(data.qvel[:3]).float(),
                     R @ torch.tensor(data.qvel[3:6]).float(), torch.tensor(q).float(), torch.tensor(data.qvel[6:]).float()]
            out = [x[0].numpy() for x in kin.forward(*(x[None] for x in state), all_bodies=True)]
            for i, name in enumerate(kin.body_names):
                bid = model.body(name).id
                jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jacBody(model, data, jp, jr, bid)
                for actual, expected in zip((out[0][i], out[1][i], out[2][i], out[3][i]),
                    (data.xpos[bid], data.xmat[bid].reshape(3, 3), jp @ data.qvel, jr @ data.qvel)):
                    np.testing.assert_allclose(actual, expected, atol=2e-5, err_msg=name)
        self.assertEqual(kin.q_lo.shape, (NUM_J,))

    def test_collision_ownership_and_primitive_distance(self):
        model = build_model(np.zeros((0, 8)))
        groups = contact_geom_ids(model)
        self.assertEqual(len(sum(groups, [])), len(set(sum(groups, []))))
        sph = BodySpheres()
        self.assertTrue(all(end > start for start, end in sph.bounds))
        if ROBOT_ID != "roban":
            return
        self.assertEqual([len(x) for x in groups[4:8]], [1, 1, 2, 2])
        data = mujoco.MjData(model)
        data.qpos[:] = np.r_[0., 0., 1., 1., 0., 0., 0., default_joint_pos_list()]
        # Foot sphere covers must remain within 5 mm of the exact plane distance even under tilt.
        for tilt in (-.3, 0., .3):
            data.qpos[3:7] = (np.cos(tilt / 2), 0., np.sin(tilt / 2), 0.)
            mujoco.mj_kinematics(model, data)
            for b in (0, 1):
                frame = model.body(list(CONTACT_BODIES.values())[b][0]).id
                start, end = sph.bounds[b]
                pts = data.xpos[frame] + sph.pts[start:end].numpy() @ data.xmat[frame].reshape(3, 3).T
                approx = (pts[:, 2] - sph.rad[start:end].numpy()).min()
                exact = min(mujoco.mj_geomDistance(model, data, g, model.geom("ground").id, 2., np.zeros(6)) for g in groups[b])
                self.assertLess(abs(approx - exact), .005)
                for g in groups[b]:self.assertTrue(np.isfinite(surface_points(model, g)).all())

    def test_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            save_checkpoint({"pi": {"weight": torch.zeros(1)}}, path)
            self.assertEqual(load_checkpoint(path)["robot"], metadata())
            torch.save({"robot": {"id": "other", "fingerprint": "wrong"}}, path)
            with self.assertRaises(ValueError):load_checkpoint(path)
            torch.save({"pi": {}}, path)
            if ROBOT_ID == "g1":load_checkpoint(path)
            else:
                with self.assertRaises(ValueError):load_checkpoint(path)

    def test_motion_shape_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_motion.npz"
            pos = np.tile(np.r_[0., 0., 1., 1., 0., 0., 0., default_joint_pos_list()], (3, 1))
            np.savez(path, joint_pos=pos, joint_names=JOINT_NAMES, fps=[50.])
            self.assertEqual(load_php_npz(path)[2].shape, (3, NUM_J))
            np.savez(path, joint_pos=pos, joint_names=JOINT_NAMES[::-1], fps=[50.])
            with self.assertRaises(ValueError):load_php_npz(path)

    def test_render_model(self):
        model = mujoco.MjModel.from_xml_string(scene_xml(np.zeros((1, 8))))
        self.assertEqual(model.nq, 2 * (7 + NUM_J))
        self.assertGreater(model.body("ref_" + ROOT_BODY).id, 0)

    def test_csv_resampling_and_root_conversion(self):
        from supermimic.acp.prepare_motion import convert
        model = get_spec().compile()
        data = mujoco.MjData(model)
        data.qpos[:] = np.r_[.2, .1, .9, 1., 0., 0., 0., default_joint_pos_list()]
        data.qpos[7] += .2
        mujoco.mj_kinematics(model, data)
        frame = "waist_yaw_link" if ROBOT_ID == "roban" else "torso_link"
        bid = model.body(frame).id
        source = np.r_[data.xpos[bid], data.xquat[bid][[1, 2, 3, 0]], data.qpos[7:]]
        with tempfile.TemporaryDirectory() as tmp:
            csv, terrain, out = (Path(tmp) / p for p in ("source.csv", "terrain.npy", "sample_motion.npz"))
            np.savetxt(csv, np.tile(source, (961, 1)), delimiter=",")
            np.save(terrain, np.zeros((0, 1, 10)))
            count = convert(csv, out, fps=120., quat_order="xyzw", root_frame=frame, terrain=terrain)
            self.assertEqual(count, 401)
            pos, quat, q, fps = load_php_npz(out)
            np.testing.assert_allclose(pos, np.tile(data.qpos[:3], (401, 1)), atol=1e-6)
            np.testing.assert_allclose(np.abs(quat[:, 0]), 1., atol=1e-6)
            self.assertEqual(fps, 50.)

    def test_sampler_excludes_terminal_only_bin(self):
        from supermimic.acp.env import TrackingSim
        motion = SimpleNamespace(num_bins=4, bin_start=torch.tensor([0, 50, 100, 150]),
            bin_clip=torch.zeros(4, dtype=torch.long), clip_end=torch.tensor([150]))
        env = SimpleNamespace(motion=motion, curriculum=False, start_phase=None, device="cpu",
            bin_G=torch.zeros(4), G_max=1., bin_w_avg=torch.ones(4))
        frames, bins = TrackingSim.sample_frames(env, 10000)
        self.assertTrue((frames < 150).all())
        self.assertTrue((bins < 3).all())

    def test_first_frame_sampling_ignores_saved_curriculum(self):
        from supermimic.acp.env import TrackingSim
        lengths = torch.tensor([31, 501, 1001])
        starts = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]))
        motion = SimpleNamespace(clip_len=lengths, clip_start=starts,
            frame_bin=torch.arange(int(lengths.sum())))
        env = SimpleNamespace(motion=motion, curriculum=False, start_phase=(0., 0.), device="cpu",
            phase=torch.tensor([.9, .8, .85]))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            frames, bins = TrackingSim.sample_frames(env, 10000)
        self.assertTrue(torch.isin(frames, starts).all())
        self.assertTrue(torch.equal(frames.unique(), starts))
        self.assertTrue(torch.equal(bins, motion.frame_bin[frames]))

    def test_network_dimensions_and_planner_gradients(self):
        from supermimic.acp.mdp import OBS_DIM, XI_DIM, STATE_DIM, INV_DIM
        from supermimic.acp.networks import WorldModel, WM_OUT
        from supermimic.acp.planner import ContactPlanner, PlannerCfg, planner_loss
        self.assertEqual((OBS_DIM, XI_DIM, STATE_DIM, WM_OUT), (3 * NUM_J + 12, NUM_J + 4, 13 + 2 * NUM_J, 12 + 2 * NUM_J))
        self.assertEqual(WorldModel(465, hidden=(16,)).in_dim, INV_DIM + NUM_J + XI_DIM + 465)
        cfg = PlannerCfg()
        planner = ContactPlanner(OBS_DIM - 6, cfg)
        pts = torch.randn(2, 187, 3)
        affordance, occupancy = planner(torch.zeros(2, 5, OBS_DIM - 6), torch.zeros(2, 3), pts)
        self.assertEqual(affordance.shape, (2, 9, 187))
        self.assertEqual(occupancy.shape, (2, 9))
        losses = planner_loss(affordance, occupancy, pts, torch.zeros(2, 9, 3), torch.ones(2, 9), .1)
        sum(losses).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in planner.parameters() if p.grad is not None))

    @unittest.skipUnless(ROBOT_ID == "roban", "RoBan explicit control")
    def test_roban_torque_law(self):
        from supermimic.acp.roban import RobanPdActuator
        kp, kd, effort, _ = [torch.tensor(x) for x in pd_params()]
        q = torch.linspace(-.1, .1, NUM_J)
        vel = torch.linspace(-2., 2., NUM_J)
        cmd = SimpleNamespace(position_target=q + .3, pos=q, velocity_target=torch.zeros_like(q),
                              vel=vel, effort_target=torch.zeros_like(q))
        result = RobanPdActuator.control_law({"stiffness": kp, "damping": kd, "force_limit": effort}, cmd)
        expected = torch.clamp(kp * .3 - kd * vel - .2 * torch.tanh(vel / .1), -effort, effort)
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
