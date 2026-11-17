"""Collection boundary and terrain fidelity tests (source collection Python)."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from collect_roban_acp import EpisodeRecorder, JOINTS
from export_roban_acp import export_episode, heights, terrain_boxes


class Scene(dict):
    pass


class CollectionTests(unittest.TestCase):
    def test_delayed_reset_preserves_terminal_and_new_initial_state(self):
        qp = torch.zeros(1, 28)
        qp[:, 3] = 1
        qp[:, 7:] = torch.arange(21)
        data = SimpleNamespace(data=SimpleNamespace(qpos=qp, qvel=torch.zeros(1, 27)),
                               indexing=SimpleNamespace(free_joint_q_adr=torch.arange(7)), joint_pos=qp[:, 7:])
        robot = SimpleNamespace(joint_names=JOINTS[::-1], data=data)
        scene = Scene(robot=robot)
        scene.terrain = SimpleNamespace(terrain_levels=torch.tensor([0]), terrain_types=torch.tensor([0]),
                                        terrain_type_names=["flat"], env_origins=torch.zeros(1, 3))
        term = SimpleNamespace(terminated=torch.tensor([False]), time_outs=torch.tensor([False]), active_terms=["fall"])
        term.compute = lambda: term.terminated | term.time_outs
        term.get_term = lambda n: term.terminated
        cmd = SimpleNamespace(pos_command_w=torch.tensor([[10., 0., 0.]]), vel_command_b=torch.zeros(1, 3))
        env = SimpleNamespace(scene=scene, termination_manager=term, episode_length_buf=torch.tensor([0]), step_dt=.02,
                              avg_feet_force=torch.tensor([[100., 90.]]),
                              command_manager=SimpleNamespace(get_term=lambda n: cmd),
                              action_manager=SimpleNamespace(action=torch.zeros(1, 22)))
        with tempfile.TemporaryDirectory() as tmp:
            rec = EpisodeRecorder(SimpleNamespace(params={"out": str(Path(tmp)/"raw")}), env)
            rec.ready = True
            ids = torch.tensor([0])
            rec.record_post_reset(ids)
            rec.just_reset.clear()
            for step in (1, 2):
                qp[:, 0] = step
                env.action_manager.action[:] = step
                env.episode_length_buf[:] = step
                term.terminated[:] = step == 2
                term.compute()
                # The source AMP implementation clears this flag during a
                # delayed camera reset; the recorder must retain first failure.
                term.terminated[:] = False
                rec.record_post_step()
            qp[:, 0] = 3
            env.action_manager.action[:] = 3
            rec.record_pre_reset(ids)
            qp[:, 0] = 100
            env.action_manager.action[:] = 0
            rec.record_post_reset(ids)
            rec.record_post_step()
            path = Path(tmp)/"raw/ep_000001"
            with np.load(path/"states.npz") as d:
                np.testing.assert_array_equal(d["qpos"][:, 0], [0, 1, 2, 3])
                np.testing.assert_array_equal(d["actions"][:, 0], [1, 2, 3])
                self.assertEqual(d["actions"].shape, (3, 22))
                self.assertEqual(d["feet_force"].shape, (3, 2))
                np.testing.assert_array_equal(d["qpos"][0, 7:], np.arange(21)[::-1])
            meta = json.loads((path/"episode.json").read_text())
            self.assertTrue(meta["first_done"]["terminated"])
            self.assertEqual(meta["first_done"]["step"], 2)
            self.assertFalse(meta["source_success"])
            self.assertEqual(len(rec.active[0]["qpos"]), 1)
            self.assertEqual(rec.active[0]["qpos"][0][0], 100)

    def test_negative_floor_is_translated_with_motion_and_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)/"run"
            folder = run/"raw/ep_000001"
            folder.mkdir(parents=True)
            qp = np.zeros((301, 28))
            qp[:, 0] = np.linspace(0, 1, 301)
            qp[:, 2:4] = [.8, 1.]
            np.savez(folder/"states.npz", qpos=qp, joint_names=np.array(JOINTS), time=np.arange(301)*.02)
            (folder/"episode.json").write_text(json.dumps(dict(source_success=True, terrain_type="stairs", terrain_row=0, terrain_col=0)))
            # Low floor -2 m and a road at z=0.
            boxes = np.array([[0, 0, -2.5, 1, 0, 0, 0, 20, 20, 1], [0, 0, -1, 1, 0, 0, 0, 4, 1, 2]], float)
            output = Path(tmp)/"motion"
            row = export_episode(run, folder, output, boxes, np.arange(2))
            with np.load(output/(row["clip"]+"_motion.npz")) as d:
                np.testing.assert_allclose(d["joint_pos"][:, 2], 2.8)
            terrain = np.load(output/(row["clip"]+"_terrain.npy"))[:, 0]
            self.assertEqual(len(terrain), 1)
            np.testing.assert_allclose(heights(np.array([[0., 0.], [0., 2.]]), terrain, floor=0), [2, 0])

    def test_nonconstant_heightfield_rejected(self):
        snapshot = dict(types=np.array([1]), pos=np.zeros((1, 3)), quat=np.array([[1., 0, 0, 0]]), size=np.ones((1, 3)),
                        names=np.array(["rough"]), dataid=np.array([0]), geom_id=np.array([1]), hfield_adr=np.array([0]),
                        hfield_nrow=np.array([2]), hfield_ncol=np.array([2]), hfield_data=np.array([0., 1., 0., 0.]),
                        hfield_size=np.array([[1., 1., .1, .1]]))
        with self.assertRaisesRegex(ValueError, "nonconstant"):
            terrain_boxes(snapshot)


if __name__ == "__main__":
    unittest.main()
