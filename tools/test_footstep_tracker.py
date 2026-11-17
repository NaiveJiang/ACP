"""Oracle timing, observations, reward masks and actor-through-world-model gradients."""
import os
os.environ["ACP_ROBOT"] = "roban"

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from supermimic.acp.footstep.data import JOINT_NAMES
from supermimic.acp.footstep.reference import FootstepMotion
from supermimic.acp.footstep.task import FootstepTask, FootstepMDP, TrackerPolicy, OBS_DIM
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.robot import default_joint_pos_list, NUM_J
from supermimic.acp.mdp import XI_DIM, normalize_xi
from supermimic.acp.networks import WorldModel
from supermimic.rotations import mat_to_quat


class TrackerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.kin = RobotKinematics('cpu')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        source = root/'future_contact_v2'
        source.mkdir()
        (source/'manifest.json').write_text(json.dumps(dict(force_threshold_N=10.)))
        meta = dict(sample_hz=50, sole_offset=[.0486,0,-.04],
                    joints=[dict(name=n,qpos_adr=7+i,dof_adr=6+i) for i,n in enumerate(JOINT_NAMES)])
        (root/'metadata.json').write_text(json.dumps(meta))
        n = 60
        q = np.zeros((n,28)); q[:,2] = .77; q[:,3] = 1.; q[:,7:] = default_joint_pos_list()
        vel = np.zeros((n,27)); vel[:,0] = .2
        vel[:,3:6] = [.1,.2,.3]
        p = torch.tensor(q[:,:3]).float(); R = torch.eye(3).expand(n,-1,-1)
        states = (p,R,torch.tensor(vel[:,:3]).float(),torch.tensor(vel[:,3:6]).float(),
                  torch.tensor(q[:,7:]).float(),torch.zeros(n,21))
        bp,bR = self.kin.forward(*states,all_bodies=True)[:2]
        ids = [self.kin.body_names.index(n) for n in ('leg_l6_link','leg_r6_link')]
        feet = np.zeros((n,2,14))
        feet[:,:,:3] = (bp[:,ids]+(bR[:,ids]@torch.tensor([.0486,0,-.04]).view(1,1,3,1)).squeeze(-1)).numpy()
        feet[:,:,3:7] = mat_to_quat(bR[:,ids]).numpy()
        feet[:,:,9] = 100; feet[:,:,13] = 1
        feet[3,0,9] = 0
        np.savez(source/'walk_labels.npz',qpos=q,qvel=vel,foot_state=feet,
                 velocity_command=np.tile([.2,0,0],(n,1)),sim_time=np.arange(n)*.02)
        interval=dict(source=str(source),case='walk',start=0,end=n)
        self.motion = FootstepMotion([interval],self.kin,'cpu',horizons=[.02,.06])
        self.task=FootstepTask(self.motion,self.kin)
        self.mdp=FootstepMDP(self.kin,self.motion,torch.tensor(default_joint_pos_list()),self.task)

    def tearDown(self):
        self.temp.cleanup()

    def test_future_margin_and_query_contact(self):
        m=self.motion; f=torch.tensor([0,0]); query=torch.tensor([0,1])
        self.assertEqual(int(m.clip_end[0]),56)
        cmd=self.task.command(m.ref_state(f),f,query)
        self.assertEqual(cmd.shape,(2,16))
        torch.testing.assert_close(cmd[:,3],torch.tensor([.02,.06]))
        torch.testing.assert_close(cmd[:,4:6],torch.tensor([[1.,1.],[0.,1.]]))
        self.assertTrue((cmd[1,6:9]==0).all())
        self.assertTrue((cmd[1,12:14]==0).all())
        f=torch.tensor([56]); self.task.command(m.ref_state(f),f,torch.tensor([1]))

    def test_changed_reference_fingerprint_is_rejected(self):
        interval=dict(source=str(Path(self.temp.name)/'future_contact_v2'),case='walk',
                      start=0,end=60,fingerprint='obsolete')
        with self.assertRaisesRegex(ValueError,'Reference data changed'):
            FootstepMotion([interval],self.kin,'cpu',horizons=[.02,.06])

    def test_oracle_uses_current_actual_base_frame(self):
        f=torch.tensor([0]); S=self.motion.ref_state(f); query=torch.tensor([0])
        original=self.task.command(S,f,query)
        moved=list(S); moved[0]=S[0]+torch.tensor([[.2,0,0]])
        after=self.task.command(moved,f,query)
        torch.testing.assert_close(after[:,[6,9]], original[:,[6,9]]-.2)

    def test_observation_has_no_contact_and_measured_angular_velocity(self):
        f=torch.tensor([0]); S=self.motion.ref_state(f)
        xi=torch.zeros(1,XI_DIM); prev=torch.ones(1,NUM_J)
        obs=self.mdp.obs(S,prev,xi)
        self.assertEqual(obs.shape,(1,OBS_DIM))
        torch.testing.assert_close(obs[0,:6],torch.tensor([0.,0.,-1.,.1,.2,.3]))
        torch.testing.assert_close(obs[:,48:],prev)

    def test_reward_evaluates_current_phase_not_future_target(self):
        f=torch.tensor([0]); S=self.motion.ref_state(f)
        before=self.task.reward_terms(S,f)
        self.motion.feet[3,:,:3] += 10
        after=self.task.reward_terms(S,f)
        for key in before:
            torch.testing.assert_close(before[key],after[key])
        self.assertGreater(float(before['position']),.999)

    def test_actor_gradient_passes_through_frozen_world_model(self):
        torch.manual_seed(7)
        f=torch.tensor([0,1]); S=self.motion.ref_state(f); xi=torch.zeros(2,XI_DIM)
        pi=TrackerPolicy(hist=2,hidden=[32])
        wm=WorldModel(self.mdp.scan_dim,hidden=[32])
        with torch.no_grad():
            wm.net[-1].weight.normal_(0,.001)
        wm.requires_grad_(False)
        obs=self.mdp.obs(S,torch.zeros(2,NUM_J),xi)
        cmd=self.task.command(S,f,torch.zeros(2,dtype=torch.long))
        action=pi(obs[:,None].expand(-1,2,-1),cmd)
        predicted=wm.step(S,action,normalize_xi(xi),self.mdp.scan(S,self.motion.boxes(f,torch.zeros_like(f))))
        loss=predicted[4].square().mean()
        loss.backward()
        gradients=[p.grad for p in pi.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0)
        self.assertTrue(all(p.grad is None for p in wm.parameters()))


if __name__ == '__main__':
    unittest.main()
