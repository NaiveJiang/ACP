"""Numerical equivalence, episode masks, and CUDA optimizer capture checks."""
import os
os.environ['ACP_ROBOT']='roban'

import unittest
import copy
import torch

from tools import test_footstep_tracker as fixture
from supermimic.acp.footstep.fast.models import FastKinematics,FastWorldModel,DeviceNorm,device_norms,FastContactTask
from supermimic.acp.footstep.fast.core import Updates,CudaUpdate,gated_step
from supermimic.acp.footstep.contact_only.task import ContactOnlyMDP,ContactOnlyPolicy
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.networks import WorldModel,RunningNorm,Critic
from supermimic.acp.mdp import flatten_state,XI_DIM
from supermimic.acp.wm_training import rollout,regression_loss
from supermimic.acp.robot import default_joint_pos_list


class FastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.kin=RobotKinematics('cpu')

    def setUp(self):
        fixture.TrackerTests.setUp(self)

    def tearDown(self):
        fixture.TrackerTests.tearDown(self)

    def test_fused_fk_keeps_fixed_frames_anchors_and_gradients(self):
        kin=FastKinematics('cpu')
        S=list(self.motion.ref_state(torch.tensor([0,1,2])))
        S[4]=S[4].clone().requires_grad_()
        expected=self.kin.forward(*S,all_bodies=True)
        actual=kin.forward(*S,all_bodies=True)
        for a,b in zip(expected,actual):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
        p,R=kin.pose(S[0],S[1],S[4])
        torch.testing.assert_close(p,expected[0],atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(R,expected[1],atol=2e-6,rtol=2e-6)
        ga=torch.autograd.grad(actual[0].square().sum(),S[4],retain_graph=True)[0]
        gb=torch.autograd.grad(expected[0].square().sum(),S[4])[0]
        torch.testing.assert_close(ga,gb,atol=1e-5,rtol=1e-5)

    def test_device_norm_matches_baseline_and_ignores_invalid_rows(self):
        a,b=RunningNorm(7),DeviceNorm(7)
        torch.manual_seed(4)
        for _ in range(3):
            x=torch.randn(9,7);a.update(x);b.update(x)
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=1e-6,rtol=1e-6)
        old=copy.deepcopy(b.state_dict())
        b.update(torch.full((3,7),float('nan')),torch.zeros(3,dtype=torch.bool))
        for k,v in old.items():torch.testing.assert_close(v,b.state_dict()[k])

    def test_structured_targets_round_trip(self):
        S=self.motion.ref_state(torch.tensor([0,1,2]))
        S1=tuple(x.clone() for x in S)
        S1=list(S1);S1[0]+=.003;S1[2]+=.02;S1[4]+=.004;S1[5]+=.03
        wm=FastWorldModel(3,[32],structured=True)
        reconstructed=wm.integrate(S,wm.targets(S,S1))
        for a,b in zip(reconstructed,S1):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)

    def test_static_mask_loss_matches_guarded_baseline_and_sanitizes_nan_suffix(self):
        kin=FastKinematics('cpu');task=FastContactTask(self.motion,kin)
        mdp=ContactOnlyMDP(kin,self.motion,torch.tensor(default_joint_pos_list()),task)
        wm=FastWorldModel(mdp.scan_dim,[32]);pi=device_norms(ContactOnlyPolicy(hidden=[32]));v=device_norms(Critic(mdp.critic_input_dim))
        with torch.no_grad():wm.net[-1].weight.normal_(0,.002)
        engine=Updates(wm,pi,v,mdp,task,None,None,None,compile=False)
        S=self.motion.ref_state(torch.tensor([0,1,2,3]));state=flatten_state(S)
        batch=dict(s=state[:,None].expand(-1,3,-1).clone(),s1=state[:,None].expand(-1,3,-1).clone(),
                   a=torch.randn(4,3,21)*.1,xi=torch.zeros(4,3,XI_DIM),
                   mask=torch.tensor([[1.,1.,0.],[1.,0.,0.],[0.,0.,0.],[1.,1.,1.]]))
        frames=torch.zeros(4,3,dtype=torch.long)
        pred=rollout(wm,mdp,self.motion,batch['s'],batch['a'],batch['xi'],frames,frames,batch['mask'])
        expected=regression_loss(wm,pred,batch['s1'],batch['mask'],engine.error_floor)[0]/3
        actual=engine._wm_loss(batch)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
        bad=copy.deepcopy(batch);inactive=~batch['mask'].bool()
        for k in ('s1','a','xi'):bad[k][inactive]=float('nan')
        bad['s'][2]=float('nan')
        loss=engine._wm_loss(bad)
        torch.testing.assert_close(loss,actual)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in wm.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA capture test')
    def test_capture_performs_one_update_and_rejects_nan_without_corrupting_state(self):
        torch.manual_seed(5)
        model=torch.nn.Linear(4,2).cuda()
        reference=copy.deepcopy(model)
        def opt(m):return torch.optim.Adam(m.parameters(),lr=torch.tensor(.001,device='cuda'),fused=True,capturable=True)
        optimizer,other=opt(model),opt(reference)
        def update(batch):
            loss=model(batch['x']).square().mean()
            good,grad=gated_step(loss,model,optimizer)
            return torch.stack((loss.detach(),good.float()))
        graph=CudaUpdate(update,[model],[optimizer],enabled=True)
        batch={'x':torch.randn(8,4,device='cuda')}
        graph(batch)
        gated_step(reference(batch['x']).square().mean(),reference,other)
        for a,b in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
        self.assertTrue(all(float(s['step'])==1 for s in optimizer.state.values()))
        before=copy.deepcopy(model.state_dict());state=copy.deepcopy(optimizer.state_dict())
        result=graph({'x':torch.full_like(batch['x'],float('nan'))})
        self.assertEqual(float(result[1]),0.)
        for k,v in before.items():torch.testing.assert_close(v,model.state_dict()[k])
        for k,v in state['state'].items():
            for name,tensor in v.items():torch.testing.assert_close(tensor,optimizer.state_dict()['state'][k][name])


if __name__=='__main__':unittest.main()
