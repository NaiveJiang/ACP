"""Physics/scan/contact parity for the heightfield dataset extension."""
import os
os.environ["ACP_ROBOT"]="roban"
import tempfile
from pathlib import Path
import unittest

import mujoco
import numpy as np
import torch

from supermimic.acp import heightfield
from supermimic.acp.terrain import box_heights,box_sdf,box_closest_point,load_terrain
from supermimic.acp.make_labels import variant_labels
from supermimic.acp.robot import JOINT_NAMES,default_joint_pos_list


class HeightfieldTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);heightfield.initialize([])
    def tearDown(self):
        heightfield.initialize([]);self.tmp.cleanup()
    def field(self,h,spacing=(1.,1.)):
        p=self.root/f"h{len(heightfield.ASSETS)}.npz"
        np.savez(p,heights=np.asarray(h,np.float32),origin_xy=np.array([-1.,-1.]),spacing=np.array(spacing))
        i=heightfield.register(p)
        return i,torch.tensor(heightfield.token(i),dtype=torch.float32)
    def test_scan_matches_mujoco_triangles_and_batch_assets(self):
        xy=np.random.default_rng(6).uniform(-.99,.99,(200,2))
        tokens=[];expected=[]
        for h in (np.array([[0,.1,.2],[.2,.4,.2],[.2,.3,.4]]),np.array([[.2,.3],[.7,.4]])):
            i,t=self.field(h,spacing=(2/(h.shape[1]-1),2/(h.shape[0]-1)));tokens.append(t)
            s=mujoco.MjSpec();heightfield.add_geom(s,i,'h');m=s.compile();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
            expected.append([2-mujoco.mj_ray(m,d,np.r_[p,2.],np.array([0.,0.,-1.]),None,1,-1,np.zeros(1,np.int32)) for p in xy])
        b=torch.stack(tokens)[:,None];q=torch.tensor(xy,dtype=torch.float32)[None].expand(2,-1,-1)
        actual=box_heights(q,b)
        np.testing.assert_allclose(actual,np.array(expected),atol=1e-6)
        x=torch.cat((q,actual[...,None]+.01),-1).requires_grad_()
        dist=box_sdf(x,b[:,0]);pt=box_closest_point(x,b[:,0]);dist.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all());self.assertTrue((dist>0).all())
        torch.testing.assert_close(box_heights(pt[...,:2],b),pt[...,2],atol=1e-6,rtol=0)
    def test_heightfield_contact_witness_matches_flat_box(self):
        i,t=self.field(np.full((3,3),.1))
        q=np.tile(np.r_[0.,0.,.91,1.,0.,0.,0.,default_joint_pos_list()],(4,1))
        path=self.root/'clip_motion.npz'
        np.savez(path,joint_pos=q,joint_names=np.array(JOINT_NAMES),fps=np.float32(50),robot_id=np.array('roban'))
        box=np.array([[0,0,.1,1,0,1,1,.05,0]],float)
        expected=variant_labels((str(path),box,.005,.1))
        actual=variant_labels((str(path),t.numpy()[None],.005,.1))
        np.testing.assert_allclose(actual['contact_dist'][:,:2],expected['contact_dist'][:,:2],atol=.001)
        contact=actual['contact_dist'][:,:2]<.05
        self.assertTrue(contact.any())
        np.testing.assert_allclose(actual['contact_pos_w'][:,:2,2][contact],.1,atol=.001)
    def test_legacy_box_padding_with_heightfield_registered(self):
        self.field(np.full((3,3),.1))
        b=torch.tensor([[[0.,0.,.3,1,0,.2,.2,.15,0]]])
        xy=torch.tensor([[[0.,0.],[1.,1.]]])
        torch.testing.assert_close(box_heights(xy,b),torch.tensor([[.3,0.]]))

    def test_invalid_queries_do_not_corrupt_cuda_context(self):
        self.field(np.full((3, 3), .1))
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                q = torch.tensor([[[0., 0.], [10., 0.], [float('nan'), 0.],
                                   [0., float('inf')], [float('-inf'), 0.], [1e30, -1e30]]], device=device)
                ids = torch.zeros(1, dtype=torch.long, device=device)
                h = heightfield.heights(q, ids)
                torch.testing.assert_close(h[:, :2], torch.tensor([[.1, 0.]], device=device))
                self.assertTrue(h[:, 2:5].isnan().all())
                self.assertEqual(h[0, 5].item(), 0.)
                # Contact queries share the same grid gather and must be safe too.
                x = torch.cat((q, torch.ones_like(q[..., :1])), -1)
                dist, point = heightfield.distance_point(x, ids)
                self.assertTrue(dist[:, :2].isfinite().all())
                self.assertTrue(point[:, :2].isfinite().all())
                torch.testing.assert_close(heightfield.heights(q[:, :1], ids), h[:, :1])

    def test_invalid_metadata_is_rejected_before_registration(self):
        for key, bad in [("origin_xy", [float('nan'), 0.]), ("spacing", [1., float('inf')]),
                         ("spacing", [1.]), ("origin_xy", [1.])]:
            with self.subTest(key=key, bad=bad):
                fields = dict(heights=np.zeros((2, 2)), origin_xy=np.zeros(2), spacing=np.ones(2))
                fields[key] = np.array(bad)
                path = self.root / 'invalid.npz'
                np.savez(path, **fields)
                with self.assertRaisesRegex(ValueError, 'Invalid heightfield'):
                    heightfield.register(path)


if __name__=='__main__':unittest.main()
