"""GPU-only norms, fused RoBan FK, and optional BlinkMimic structured dynamics.

Structured integration and mm3/mv3 follow BlinkMimic (Fuyi Chen et al., MIT);
see vendor/LICENSE. RoBan fixed frames and hinge anchors remain supported.
"""
import torch
from torch import nn

from ...kinematics import RobotKinematics
from ...networks import WorldModel, RunningNorm, WM_OUT
from ...robot import NUM_J
from ....rotations import axis_angle_to_mat, yaw_mat, mat_to_6d, so3_log, skew
from ..task import FootstepTask
from ..contact_only.task import ContactOnlyTask
from ..stable_contact import StableContactTask
from ..upper_supervised import UpperSupervisedTask
from ..gait_style import GaitStyleTask
from ..leg_collision import LegCollisionTask
from ..body_goal import BodyGoalTask
from ..joint import JointTask
from ..consistent_joint import ConsistentJointTask


def mm3(a, b):
    return (a.unsqueeze(-1)*b.unsqueeze(-3)).sum(-2)


def mv3(a, b):
    return (a*b.unsqueeze(-2)).sum(-1)


def exp3(v):
    t2 = v.square().sum(-1, keepdim=True)
    t = (t2+1e-12).sqrt()
    a = torch.where(t2 < 1e-8, 1-t2/6, torch.sin(t)/t)
    b = torch.where(t2 < 1e-8, .5-t2/24, (1-torch.cos(t))/(t2+1e-12))
    k = skew(v)
    return torch.eye(3, device=v.device)+a[..., None]*k+b[..., None]*mm3(k, k)


def invariant(S):
    p, R, v, w, q, qd = S
    Rt = R.transpose(-1, -2)
    return torch.cat((p[..., 2:3], mat_to_6d(mm3(yaw_mat(R).transpose(-1, -2), R)),
                      mv3(Rt, v), mv3(Rt, w), q, qd), -1)


class DeviceNorm(RunningNorm):
    @torch.no_grad()
    def update(self, x, valid=None):
        x = x.reshape(-1, x.shape[-1]).float()
        if valid is None:
            valid = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        valid = valid.reshape(-1)
        x = torch.where(valid[:, None], x, 0.)
        n = valid.float().sum()
        denom = n.clamp_min(1)
        mean = x.sum(0)/denom
        var = torch.where(valid[:, None], (x-mean).square(), 0.).sum(0)/denom
        total = (self.count+n).clamp_min(1)
        weight = n/total
        delta = mean-self.mean
        m = self.mean+delta*weight
        v = self.var*(1-weight)+var*weight+delta.square()*(self.count/total)*weight
        good = ((n > 0) & (self.count < self.max_count)
                & torch.isfinite(m).all() & torch.isfinite(v).all())
        self.mean.copy_(torch.where(good, m, self.mean))
        self.var.copy_(torch.where(good, v, self.var))
        self.count.copy_(torch.where(good, self.count+n, self.count))


def device_norms(module):
    for name, child in list(module.named_children()):
        if isinstance(child, RunningNorm):
            norm = DeviceNorm(len(child.mean), child.max_count, child.eps).to(child.mean.device)
            norm.load_state_dict(child.state_dict())
            setattr(module, name, norm)
        else:
            device_norms(child)
    return module


class FastKinematics(RobotKinematics):
    def pose(self, p, R, q):
        P, Rs = [None]*self.num_bodies, [None]*self.num_bodies
        for i, parent in enumerate(self.parent):
            if parent < 0:
                P[i], Rs[i] = p, R
                continue
            Rp = Rs[parent]
            pi = P[parent]+mv3(Rp, self.off_p[i])
            Roff = mm3(Rp, self.off_R[i])
            j = self.joint_idx[i]
            if j >= 0:
                Ri = mm3(Roff, axis_angle_to_mat(self.axes[i], q[..., j]-self.refs[i]))
                pi = pi+mv3(Roff, self.anchors[i])-mv3(Ri, self.anchors[i])
            else:
                Ri = Roff
            P[i], Rs[i] = pi, Ri
        return torch.stack(P, -2), torch.stack(Rs, -3)

    def forward(self, p, R, v, w, q, qd, all_bodies=False):
        P, Rs, V, W = ([None]*self.num_bodies for _ in range(4))
        for i, parent in enumerate(self.parent):
            if parent < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            Rp = Rs[parent]
            pi = P[parent]+mv3(Rp, self.off_p[i])
            Roff = mm3(Rp, self.off_R[i])
            Ri, wi = Roff, W[parent]
            joint_w, anchor = torch.zeros_like(w), torch.zeros_like(p)
            j = self.joint_idx[i]
            if j >= 0:
                Ri = mm3(Roff, axis_angle_to_mat(self.axes[i], q[..., j]-self.refs[i]))
                anchor = mv3(Ri, self.anchors[i])
                pi = pi+mv3(Roff, self.anchors[i])-anchor
                joint_w = mv3(Roff, self.axes[i])*qd[..., j:j+1]
                wi = wi+joint_w
            vi = V[parent]+torch.cross(W[parent], pi-P[parent], dim=-1)-torch.cross(joint_w, anchor, dim=-1)
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = list(range(self.num_bodies)) if all_bodies else self.tracked_idx
        return tuple(torch.stack([xs[i] for i in idx], axis) for xs, axis in ((P,-2),(Rs,-3),(V,-2),(W,-2)))


class Geometry:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A Python list used as a CUDA advanced index copies a host tensor on
        # every call and is illegal during graph capture.
        self.foot_ids = torch.tensor(self.foot_ids,device=self.motion.device)

    def foot_geometry(self, S):
        p, R = self.kin.pose(S[0], S[1], S[4])
        p, R = p[:, self.foot_ids], R[:, self.foot_ids]
        sole = p+mv3(R, self.motion.sole_offset)
        points = p[:, self.sphere_owner]+mv3(R[:, self.sphere_owner], self.sphere_points)
        gap = points[..., 2]-self.sphere_radii
        distance = torch.stack([gap[:, s:e].min(-1).values for s,e in self.sphere_bounds], -1)
        return sole, R, distance


class FastContactTask(Geometry, ContactOnlyTask):
    pass


class FastStableContactTask(Geometry, StableContactTask):
    pass


class FastUpperSupervisedTask(Geometry, UpperSupervisedTask):
    pass


class FastGaitStyleTask(Geometry, GaitStyleTask):
    pass


from ..next_contact import NextContactTask


class FastNextContactTask(Geometry, NextContactTask):
    pass


class FastLegCollisionTask(Geometry, LegCollisionTask):
    pass


class FastBodyGoalTask(Geometry, BodyGoalTask):
    pass


class FastJointTask(Geometry, JointTask):
    pass


from ..walking_planner import WalkingPlannerTask


class FastWalkingPlannerTask(Geometry, WalkingPlannerTask):
    pass


class FastConsistentJointTask(Geometry, ConsistentJointTask):
    pass


class FastVelocityTask(Geometry, FootstepTask):
    pass


class FastWorldModel(WorldModel):
    def __init__(self, scan_dim, hidden=(512,512,512), structured=False, amp=False):
        super().__init__(scan_dim, hidden)
        device_norms(self)
        self.inc_norm = DeviceNorm(WM_OUT, eps=1e-4)
        self.structured, self.amp = structured, amp

    def inputs(self, S, a, xi_n, hs):
        return torch.cat((invariant(S), a, xi_n, hs), -1)

    def drift(self, S, v1, w1, qd1):
        p, R, v, w, q, qd = S
        # BlinkMimic: four semi-implicit physics substeps -> new-velocity weight .625.
        old, new = .02*.375, .02*.625
        return mv3(yaw_mat(R).transpose(-1,-2),old*v+new*v1), mv3(R.transpose(-1,-2),old*w+new*w1),old*qd+new*qd1

    def targets(self, S, S1):
        inc = WorldModel.targets(S, S1)
        if not self.structured:
            return inc
        dp, dt, dv, dw, dq, dqd = inc.split((3,3,3,3,NUM_J,NUM_J), -1)
        gp, gt, gq = self.drift(S, S1[2], S1[3], S1[5])
        return torch.cat((dp-gp,dt-gt,dv,dw,dq-gq,dqd), -1)

    def integrate(self, S, d):
        p, R, v, w, q, qd = S
        dp, dt, dv, dw, dq, dqd = d.split((3,3,3,3,NUM_J,NUM_J), -1)
        v1, w1, qd1 = v+mv3(R,dv), w+mv3(R,dw), qd+dqd
        if self.structured:
            gp, gt, gq = self.drift(S,v1,w1,qd1)
            dp, dt, dq = dp+gp, dt+gt, dq+gq
        return p+mv3(yaw_mat(R),dp),mm3(R,exp3(dt)),v1,w1,q+dq,qd1

    def step(self, S, a, xi_n, hs):
        x = self.in_norm(self.inputs(S,a,xi_n,hs))
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp):
            y = self.net(x).float()
        return self.integrate(S, y*self.out_norm.std()+self.out_norm.mean)


def flat_scan(S, count):
    return S[0][..., 2:3].expand(*S[0].shape[:-1], count)
