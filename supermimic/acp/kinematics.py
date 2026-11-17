"""Differentiable robot forward kinematics in torch, including fixed frames and hinge anchors.

The same FK computes tracked-body poses and velocities for the reference motion, for logged
simulator states, and for world-model predictions, so the analytic reward head matches the
reward the simulator transitions are labeled with.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .robot import JOINT_NAMES, TRACKED_BODIES, get_spec
from ..rotations import axis_angle_to_mat, quat_to_mat


class RobotKinematics:
    def __init__(self, device: str, tracked: list[str] = TRACKED_BODIES):
        m = get_spec().compile()
        self.device = device
        jname_to_idx = {n: i for i, n in enumerate(JOINT_NAMES)}
        # All bodies in tree order (MuJoCo orders parents before children).
        bodies, parent, joint_idx, axes, offs_p, offs_R, anchors, refs = [], [], [], [], [], [], [], []
        mj_to_local = {}
        for b in range(1, m.nbody):
            name = m.body(b).name
            nj = m.body_jntnum[b]
            if nj > 1:
                raise ValueError(f"FK supports one hinge per body, got {nj} on {name}")
            jid = int(m.body_jntadr[b]) if nj else -1
            mj_to_local[b] = len(bodies)
            bodies.append(name)
            if nj and m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                if m.body_parentid[b] != 0:
                    raise ValueError("Free joint must be the robot root")
                parent.append(-1)
                joint_idx.append(-1)
            else:
                pb = m.body_parentid[b]
                if pb not in mj_to_local:
                    raise ValueError(f"Body {name} is outside the free-root robot tree")
                parent.append(mj_to_local[pb])
                if nj and m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError(f"Unsupported joint on {name}: {m.jnt_type[jid]}")
                joint_idx.append(jname_to_idx[m.joint(jid).name] if nj else -1)
            axes.append(m.jnt_axis[jid].copy() if nj else np.zeros(3))
            anchors.append(m.jnt_pos[jid].copy() if nj else np.zeros(3))
            refs.append(m.qpos0[m.jnt_qposadr[jid]] if nj else 0.)
            offs_p.append(m.body_pos[b].copy())
            offs_R.append(m.body_quat[b].copy())
        self.body_names = bodies
        self.parent = parent
        self.joint_idx = joint_idx
        self.axes = torch.tensor(np.array(axes), dtype=torch.float32, device=device)
        self.anchors = torch.tensor(np.array(anchors), dtype=torch.float32, device=device)
        self.refs = torch.tensor(refs, dtype=torch.float32, device=device)
        self.off_p = torch.tensor(np.array(offs_p), dtype=torch.float32, device=device)
        self.off_R = quat_to_mat(torch.tensor(np.array(offs_R), dtype=torch.float32, device=device))
        self.tracked_idx = [bodies.index(n) for n in tracked]
        self.num_bodies = len(bodies)
        # joint limits (in JOINT_NAMES order)
        lo, hi = np.zeros(len(JOINT_NAMES)), np.zeros(len(JOINT_NAMES))
        for j in range(m.njnt):
            n = m.joint(j).name
            if n in jname_to_idx:
                lo[jname_to_idx[n]], hi[jname_to_idx[n]] = m.jnt_range[j]
        self.q_lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self.q_hi = torch.tensor(hi, dtype=torch.float32, device=device)

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Tracked-body world positions, rotations, linear (link-origin) and angular velocities.

        Args:
            p, R, v, w: root position (B,3), rotation (B,3,3), world lin/ang velocity (B,3).
            q, qd: joint positions/velocities (B,NUM_J) in JOINT_NAMES order.
        """
        P, Rs, V, W = [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies
        for i in range(self.num_bodies):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint_idx[i]
            Rp = Rs[par]
            pi = P[par] + (Rp @ self.off_p[i].unsqueeze(-1)).squeeze(-1)
            R_off = Rp @ self.off_R[i]
            Ri, wi = R_off, W[par]
            joint_w = torch.zeros_like(w)
            rotated_anchor = torch.zeros_like(p)
            if j >= 0:
                Ri = R_off @ axis_angle_to_mat(self.axes[i], q[..., j] - self.refs[i])
                rotated_anchor = (Ri @ self.anchors[i].unsqueeze(-1)).squeeze(-1)
                pi = pi + (R_off @ self.anchors[i].unsqueeze(-1)).squeeze(-1) - rotated_anchor
                axis_w = (R_off @ self.axes[i].unsqueeze(-1)).squeeze(-1)
                joint_w = axis_w * qd[..., j : j + 1]
                wi = wi + joint_w
            vi = (V[par] + torch.cross(W[par], pi - P[par], dim=-1)
                  - torch.cross(joint_w, rotated_anchor, dim=-1))
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([P[i] for i in idx], -2),
            torch.stack([Rs[i] for i in idx], -3),
            torch.stack([V[i] for i in idx], -2),
            torch.stack([W[i] for i in idx], -2),
        )


# Compatibility for external callers of the original G1-only interface.
G1Kinematics = RobotKinematics
