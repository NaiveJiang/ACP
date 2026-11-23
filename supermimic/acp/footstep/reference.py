"""Measured 50 Hz reference states and oracle contacts, isolated from the legacy loader."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .data import KINEMATIC_REFERENCE, JOINT_NAMES, reference_fingerprint
from ..motion import MotionLib
from ...rotations import quat_to_mat


class FootstepMotion(MotionLib):
    def __init__(self, intervals, kin, device, horizons=(.1, .2, .3, .5, .8, 1.), max_clips=0, *, reserve_steps=0):
        if max_clips:
            intervals = intervals[:max_clips]
        self.device = device
        self.horizons = torch.tensor(horizons, dtype=torch.float32, device=device)
        steps = np.rint(np.asarray(horizons)*50).astype(int)
        if (not len(steps) or np.any(steps < 1) or len(set(steps)) != len(steps)
                or not np.allclose(steps*.02, horizons, atol=1e-7, rtol=0)):
            raise ValueError("Horizons must be distinct positive 20 ms multiples")
        self.horizon_steps = torch.tensor(steps, device=device)
        margin = max(int(steps.max()), reserve_steps)
        arrays = {k: [] for k in ("p", "R", "v", "w", "q", "qd", "feet", "contact", "velocity", "requested")}
        lengths, usable, offsets = [], [], []
        self.names, self.files, self.sources = [], [], []
        self.geometry_reports = []
        self.sole_offset = torch.tensor([.0486, 0., -.04], device=device)
        self.foot_ids = [kin.body_names.index(n) for n in ("leg_l6_link", "leg_r6_link")]
        cursor = 0
        for part in intervals:
            source = Path(part["source"])
            metadata_text = (source.parent/"metadata.json").read_text()
            meta = json.loads(metadata_text)
            if abs(float(meta.get("sample_hz", 50))-50.) > 1e-6:
                raise ValueError("Tracker requires 50 Hz reference")
            if not np.allclose(meta.get("sole_offset"), self.sole_offset.cpu().numpy()):
                raise ValueError("Unexpected sole reference point")
            joints = {j["name"]: j for j in meta["joints"]}
            qidx = [joints[n]["qpos_adr"] for n in JOINT_NAMES]
            vidx = [joints[n]["dof_adr"] for n in JOINT_NAMES]
            lo, hi = part["start"], part["end"]
            if hi-lo <= margin+2:
                continue
            path = source/(part["case"]+"_labels.npz")
            if "fingerprint" in part and reference_fingerprint(path, metadata_text) != part["fingerprint"]:
                raise ValueError("Reference data changed since split/checkpoint: "+str(path))
            kinematic = part.get("reference_schema") == KINEMATIC_REFERENCE
            with np.load(path, allow_pickle=False) as d:
                qpos, qvel = d["qpos"][lo:hi], d["qvel"][lo:hi]
                feet, cmd = d["foot_state"][lo:hi], d["velocity_command"][lo:hi]
                requested = d["requested"][lo:hi] if reserve_steps and "requested" in d else cmd
                times = d["sim_time"][lo:hi]
                if kinematic:
                    contact = d["contact"][lo:hi]
                    if contact.dtype != np.bool_ or contact.shape != (hi-lo, 2):
                        raise ValueError("Invalid kinematic contact array in "+str(path))
                else:
                    manifest = json.loads((source/"manifest.json").read_text())
                    contact = (feet[:, :, 13]>0)&(feet[:, :, 9]>=manifest["force_threshold_N"])
            if (not all(np.isfinite(x).all() for x in (qpos, qvel, feet, cmd, requested))
                    or not np.allclose(np.diff(times), .02, atol=1e-5, rtol=0)):
                raise ValueError("Missing state/command or irregular time in " + str(path))
            t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
            p, R = t(qpos[:, :3]), quat_to_mat(t(qpos[:, 3:7]))
            v, w = t(qvel[:, :3]), (R@t(qvel[:, 3:6]).unsqueeze(-1)).squeeze(-1)
            q, qd = t(qpos[:, qidx]), t(qvel[:, vidx])
            data = dict(p=p, R=R, v=v, w=w, q=q, qd=qd, feet=t(feet), velocity=t(cmd), requested=t(requested),
                        contact=torch.as_tensor(contact, device=device))
            # Keep measured feet as command targets; report compatibility with the training asset.
            pick = torch.linspace(0, len(p)-1, min(32, len(p)), device=device).long()
            bp, bR = kin.forward(p[pick], R[pick], v[pick], w[pick], q[pick], qd[pick], all_bodies=True)[:2]
            sole = bp[:, self.foot_ids]+(bR[:, self.foot_ids]@self.sole_offset.view(1, 1, 3, 1)).squeeze(-1)
            err = torch.linalg.vector_norm(sole-data["feet"][pick, :, :3], dim=-1)
            self.geometry_reports.append(dict(source=str(source), case=part["case"],
                                              mean_m=float(err.mean()), max_m=float(err.max())))
            for key in arrays:
                arrays[key].append(data[key])
            lengths.append(len(p)); usable.append(len(p)-margin); offsets.append(cursor)
            cursor += len(p)
            self.names.append(source.parent.name+"/"+part["case"]+f"[{lo}:{hi}]")
            self.files.append(str(path))
            self.sources.append(part)
        if not lengths:
            raise ValueError("No clips long enough for selected horizons")
        values = {k: torch.cat(v) for k, v in arrays.items()}
        self.root_p, self.root_R, self.root_v, self.root_w = (values[k] for k in ("p", "R", "v", "w"))
        self.q, self.qd = values["q"], values["qd"]
        self.feet, self.contact, self.velocity_command = (values[k] for k in ("feet", "contact", "velocity"))
        self.requested_velocity = values["requested"]
        self.num_frames = cursor
        self.clip_start = torch.tensor(offsets, device=device)
        self.clip_len = torch.tensor(usable, device=device)
        self.clip_end = self.clip_start+self.clip_len-1
        self.data_clip_end = self.clip_start+torch.tensor(lengths, device=device)-1
        self.clip_id = torch.cat([torch.full((n,), c, device=device, dtype=torch.long) for c, n in enumerate(lengths)])
        body = [[] for _ in range(4)]
        with torch.no_grad():
            for start in range(0, cursor, 8192):
                frames = torch.arange(start, min(cursor, start+8192), device=device)
                out = kin.forward(*self.ref_state(frames))
                for i in range(4):
                    body[i].append(out[i])
        self.body_p, self.body_R, self.body_v, self.body_w = [torch.cat(x) for x in body]
        self.clip_boxes = torch.zeros(len(lengths), 1, 1, 9, device=device)
        self.num_variants, self.goal_boxes = 1, None
        self.variant_ok = torch.ones(len(lengths), 1, dtype=torch.bool, device=device)
        starts, owners = [], []
        self.frame_bin = torch.zeros(cursor, device=device, dtype=torch.long)
        for c, (start, length) in enumerate(zip(offsets, usable)):
            for s in range(start, start+length, 50):
                self.frame_bin[s:min(s+50, start+length)] = len(starts)
                starts.append(s); owners.append(c)
        self.bin_start = torch.tensor(starts, device=device)
        self.bin_clip = torch.tensor(owners, device=device)
        self.num_bins = len(starts)
