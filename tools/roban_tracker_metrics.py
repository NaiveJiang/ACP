"""Read-only rollout measurements; no RNG draws or changes to policy/simulator."""
import numpy as np
import torch

from supermimic.acp.robot import CONTACT_BODIES
from supermimic.acp.task import PLACE_WINDOW, placement_error, to_base
from supermimic.rotations import rot_angle_sq


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return dict(count=0, mean=None, rmse=None, p50=None, p95=None, max=None)
    return dict(count=int(x.size), mean=float(x.mean()), rmse=float(np.sqrt(np.mean(x*x))),
                p50=float(np.quantile(x, .5)), p95=float(np.quantile(x, .95)), max=float(x.max()))


class TrackerMetrics:
    names = ["root_xy_error_m", "root_z_abs_error_m", "root_orientation_error_rad", "root_yaw_error_rad",
             "root_linear_velocity_error_m_s", "joint_angle_rmse_rad",
             "left_foot_link_world_error_m", "right_foot_link_world_error_m",
             "left_foot_link_local_error_m", "right_foot_link_local_error_m",
             "left_foot_orientation_error_rad", "right_foot_orientation_error_rad",
             "left_foot_link_horizontal_speed_m_s", "right_foot_link_horizontal_speed_m_s"]

    def __init__(self, ctc, n, device):
        self.ctc, self.motion, self.kin = ctc, ctc.motion, ctc.mdp.kin
        self.feet = [list(CONTACT_BODIES).index(n) for n in ("left_foot", "right_foot")]
        self.link_ids = ctc.link_idx[self.feet]
        self.tw = torch.zeros(PLACE_WINDOW, n, len(CONTACT_BODIES), 3, device=device)
        self.tv = torch.zeros(PLACE_WINDOW, n, len(CONTACT_BODIES), dtype=torch.bool, device=device)
        self.previous_touch = torch.ones(n, len(CONTACT_BODIES), dtype=torch.bool, device=device)
        self.previous_action = None
        self.rows = []

    def command(self, f, variant, targets_w=None, target_valid=None):
        # Measure against the selected intended targets before command noise.
        # Defaults preserve the reference-only protocol; videos show noisy commands.
        if targets_w is None:
            targets_w = self.ctc.next_pos[f, self.ctc.rv(variant)]
        if target_valid is None:
            target_valid = self.ctc.next_valid[f, self.ctc.rv(variant)]
        self.tw = torch.cat((self.tw[1:], targets_w[None]))
        self.tv = torch.cat((self.tv[1:], target_valid[None]))

    @torch.no_grad()
    def observe(self, state, f, variant, valid_active, action):
        p, R, v, _, q, _ = state
        m, c = self.motion, self.ctc
        rp, rR = m.root_p[f], m.root_R[f]
        fk = self.kin.forward(*state)
        bp, bR, bv = fk[:3]
        feet, rfeet = bp[:, self.link_ids], m.body_p[f][:, self.link_ids]
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0]) - torch.atan2(rR[:, 1, 0], rR[:, 0, 0])
        base = torch.stack(((p[:, :2]-rp[:, :2]).norm(dim=-1), (p[:, 2]-rp[:, 2]).abs(),
                            rot_angle_sq(R, rR).clamp_min(0).sqrt(),
                            torch.atan2(yaw.sin(), yaw.cos()).abs(),
                            (v-m.root_v[f]).norm(dim=-1), (q-m.q[f]).square().mean(-1).sqrt()), -1)
        values = torch.cat((base, (feet-rfeet).norm(dim=-1),
                            (to_base(feet,p,R)-to_base(rfeet,rp,rR)).norm(dim=-1),
                            rot_angle_sq(bR[:, self.link_ids],m.body_R[f][:, self.link_ids]).clamp_min(0).sqrt(),
                            bv[:, self.link_ids, :2].norm(dim=-1)), -1)
        boxes = m.boxes(f, variant)
        touch = (c.body_distance(state, boxes, fk=fk) <= c.cfg.touch_dist) & valid_active[:, None]
        labels = c.labels[f, c.rv(variant)] & valid_active[:, None]
        onsets = touch & ~self.previous_touch
        self.previous_touch = touch
        pe = placement_error(c.contact_point(state, boxes, fk=fk), self.tw, self.tv)
        pe = torch.where(onsets & pe.isfinite(), pe, torch.nan)
        delta = (action-self.previous_action).square().mean(-1).sqrt() if self.previous_action is not None else torch.full_like(p[:, 0],torch.nan)
        self.previous_action = action.clone()
        self.rows.append(tuple(x.detach().cpu().numpy().copy() for x in (values,valid_active,touch,labels,onsets,pe,delta)))

    def finish(self):
        values, active, touch, labels, onsets, pe, delta = [np.stack([r[i] for r in self.rows]) for i in range(7)]
        result = []
        for j in range(active.shape[1]):
            mask = active[:,j]
            tracking = {name:stats(values[mask,j,k]) for k,name in enumerate(self.names)}
            for side,b,k in [("left",self.feet[0],12),("right",self.feet[1],13)]:
                tracking[side+"_support_link_horizontal_speed_m_s"] = stats(values[touch[:,j,b]&labels[:,j,b],j,k])
            contacts = {}
            for b,name in enumerate(CONTACT_BODIES):
                tp = int((touch[:,j,b]&labels[:,j,b]).sum());nt=int(touch[:,j,b].sum());nl=int(labels[:,j,b].sum())
                placements=pe[:,j,b];matched=placements[np.isfinite(placements)]
                contacts[name]=dict(true_positive_frames=tp,touch_frames=nt,label_frames=nl,
                    precision=tp/nt if nt else None,recall=tp/nl if nl else None,f1=2*tp/(nt+nl) if nt+nl else None,
                    onsets=int(onsets[:,j,b].sum()),matched_onsets=len(matched),placement_error_m=stats(matched),
                    placement_within_0_1=float((matched<=.1).mean()) if len(matched) else None)
            good=values[mask,j]
            result.append(dict(tracking=tracking,contacts=contacts,action_increment_rmse=stats(delta[mask,j]),
                metric_frames=int(mask.sum()),finite_tracking_metrics=bool(np.isfinite(good).all()),
                terminal_root_xy_error_m=float(good[-1,0]) if len(good) else None,
                terminal_root_yaw_error_rad=float(good[-1,3]) if len(good) else None))
        return result
