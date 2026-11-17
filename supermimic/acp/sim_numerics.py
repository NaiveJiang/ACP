"""Quarantine numerical simulator failures before replay and normalization."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


def finite_step_state(previous, current):
    """Finite bookkeeping state plus an explicit transition-validity mask.

    Failed transitions are terminated and excluded from learning. The caller
    retains the original invalid state in diagnostics before resetting the world.
    """
    valid = torch.stack([torch.isfinite(x).flatten(1).all(1) for x in current]).all(0)
    safe = tuple(torch.where(valid.reshape((-1,) + (1,) * (x.ndim - 1)), x, old)
                 for old, x in zip(previous, current))
    return safe, valid


STATE_REASONS = {
    1: "nonfinite_simulator_state",
    2: "joint_position_outlier",
    4: "joint_velocity_outlier",
    8: "joint_position_jump",
    16: "root_linear_velocity_outlier",
    32: "root_angular_velocity_outlier",
    64: "root_position_jump",
}


@dataclass(frozen=True)
class PhysicalLimits:
    """Loose numerical quarantine bounds, NOT motor limits or task termination.

    Keep ordinary falls, impacts and small joint-limit violations as real dynamics.
    Fixed bounds must never adapt to the running statistics they protect.
    """
    joint_position_margin: float = 1.0  # rad beyond the MJCF joint range
    joint_speed_multiplier: float = 10.0
    joint_step_speed: float = 200.0  # rad/s averaged over one control step
    root_linear_speed: float = 50.0  # m/s
    root_angular_speed: float = 200.0  # rad/s


class PhysicalStateGuard:
    """Reject gross finite simulator outliers before reward, replay and norms.

    This operates on the pre-reset transition, independently of done/termination.
    The unchanged raw state remains available for diagnostics; safe placeholders
    only serve bookkeeping and must always be accompanied by valid=False.
    """

    def __init__(self, q_lo, q_hi, joint_speed, control_dt, limits=PhysicalLimits()):
        self.limits, self.control_dt = limits, control_dt
        self.q_lo = q_lo - limits.joint_position_margin
        self.q_hi = q_hi + limits.joint_position_margin
        self.joint_speed = torch.as_tensor(joint_speed, device=q_lo.device, dtype=q_lo.dtype) * limits.joint_speed_multiplier

    def __call__(self, previous, current):
        p, _, v, w, q, qd = current
        finite = torch.stack([torch.isfinite(x).flatten(1).all(1) for x in current]).all(0)
        checks = (
            ~finite,
            ((q < self.q_lo) | (q > self.q_hi)).any(-1),
            (qd.abs() > self.joint_speed).any(-1),
            ((q - previous[4]).abs() > self.limits.joint_step_speed * self.control_dt).any(-1),
            v.norm(dim=-1) > self.limits.root_linear_speed,
            w.norm(dim=-1) > self.limits.root_angular_speed,
            (p - previous[0]).norm(dim=-1) > self.limits.root_linear_speed * self.control_dt,
        )
        reasons = torch.zeros_like(finite, dtype=torch.int32)
        for bit, bad in zip(STATE_REASONS, checks):
            reasons.bitwise_or_(bad.to(torch.int32) * bit)
        valid = reasons == 0
        safe = tuple(torch.where(valid.reshape((-1,) + (x.ndim - 1) * (1,)), x, old)
                     for old, x in zip(previous, current))
        return safe, valid, reasons

    def metadata(self):
        return dict(limits=asdict(self.limits), control_dt=self.control_dt,
                    joint_position_lo=self.q_lo, joint_position_hi=self.q_hi,
                    joint_speed=self.joint_speed, reason_codes=STATE_REASONS)


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(v) for v in value)
    return value


class TransitionGuard:
    """Record isolated failures; stop a persistently failing simulator."""

    def __init__(self, log_dir, max_consecutive=3):
        self.directory = Path(log_dir) / "numerics"
        self.consecutive = 0
        self.rejected = 0
        self.max_consecutive = max_consecutive

    def observe(self, out, motion, sim_steps, iteration):
        bad = ~out["valid"]
        if not bool(bad.any()):
            self.consecutive = 0
            return
        ids = bad.nonzero(as_tuple=False).squeeze(-1)
        count = len(ids)
        self.rejected += count
        self.consecutive += 1
        clips = motion.clip_id[out["f"][ids]].tolist()
        codes = out.get("invalid_reason", (~out["valid"]).to(torch.int32))[ids].tolist()
        reasons = [[name for bit, name in STATE_REASONS.items() if code & bit] for code in codes]
        event = dict(iteration=iteration, sim_steps=sim_steps, invalid_worlds=ids.tolist(),
                     clips=[motion.names[c] for c in clips], rejected_total=self.rejected,
                     consecutive_control_steps=self.consecutive,
                     reason="nonfinite_simulator_state" if all(code == 1 for code in codes) else "invalid_simulator_state",
                     reason_codes=codes, reasons=reasons)
        self.directory.mkdir(parents=True, exist_ok=True)
        diagnostics = [self.directory / "first_bad_sim_transition.pt"]
        if any(code and not (code & 1) for code in codes):
            # A resumed run may already have the first NaN on disk. Preserve the
            # first finite outlier separately so that it still has a raw snapshot.
            diagnostics.append(self.directory / "first_bad_finite_sim_transition.pt")
        missing = [path for path in diagnostics if not path.exists()]
        if missing:
            selected = {k: v[ids] for k, v in out.items()
                        if isinstance(v, torch.Tensor) and v.ndim and v.shape[0] == len(bad)
                        and not k.startswith("_invalid")}
            raw = {k: v for k, v in out.items() if k.startswith("_invalid")}
            evidence = _cpu(dict(event=event, transition=selected, raw=raw))
            for diagnostic in missing:
                torch.save(evidence, diagnostic)
        with (self.directory / "sim_rejections.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")
        print(f"Simulator transitions quarantined: {event}; diagnostics: {self.directory}", flush=True)
        if count == len(bad) or self.consecutive >= self.max_consecutive:
            raise FloatingPointError(f"Persistent simulator numerical failures; see {self.directory}")
