"""Check gross position bounds against archived active frames, including terminals.

Recordings do not contain velocities: this audit deliberately cannot certify the
velocity checks. Use live simulation to exercise those checks as well.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("ACP_ROBOT", "roban")
import numpy as np
from supermimic.acp.kinematics import RobotKinematics
from supermimic.acp.sim_numerics import PhysicalLimits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    kin, cfg = RobotKinematics("cpu"), PhysicalLimits()
    lo, hi = kin.q_lo.numpy(), kin.q_hi.numpy()
    rows = []
    for path in sorted(args.recordings.glob("*/*.npz")):
        with np.load(path) as data:
            if "post_q" not in data.files:
                continue
            q, p, alive = data["post_q"], data["post_root_p"], data["alive"]
            prev_q = np.concatenate((data["initial_q"][None], q[:-1]))
            prev_p = np.concatenate((data["initial_root_p"][None], p[:-1]))
            excess = np.maximum(np.maximum(lo - q, q - hi), 0).max(-1)
            dq = np.abs(q - prev_q).max(-1)
            dp = np.linalg.norm(p - prev_p, axis=-1)
            checks = dict(nonfinite_position=~(np.isfinite(q).all(-1) & np.isfinite(p).all(-1)),
                          joint_position_outlier=excess > cfg.joint_position_margin,
                          joint_position_jump=dq > cfg.joint_step_speed * .02,
                          root_position_jump=dp > cfg.root_linear_speed * .02)
            bad = np.logical_or.reduce(list(checks.values())) & alive
            last = alive.sum(0) - 1
            assert (last >= 0).all()
            eps = json.loads((path.parent / "results.json").read_text())["episodes"]
            row = dict(recording=str(path.resolve()), episodes=len(last),
                       ordinary_failed_episodes=sum(e["failed"] and e["valid_physics"] for e in eps),
                       active_transitions=int(alive.sum()), rejected_transitions=int(bad.sum()),
                       rejected_terminal_transitions=int(bad[last, np.arange(len(last))].sum()),
                       reason_counts={k: int((v & alive).sum()) for k, v in checks.items()},
                       max_joint_range_excess_rad=float(excess[alive].max()),
                       max_joint_step_rad=float(dq[alive].max()), max_root_step_m=float(dp[alive].max()),
                       rejected_clips=data["names"][bad.any(0)].tolist())
            rows.append(row)
    if not rows:
        raise ValueError("No recorded trajectories found")
    totals = {k: sum(r[k] for r in rows) for k in ("episodes", "ordinary_failed_episodes", "active_transitions",
                                                "rejected_transitions", "rejected_terminal_transitions")}
    maxima = {k: max(r[k] for r in rows) for k in ("max_joint_range_excess_rad", "max_joint_step_rad", "max_root_step_m")}
    report = dict(scope="Archived positions only; no measured velocities stored in recordings. Includes terminal frames.",
                  totals=totals, maxima=maxima, recordings=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(totals=totals, maxima=maxima), indent=2))


if __name__ == "__main__":
    main()
