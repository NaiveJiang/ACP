"""Summarize strict course outcomes and generated geometry, not curriculum latches."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import mujoco


def stone_gaps(course):
    start, goal = np.array(course["local_start"]), np.array(course["local_goal"])
    intervals = []
    for g in course["geometry"]:
        p, s = np.array(g["pos"]), np.array(g["size"])
        if g["type"] != 6 or p[2] + s[2] < -.5 or abs(p[1] - start[1]) > s[1]:
            continue
        a, b = max(start[0], p[0] - s[0]), min(goal[0], p[0] + s[0])
        if b > a:
            intervals.append((a, b))
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return [{"from_x": a[1], "to_x": b[0], "width_m": b[0] - a[1]}
            for a, b in zip(merged, merged[1:])]


def summarize(run):
    episodes = [json.loads(x) for x in (run / "episodes.jsonl").read_text().splitlines()]
    model = mujoco.MjModel.from_binary_path(str(run / "source_model.mjb"))
    data = mujoco.MjData(model)
    terrain_body = model.body("terrain").id
    rows = []
    groups = {}
    for e in episodes:
        if e["incomplete"]:
            continue
        name, d = e["terrain_type"], e["difficulty"]
        q = np.load(run / "raw" / e["episode_id"] / "states.npz")["qpos"]
        data.qpos[:] = q[0]
        mujoco.mj_forward(model, data)
        initial_penetration = min([float(c.dist) for c in data.contact
            if terrain_body in model.geom_bodyid[c.geom]] + [0.])
        first = e["first_done"] or {}
        terms = first.get("terms", [])
        limit = .5 if name in ("stepping_stones", "narrow_beams") or (name.endswith("_narrow") and d >= .3) else .8
        if initial_penetration < -.01:
            outcome = "invalid_spawn"
        elif not e["finite"]:
            outcome = "nonfinite"
        elif first.get("terminated"):
            outcome = "fall_or_termination"
        elif e["max_lateral_deviation"] > limit:
            outcome = "off_route"
        elif "course_complete" in terms:
            outcome = "success"
        else:
            outcome = "timeout"
        row = {**e, "outcome": outcome, "lateral_limit_m": limit,
               "initial_min_contact_distance_m": initial_penetration}
        rows.append(row)
        key = (name, d)
        if key not in groups:
            groups[key] = {"terrain": name, "difficulty": d, "counts": Counter(), "geometry_instances": set()}
        groups[key]["counts"][outcome] += 1
        groups[key]["geometry_instances"].add((e["terrain_row"], e["terrain_col"]))
    courses = json.loads((run / "courses.json").read_text())
    geometry = []
    for c in courses:
        name, d = c["terrain"], c["difficulty"]
        rec = {k: v for k, v in c.items() if k != "geometry"}
        if "slope" in name:
            rec.update(gradient=.4*d, angle_deg=float(np.degrees(np.arctan(.4*d))),
                       corridor_width_m=1. if name.endswith("_narrow") and d >= .3 else None)
        if name == "stepping_stones":
            controlled = c.get("controlled_gap_m")
            rec.update(route_gaps=stone_gaps(c), nominal_stone_size_m=7.5/7-controlled if controlled is not None else .8-.4*d,
                       nominal_grid_pitch_m=7.5/7, pit_floor_z_m=-2.)
        if name == "narrow_beams":
            rec.update(beam_width_m=.8-.6*d, pit_floor_z_m=-2.)
        if name == "random_rough":
            rec["noise_scale"] = .1*d
        if name in ("threshold", "random_spread_boxes"):
            heights = [g["pos"][2]+g["size"][2] for g in c["geometry"] if g["type"] == 6 and g["pos"][2]+g["size"][2] > 1e-6]
            rec["obstacle_top_height_range_m"] = [min(heights), max(heights)] if heights else None
        geometry.append(rec)
    summary = []
    for (name, d), g in groups.items():
        count = dict(g["counts"])
        n = sum(count.values())
        summary.append(dict(terrain=name, difficulty=d, trials=n, counts=count,
                            success_rate=count.get("success", 0)/n, geometry_instances=len(g["geometry_instances"])))
    result = dict(total_closed=len(rows), invalid_spawns=sum(e["outcome"] == "invalid_spawn" for e in rows), incomplete=sum(e["incomplete"] for e in episodes),
                  summary=summary, geometry=geometry, episodes=rows,
                  criterion="course_complete + 0.4 s follow-through, no source termination/pit_fall, finite, route corridor respected; 24 s deadline")
    (run / "analysis.json").write_text(json.dumps(result, indent=2)+"\n")
    print("terrain difficulty trials success failures off_route timeout invalid_spawn")
    for g in summary:
        c = g["counts"]
        print(g["terrain"], g["difficulty"], g["trials"], c.get("success", 0), c.get("fall_or_termination", 0), c.get("off_route", 0), c.get("timeout", 0), c.get("invalid_spawn", 0))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    summarize(ap.parse_args().run.resolve())
