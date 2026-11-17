"""Audit terrain profiles and real foot support in successful source rollouts."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def audit(run):
    report = json.loads((run / "analysis.json").read_text())
    model = mujoco.MjModel.from_binary_path(str(run / "source_model.mjb"))
    data = mujoco.MjData(model)
    terrain = model.body("terrain").id
    feet = [model.body("robot/leg_l6_link").id, model.body("robot/leg_r6_link").id]
    courses = json.loads((run / "courses.json").read_text())
    snapshot = np.load(run / "terrain_snapshot.npz")
    profiles = []
    data.qpos[2] = -100
    mujoco.mj_forward(model, data)
    for i, c in enumerate(courses):
        if c["terrain"] != "random_rough":
            continue
        row, col = i % 4, i // 4
        offset = snapshot["terrain_origins"][row, col] - c["local_start"]
        a, b = np.array(c["local_start"])+offset, np.array(c["local_goal"])+offset
        heights = []
        geom_id = np.zeros(1, np.int32)
        for xy in np.linspace(a[:2], b[:2], 346):
            dist = mujoco.mj_ray(model, data, np.array([*xy, 20.]), np.array([0., 0., -1.]), None, 1, -1, geom_id)
            assert dist >= 0
            heights.append(20-dist)
        profiles.append(dict(terrain=c["terrain"], difficulty=c["difficulty"], row=row, col=col,
                             route_height_min=min(heights), route_height_max=max(heights),
                             route_height_peak_to_peak=float(np.ptp(heights)), sampling_step_m=.01))
    if profiles:
        (run / "rough_profile.json").write_text(json.dumps(profiles, indent=2)+"\n")
    support = []
    for e in report["episodes"]:
        if e["outcome"] != "success" or e["terrain_type"] not in ("stepping_stones", "random_spread_boxes"):
            continue
        states = np.load(run / "raw" / e["episode_id"] / "states.npz")
        goal = np.array(e["route_goal"])
        goal_geoms = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == terrain and model.geom_type[g] == 6:
                if np.all(np.abs(model.geom_pos[g, :2]-goal[:2]) <= model.geom_size[g, :2]+1e-5) and model.geom_pos[g, 2]+model.geom_size[g, 2] > 0:
                    goal_geoms.append(g)
        supported, tops = set(), []
        for k in range(1, len(states["qpos"])):
            data.qpos[:] = states["qpos"][k]
            mujoco.mj_forward(model, data)
            for contact in data.contact:
                if contact.dist > .002:
                    continue
                g0, g1 = contact.geom
                for f, body in enumerate(feet):
                    if states["feet_force"][k-1, f] <= 20:
                        continue
                    other = g1 if model.geom_bodyid[g0] == body else g0 if model.geom_bodyid[g1] == body else -1
                    if other < 0 or model.geom_bodyid[other] != terrain or model.geom_type[other] != 6:
                        continue
                    top = model.geom_pos[other, 2]+model.geom_size[other, 2]
                    if abs(contact.frame[2]) >= .8 and abs(contact.pos[2]-top) <= .02:
                        tops.append(float(top))
                        if other in goal_geoms:
                            supported.add(k)
        support.append(dict(episode=e["episode_id"], terrain=e["terrain_type"], difficulty=e["difficulty"],
                            goal_stone_geom_ids=goal_geoms, supported_frame_count=len(supported),
                            first_support_frame=min(supported) if supported else None,
                            max_loaded_top_contact_height_m=max(tops) if tops else None))
    (run / "foot_support_audit.json").write_text(json.dumps(support, indent=2)+"\n")
    stones = [r for r in support if r["terrain"] == "stepping_stones"]
    print("stone successes / confirmed goal-stone support:", len(stones), sum(r["supported_frame_count"] > 0 for r in stones))
    for d in (0., .3, .6, 1.):
        boxes = [r["max_loaded_top_contact_height_m"] for r in support if r["terrain"] == "random_spread_boxes" and r["difficulty"] == d]
        if boxes:
            print("box successes", d, "maximum foot-supported obstacle top heights", boxes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    audit(ap.parse_args().run.resolve())
