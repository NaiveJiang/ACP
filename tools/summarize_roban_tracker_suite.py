"""Validate and aggregate a completed fixed-checkpoint tracker evaluation suite.

RMSEs are pooled by frame count; contact rates use summed counts. No pooled
quantiles are inferred from per-episode quantiles. Output includes JSON,
Markdown tables, and a completion heatmap.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


LABELS = {
    "nominal": "无噪声重复", "obs_noise": "观测噪声", "command_noise": "目标与扫描噪声",
    "action_noise": "动作噪声", "all_noise": "三类噪声", "domain_all_noise": "物理/初态随机化＋三类噪声",
    "friction_low": "摩擦系数 0.3", "friction_high": "摩擦系数 1.2",
    "goal_short": "根目标前瞻 0.5 s", "goal_long": "根目标前瞻 1.5 s",
}
CATEGORIES = [
    "flat", "flat_backward", "flat_left", "flat_right", "flat_turn_left", "flat_turn_right",
    "pyramid_stairs", "pyramid_stairs_narrow", "pyramid_inv_stairs", "pyramid_inv_stairs_narrow",
    "variable_height_stairs_up", "variable_height_stairs_down", "hf_pyramid_slope",
    "hf_pyramid_slope_inv", "hf_pyramid_slope_narrow", "hf_pyramid_slope_inv_narrow",
    "random_rough", "threshold", "narrow_beams", "random_spread_boxes",
]


def pooled_stats(items):
    items = [s for s in items if s["count"]]
    count = sum(s["count"] for s in items)
    if not count:
        return dict(count=0, mean=None, rmse=None, max=None)
    return dict(count=count, mean=sum(s["mean"]*s["count"] for s in items)/count,
                rmse=math.sqrt(sum(s["rmse"]**2*s["count"] for s in items)/count),
                max=max(s["max"] for s in items))


def aggregate(episodes):
    n = len(episodes)
    contacts = {}
    for body in episodes[0]["contacts"]:
        rows = [e["contacts"][body] for e in episodes]
        tp = sum(r["true_positive_frames"] for r in rows)
        nt = sum(r["touch_frames"] for r in rows)
        nl = sum(r["label_frames"] for r in rows)
        matched = sum(r["matched_onsets"] for r in rows)
        contacts[body] = dict(true_positive_frames=tp, touch_frames=nt, label_frames=nl,
                              precision=tp/nt if nt else None, recall=tp/nl if nl else None,
                              f1=2*tp/(nt+nl) if nt+nl else None,
                              onsets=sum(r["onsets"] for r in rows), matched_onsets=matched,
                              placement_error_m=pooled_stats([r["placement_error_m"] for r in rows]),
                              placement_within_0_1=sum((r["placement_within_0.1"] or 0)*r["matched_onsets"]
                                                      for r in rows)/matched if matched else None)
    return dict(episodes=n, completed=sum(e["completed"] for e in episodes),
                completion_rate=sum(e["completed"] for e in episodes)/n,
                invalid_physics=sum(not e["valid_physics"] for e in episodes),
                valid_completed=sum(e["valid_physics"] and e["completed"] for e in episodes),
                active_overflow_episodes=sum(bool(e["active_physics_overflow_flags"]) for e in episodes),
                inactive_overflow_episodes=sum(bool(e["inactive_physics_overflow_flags"]) for e in episodes),
                simulated_seconds=sum(e["duration_s"] for e in episodes),
                reference_seconds=sum(e["reference_steps"]*.02 for e in episodes),
                mean_progress=sum(e["reference_progress"] for e in episodes)/n,
                mean_normalized_return=sum(e["normalized_return"] for e in episodes)/n,
                tracking={key:pooled_stats([e["tracking"][key] for e in episodes])
                          for key in episodes[0]["tracking"]}, contacts=contacts,
                action_increment_rmse=pooled_stats([e["action_increment_rmse"] for e in episodes]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    directory = args.directory.resolve()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((directory/"manifest.json").read_text())
    status = json.loads((directory/"status.json").read_text())
    assert status["phase"] == "complete" and not status["failed"], "Suite is not complete"
    assert hashlib.sha256(Path(manifest["checkpoint"]).read_bytes()).hexdigest() == manifest["checkpoint_sha256"]
    assert hashlib.sha256((root/"data/roban_motion/manifest.json").read_bytes()).hexdigest() == manifest["training_manifest_sha256"]
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((directory/"source"/name).read_bytes()).hexdigest() == digest
    data_manifest = json.loads((root/"data/roban_motion/manifest.json").read_text())
    dataset_names = {r["clip"] for r in data_manifest}
    true_flat = {r["clip"] for r in data_manifest if r["clip"].startswith("flat/") and r["box_count"] == 0}
    assert len(dataset_names) == 200 and len(true_flat) == 9
    per_trial, groups, failures = {}, defaultdict(list), []
    for case in manifest["trials"]:
        name = case["name"]
        result = json.loads((directory/(name+".json")).read_text())
        assert result["checkpoint_sha256"] == manifest["checkpoint_sha256"]
        ep = result["episodes"]
        assert len(ep) == 200 and {r["clip"] for r in ep} == dataset_names
        assert Counter(r["category"] for r in ep) == {k:10 for k in CATEGORIES}
        protocol = result["protocol"]
        options = case["options"]
        expected_noise = options[options.index("--noise")+1:] if "--noise" in options else []
        assert protocol["seed"] == case["seed"] and protocol["noise"] == expected_noise
        assert protocol["domain_randomization"] == ("--domain-randomization" in options)
        expected_friction = float(options[options.index("--friction")+1]) if "--friction" in options else None
        expected_goal = float(options[options.index("--goal-s")+1]) if "--goal-s" in options else 1.
        assert protocol["friction_override"] == expected_friction and protocol["goal_s"] == expected_goal
        assert protocol["finished_worlds_held_above_terrain"] and protocol["reference_commands"]
        assert protocol["first_variant_only"] and protocol["metrics_after_transition_include_failure_step"]
        assert result["original_evaluation"]["term_window"] == 0
        assert sum(e["failed"] for e in ep) == result["original_evaluation"]["n_failed"]
        for e in ep:
            assert e["completed"] != e["failed"]
            assert 0 < e["steps"] <= e["reference_steps"]
            if protocol["domain_randomization"]:
                assert .3-1e-6 <= e["friction"] <= 1.2+1e-6
                assert max(map(abs,e["joint_offsets"])) <= .01+1e-6
                assert all(abs(x) <= limit+1e-6 for x,limit in zip(e["com_offset"],[.025,.05,.05]))
            else:
                assert abs(e["friction"]-(expected_friction or 1.)) < 1e-6
                assert max(map(abs,e["joint_offsets"])) == 0
                assert max(map(abs,e["com_offset"])) == 0
            for metric, values in e["tracking"].items():
                if "support" not in metric:
                    assert values["count"] == e["steps"]
            for body, values in e["contacts"].items():
                assert values["matched_onsets"] == values["placement_error_m"]["count"] <= values["onsets"]
                assert values["true_positive_frames"] <= min(values["touch_frames"], values["label_frames"])
            if e["failed"]:
                failures.append({"trial":name,"condition":case["condition"],"seed":case["seed"],
                                 **{k:e[k] for k in ["clip","category","duration_s","reference_progress",
                                                     "failure_height","failure_tilt","valid_physics",
                                                     "root_xy_error_at_end_m"]}})
        per_trial[name] = aggregate(ep)
        assert abs(per_trial[name]["mean_normalized_return"]-result["original_evaluation"]["norm_return"]) <= .000051
        # Check against the existing evaluator's independently accumulated counts.
        for foot in ["left_foot","right_foot"]:
            c = per_trial[name]["contacts"][foot]
            raw = result["original_evaluation"]["contacts"][foot]
            for count_key in ["touch_frames","label_frames"]:
                assert c[count_key] == raw[count_key]
            for rate in ["precision","recall"]:
                assert abs(c[rate]-raw[rate]) <= .000501
            assert abs(c["placement_error_m"]["mean"]-raw["placement_err"]) <= .000501
        groups[case["condition"]].extend(ep)
    assert sum(len(v) for v in groups.values()) == manifest["expected_episodes"]
    conditions = {}
    for condition, ep in groups.items():
        names = [c["name"] for c in manifest["trials"] if c["condition"] == condition]
        conditions[condition] = {**aggregate(ep), "label":LABELS[condition], "trials":names,
                                "per_seed_completed":[per_trial[n]["completed"] for n in names],
                                "per_category":{c:aggregate([r for r in ep if r["category"]==c]) for c in CATEGORIES},
                                "true_flat":aggregate([r for r in ep if r["clip"] in true_flat])}
    base = json.loads((directory/"nominal_seed0.json").read_text())
    oracle = {b:pooled_stats([r["reference_pose_contact_point_disagreement_m"] for r in base["contact_geometry_calibration"]
                             if r["body"] == b]) for b in ["left_foot","right_foot"]}
    failing_clips = Counter(r["clip"] for r in failures)
    result = dict(checkpoint=manifest["checkpoint"],checkpoint_sha256=manifest["checkpoint_sha256"],
                  trials=per_trial,conditions=conditions,overall=aggregate([r for g in groups.values() for r in g]),
                  oracle_contact_geometry=oracle,failures=failures,
                  failure_counts_by_clip=dict(failing_clips.most_common()),
                  unique_failed_clips=len(failing_clips),validation="passed",
                  scope="training-reference clips, first terrain variant, reference commands; no held-out or planner evaluation",
                  pooling="tracking RMSE weighted by frames, contact rates by counts; no pooled quantiles estimated")
    (directory/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    table = ["| 条件 | 完成 / 回合 | 单次完成数（各 200 条） | 根 XY RMSE cm | 左/右脚世界 RMSE cm | 平均归一化回报 |",
             "|---|---:|---|---:|---:|---:|"]
    for condition, v in conditions.items():
        t=v["tracking"]
        table.append(f"| {LABELS[condition]} | {v['completed']}/{v['episodes']}（{v['completion_rate']:.1%}） | "
                     f"{', '.join(map(str,v['per_seed_completed']))} | {t['root_xy_error_m']['rmse']*100:.2f} | "
                     f"{t['left_foot_link_world_error_m']['rmse']*100:.2f}/{t['right_foot_link_world_error_m']['rmse']*100:.2f} | "
                     f"{v['mean_normalized_return']:.4f} |")
    table += ["", "| 地形/运动类别 | 无噪声 | 三类噪声 | 随机化＋噪声 | μ=0.3 | 前瞻 0.5 s | 前瞻 1.5 s |", "|---|---:|---:|---:|---:|---:|---:|"]
    selected = ["nominal","all_noise","domain_all_noise","friction_low","goal_short","goal_long"]
    for c in CATEGORIES:
        cells=[conditions[x]["per_category"][c] for x in selected]
        table.append("| `"+c+"` | "+" | ".join(f"{s['completed']}/{s['episodes']}" for s in cells)+" |")
    table += ["", "| 条件 | 左/右脚 precision | 左/右脚 recall | 左/右脚落点平均误差 cm | 左/右脚 ≤10 cm |", "|---|---:|---:|---:|---:|"]
    for condition, v in conditions.items():
        feet=[v['contacts'][b] for b in ['left_foot','right_foot']]
        cell=["/".join(f"{r[key]:.1%}" for r in feet) for key in ['precision','recall']]
        cell += ["/".join(f"{r['placement_error_m']['mean']*100:.2f}" for r in feet),
                 "/".join(f"{r['placement_within_0_1']:.1%}" for r in feet)]
        table.append("| "+LABELS[condition]+" | "+" | ".join(cell)+" |")
    (directory/"tables.md").write_text("\n".join(table)+"\n")
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order=list(conditions)
    heat=np.array([[conditions[k]["per_category"][c]["completion_rate"]*100 for k in order] for c in CATEGORIES])
    fig, ax=plt.subplots(figsize=(15,10),layout="constrained")
    im=ax.imshow(heat,cmap="RdYlGn",vmin=50,vmax=100,aspect="auto")
    ax.set_xticks(range(len(order)),['Nominal\n2 repeats','Obs\nnoise','Command\nnoise','Action\nnoise','All noise\n3 seeds',
                                    'Domain + noise\n3 seeds','Friction\n0.3','Friction\n1.2','Goal\n0.5 s','Goal\n1.5 s'])
    ax.set_yticks(range(len(CATEGORIES)),CATEGORIES)
    for i,c in enumerate(CATEGORIES):
        for j,k in enumerate(order):
            v=conditions[k]['per_category'][c]
            ax.text(j,i,f"{v['completed']}/{v['episodes']}",ha="center",va="center",fontsize=9,
                    color="white" if heat[i,j] >= 95 else "#202020")
    ax.set_title("RoBan tracker checkpoint 011500 — reference-clip completion\n200 training clips per trial; 15 trials / 3000 episodes",pad=16)
    fig.colorbar(im,ax=ax,label="Completion (%)",shrink=.65)
    fig.savefig(directory/"completion_heatmap.png",dpi=160)
    plt.close(fig)
    print(json.dumps({"out":str(directory/"summary.json"),"completed":result['overall']['completed'],
                      "episodes":result['overall']['episodes'],"invalid_physics":result['overall']['invalid_physics'],
                      "unique_failed_clips":len(failing_clips)},ensure_ascii=False))


if __name__ == "__main__":
    main()
