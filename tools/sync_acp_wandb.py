"""Backfill and follow an existing ACP CSV run without restarting its trainer.

Uses existing W&B credentials. Uploads numeric metrics and run configuration;
does not upload checkpoints, training data, source code, or console output.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import time
import uuid


METRICS = {
    "iteration": "train/iteration",
    "sim_steps": "train/simulation_steps",
    "wall_h": "performance/training_wall_hours",
    "wm_loss": "world_model/loss",
    "wm_err8": "world_model/window_end_error",
    "pi_loss": "tracker/loss",
    "pi_gradnorm": "tracker/gradient_norm_before_clip",
    "v_loss": "critic/loss",
    "imag_rew": "imagination/reward_normalized",
    "imag_survival": "imagination/survival",
    "sim_rew": "simulation/reward_normalized",
    "ep_len": "simulation/episode_length_steps",
    "ep_ret_norm": "simulation/episode_return_normalized",
    "term_frac": "simulation/termination_fraction",
    "touch_precision": "contacts/precision",
    "touch_recall": "contacts/recall",
    "it_per_s": "performance/iterations_per_second",
    "lr": "optimization/learning_rate",
    "phase_zero_frac": "curriculum/fraction_starting_at_zero",
    "eta_obs": "schedule/planner_target_probability",
    "eta_rew": "schedule/planned_reward_weight",
    "plan_frac": "planner/observed_target_fraction",
    "onset_err_plan": "contacts/planner_target_placement_error_m",
    "onset_err_ref": "contacts/reference_target_placement_error_m",
    "ended": "collection/ended_episodes",
    "admitted": "collection/admitted_episodes",
    "labeled": "collection/training_samples_added",
    "planner_buffer": "planner/buffer_samples",
    "planner_aff": "planner/affordance_loss",
    "planner_occ": "planner/occupancy_loss",
    "fresh_aff": "planner/fresh_affordance_loss",
    "fresh_occ": "planner/fresh_occupancy_loss",
    "sched_it": "schedule/iteration",
    "admit_rate_window": "collection/admission_rate_window",
    "held": "schedule/held",
    "v_m": "critic/reference_loss",
    "v_fs": "critic/goal_loss",
    "imag_r_ref": "imagination/reference_reward_normalized",
    "imag_r_goal": "imagination/goal_reward_normalized",
    "surv_ref": "imagination/reference_survival",
    "surv_goal": "imagination/goal_survival",
    "scale_ref": "imagination/reference_return_scale",
    "scale_goal": "imagination/goal_return_scale",
    "sim_r_ref": "simulation/reference_reward_normalized",
    "sim_r_goal": "simulation/goal_reward_normalized",
    "ep_len_ref": "simulation/reference_episode_length_steps",
    "ep_len_goal": "simulation/goal_episode_length_steps",
    "term_ref": "simulation/reference_termination_fraction",
    "fell_goal": "simulation/goal_fall_fraction",
    "reached_goal": "goal/reached_fraction",
    "success_goal": "goal/success_fraction",
    "admitted_ref": "collection/admitted_reference_episodes",
    "admitted_goal": "collection/admitted_goal_episodes",
    "planner_ref_buffer": "planner/reference_buffer_samples",
    "planner_goal_buffer": "planner/goal_buffer_samples",
    "actor_updates_enabled": "train/actor_updates_enabled",
    "goal_actor_enabled": "train/goal_actor_updates_enabled",
}
ACTOR_METRICS = {"pi_loss", "pi_gradnorm", "v_loss", "imag_rew", "imag_survival"}


def read_rows(path: Path):
    """Ignore an in-progress final CSV line; the trainer flushes complete rows."""
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return []  # A fresh trainer creates the CSV only after its initial replay collection.
    complete = raw[:raw.rfind("\n") + 1]
    rows = list(csv.DictReader(io.StringIO(complete)))
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"Malformed completed CSV row: {path}")
    return rows


def metrics(row, wm_only_iters):
    iteration = int(row["iteration"])
    active = bool(int(row["actor_updates_enabled"])) if "actor_updates_enabled" in row else iteration > wm_only_iters
    out = {"train/actor_updates_enabled": int(active)}
    bad = 0
    for key, value in row.items():
        if not active and key in ACTOR_METRICS:
            continue  # CSV zeros during pretraining are not actual actor losses.
        number = float(value)
        if not math.isfinite(number):
            bad += 1
            continue
        name = METRICS.get(key)
        if name is None:
            name = ("curriculum/start_phase/" + key[6:] if key.startswith("phase_") else
                    "curriculum/terrain_level/" + key[6:] if key.startswith("level_") else "extra/" + key)
        out[name] = int(number) if key in ("iteration", "sim_steps") else number
    out["diagnostics/nonfinite_metrics"] = bad
    return out


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def process_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        stat = Path(f"/proc/{pid}/stat")
        return not stat.exists() or stat.read_text().split(")", 1)[1].split()[0] != "Z"
    except ProcessLookupError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", default="acp-roban")
    ap.add_argument("--poll-seconds", type=float, default=5)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    directory = args.run_dir.resolve()
    state_path = directory / "wandb_sync_state.json"
    # Prevent two followers from writing the same W&B run concurrently.
    lock = (directory / ".wandb_sync.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    train_args = json.loads((directory / "args.json").read_text())
    release = json.loads((directory / "data_release.json").read_text())
    launch = json.loads((directory / "launch_config.json").read_text())
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "id": uuid.uuid4().hex[:8], "entity": args.entity, "project": args.project,
    }
    if (state["entity"], state["project"]) != (args.entity, args.project):
        raise ValueError("Existing sync destination differs; do not silently move a run")
    state.update(pid=os.getpid(), status="connecting", updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    write_json(state_path, state)
    import wandb
    config = {k: v for k, v in train_args.items() if k not in {"log_dir", "resume", "init"}}
    config.update(robot="roban", dataset=release, dataset_manifest_sha256=launch["dataset_manifest_sha256"],
                  logging_source="existing train_log.csv; original iteration axis",
                  torch_dynamo_disabled=launch["environment"].get("TORCHDYNAMO_DISABLE") == "1")
    stage = launch.get("stage") or ("stage3" if "goal_frac" in train_args else "stage2" if "planner_updates" in train_args else "stage1")
    config["stage"] = stage
    if launch.get("bootstrap"):
        config["bootstrap"] = launch["bootstrap"]
    if "resume_iteration" in launch:
        config["resume_iteration"] = launch["resume_iteration"]
    if launch.get("init_checkpoint_sha256"):
        config["init_checkpoint_sha256"] = launch["init_checkpoint_sha256"]
    run = wandb.init(entity=args.entity, project=args.project, id=state["id"], resume="allow",
                     name=directory.name, dir=str(directory), config=config,
                     job_type="tracker-planner-" + stage if stage != "stage1" else "tracker-stage1",
                     tags=["roban", "dataset-" + release.get("release", "unknown"),
                           "joint-tracker-planner" if stage != "stage1" else "reference-tracker", stage],
                     save_code=False, settings=wandb.Settings(disable_git=True, disable_code=True,
                         console="off", x_disable_stats=True, init_timeout=90))
    # On resumption W&B can retain the old config; local args describe the active trainer.
    run.config.update(config, allow_val_change=True)
    run.define_metric("train/iteration")
    run.define_metric("*", step_metric="train/iteration", step_sync=False)
    # Resume from server history, not the local enqueue cursor: queued data may
    # not have reached the server before a crash. CSV remains the source of truth.
    last = int(run.step) - 1
    wm_only = 0 if train_args.get("init") else train_args.get("wm_only_iters", 500)
    state.update(url=run.url, status="following", last_enqueued_iteration=last)
    write_json(state_path, state)
    print(json.dumps({"url": run.url, "resume_after_iteration": last}), flush=True)
    code = 0
    try:
        while True:
            count = 0
            for row in read_rows(directory / "train_log.csv"):
                iteration = int(row["iteration"])
                if iteration <= last:
                    continue
                run.log(metrics(row, wm_only), step=iteration)
                last = iteration
                count += 1
            if count:
                state.update(last_enqueued_iteration=last, updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
                write_json(state_path, state)
                print(f"enqueued {count} rows through iteration {last}", flush=True)
            status_file = directory / "status.json"
            status = json.loads(status_file.read_text()) if status_file.exists() else {}
            phase = status.get("phase", "unknown")
            run.summary["training_status"] = phase
            run.summary["last_csv_iteration"] = last
            terminal = phase in {"complete", "paused_for_benchmark", "superseded", "stopped_by_user"} or phase.endswith("_failed")
            stopped = status and not process_alive(status.get("supervisor_pid")) and not process_alive(status.get("child_pid"))
            if args.once or terminal or stopped:
                if terminal and phase == "complete":
                    evaluation = directory / "eval_reference_final.json"
                    if evaluation.exists():
                        result = json.loads(evaluation.read_text())
                        run.summary["evaluation/reference"] = result
                code = int(phase.endswith("_failed") or bool(stopped and not terminal))
                state.update(status="finished" if code == 0 else "training_failed", training_status=phase)
                break
            time.sleep(args.poll_seconds)
    except BaseException:
        code = 1
        state.update(status="sync_interrupted")
        raise
    finally:
        run.finish(exit_code=code)
        state.update(updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        write_json(state_path, state)


if __name__ == "__main__":
    main()
