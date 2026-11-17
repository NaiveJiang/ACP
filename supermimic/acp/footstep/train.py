"""Offline footstep planner: python -m supermimic.acp.footstep.train --help."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .data import ContactDataset, JOINT_NAMES, STATE_FIELDS, prepare, split_clips
from .model import FootstepPlanner, planner_loss


def move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, horizons):
    model.eval()
    sums = {k: 0. for k in ("total", "contact", "position", "heading")}
    values = {k: [] for k in ("logits", "contact", "position_error", "heading_error", "mask", "horizon")}
    count = 0
    for batch in loader:
        batch = move(batch, device)
        pred = model(batch["state_history"], batch["velocity_command"], batch["query_dt"])
        n = len(batch["query_dt"])
        for k, v in planner_loss(pred, batch).items():
            sums[k] += float(v)*n
        count += n
        values["logits"].append(pred["contact_logits"].cpu())
        values["contact"].append(batch["target_contact"].cpu())
        values["mask"].append(batch["support_mask"].bool().cpu())
        values["position_error"].append(torch.linalg.vector_norm(pred["position"]-batch["target_position"], dim=-1).cpu())
        dot = (pred["heading"]*batch["target_heading"]).sum(-1).clamp(-1, 1)
        values["heading_error"].append(torch.rad2deg(torch.acos(dot)).cpu())
        values["horizon"].append(batch["horizon_index"].cpu())
    a = {k: torch.cat(v).numpy() for k, v in values.items()}

    def metrics(sel):
        actual, predicted = a["contact"][sel] > .5, a["logits"][sel] > 0
        tp = (actual & predicted).sum(0)
        fp = (~actual & predicted).sum(0)
        fn = (actual & ~predicted).sum(0)
        precision = tp/np.maximum(tp+fp, 1)
        recall = tp/np.maximum(tp+fn, 1)
        mask = a["mask"][sel]
        pos, angle = a["position_error"][sel][mask], a["heading_error"][sel][mask]
        return dict(samples=int(np.sum(sel)), contact_accuracy=float((actual == predicted).mean()),
                    precision=precision.tolist(), recall=recall.tolist(),
                    f1=(2*precision*recall/np.maximum(precision+recall, 1e-12)).tolist(),
                    support_position_mean_m=float(pos.mean()) if len(pos) else None,
                    support_position_p95_m=float(np.quantile(pos, .95)) if len(pos) else None,
                    support_yaw_mean_deg=float(angle.mean()) if len(angle) else None)

    result = dict(loss={k: v/count for k, v in sums.items()}, **metrics(np.ones(count, dtype=bool)))
    result["per_horizon"] = {str(h): metrics(a["horizon"] == i) for i, h in enumerate(horizons)
                             if (a["horizon"] == i).any()}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", nargs="+", required=True, help="Capture roots or their future_contact_v2 directories")
    p.add_argument("--log-dir", required=True)
    p.add_argument("--cache-dir", default="data/cache/footstep_v2")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hist", type=int, default=5)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 128])
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-fraction", type=float, default=.2)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--val-samples", type=int, default=8192)
    p.add_argument("--norm-samples", type=int, default=8192)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--eval-only", help="Evaluate a checkpoint; uses its saved split and normalization")
    args = p.parse_args()
    if any(v < 1 for v in (args.hist, args.batch_size, args.val_every, args.val_samples,
                           args.norm_samples, args.log_every, args.threads, *args.hidden)) or args.steps < 1:
        p.error("Dimensions, counts and intervals must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    out = Path(args.log_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    device = torch.device(args.device)
    clips = prepare(args.data, args.cache_dir)
    if args.eval_only:
        ck = torch.load(args.eval_only, map_location="cpu", weights_only=True)
        config = ck["config"]
        current = {c["fingerprint"] for c in clips}
        expected = {c["fingerprint"] for intervals in ck["split"].values() for c in intervals}
        if current != expected:
            raise ValueError("Evaluation data differ from checkpoint fingerprints")
        cache_paths = {c["fingerprint"]: c["cache"] for c in clips}
        split = {k: [dict(c, cache=cache_paths[c["fingerprint"]]) for c in v] for k, v in ck["split"].items()}
    else:
        config = dict(hist=args.hist, hidden=args.hidden)
        split = split_clips(clips, args.val_fraction, args.seed)
    train = ContactDataset(split["train"], config["hist"])
    val = ContactDataset(split["val"], config["hist"])
    horizons = clips[0]["horizons"]
    report = dict(config=config, joint_names=JOINT_NAMES, state_fields=STATE_FIELDS,
                  horizons_s=horizons, train=train.summary(), val=val.summary(), split=split)
    (out / "dataset.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("config", "train", "val")}), flush=True)
    model = FootstepPlanner(**config).to(device)
    # Fixed validation subset, sampled without replacement; independent of train sampler.
    validation_seed = ck["args"]["seed"] if args.eval_only else args.seed
    ids = np.random.default_rng(validation_seed+1).choice(len(val), min(len(val), args.val_samples), replace=False)
    vloader = DataLoader(Subset(val, ids.tolist()), batch_size=args.batch_size, shuffle=False)
    if args.eval_only:
        model.load_state_dict(ck["model"])
        metrics = evaluate(model, vloader, device, horizons)
        (out / "evaluation.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics), flush=True)
        return
    # Normalize using only training histories. No validation or future target state.
    ids = np.random.default_rng(args.seed+2).choice(len(train), min(len(train), args.norm_samples), replace=False)
    states, cmds = [], []
    for batch in DataLoader(Subset(train, ids.tolist()), batch_size=args.batch_size):
        states.append(batch["state_history"].reshape(-1, 48))
        cmds.append(torch.cat((batch["velocity_command"], batch["query_dt"]), dim=-1))
    model.set_normalization(torch.cat(states).to(device), torch.cat(cmds).to(device))
    del states, cmds
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sampler = WeightedRandomSampler(train.balanced_weights(), args.steps*args.batch_size, replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler)
    best = float("inf")
    initial = evaluate(model, vloader, device, horizons)
    (out / "initial_validation.json").write_text(json.dumps(initial, indent=2))
    print("initial validation " + json.dumps(initial["loss"]), flush=True)
    start = time.monotonic()

    def save(path, step, metrics):
        payload = dict(format="two_foot_future_contact_v1", model=model.state_dict(),
                       optimizer=optimizer.state_dict(), config=config, step=step, split=split,
                       joint_names=JOINT_NAMES, state_fields=STATE_FIELDS, horizons_s=horizons,
                       args=vars(args), validation=metrics,
                       torch_rng_state=torch.get_rng_state())
        temp = path.with_suffix(".tmp")
        torch.save(payload, temp)
        temp.replace(path)

    with (out / "train.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "wall_s", "train_total", "train_contact", "train_position", "train_heading", "val_total"])
        for step, batch in enumerate(loader, 1):
            model.train()
            batch = move(batch, device)
            losses = planner_loss(model(batch["state_history"], batch["velocity_command"], batch["query_dt"]), batch)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            metrics = None
            if step % args.val_every == 0 or step == args.steps:
                metrics = evaluate(model, vloader, device, horizons)
                (out / "validation_{:06d}.json".format(step)).write_text(json.dumps(metrics, indent=2))
                if metrics["loss"]["total"] < best:
                    best = metrics["loss"]["total"]
                    save(out / "best.pt", step, metrics)
                save(out / "last.pt", step, metrics)
            if step % args.log_every == 0 or metrics is not None or step == 1:
                row = [step, time.monotonic()-start, *[float(losses[k].detach()) for k in ("total", "contact", "position", "heading")],
                       metrics["loss"]["total"] if metrics is not None else ""]
                writer.writerow(row)
                file.flush()
                print("step {} train {:.4f} val {}".format(step, row[2], row[-1]), flush=True)
    print("Saved planner: " + str(out / "best.pt"), flush=True)


if __name__ == "__main__":
    main()
