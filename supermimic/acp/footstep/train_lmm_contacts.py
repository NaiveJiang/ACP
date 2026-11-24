"""Stage 1: fit the LMM foothold head to measured Frax rollout contacts."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .lmm_contact import (ContactHead, EVENT_FRAME, FORMAT, LOOKAHEAD, SeededForecast, decode_events,
                          label_rows, recorded_history)
from .motion_matching import heading
from .rollout_controller import rollout_clips, split_rollouts
from ..motion import CONTROL_DT


def tensor(array, device):
    return torch.as_tensor(array, device=device)


def episode_rows(part, forecast, stride, batch_size, limit):
    with np.load(part["sample"], allow_pickle=False) as source:
        data = {key: source[key] for key in ("qpos", "qvel", "previous_action", "upstream_frame",
                                             "sample_valid", "foot_support", "sole_pos_w", "sole_quat_w",
                                             "event_frame", "event_foot")}
    starts = np.flatnonzero(data["sample_valid"])
    starts = starts[starts + LOOKAHEAD < len(data["sample_valid"])][::stride]
    case = part["case"].split("/")[0]
    begin, _ = forecast.cases[case]
    playable = forecast.rows[begin + np.floor(data["upstream_frame"][starts]).astype(int)] >= 0
    starts = starts[playable][:limit or None]
    event = np.zeros_like(data["foot_support"])
    event[data["event_frame"], data["event_foot"]] = True
    for offset in range(0, len(starts), batch_size):
        frames = starts[offset:offset + batch_size]
        index = frames[:, None] + np.arange(LOOKAHEAD + 1)
        preview = forecast(case, data["upstream_frame"][frames])
        device = preview.feature.device
        rows = label_rows(preview, tensor(data["qpos"][frames, :7], device),
                          tensor(data["foot_support"][index], device),
                          tensor(data["sole_pos_w"][index], device),
                          tensor(heading(data["sole_quat_w"][index]), device),
                          tensor(event[index], device), torch.ones(index.shape, dtype=torch.bool, device=device))
        rows["history"] = recorded_history(data, frames).to(device)
        yield {key: value.cpu() for key, value in rows.items()}


def stack_rows(parts):
    parts = list(parts)
    if not parts:
        raise ValueError("No complete foothold forecast windows remained in this split")
    return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}


def slice_rows(rows, index, device):
    return {key: values[index].to(device) for key, values in rows.items()}


class EventScore:
    def __init__(self):
        self.count = np.zeros(3, dtype=np.int64)
        self.error = np.zeros(3)

    def update(self, prediction, batch):
        valid, sides, times, positions, yaw = (x.cpu().numpy() for x in decode_events(*prediction, 150))
        target = batch["event"].cpu().numpy().copy()
        target[:, 0] = False
        from .event_planner import WINDOW
        target[:, WINDOW + 1:] = False
        foot_pos = batch["foot_position"].cpu().numpy()
        foot_yaw = batch["foot_heading"].cpu().numpy()
        for i in range(len(valid)):
            truth = list(zip(*np.nonzero(target[i])))
            self.count[:2] += [valid[i].sum(), len(truth)]
            for slot in np.flatnonzero(valid[i]):
                side = sides[i, slot]
                options = [(abs(t * CONTROL_DT - times[i, slot]), j, t)
                           for j, (t, foot) in enumerate(truth) if foot == side]
                if not options:
                    continue
                dt, j, t = min(options)
                if dt > .2:
                    continue
                truth.pop(j)
                self.count[2] += 1
                xy = np.linalg.norm(positions[i, slot, :2] - foot_pos[i, t, side, :2])
                sine, cosine = foot_yaw[i, t, side]
                error = yaw[i, slot] - np.arctan2(sine, cosine)
                self.error += [dt, xy, abs(np.arctan2(np.sin(error), np.cos(error)))]

    def report(self):
        predicted, target, matched = map(int, self.count)
        error = (self.error / matched).tolist() if matched else [None] * 3
        return dict(predicted=predicted, target=target, matched=matched,
                    precision=matched / max(predicted, 1), recall=matched / max(target, 1),
                    time_mae_s=error[0], xy_mae_m=error[1], yaw_mae_rad=error[2])


@torch.no_grad()
def evaluate(head, validation, batch_size, device):
    learned, geometric = EventScore(), EventScore()
    loss = total = 0
    for offset in range(0, len(validation["history"]), batch_size):
        batch = slice_rows(validation, slice(offset, offset + batch_size), device)
        learned.update(head.predict(batch), batch)
        ground = batch["feature"][..., 27:33].unflatten(-1, (2, 3))
        speed = batch["feature"][..., 33:39].unflatten(-1, (2, 3)) * 3
        estimate = (ground[..., 2] < .045) & (speed[..., :2].norm(dim=-1) < .30)
        geometric.update((estimate.float() - .5, batch["sole"], batch["yaw"]), batch)
        n = len(batch["history"])
        loss += float(head.loss(batch)) * n
        total += n
    return dict(loss=loss / total, learned=learned.report(), geometry=geometric.report())


def save(path, head, identity, split, **fields):
    pending = path.with_suffix(".tmp")
    torch.save(dict(format=FORMAT, event_frame=EVENT_FRAME, head=head.state_dict(), hidden=head.hidden,
                    lmm=identity, split=split, **fields), pending)
    pending.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("lmm", "codec", "upstream-roban", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--max-windows-per-episode", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if min(args.steps, args.batch, args.stride, args.eval_every) < 1 or args.max_windows_per_episode < 0:
        p.error("Training counts must be positive")
    if args.out.exists():
        p.error("Choose a new --out directory")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    forecast = SeededForecast(args.lmm, args.codec, args.upstream_roban, args.device)
    identity = forecast.identity
    print(f"Stepper trained with {forecast.window} frames; evaluating the full 1.5 s window", flush=True)
    parts = rollout_clips(args.data, 2)
    split = split_rollouts(parts, .25, args.seed)
    data = {name: stack_rows(rows for part in episodes for rows in episode_rows(
        part, forecast, args.stride, min(args.batch, 32), args.max_windows_per_episode))
            for name, episodes in split.items()}
    if not data["train"]["event"].any() or not data["val"]["event"].any():
        p.error("Both partitions must contain measured touchdown events")
    print(f"Prepared {len(data['train']['history'])} training and {len(data['val']['history'])} held-out windows", flush=True)
    head = ContactHead().to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4)
    args.out.mkdir(parents=True, exist_ok=False)
    best = (-1., -float("inf"))
    for step in range(1, args.steps + 1):
        ids = torch.randint(len(data["train"]["history"]), (args.batch,))
        loss = head.loss(slice_rows(data["train"], ids, args.device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(head, data["val"], args.batch, args.device)
            metrics.update(step=step, train_loss=float(loss.detach()))
            print(json.dumps(metrics), flush=True)
            save(args.out / "last.pt", head, identity, split, step=step, metrics=metrics)
            learned = metrics["learned"]
            f1 = 2 * learned["matched"] / max(1, learned["target"] + learned["predicted"])
            quality = (f1, -metrics["loss"])
            if quality > best:
                best = quality
                save(args.out / "best.pt", head, identity, split, step=step, metrics=metrics)
                (args.out / "validation.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
