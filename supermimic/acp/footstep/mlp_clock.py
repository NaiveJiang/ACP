"""Supervised MLP for a standing start followed by constant forward walking."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .gait_clock import ForwardClock, calibrate


def network():
    return nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 6))


def batch(rows):
    speed = torch.tensor([[r[0]] for r in rows], dtype=torch.float32)
    times = torch.tensor(np.array([[r[1], *r[3]] for r in rows]), dtype=torch.float32)
    first = torch.tensor([r[2] == 1 for r in rows], dtype=torch.float32)
    return speed, times, first


def load_clock(path, speed):
    ck = torch.load(path, weights_only=True, map_location="cpu")
    if ck["format"] != "forward_mlp_clock_v1" or not ck["min_speed"]-1e-6 <= speed <= ck["max_speed"]+1e-6:
        raise ValueError("Incompatible clock or speed outside training range")
    model = network().eval()
    model.load_state_dict(ck["model"])
    with torch.no_grad():
        raw = model(torch.tensor([[speed]], dtype=torch.float32))[0]
        times = F.softplus(raw[:5]).numpy() + .001
    return ForwardClock([(speed, times[0], 1 if raw[5] >= 0 else 2, times[1:])], speed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=1200)
    args = parser.parse_args()
    if args.steps <= 0 or args.out.exists():
        parser.error("Choose a new output file and a positive update count")
    torch.manual_seed(42)
    torch.set_num_threads(1)
    split = torch.load(args.checkpoint, weights_only=True, map_location="cpu")["split"]
    train = calibrate(split["train"], individual_cycles=True)
    val = calibrate(split["val"], individual_cycles=True)
    speed, times, first = batch(train)
    # Give each command speed equal influence despite different cycle counts.
    _, inv, counts = speed.flatten().unique(return_inverse=True, return_counts=True)
    weight = counts[inv].float().reciprocal()
    model = network()
    opt = torch.optim.AdamW(model.parameters(), lr=.003)
    for _ in range(args.steps):
        raw = model(speed)
        predicted = F.softplus(raw[:, :5]) + .001
        duration_loss = F.smooth_l1_loss(predicted.log(), times.clamp_min(.01).log(),
                                         reduction="none").mean(1)
        side_loss = F.binary_cross_entropy_with_logits(raw[:, 5], first, reduction="none")
        loss = ((duration_loss + .15*side_loss)*weight).sum()/weight.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x, target, side = batch(val)
        raw = model(x)
        error = (F.softplus(raw[:, :5])+.001-target).abs().mean(0)
        print(f"Train/validation cycles: {len(train)}/{len(val)}. Validation delay and stage MAE (s):",
              error.tolist(), "first support accuracy:", (side == (raw[:,5]>=0)).float().mean().item())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as stream:
        torch.save(dict(format="forward_mlp_clock_v1", model=model.state_dict(),
                        min_speed=float(speed.min()), max_speed=float(speed.max())), stream)
    print(args.out)


if __name__ == "__main__":
    main()
