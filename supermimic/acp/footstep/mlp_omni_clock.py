"""Train a constant-command omnidirectional gait clock from labeled starts and cycles."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .gait_clock import ForwardClock, calibrate

FORMAT = "omni_mlp_clock_v1"


def network():
    return nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 6))


def batch(rows):
    commands = torch.tensor([x[0] for x in rows], dtype=torch.float32)
    times = torch.tensor(np.array([[r[1], *r[3]] for r in rows]), dtype=torch.float32)
    first = torch.tensor([r[2] == 1 for r in rows], dtype=torch.float32)
    return commands,times,first


def load_clock(checkpoint, velocity):
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Expected an omnidirectional MLP clock checkpoint")
    v = torch.tensor(velocity,dtype=torch.float32)
    measured = checkpoint["commands"]
    signatures = torch.sign(torch.where(measured.abs()>1e-5,measured,0.))
    sign = torch.sign(torch.where(v.abs()>1e-5,v,0.))
    group = measured[(signatures==sign).all(dim=1)]
    if not len(group) or (v<group.min(0).values-1e-5).any() or (v>group.max(0).values+1e-5).any():
        raise ValueError(f"Velocity {velocity} is outside the measured command range for its direction")
    model = network().eval()
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        raw = model((v/checkpoint["scale"])[None])[0]
        times = (F.softplus(raw[:5])+.001).numpy()
    return ForwardClock([(1.,times[0],1 if raw[5]>=0 else 2,times[1:])],1.)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--steps",type=int,default=1600)
    args = p.parse_args()
    if args.steps<1 or args.out.exists():
        p.error("Use a fresh output path and positive steps")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    split = torch.load(args.checkpoint,weights_only=True,map_location="cpu")["split"]
    train,val = (calibrate(split[s],individual_cycles=True,omnidirectional=True) for s in ("train","val"))
    cmd,times,first = batch(train)
    scale = cmd.abs().amax(0).clamp_min(.1)
    commands,inv,count = cmd.unique(dim=0,return_inverse=True,return_counts=True)
    weight = count[inv].float().reciprocal()
    model = network()
    opt = torch.optim.AdamW(model.parameters(),lr=.003)
    x = cmd/scale
    for _ in range(args.steps):
        raw = model(x)
        duration = F.smooth_l1_loss((F.softplus(raw[:,:5])+.001).log(),
                                   times.clamp_min(.01).log(),reduction="none").mean(-1)
        support = F.binary_cross_entropy_with_logits(raw[:,5],first,reduction="none")
        loss = ((duration+.15*support)*weight).sum()/weight.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        v,t,side = batch(val)
        raw = model(v/scale)
        mae = ((F.softplus(raw[:,:5])+.001-t).abs().mean(0)*1000).tolist()
        print(f"Train/validation cycles: {len(train)}/{len(val)}; unique training commands: {len(commands)}")
        print(f"Delay and stage MAE (ms): {mae}; first support (cycle weighted): "
              f"{float(((raw[:,5]>=0)==side).float().mean()):.3f}")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("xb") as stream:
        torch.save(dict(format=FORMAT,model=model.state_dict(),scale=scale,commands=commands),stream)


if __name__=="__main__":
    main()
