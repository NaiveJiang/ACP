"""Train contact position and heading conditioned on future gait phase."""
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

FORMAT = "phase_foot_pose_v1"


def network(width):
    return nn.Sequential(nn.Linear(width, 48), nn.Tanh(), nn.Linear(48, 10))


def predict(model, checkpoint, inputs):
    pose = model((inputs-checkpoint["mean"])/checkpoint["scale"]).reshape(-1,2,5)
    heading = F.normalize(pose[...,3:], dim=-1, eps=1e-6)
    fallback = heading.new_tensor([0.,1.])
    heading = torch.where(heading.norm(dim=-1,keepdim=True)>.5, heading, fallback)
    return pose[...,:3], heading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=800)
    args = parser.parse_args()
    if args.out.exists() or args.steps < 1:
        parser.error("Use a fresh output path and positive steps")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    data = [torch.load(path,weights_only=True,map_location="cpu") for path in args.samples]
    x, position, heading, support = (torch.cat([sample[key] for sample in data])
                                       for key in ("features","position","heading","support"))
    if not all(t.isfinite().all() for t in (x,position,heading,support)) or not support.any():
        raise ValueError("Need finite data with at least one support target")
    mean, scale = x.mean(0), x.std(0,unbiased=False).clamp_min(.05)
    scale[:-8] *= 10
    x_norm = (x-mean)/scale
    model = network(x.shape[1])
    # Initialize around observed foot poses, especially the negative local z.
    target = torch.cat((position,heading),dim=-1)
    with torch.no_grad():
        center = (target*support[...,None]).sum(0)/support.sum(0).clamp_min(1)[...,None]
        model[-1].bias.copy_(center.flatten())
    opt = torch.optim.AdamW(model.parameters(),lr=.002)
    length_scale = x.new_tensor([.10,.10,.04])
    for _ in range(args.steps):
        pred = model(x_norm).reshape(-1,2,5)
        pos_loss = F.smooth_l1_loss(pred[...,:3]/length_scale,position/length_scale,
                                    reduction="none").mean(-1)
        yaw_loss = (pred[...,3:]-heading).square().sum(-1)
        loss = ((pos_loss+yaw_loss)*support).sum()/support.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    weights = dict(format=FORMAT,weights=model.state_dict(),mean=mean,scale=scale)
    with torch.no_grad():
        xyz, yaw = predict(model,weights,x)
        mask = support.bool()
        error = (xyz-position).abs()[mask].mean(0)*100
        angle = (yaw*heading).sum(-1).clamp(-1,1).acos()[mask].rad2deg().mean()
        print(f"Train frames: {len(x)}. XYZ MAE (cm): {error.tolist()}. Yaw MAE (deg): {angle:.2f}")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("xb") as f:
        torch.save(weights,f)


if __name__=="__main__":
    main()
