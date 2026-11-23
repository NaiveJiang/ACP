"""Small supervised forward-foot-position experiment on recorded walking cycles."""
import argparse
import numpy as np
import torch
from torch import nn


def cycle_features(contact):
    left = contact[:, 0].cpu().numpy()
    strikes = np.flatnonzero(left[1:] & ~left[:-1]) + 1
    if len(strikes) < 2:
        raise ValueError("Need two left touchdowns for recorded phase")
    frames = np.arange(len(left))
    # At the window edges extrapolate the nearest measured cycle.
    interval = np.clip(np.searchsorted(strikes, frames, side="right") - 1, 0, len(strikes)-2)
    angle = 2*np.pi*(frames-strikes[interval]) / (strikes[interval+1]-strikes[interval])
    return torch.as_tensor(np.stack((np.sin(angle), np.cos(angle)), -1),
                           dtype=torch.float32, device=contact.device)


def features(history, command, support, phase):
    return torch.cat((history[..., :48].flatten(1), command, support, phase.reshape(1, 2)), -1)


def network(width):
    return nn.Sequential(nn.Linear(width, 32), nn.Tanh(), nn.Linear(32, 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("samples", nargs="+")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    data = [torch.load(path, weights_only=True) for path in args.samples]
    x, y, mask = [torch.cat([d[key] for d in data]) for key in ("features", "target", "support")]
    torch.set_num_threads(2)
    torch.manual_seed(42)
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(.05)
    # Keep phase and support prominent when fitting a few short rollouts.
    scale[:-8] *= 10
    model = network(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    for _ in range(500):
        pred = model((x-mean)/scale)
        loss = (((pred-y)/.1).square()*mask).sum()/mask.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with open(args.out, "xb") as f:
        torch.save(dict(weights=model.state_dict(), mean=mean, scale=scale), f)


if __name__ == "__main__":
    main()
