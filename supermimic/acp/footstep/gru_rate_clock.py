"""GRU cadence correction for the supervised forward walking clock."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .gait_clock import ForwardClock
from .gru_clock import DT, WINDOW
from .mlp_clock import network

FORMAT = "forward_gru_rate_clock_v2"
MAX_CHANGE = .15


class RateGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(49))
        self.register_buffer("scale", torch.ones(49))
        self.gru = nn.GRU(49, 16, batch_first=True)
        self.output = nn.Linear(16, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, history):
        _, hidden = self.gru(((history-self.mean)/self.scale).clamp(-8, 8))
        return MAX_CHANGE * self.output(hidden[0]).squeeze(-1).tanh()


def dataset(parts):
    histories, periods, speeds = [], [], []
    for part in parts:
        if part["category"] != "forward":
            continue
        sl = slice(part["start"], part["end"])
        cache, source = Path(part["cache"]), Path(part["source"])
        vx = np.load(cache/"velocity.npy", mmap_mode="r")[sl, 0]
        obs = np.load(cache/"state.npy", mmap_mode="r")[sl]
        threshold = json.loads((source/"manifest.json").read_text())["force_threshold_N"]
        with np.load(source/(part["case"]+"_labels.npz")) as data:
            left = data["foot_state"][sl, 0]
        support = (left[:,13]>0) & (left[:,9]>=threshold)
        support[1:-1] = np.where(support[:-2]==support[2:], support[:-2], support[1:-1])
        touchdown = np.flatnonzero(support[1:] & ~support[:-1])+1
        state = np.column_stack((obs, vx))
        for start, end in zip(touchdown[:-1], touchdown[1:]):
            duration = (end-start)*DT
            if not (.4 <= duration <= 2 and vx[start]>0 and np.ptp(vx[start:end+1])<1e-6):
                continue
            # Equal weight per cycle. Each input ends at the sampled instant.
            for frame in np.linspace(start, end-1, 8, dtype=int):
                indices = np.maximum(frame + np.arange(1-WINDOW, 1), 0)
                histories.append(state[indices])
                periods.append(duration)
                speeds.append(vx[start])
    if not periods:
        raise ValueError("No complete constant-speed forward cycles")
    return tuple(torch.as_tensor(np.asarray(a),dtype=torch.float32) for a in (histories,periods,speeds))


def baseline(checkpoint):
    if checkpoint.get("format") != "forward_mlp_clock_v1":
        raise ValueError("Expected the trained MLP clock checkpoint")
    model = network().eval()
    model.load_state_dict(checkpoint["model"])
    return model.requires_grad_(False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--base-clock",required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--steps",type=int,default=600)
    args = p.parse_args()
    if args.out.exists() or args.steps<1:
        p.error("Use positive steps and a new output file")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    load = lambda path: torch.load(path,weights_only=True,map_location="cpu")
    split = load(args.checkpoint)["split"]
    base_ck = load(args.base_clock)
    base = baseline(base_ck)
    history, period, speed = dataset(split["train"])
    vh, vp, vs = dataset(split["val"])
    def base_rate(v):
        return (F.softplus(base(v[:,None])[:,1:5])+.001).sum(-1).reciprocal()
    rate = base_rate(speed)
    model = RateGRU()
    model.mean.copy_(history.mean((0,1)))
    model.scale.copy_(history.std((0,1)).clamp_min(.03))
    opt = torch.optim.AdamW(model.parameters(),lr=.001)
    for _ in range(args.steps):
        idx = torch.randint(len(period),(64,))
        residual = model(history[idx])
        predicted = 1/(rate[idx]*(1+residual))
        loss = F.smooth_l1_loss(predicted,period[idx],beta=.05)+.03*residual.square().mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
    with torch.no_grad():
        vr = base_rate(vs)
        pred = 1/(vr*(1+model(vh)))
        print(f"Train/val cycles: {len(period)//8}/{len(vp)//8}; period MAE (ms), MLP -> GRU: "
              f"{1000*(1/vr-vp).abs().mean():.2f} -> {1000*(pred-vp).abs().mean():.2f}")
    args.out.parent.mkdir(exist_ok=True,parents=True)
    with args.out.open("xb") as f:
        torch.save(dict(format=FORMAT,model=model.state_dict(),base_clock=base_ck),f)


class RateClock:
    def __init__(self, checkpoint, device, speed):
        base_ck = checkpoint["base_clock"]
        if not base_ck["min_speed"]-1e-6 <= speed <= base_ck["max_speed"]+1e-6:
            raise ValueError("Speed outside MLP clock calibration")
        raw = baseline(base_ck)(torch.tensor([[speed]],dtype=torch.float32))[0]
        times = (F.softplus(raw[:5])+.001).tolist()
        self.clock = ForwardClock([(speed,times[0],1 if raw[5]>=0 else 2,times[1:])],speed)
        self.model = RateGRU().to(device).eval()
        self.model.load_state_dict(checkpoint["model"])
        self.reset()

    def reset(self):
        self.history = None
        self.elapsed = 0.
        self.rate = 1/self.clock.period
        self.phase = (self.clock.offset-self.clock.delay)*self.rate

    def query(self, obs, command):
        x = torch.cat((obs[:,:48],command[:,:1]),1)[:,None]
        self.history = (x.expand(-1,WINDOW,-1) if self.history is None else
                        torch.cat((self.history[:,1:],x),1))
        horizon = float(command[0,3])
        # Preserve baseline startup timing and first support. Limit rate changes so
        # even the look-ahead phase progresses monotonically for fixed horizon.
        if self.elapsed >= self.clock.delay:
            desired = (1+float(self.model(self.history)[0]))/self.clock.period
            limit = .25*self.rate*DT/max(horizon,DT)
            self.rate += np.clip(desired-self.rate,-limit,limit)
        future = (self.phase+horizon*self.rate)%1
        waiting = self.elapsed+horizon < self.clock.delay
        stage = np.searchsorted(self.clock.ends/self.clock.period,future,side="right")%4
        self.phase += self.rate*DT
        self.elapsed += DT
        return future, (3 if waiting else (3,1,3,2)[stage]), waiting


if __name__=="__main__":
    main()
