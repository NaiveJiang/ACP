"""Standalone GRU predicting durations for the ordered forward gait clock."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .gait_clock import calibrate
from .gru_clock import DT, WINDOW

FORMAT = "forward_gru_duration_clock_v1"


class DurationGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(49))
        self.register_buffer("scale", torch.ones(49))
        self.encoder = nn.GRU(49, 16, batch_first=True)
        self.output = nn.Linear(16, 6)

    def forward(self, history):
        _, hidden = self.encoder(((history-self.mean)/self.scale).clamp(-8,8))
        return self.output(hidden[0])


def data(parts):
    rows = calibrate(parts, individual_cycles=True, with_events=True)
    start_inputs, cycle_inputs, targets, sides, speeds = [], [], [], [], []
    for speed, delay, side, durations, part, onset, strike in rows:
        obs = np.load(Path(part["cache"])/"state.npy", mmap_mode="r")[part["start"]:part["end"]]
        # At play reset only the initial pose and the newly issued command are known.
        initial = np.tile(np.r_[obs[onset],speed], (WINDOW,1))
        for steps in (15,25):
            idx = np.maximum(strike-steps+np.arange(1-WINDOW,1),0)
            history = np.column_stack((obs[idx], np.where(idx>=onset,speed,0)))
            start_inputs.append(initial)
            cycle_inputs.append(history)
            targets.append(np.r_[delay,durations])
            sides.append(side==1)
            speeds.append(speed)
    return tuple(torch.tensor(np.asarray(x),dtype=torch.float32)
                 for x in (start_inputs,cycle_inputs,targets,sides,speeds))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--steps",type=int,default=1200)
    args=p.parse_args()
    if args.steps<1 or args.out.exists():
        p.error("Use positive steps and a fresh output path")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    split=torch.load(args.checkpoint,weights_only=True,map_location="cpu")["split"]
    a,b,t,side,v=data(split["train"])
    va,vb,vt,vs,_=data(split["val"])
    net=DurationGRU()
    histories=torch.cat((a,b))
    net.mean.copy_(histories.mean((0,1)))
    net.scale.copy_(histories.std((0,1)).clamp_min(.05))
    _,inv,n=v.unique(return_inverse=True,return_counts=True)
    weight=n[inv].float().reciprocal()
    opt=torch.optim.AdamW(net.parameters(),lr=.003)
    for _ in range(args.steps):
        start,cycle=net(a),net(b)
        def time_error(raw, target):
            return F.smooth_l1_loss((F.softplus(raw)+.001).log(),target.clamp_min(.01).log(),reduction="none")
        err=time_error(start[:,:5],t)
        err[:,1:] = .5*(err[:,1:] + time_error(cycle[:,1:5],t[:,1:]))
        loss=err.mean(1)+.15*F.binary_cross_entropy_with_logits(start[:,5],side,reduction="none")
        opt.zero_grad()
        ((loss*weight).sum()/weight.sum()).backward()
        nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.step()
    with torch.no_grad():
        start,cycle=net(va),net(vb)
        times=F.softplus(torch.cat((start[:,:1],cycle[:,1:5]),1))+.001
        print(f"Train/validation cycles: {len(t)//2}/{len(vt)//2}; delay and stage MAE (s):",
              (times-vt).abs().mean(0).tolist(),"first support accuracy:",float(((start[:,5]>=0)==vs).float().mean()))
    args.out.parent.mkdir(exist_ok=True,parents=True)
    with args.out.open("xb") as f:
        torch.save(dict(format=FORMAT,model=net.state_dict(),min_speed=float(v.min()),max_speed=float(v.max())),f)


class DurationClock:
    def __init__(self,ck,device,speed,horizon):
        if horizon not in (.3,.5) or not ck["min_speed"]-1e-6 <= speed <= ck["max_speed"]+1e-6:
            raise ValueError("Unsupported GRU clock speed or query horizon")
        self.model=DurationGRU().to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.reset()

    def reset(self):
        self.history=None
        self.step=0

    def query(self,obs,command):
        x=torch.cat((obs[:,:48],command[:,:1]),1)[:,None]
        first=self.history is None
        self.history=x.expand(-1,WINDOW,-1) if first else torch.cat((self.history[:,1:],x),1)
        future=self.step*DT+float(command[0,3])
        self.step+=1
        if first or future>=self.origin+self.ends[-1]:
            raw=self.model(self.history)[0]
            times=(F.softplus(raw[:5])+.001).tolist()
            ends=np.cumsum(times[1:])
            if first:
                self.delay=times[0]
                self.origin=self.delay-ends[0 if raw[5]>=0 else 2]
            else:
                self.origin+=self.ends[-1]
            self.ends=ends
        phi=((future-self.origin)/self.ends[-1])%1
        stage=np.searchsorted(self.ends,phi*self.ends[-1],side="right")%4
        waiting=future<self.delay
        return phi,3 if waiting else (3,1,3,2)[stage],waiting


if __name__=="__main__":
    main()
