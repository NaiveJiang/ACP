"""Future support and left-touchdown phase from causal proprioception and vx history."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DT = .02
WINDOW = 32
FORMAT = "forward_gru_clock_v1"


class ClockGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(49))
        self.register_buffer("scale", torch.ones(49))
        self.gru = nn.GRU(49, 32, batch_first=True)
        self.head = nn.Sequential(nn.Linear(33, 32), nn.Tanh(), nn.Linear(32, 6))

    def forward(self, history, horizon):
        _, hidden = self.gru(((history-self.mean)/self.scale).clamp(-10, 10))
        return self.head(torch.cat((hidden[0], horizon), dim=-1))


def make_data(parts):
    states, windows, labels, angles, valid_phase, horizons = [], [], [], [], [], []
    offset = 0
    for part in parts:
        if part["category"] != "forward":
            continue
        sl = slice(part["start"], part["end"])
        cache = Path(part["cache"])
        source = Path(part["source"])
        vx = np.load(cache/"velocity.npy", mmap_mode="r")[sl, 0].copy()
        with np.load(source/(part["case"]+"_labels.npz")) as raw:
            feet = raw["foot_state"][sl]
        threshold = json.loads((source/"manifest.json").read_text())["force_threshold_N"]
        contact = (feet[:,:,13] > 0) & (feet[:,:,9] >= threshold)
        for i in range(1,len(contact)-1):
            same = contact[i-1] == contact[i+1]
            contact[i,same] = contact[i-1,same]
        # Stop/start recordings ramp to one known requested speed. Use the step
        # request at ramp onset, matching the constant command used by play.
        if part["case"] != "continuous" and vx.max() > 0:
            moving = np.flatnonzero(vx > 1e-6)
            plateau = np.flatnonzero(np.isclose(vx, vx.max()))
            vx[moving[0]:plateau[-1]+1] = vx.max()
        state = np.concatenate((np.load(cache/"state.npy",mmap_mode="r")[sl],vx[:,None]),1)
        code = contact[:,0].astype(int) + 2*contact[:,1].astype(int)
        strikes = np.flatnonzero(contact[1:,0] & ~contact[:-1,0])+1
        phase = np.zeros(len(vx))
        has_phase = np.zeros(len(vx),bool)
        for a,b in zip(strikes[:-1],strikes[1:]):
            if 15 <= b-a <= 100:
                phase[a:b] = 2*np.pi*np.arange(b-a)/(b-a)
                has_phase[a:b] = True
        # Class 0 holds stance at startup. Double support in a cycle is class 3.
        hold = np.zeros(len(vx),bool)
        if part["case"] != "continuous":
            singles = np.flatnonzero((code==1)|(code==2))
            if len(singles):
                hold[:singles[0]] = True
                # Extend the first measured cycle backwards to cover the first swing.
                after = strikes[strikes >= singles[0]]
                if len(after) >= 2 and 15 <= after[1]-after[0] <= 100:
                    start, end = singles[0], after[0]
                    phase[start:end] = 2*np.pi*(np.arange(start, end)-end)/(after[1]-end)
                    has_phase[start:end] = True
        history_offsets = np.arange(1-WINDOW,1)
        for steps in (15,25):
            origin = np.arange(max(0,len(vx)-steps))
            target = origin+steps
            changes = np.r_[0,np.cumsum(np.abs(np.diff(vx))>1e-6)]
            keep = ((vx[origin]>0) & (code[target]>0) & (changes[target]==changes[origin])
                    & (has_phase[target] | hold[target]))
            origin,target = origin[keep],target[keep]
            windows.append(np.maximum(0,origin[:,None]+history_offsets)+offset)
            labels.append(np.where(hold[target],0,code[target]))
            angles.append(np.stack((np.sin(phase[target]),np.cos(phase[target])),1))
            valid_phase.append(has_phase[target] & ~hold[target])
            horizons.append(np.full((len(target),1),steps*DT))
        offset += len(state)
        states.append(state)
    pack = lambda xs,dtype: torch.as_tensor(np.concatenate(xs),dtype=dtype)
    return (pack(states,torch.float32),pack(windows,torch.long),pack(labels,torch.long),
            pack(angles,torch.float32),pack(valid_phase,torch.bool),pack(horizons,torch.float32))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--steps",type=int,default=1200)
    args=p.parse_args()
    if args.steps<=0 or args.out.exists():
        p.error("Use positive steps and a new output path")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    split=torch.load(args.checkpoint,map_location="cpu",weights_only=True)["split"]
    train=make_data(split["train"])
    val=make_data(split["val"])
    state,window,label,phase,mask,dt=train
    model=ClockGRU()
    model.mean.copy_(state.mean(0))
    model.scale.copy_(state.std(0).clamp_min(.03))
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    buckets=[torch.where(label==k)[0] for k in range(4)]
    if any(not len(b) for b in buckets):
        raise ValueError("Missing standing or support phase in training data")
    for _ in range(args.steps):
        idx=torch.cat([b[torch.randint(len(b),(16,))] for b in buckets])
        pred=model(state[window[idx]],dt[idx])
        err=(pred[:,4:]-phase[idx]).square().sum(1)
        loss=F.cross_entropy(pred[:,:4],label[idx])+(err*mask[idx]).sum()/mask[idx].sum().clamp_min(1)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
    s,w,y,angle,valid,h=val
    with torch.no_grad():
        pred=torch.cat([model(s[w[i:i+256]],h[i:i+256]) for i in range(0,len(w),256)])
        delta=(torch.atan2(pred[:,4],pred[:,5])-torch.atan2(angle[:,0],angle[:,1]))[valid]
        print(f"Train/validation windows: {len(window)}/{len(w)}; support accuracy: {(pred[:,:4].argmax(1)==y).float().mean():.3f}; phase MAE: {torch.atan2(delta.sin(),delta.cos()).abs().rad2deg().mean():.1f} degrees")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("xb") as f:
        torch.save(dict(format=FORMAT,model=model.state_dict(),horizons=[.3,.5]),f)


class OnlineClock:
    def __init__(self, checkpoint, device, horizon):
        if checkpoint.get("format")!=FORMAT or horizon not in checkpoint["horizons"]:
            raise ValueError("Need a forward GRU clock and a trained query horizon")
        self.model=ClockGRU().to(device).eval()
        self.model.load_state_dict(checkpoint["model"])
        self.history=None

    def reset(self):
        self.history = None

    def query(self, obs, command):
        x=torch.cat((obs[:,:48],command[:,:1]),1)[:,None]
        self.history=(x.expand(-1,WINDOW,-1) if self.history is None
                      else torch.cat((self.history[:,1:],x),1))
        pred=self.model(self.history,command[:,3:4])[0]
        code=int(pred[:4].argmax())
        return float(torch.atan2(pred[4],pred[5])/(2*np.pi))%1, code or 3, code==0


if __name__=="__main__":
    main()
