"""Train the phase-conditioned foot pose planner on all valid labeled trajectories."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import ContactDataset
from .phase_pose import FORMAT, network, predict


def phase_labels(contact, velocity):
    left = contact[:, 0].copy()
    left[1:-1] = np.where(left[:-2] == left[2:], left[:-2], left[1:-1])
    strikes = np.flatnonzero(left[1:] & ~left[:-1]) + 1
    phase = np.zeros((len(left), 2), dtype=np.float32)
    for a,b in zip(strikes[:-1], strikes[1:]):
        if not 15 <= b-a <= 100:
            continue
        frames = np.arange(a,b)
        angle = 2*np.pi*(frames-a)/(b-a)
        phase[frames] = np.stack((np.sin(angle),np.cos(angle)),axis=1)
    phase[np.linalg.norm(velocity,axis=1)<1e-5] = 0
    return phase


class PoseData:
    def __init__(self, intervals, hist):
        contact_data = ContactDataset(intervals, hist)
        self.hist = hist
        columns = {key: [] for key in ("state","velocity","dt","contact","position","heading","phase")}
        indices = []
        offset = 0
        for part,pairs in contact_data.entries:
            folder, source = Path(part["cache"]), Path(part["source"])
            lo,hi = part["start"],part["end"]
            source_file = source/(part["case"]+"_labels.npz")
            threshold = json.loads((source/"manifest.json").read_text())["force_threshold_N"]
            with np.load(source_file) as raw:
                feet = raw["foot_state"][lo:hi]
            arrays = {k: np.load(folder/(k+".npy"), mmap_mode="r")[lo:hi].copy()
                      for k in columns if k != "phase"}
            current = (feet[:,:,13]>0) & (feet[:,:,9]>=threshold)
            arrays["phase"] = phase_labels(current,arrays["velocity"])
            for key,value in arrays.items():
                columns[key].append(value)
            frames,horizon = pairs.T
            keep = arrays["contact"][frames-lo,horizon].any(-1)
            future = np.load(folder/"future_index.npy",mmap_mode="r")
            indices.append(np.stack((frames[keep]-lo+offset,horizon[keep],
                                     future[frames[keep],horizon[keep]]-lo+offset),-1))
            offset += hi-lo
        self.arrays = {k:torch.from_numpy(np.concatenate(v)) for k,v in columns.items()}
        self.index = torch.from_numpy(np.concatenate(indices)).long()
        self.back = torch.arange(1-hist,1)
        self.intervals = len(contact_data.entries)

    def __len__(self):
        return len(self.index)

    def batch(self, indices, device):
        frame,horizon,target = self.index[indices].unbind(1)
        d = self.arrays
        support = d["contact"][frame,horizon]
        x = torch.cat((d["state"][frame[:,None]+self.back].flatten(1),d["velocity"][frame],
                       d["dt"][frame,horizon,None],support,d["phase"][target]),1)
        return tuple(t.to(device) for t in (x,d["position"][frame,horizon],d["heading"][frame,horizon],support))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=8)
    p.add_argument("--batch-size",type=int,default=2048)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--init", help="Warm-start from a phase pose checkpoint")
    p.add_argument("--lr",type=float,default=.002)
    p.add_argument("--rollout-ratio",type=float,default=.25,
                   help="Number of closed-loop targets per offline target in each batch")
    p.add_argument("--rollout-samples",nargs="+",default=[])
    args = p.parse_args()
    if args.out.exists() or min(args.epochs,args.batch_size)<1 or not (0 < args.lr and 0 < args.rollout_ratio <= 1):
        p.error("Choose a new output file and positive epochs and batch size")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    ck = torch.load(args.checkpoint,weights_only=True,map_location="cpu")
    hist = ck["config"]["hist"]
    train,val = [PoseData(ck["split"][split],hist) for split in ("train","val")]
    print(f"Train {train.intervals} intervals / {len(train)} targets, validation {val.intervals} intervals / {len(val)} targets",flush=True)
    rollout = []
    if args.rollout_samples:
        files = [torch.load(path,weights_only=True,map_location="cpu") for path in args.rollout_samples]
        rollout = [torch.cat([r[k] for r in files]) for k in ("features","position","heading","support")]
        if rollout[0].shape[1] != 48*hist+8 or not all(t.isfinite().all() for t in rollout):
            raise ValueError("Incompatible rollout samples")
    sample = torch.randperm(len(train))[:min(32768,len(train))]
    x,pos,heading,support = train.batch(sample,args.device)
    mean,scale = x.mean(0),x.std(0,unbiased=False).clamp_min(.05)
    scale[:-8] *= 4
    model = network(x.shape[-1]).to(args.device)
    with torch.no_grad():
        target = torch.cat((pos,heading),-1)
        model[-1].bias.copy_(((target*support[...,None]).sum(0)/support.sum(0).clamp_min(1)[:,None]).flatten())
    if args.init:
        init = torch.load(args.init,weights_only=True,map_location=args.device)
        if init.get("format") != FORMAT:
            raise ValueError("Expected a phase pose checkpoint for --init")
        model.load_state_dict(init["weights"])
        mean,scale = init["mean"],init["scale"]
        if mean.shape != x.shape[1:] or scale.shape != mean.shape or not (scale>0).all():
            raise ValueError("Incompatible phase pose normalization")
    opt = torch.optim.AdamW(model.parameters(),lr=args.lr)
    norms = x.new_tensor([.1,.1,.04])
    weights = dict(format=FORMAT,mean=mean,scale=scale)
    for epoch in range(args.epochs):
        model.train()
        for idx in torch.randperm(len(train)).split(args.batch_size):
            inputs,xyz,yaw,mask = train.batch(idx,args.device)
            if rollout:
                extra = torch.randint(len(rollout[0]),(max(1,round(len(idx)*args.rollout_ratio)),))
                inputs,xyz,yaw,mask = (torch.cat((a,b[extra].to(args.device))) for a,b in zip((inputs,xyz,yaw,mask),rollout))
            raw = model((inputs-mean)/scale).reshape(-1,2,5)
            position = F.smooth_l1_loss(raw[...,:3]/norms,xyz/norms,reduction="none").mean(-1)
            facing = (raw[...,3:]-yaw).square().sum(-1)
            loss = ((position+facing)*mask).sum()/mask.sum().clamp_min(1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            total = torch.zeros(5,device=args.device)
            for idx in torch.arange(len(val)).split(args.batch_size):
                inputs,xyz,yaw,mask = val.batch(idx,args.device)
                pxyz,pyaw = predict(model,weights,inputs)
                position = ((pxyz-xyz).abs()*mask[...,None]).sum((0,1))
                angle = ((pyaw*yaw).sum(-1).clamp(-1,1).acos().rad2deg()*mask).sum()
                total += torch.cat((position,angle.reshape(1),mask.sum().reshape(1)))
            print(f"Epoch {epoch+1}: validation XYZ MAE (cm) = {(100*total[:3]/total[4]).tolist()}, yaw MAE (deg) = {float(total[3]/total[4]):.2f}",flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    serialized = dict(format=FORMAT,weights={k:v.cpu() for k,v in model.state_dict().items()},
                      mean=mean.cpu(),scale=scale.cpu())
    with args.out.open("xb") as f:
        torch.save(serialized,f)


if __name__=="__main__":
    main()
