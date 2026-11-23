"""Distil classic RoBan motion matching into a neural candidate projector."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F

from .motion_matching import Database, INTENT_MODES, Robot, command_trajectory, intent_mode

FORMAT = "roban_motion_projector_v1"
FEATURE_DIM = 27


class Projector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, dim))

    def forward(self, value):
        return value + self.net(value)


def weights(dim, mode):
    scale = np.ones(dim, dtype=np.float32)
    scale[FEATURE_DIM:FEATURE_DIM+3] = INTENT_MODES[mode]
    return scale


class MotionProjector:
    def __init__(self, path, db, *, pool=None):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("format") != FORMAT or saved["database"] != db.projection_signature():
            raise ValueError("Projector requires the same motions, clip filters and frame rate used for training")
        keys = db.projection_keys()
        if pool is not None:
            keys = keys[pool]
        self.k = min(saved["shortlist"], len(keys))
        self.input_mean, self.input_std = saved["mean"], saved["std"]
        self.net = Projector(keys.shape[1]).eval()
        self.net.load_state_dict(saved["model"])
        self.scale = [weights(keys.shape[1], mode) for mode in range(len(INTENT_MODES))]
        self.trees = [cKDTree(keys * scale) for scale in self.scale]

    @torch.inference_mode()
    def propose(self, query, mode):
        query = (torch.from_numpy(query) - self.input_mean)/self.input_std
        projected = (self.input_mean + self.net(query)*self.input_std).numpy()
        _, index = self.trees[mode].query(projected*self.scale[mode], k=self.k)
        return np.atleast_1d(index)


def examples(db, pool, count, rng):
    ids = rng.choice(pool, size=count, replace=True)
    query = db.features[ids].copy()
    pose = db.pose_features[ids].copy()
    command = db.intent[ids].copy()
    command += rng.normal(size=command.shape)*[.13, .10, .18]
    # Deliberately include starts, stops and commands requested while moving in another direction.
    novel = rng.random(count)
    command[novel < .12] = 0.
    jump = (novel >= .12) & (novel < .38)
    command[jump] = rng.uniform([-1., -.65, -1.5], [1., .65, 1.5], size=(jump.sum(), 3))
    command = np.clip(command, [-1., -.65, -1.5], [1., .65, 1.5])
    displacement, facing = command_trajectory(command)
    query[:, 15:21] = displacement.reshape(-1, 6)
    query[:, 21:27] = facing.reshape(-1, 6)
    query[:, :15] += rng.normal(size=(count, 15))*np.repeat([.017, .14, .08], [6, 6, 3])
    pose += rng.normal(size=pose.shape)*.12
    return db.projection_inputs(query, command, pose), ids, intent_mode(command)


@torch.no_grad()
def teach(db, keys, item, device, candidate_frames=None):
    query, frames, modes = item
    target = np.empty(len(query), dtype=np.int64)
    index = torch.as_tensor(db.candidates if candidate_frames is None else candidate_frames, device=device)
    owner = torch.as_tensor(db.clip, device=device)
    for mode in range(len(INTENT_MODES)):
        scale = torch.from_numpy(weights(keys.shape[-1], mode)).to(device)
        bank = keys*scale
        norm = bank.square().sum(-1)
        selected = np.flatnonzero(modes == mode)
        for batch in np.array_split(selected, max(1, int(np.ceil(len(selected)/128)))):
            if not len(batch):
                continue
            current = torch.as_tensor(frames[batch], device=device)
            q = torch.from_numpy(query[batch]).to(device)*scale
            cost = norm - 2 * (q @ bank.T)
            blocked = ((owner[current, None] == owner[index][None, :])
                       & ((current[:, None]-index[None, :]).abs() <= db.ignore_neighbor))
            cost.masked_fill_(blocked, torch.inf)
            target[batch] = cost.argmin(-1).cpu().numpy()
    return target


@torch.no_grad()
def assess(db, model, norm, item, target, keys, shortlist):
    query, frames, modes = item
    mean, std = norm
    projected = (model((torch.from_numpy(query).to(mean)-mean)/std)*std+mean).cpu().numpy()
    hit, gap, agreement = [], [], []
    k = min(shortlist, len(keys))
    for mode in range(len(INTENT_MODES)):
        selection = modes == mode
        if not selection.any():
            continue
        scale = weights(keys.shape[1], mode)
        tree = cKDTree(keys * scale)
        _, props = tree.query(projected[selection]*scale, k=k, workers=2)
        props = np.asarray(props).reshape(-1, k)
        truth = target[selection]
        q = query[selection]
        cost = np.square((keys[props]-q[:, None])*scale).sum(-1)
        current = frames[selection]
        candidate = db.candidates[props]
        blocked = ((db.clip[candidate] == db.clip[current, None])
                   & (np.abs(candidate-current[:, None]) <= db.ignore_neighbor))
        cost[blocked] = np.inf
        order = cost.argmin(-1)
        nearest = props[np.arange(len(props)), order]
        best_cost = cost.min(-1)
        actual = np.square((keys[truth]-q)*scale).sum(-1)
        hit.extend((props == truth[:, None]).any(-1))
        agreement.extend(nearest == truth)
        gap.extend((best_cost-actual).tolist())
    gap = np.asarray(gap)
    return dict(teacher_in_shortlist=float(np.mean(hit)), exact_after_rerank=float(np.mean(agreement)),
                mean_extra_cost=float(gap[np.isfinite(gap)].mean()) if np.isfinite(gap).any() else None,
                no_eligible_candidate=float((~np.isfinite(gap)).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path.home()/"roban数据/lafan1")
    parser.add_argument("--clips", nargs="+", default=["walk*.csv", "run*.csv"])
    parser.add_argument("--fps", type=float, default=30.)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=48000)
    parser.add_argument("--validation", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=2800)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--shortlist", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if min(args.samples, args.validation, args.steps, args.batch, args.shortlist) < 1 or not 10 <= args.fps:
        parser.error("Use positive sample and training counts and a frame rate of at least 10 Hz")
    if args.out.exists():
        parser.error(f"{args.out} already exists")
    os.environ["ACP_ROBOT"] = "roban"
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    db = Database(Robot(), args.data, args.fps, args.clips)
    bank = db.projection_keys()
    keys = torch.as_tensor(bank, device=args.device)
    owners = np.unique(db.clip[db.candidates])
    if len(owners) < 2:
        parser.error("At least two clips with eligible windows are required")
    validation_clips = rng.permutation(owners)[:max(1, len(owners)//5)]
    held_out = np.isin(db.clip[db.candidates], validation_clips)
    train = examples(db, db.candidates[~held_out], args.samples, rng)
    val = examples(db, db.candidates[held_out], args.validation, rng)
    print(f"Labelling {args.samples} training queries and {args.validation} held-out queries "
          f"against {len(bank)} frames", flush=True)
    train_target = teach(db, keys, train, args.device)
    val_target = teach(db, keys, val, args.device)
    query = torch.from_numpy(train[0]).to(args.device)
    mean, std = query.mean(0), query.std(0, unbiased=False).clamp_min(.06)
    source = (query-mean)/std
    target = (keys[train_target]-mean)/std
    model = Projector(bank.shape[1]).to(args.device)
    nn.init.zeros_(model.net[-1].weight)
    nn.init.zeros_(model.net[-1].bias)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=7e-5)
    for step in range(1, args.steps+1):
        ids = torch.randint(len(source), (args.batch,), device=args.device)
        loss = F.smooth_l1_loss(model(source[ids]), target[ids])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        schedule.step()
        if step % 500 == 0 or step == args.steps:
            print(f"Step {step}/{args.steps}, projection loss {loss.item():.4f}", flush=True)
    metrics = assess(db, model, (mean, std), val, val_target, bank, args.shortlist)
    metrics["held_out_query_clips"] = [db.names[i] for i in validation_clips]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as file:
        torch.save(dict(format=FORMAT, database=db.projection_signature(), model=model.cpu().state_dict(),
                        mean=mean.cpu(), std=std.cpu(), shortlist=args.shortlist, validation=metrics), file)
    print(json.dumps(metrics), flush=True)
    print(args.out)


if __name__ == "__main__":
    main()
