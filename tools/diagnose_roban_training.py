"""Bounded, isolated resume with first-invalid-value diagnostics; never writes the source run."""
import argparse
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import traceback

os.environ.setdefault("ACP_ROBOT", "roban")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--source-run", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--iterations", type=int, default=300)
ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=False)

import torch
from supermimic.acp import buffer, heightfield, networks

torch.set_num_threads(4)
source = args.source_run.resolve()
ck = torch.load(source / "last.pt", map_location="cpu", weights_only=False)
cmd = json.loads((source / "launch_config.json").read_text())["train_command"][2:]
def option(name, value):
    if name in cmd:
        cmd[cmd.index(name) + 1] = str(value)
    else:
        cmd.extend([name, str(value)])
option("--resume", source / "last.pt")
option("--log_dir", args.out.resolve())
option("--iterations", ck["iteration"] + args.iterations)
option("--log_every", 10)
option("--ckpt_every", 1000000)
torch.save({"source": str(source), "iteration": ck["iteration"], "command": cmd}, args.out / "invocation.pt")
del ck
last_window = None
last_model = None
counts = {"wm_windows": 0, "wm_steps": 0, "gradient_checks": 0, "cells": 0}

def cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: cpu(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return type(x)(cpu(v) for v in x)
    return x

def fail(reason, payload):
    frames = inspect.stack()
    context = []
    for frame in frames:
        if frame.function in ("wm_update", "_wm_rollout", "policy_update", "<module>"):
            context.append({"file": frame.filename, "line": frame.lineno, "function": frame.function,
                            "iteration": frame.frame.f_locals.get("it"),
                            "rollout_step": frame.frame.f_locals.get("j")})
    report = {"reason": reason, "counts": counts, "context": context}
    (args.out / "failure.json").write_text(json.dumps(report, indent=2))
    torch.save(cpu({"payload": payload, "window": last_window,
                    "wm": last_model.state_dict() if last_model else None,
                    "heightfield_paths": [a["path"] for a in heightfield.ASSETS]}), args.out / "failure.pt")
    raise RuntimeError("FIRST INVALID VALUE: " + reason)

old_sample = buffer.ReplayBuffer.sample_windows
def sample(self, *a, **kw):
    global last_window
    last_window = old_sample(self, *a, **kw)
    counts["wm_windows"] += 1
    return last_window
buffer.ReplayBuffer.sample_windows = sample

old_step = networks.WorldModel.step
def step(self, S, a, xi_n, hs):
    global last_model
    last_model = self
    counts["wm_steps"] += 1
    result = old_step(self, S, a, xi_n, hs)
    if not bool(torch.stack([t.isfinite().all() for t in result]).all()):
        fail("world_model_nonfinite_output", dict(S=S, action=a, xi_n=xi_n, scan=hs, output=result))
    return result
networks.WorldModel.step = step

old_cells = heightfield._cells
def cells(xy, indices):
    counts["cells"] += 1
    if not bool(xy.isfinite().all()):
        fail("nonfinite_heightfield_xy", dict(xy=xy, indices=indices))
    if not bool(((indices >= 0) & (indices < len(heightfield.ASSETS))).all()):
        fail("invalid_heightfield_asset_id", dict(xy=xy, indices=indices))
    result = old_cells(xy, indices)
    info, values, u, v, ix, iy = result
    off, nx = info[:, :1].long(), info[:, 1:2].long()
    adr = off + iy * nx + ix
    if not bool(((adr >= 0) & (adr + nx + 1 < values.numel())).all()):
        fail("invalid_heightfield_grid_address", dict(xy=xy, indices=indices, info=info, u=u, v=v, ix=ix, iy=iy))
    return result
heightfield._cells = cells

old_clip = torch.nn.utils.clip_grad_norm_
def clip(parameters, *a, **kw):
    params = list(parameters)
    norm = old_clip(params, *a, **kw)
    counts["gradient_checks"] += 1
    caller = inspect.currentframe().f_back
    # The actor already skips nonfinite gradients. WM/critic previously did not.
    if not bool(norm.isfinite()) and caller.f_code.co_name == "wm_update":
        fail("world_model_nonfinite_gradient", {"gradnorm": norm, "loss": caller.f_locals.get("loss"),
             "S_hat": caller.f_locals.get("S_hat"), "parameters": params,
             "gradients": [p.grad for p in params]})
    return norm
torch.nn.utils.clip_grad_norm_ = clip

sys.argv = [f"train_stage{args.stage}", *cmd]
try:
    runpy.run_module(f"supermimic.acp.train_stage{args.stage}", run_name="__main__")
except Exception:
    (args.out / "traceback.txt").write_text(traceback.format_exc())
    raise
else:
    (args.out / "diagnostic_pass.json").write_text(json.dumps(counts, indent=2))
