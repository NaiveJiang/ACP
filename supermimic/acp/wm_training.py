"""Episode-safe world-model regression and guarded optimizer updates."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .mdp import normalize_xi, unflatten_state


def rollout(wm, mdp, motion, s, a, xi, f, v, mask, *, delay_steps=0, boxes=None):
    """Roll out only the valid prefix of each replay window, including its done step.

    The following episode's action/terrain must never be applied to the previous
    episode's predicted state. Invalid rows are not forwarded through the model.
    """
    state = unflatten_state(s[:, 0])
    predictions = []
    for j in range(a.shape[1]):
        ids = mask[:, j].bool().nonzero(as_tuple=False).squeeze(-1)
        if ids.numel():
            active = tuple(x[ids] for x in state)
            terrain = motion.boxes(f[ids, j], v[ids, j]) if boxes is None else boxes[ids, j]
            context = normalize_xi(xi[ids, j], delay_steps)
            next_state = wm.step(active, a[ids, j], context, mdp.scan(active, terrain))
            state = tuple(x.index_copy(0, ids, y) for x, y in zip(state, next_state))
        predictions.append(state)
    return tuple(torch.stack([p[i] for p in predictions], 1) for i in range(6))


def regression_loss(wm, predictions, target, mask, error_floor):
    """Select valid transitions before arithmetic: NaN * 0 is still NaN."""
    valid = mask.bool()
    std = torch.maximum(getattr(wm, "inc_norm", wm.out_norm).std(), error_floor)
    predicted = tuple(x[valid] for x in predictions)
    error = (wm.state_error(predicted, unflatten_state(target[valid])) / std).square().mean(-1)
    per_step = error.new_zeros(mask.shape).masked_scatter(valid, error)
    return error.sum() / target.shape[0], per_step[:, -1]


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(v) for v in value)
    return value


class UpdateGuard:
    """Never feed nonfinite WM gradients to Adam; retain evidence of rejected updates.

    Reject isolated bad batches. Repeated failures stop with a usable CUDA context
    and a reproducible batch instead of silently continuing a broken training run.
    """
    def __init__(self, log_dir, max_consecutive=3):
        self.directory = Path(log_dir) / "numerics"
        self.skipped = 0
        self.consecutive = 0
        self.max_consecutive = max_consecutive

    def step(self, loss, model, optimizer, window, iteration):
        optimizer.zero_grad(set_to_none=True)
        reason = None
        gradnorm = loss.new_tensor(float("nan"))
        if not bool(loss.isfinite()):
            reason = "nonfinite_loss"
        else:
            loss.backward()
            gradnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not bool(gradnorm.isfinite()):
                reason = "nonfinite_gradient"
        if reason is None:
            optimizer.step()
            self.consecutive = 0
            return True
        self.skipped += 1
        self.consecutive += 1
        self.directory.mkdir(parents=True, exist_ok=True)
        diagnostic = self.directory / "first_bad_wm_update.pt"
        if self.skipped == 1 and not diagnostic.exists():
            from . import heightfield, networks
            torch.save(_cpu({"iteration": iteration, "reason": reason, "loss": loss,
                            "gradnorm": gradnorm, "window": window, "wm": model.state_dict(),
                            "optimizer": optimizer.state_dict(), "amp_dtype": str(networks.AMP_DTYPE),
                            "heightfield_paths": [a["path"] for a in heightfield.ASSETS]}),
                       diagnostic)
        event = dict(iteration=iteration, reason=reason, skipped=self.skipped, consecutive=self.consecutive)
        with (self.directory / "wm_skips.jsonl").open("a") as file:
            file.write(json.dumps(event) + "\n")
        print(f"WM update rejected: {event}; diagnostics: {self.directory}", flush=True)
        optimizer.zero_grad(set_to_none=True)
        if self.consecutive >= self.max_consecutive:
            raise FloatingPointError(f"{self.consecutive} consecutive invalid WM updates; see {self.directory}")
        return False
