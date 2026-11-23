"""The WBC learning-rate and replay settings for the flat RoBan contact task."""
from dataclasses import dataclass
import math

import torch

from ...mdp import XI_DIM, flatten_state, unflatten_state
from ...motion import BIN_FRAMES
from ...robot import NUM_J


@dataclass
class WbcRates:
    base: float
    multiplier: float
    decay_start: int | None = None
    training_return: float = 0.

    def rates(self, iteration):
        if self.decay_start is None and (self.training_return >= .75 or iteration >= 1400):
            self.decay_start = iteration
        progress = min(iteration / 250_000, 1.)
        base = self.base * (.03 + .97 * .5 * (1 + math.cos(math.pi * progress)))
        if self.decay_start is None:
            return base, base * self.multiplier
        fraction = min((iteration - self.decay_start) / 300, 1.)
        return base, base * (self.multiplier + (1 - self.multiplier) * fraction)


class FlatWindowMirror:
    """Reflect only world-model inputs and targets in the flat ground frame."""

    def __init__(self, device):
        self.order = torch.tensor(
            [0, *range(7, 13), *range(1, 7), *range(17, 21), *range(13, 17)], device=device
        )
        self.sign = torch.tensor(
            [-1, *([-1, -1, -1, 1, 1, -1] * 2), *([1, -1, -1, 1] * 2)], device=device
        )
        self.polar = torch.tensor([1, -1, 1], device=device)

    def _joints(self, values):
        return values[..., self.order] * self.sign

    def _state(self, flat):
        p, R, v, w, q, qd = unflatten_state(flat)
        polar = self.polar
        return flatten_state((
            p * polar, R * polar[:, None] * polar[None, :], v * polar, -w * polar,
            self._joints(q), self._joints(qd),
        ))

    def __call__(self, batch, fraction):
        count = int(len(batch['s']) * fraction)
        if not count:
            return batch
        mirrored = dict(batch)
        for name in ('s', 's1', 'a', 'xi'):
            values = batch[name][:count]
            if name == 'xi':
                values = torch.cat((values[..., :1], self._joints(values[..., 1:1+NUM_J]),
                                    values[..., 1+NUM_J:XI_DIM] * self.polar), dim=-1)
            elif name == 'a':
                values = self._joints(values)
            else:
                values = self._state(values)
            mirrored[name] = torch.cat((values, batch[name][count:]))
        return mirrored


@torch.no_grad()
def probe_failures(env, policy, task):
    """Find failures with the deployment EMA policy in separate flat worlds."""
    motion = env.motion
    ids = torch.arange(env.N, device=env.device)
    order = torch.randperm(len(motion.clip_start), device=env.device)
    clips = order[ids % len(order)]
    lo, hi = motion.clip_start[clips], motion.clip_end[clips]
    sampled = lo + (torch.rand(env.N, device=env.device) * (hi - lo)).long()
    earlier = torch.maximum(sampled - 2 * BIN_FRAMES, lo)
    env.done_eval.zero_()
    env.failed_eval.zero_()
    env.reset(ids, frames=torch.where(ids % 2 == 0, lo, earlier))
    query = task.sample_queries(env.N)
    history = env.observe()[:, None].expand(-1, policy.hist, -1)
    failed = torch.full_like(env.f, -1)
    while not bool(env.done_eval.all()):
        active = ~env.done_eval
        command = task.command(env.state(), env.f, query)
        transition = env.step(policy(history, command))
        failed = torch.where(active & transition['term'] & transition['valid'], env.f, failed)
        # The finished worlds still share the same simulator; keep their lookups in range.
        env.f = torch.minimum(env.f, hi - 1)
        history = torch.cat((history[:, 1:], env.observe()[:, None]), dim=1)
    return failed[failed >= 0]
