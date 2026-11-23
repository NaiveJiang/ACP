"""EMA rollouts in separate worlds for finding reset segments."""

import random
from contextlib import contextmanager

import numpy as np
import torch

from .env import TrackingSim
from .motion import BIN_FRAMES
from .reference_mimic import ReferenceTask


@contextmanager
def isolated_probe_rng(device):
    device = torch.device(device)
    devices = [device] if device.type == "cuda" else []
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class BlinkProbe:
    def __init__(self, count, mdp, reference_cfg, env_options):
        self.task = ReferenceTask(reference_cfg, mdp)
        options = dict(env_options)
        for key in ("ground_friction_range", "box_friction_range"):
            bounds = options[key]
            if bounds is not None:
                options[key] = (sum(bounds) / 2.0,) * 2
        self.env = TrackingSim(
            count, mdp.motion, mdp, str(mdp.default_q.device), eval_mode=True, **options
        )

    @torch.no_grad()
    def run(self, policy):
        env = self.env
        motion = env.motion
        ids = torch.arange(env.N, device=env.device)
        order = torch.randperm(len(motion.names), device=env.device)
        clips = order[ids % len(order)]
        lo, hi = motion.clip_start[clips], motion.clip_end[clips]
        sampled = lo + (torch.rand(env.N, device=env.device) * (hi - lo)).long()
        earlier = torch.maximum(sampled - 2 * BIN_FRAMES, lo)
        frames = torch.where(ids % 2 == 0, lo, earlier)
        env.done_eval.zero_()
        env.failed_eval.zero_()
        env.reset(ids, frames)
        history = env.observe()[:, None].expand(-1, policy.hist, -1)
        failures = []
        while not bool(env.done_eval.all()):
            command = self.task.command(env.state(), env.f, env.v)
            alive = ~env.done_eval
            transition = env.step(policy(history, command))
            failures.append(
                transition["f"][alive & transition["term"] & transition["valid"]]
            )
            env.f = torch.minimum(env.f, hi - 1)
            env.park_finished_terrains((alive & env.done_eval).nonzero().flatten())
            history = torch.cat((history[:, 1:], env.observe()[:, None]), dim=1)
        return torch.cat(failures)
