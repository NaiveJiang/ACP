"""Clip-conditioned BlinkMimic reset sampling, adapted from Kuavo's learner."""

from dataclasses import dataclass

import torch

from .motion import BIN_FRAMES


@dataclass(frozen=True)
class BlinkResetCfg:
    failure_share: float = 0.30
    probe_share: float = 0.15
    start_fraction: float = 0.30


class BlinkResetSampler:
    def __init__(self, motion, cfg, bin_returns, bin_average, max_return):
        self.motion = motion
        self.cfg = cfg
        self.bin_returns, self.bin_average = bin_returns, bin_average
        self.max_return = max_return
        self.failures = torch.zeros_like(bin_returns)
        self.focus = torch.zeros_like(bin_returns)
        self.device = bin_returns.device
        bins = torch.arange(motion.num_bins, device=self.device)
        self.previous = bins.clone()
        self.previous[1:] = torch.where(
            motion.bin_clip[1:] == motion.bin_clip[:-1], bins[:-1], bins[1:]
        )
        self.can_reset = motion.bin_start < motion.clip_end[motion.bin_clip]

    @torch.no_grad()
    def observe(self, frames, failed):
        self.failures.mul_(0.99).index_add_(
            0, self.motion.frame_bin[frames], failed.to(self.failures)
        )

    @torch.no_grad()
    def set_focus(self, failed_frames, preceding_bins):
        self.focus.mul_(0.7)
        bins = self.motion.frame_bin[failed_frames]
        active = torch.ones_like(bins, dtype=torch.bool)
        for _ in range(preceding_bins + 1):
            self.focus.index_add_(0, bins, active.to(self.focus))
            previous = self.previous[bins]
            active &= previous != bins
            bins = previous

    def probabilities(self, clips):
        """Normalize each source inside its selected clip before mixing."""
        motion = self.motion
        soft = torch.softmax(-25.0 * self.bin_returns / self.max_return, dim=0)
        self.bin_average.lerp_(soft, 0.001)
        base = 0.5 / motion.num_bins + 0.4 * soft + 0.1 * self.bin_average
        allowed = (motion.bin_clip == clips[:, None]) & self.can_reset
        base = base * allowed
        base = base / base.sum(-1, keepdim=True)
        failure = torch.zeros_like(self.failures).index_add_(
            0, self.previous, self.failures
        )
        probabilities = base
        for weights, share in (
            (failure, self.cfg.failure_share),
            (self.focus, self.cfg.probe_share),
        ):
            masked = weights * allowed
            total = masked.sum(-1, keepdim=True)
            source = torch.where(
                total > 0, masked / total.clamp_min(1e-9), probabilities
            )
            probabilities = probabilities.lerp(source, share)
        return probabilities

    @torch.no_grad()
    def sample(self, count):
        motion = self.motion
        clips = torch.randint(len(motion.clip_start), (count,), device=self.device)
        bins = torch.multinomial(self.probabilities(clips), 1).squeeze(-1)
        lo = motion.bin_start[bins]
        hi = torch.minimum(lo + BIN_FRAMES, motion.clip_end[clips])
        frames = lo + (torch.rand(count, device=self.device) * (hi - lo)).long()
        from_start = torch.rand(count, device=self.device) < self.cfg.start_fraction
        frames = torch.where(from_start, motion.clip_start[clips], frames)
        return frames, motion.frame_bin[frames]

    def state_dict(self):
        return {
            "clips": tuple(self.motion.names),
            "clip_len": self.motion.clip_len.tolist(),
            "failures": self.failures.clone(),
            "focus": self.focus.clone(),
            "bin_returns": self.bin_returns.clone(),
            "bin_average": self.bin_average.clone(),
        }

    def load_state_dict(self, state):
        if (
            tuple(state["clips"]) != tuple(self.motion.names)
            or state["clip_len"] != self.motion.clip_len.tolist()
        ):
            raise ValueError("Blink reset state belongs to a different motion library")
        for key in ("failures", "focus", "bin_returns", "bin_average"):
            getattr(self, key).copy_(state[key])
