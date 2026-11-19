"""Kuavo BlinkMimic's return-triggered actor learning rate multiplier."""

from dataclasses import dataclass


@dataclass
class BlinkActorRate:
    multiplier: float = 4.0
    decay_return: float = 0.75
    decay_by: int = 1400
    decay_length: int = 300
    decay_start: int | None = None
    training_return: float = 0.0

    def factor(self, iteration):
        if self.decay_start is None and (
            self.training_return >= self.decay_return or iteration >= self.decay_by
        ):
            self.decay_start = iteration
        if self.decay_start is None:
            return self.multiplier
        progress = min((iteration - self.decay_start) / self.decay_length, 1.0)
        return self.multiplier + (1.0 - self.multiplier) * progress

    def state_dict(self):
        return {
            "decay_start": self.decay_start,
            "training_return": self.training_return,
        }

    def load_state_dict(self, state):
        self.decay_start = state["decay_start"]
        self.training_return = state["training_return"]
