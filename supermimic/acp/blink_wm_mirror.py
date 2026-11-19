"""Mirror RoBan replay windows and their box terrain together."""

import torch

from .mdp import XI_DIM, flatten_state, unflatten_state
from .robot import NUM_J


class BlinkWindowMirror:
    def __init__(self, device, delay_steps):
        self.order = torch.tensor(
            [0, *range(7, 13), *range(1, 7), *range(17, 21), *range(13, 17)],
            device=device,
        )
        self.sign = torch.tensor(
            [-1, *([-1, -1, -1, 1, 1, -1] * 2), *([1, -1, -1, 1] * 2)],
            device=device,
        )
        self.polar = torch.tensor([1, -1, 1], device=device)
        self.delay_steps = delay_steps

    def joints(self, values):
        return values[..., self.order] * self.sign

    def state(self, state):
        p, R, v, w, q, qd = unflatten_state(state)
        polar = self.polar
        return flatten_state(
            (
                p * polar,
                R * polar[:, None] * polar[None, :],
                v * polar,
                w * -polar,
                self.joints(q),
                self.joints(qd),
            )
        )

    def context(self, xi):
        result = xi.clone()
        result[..., 1 : 1 + NUM_J] = self.joints(xi[..., 1 : 1 + NUM_J])
        result[..., 1 + NUM_J : XI_DIM] *= self.polar
        if self.delay_steps:
            queue = xi[..., -self.delay_steps * NUM_J :].unflatten(
                -1, (self.delay_steps, NUM_J)
            )
            result[..., -self.delay_steps * NUM_J :] = self.joints(queue).flatten(-2)
        return result

    def windows(self, window, boxes, fraction):
        count = int(len(window["s"]) * fraction)
        # A heightfield token points to an unreflected raster. Keep those
        # windows intact, while plane and box windows can be reflected exactly.
        with_hf = (boxes[:count, ..., 8] > 0).any(-1) & window["mask"][:count].bool()
        rows = (~with_hf.any(-1)).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return window, boxes
        result = dict(window)
        for name, mirror in (
            ("s", self.state),
            ("s1", self.state),
            ("a", self.joints),
            ("xi", self.context),
        ):
            source = window[name]
            result[name] = source.index_copy(0, rows, mirror(source[rows]))
        signs = boxes.new_tensor([1, -1, 1, 1, -1, 1, 1, 1, 1])
        return result, boxes.index_copy(0, rows, boxes[rows] * signs)
