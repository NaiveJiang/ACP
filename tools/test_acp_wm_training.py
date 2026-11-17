"""Regression tests for episode boundaries and optimizer contamination."""
import os
os.environ["ACP_ROBOT"] = "roban"
from pathlib import Path
import tempfile
import unittest

import torch
from supermimic.acp.mdp import XI_DIM, flatten_state, unflatten_state
from supermimic.acp.networks import WorldModel, RunningNorm, WM_OUT
from supermimic.acp.robot import NUM_J
from supermimic.acp.wm_training import rollout, regression_loss, UpdateGuard


class TinyWM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.1))
        self.out_norm = RunningNorm(WM_OUT)
        self.batch_sizes = []

    def step(self, state, action, xi, scan):
        self.batch_sizes.append(len(action))
        assert action.isfinite().all()
        p, R, v, w, q, qd = state
        return p + self.weight * action[:, :3], R, v, w, q + self.weight * action, qd

    state_error = staticmethod(WorldModel.state_error)


class Motion:
    def boxes(self, f, v):
        assert (f < 100).all(), "Queried the terrain of an expired window"
        return f


class MDP:
    def scan(self, state, boxes):
        assert all(x.isfinite().all() for x in state)
        return torch.zeros(len(boxes), 1)


class WMTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        B, T = 2, 4
        state = (torch.zeros(B, 3), torch.eye(3).expand(B, 3, 3), torch.zeros(B, 3),
                 torch.zeros(B, 3), torch.zeros(B, NUM_J), torch.zeros(B, NUM_J))
        s = flatten_state(state)[:, None].repeat(1, T, 1)
        self.window = dict(s=s, s1=s.clone(), a=torch.ones(B, T, NUM_J), xi=torch.zeros(B, T, XI_DIM),
                           f=torch.arange(T).expand(B, T).clone(), v=torch.zeros(B, T, dtype=torch.long),
                           mask=torch.ones(B, T))
        self.model = TinyWM()
        self.floor = torch.full((WM_OUT,), .01)

    def forward(self):
        w = self.window
        return rollout(self.model, MDP(), Motion(), *(w[k] for k in ("s", "a", "xi", "f", "v", "mask")))

    def test_expired_rows_never_enter_model_or_terrain(self):
        w = self.window
        w["mask"][0, 1:] = 0  # includes the first/terminal transition
        w["mask"][1, 3:] = 0
        expired = ~w["mask"].bool()
        for key in ("s", "s1", "a", "xi"):
            w[key][expired] = float("nan")
        w["f"][expired] = 999
        pred = self.forward()
        loss, last = regression_loss(self.model, pred, w["s1"], w["mask"], self.floor)
        loss.backward()
        self.assertEqual(self.model.batch_sizes, [2, 1, 1])
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(self.model.weight.grad.isfinite())
        self.assertTrue((last == 0).all())
        self.assertGreater(pred[0][0, 0, 0], 0.)

    def test_valid_prefix_loss_and_gradient_match_original(self):
        for truncate in (False, True):
            with self.subTest(truncate=truncate):
                w = self.window
                if truncate:
                    w["mask"][0, 2:] = 0
                new_loss, new_last = regression_loss(self.model, self.forward(), w["s1"], w["mask"], self.floor)
                new_grad = torch.autograd.grad(new_loss, self.model.weight)[0]
                state = unflatten_state(w["s"][:, 0]); preds = []
                for j in range(4):
                    state = self.model.step(state, w["a"][:, j], w["xi"][:, j], None)
                    preds.append(state)
                pred = tuple(torch.stack([p[i] for p in preds], 1) for i in range(6))
                err = (self.model.state_error(pred, unflatten_state(w["s1"])) /
                       torch.maximum(self.model.out_norm.std(), self.floor)).square().mean(-1)
                old_loss = (err * w["mask"]).sum() / 2
                old_grad = torch.autograd.grad(old_loss, self.model.weight)[0]
                torch.testing.assert_close(new_loss, old_loss)
                torch.testing.assert_close(new_grad, old_grad)
                torch.testing.assert_close(new_last, err[:, -1] * w["mask"][:, -1])

    def test_guard_skips_bad_loss_and_gradient_without_poisoning_adam(self):
        class BadBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()
            @staticmethod
            def backward(ctx, grad):
                return grad * float("nan")

        with tempfile.TemporaryDirectory() as tmp:
            model = self.model
            opt = torch.optim.Adam(model.parameters(), lr=.01)
            guard = UpdateGuard(tmp)
            self.assertTrue(guard.step(model.weight.square(), model, opt, self.window, 1))
            weight = model.weight.detach().clone()
            saved = {k: v.clone() for k, v in opt.state[model.weight].items()}
            self.assertFalse(guard.step(model.weight * float('nan'), model, opt, self.window, 2))
            self.assertFalse(guard.step(BadBackward.apply(model.weight), model, opt, self.window, 3))
            torch.testing.assert_close(model.weight, weight)
            for key, val in saved.items():
                torch.testing.assert_close(opt.state[model.weight][key], val)
            self.assertTrue((Path(tmp) / "numerics/first_bad_wm_update.pt").exists())
            self.assertTrue(guard.step(model.weight.square(), model, opt, self.window, 4))
            self.assertEqual(guard.consecutive, 0)
            self.assertEqual(guard.skipped, 2)

    def test_repeated_invalid_updates_stop_with_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard = UpdateGuard(tmp, max_consecutive=2)
            opt = torch.optim.Adam(self.model.parameters())
            guard.step(self.model.weight * float('nan'), self.model, opt, self.window, 1)
            with self.assertRaisesRegex(FloatingPointError, '2 consecutive invalid WM updates'):
                guard.step(self.model.weight * float('nan'), self.model, opt, self.window, 2)


if __name__ == "__main__":
    unittest.main()
