from types import SimpleNamespace

import torch

from supermimic.acp.blink_reset_sampling import BlinkResetCfg, BlinkResetSampler


def make_sampler(cfg):
    motion = SimpleNamespace(
        names=["first", "second"],
        clip_start=torch.tensor([0, 151]),
        clip_end=torch.tensor([150, 251]),
        clip_len=torch.tensor([151, 101]),
        bin_start=torch.tensor([0, 50, 100, 150, 151, 201, 251]),
        bin_clip=torch.tensor([0, 0, 0, 0, 1, 1, 1]),
        frame_bin=torch.arange(7).repeat_interleave(
            torch.tensor([50, 50, 50, 1, 50, 50, 1])
        ),
        num_bins=7,
    )
    return BlinkResetSampler(
        motion, cfg, torch.zeros(7), torch.full((7,), 1 / 7), max_return=10.0
    )


def test_observed_failure_biases_the_preceding_bin_inside_its_clip():
    sampler = make_sampler(BlinkResetCfg(1.0, 0.0, 0.0))
    sampler.observe(torch.tensor([125, 200]), torch.tensor([True, False]))
    probabilities = sampler.probabilities(torch.tensor([0, 1]))
    torch.testing.assert_close(
        probabilities,
        torch.tensor([[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0.5, 0.5, 0]]),
    )
    frames, bins = sampler.sample(100)
    assert (bins[frames < 151] == 1).all()
    assert not torch.isin(frames, sampler.motion.clip_end).any()


def test_focus_and_checkpoint_keep_preceding_bins_in_the_same_clip():
    cfg = BlinkResetCfg(0.30, 1.0, 0.0)
    sampler = make_sampler(cfg)
    sampler.set_focus(torch.tensor([151, 125]), preceding_bins=2)
    sampler.observe(torch.tensor([125]), torch.tensor([True]))
    sampler.bin_returns[2] = 0.5
    restored = make_sampler(cfg)
    restored.load_state_dict(sampler.state_dict())
    torch.testing.assert_close(restored.failures, sampler.failures)
    torch.testing.assert_close(restored.bin_returns, sampler.bin_returns)
    torch.testing.assert_close(restored.focus, torch.tensor([1, 1, 1, 0, 1, 0, 0.0]))
    probabilities = restored.probabilities(torch.tensor([1]))
    torch.testing.assert_close(probabilities, torch.tensor([[0, 0, 0, 0, 1, 0, 0.0]]))


def test_start_fraction_one_uses_only_the_first_frame():
    sampler = make_sampler(BlinkResetCfg(0.30, 0.15, 1.0))
    frames, _ = sampler.sample(100)
    assert torch.isin(frames, sampler.motion.clip_start).all()
