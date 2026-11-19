import math

import torch

from supermimic.acp.blink_learning_rate import BlinkActorRate
from supermimic.acp.blink_wm_mirror import BlinkWindowMirror
from supermimic.acp.mdp import XI_DIM
from supermimic.acp.robot import NUM_J
from supermimic.acp.terrain import box_heights


def test_mirrors_half_the_windows_with_rotated_boxes_and_delay_targets():
    delay_steps = 10
    mirror = BlinkWindowMirror("cpu", delay_steps)
    state = torch.zeros(4, 2, 13 + 2 * NUM_J)
    state[..., 1] = 0.45
    state[..., 3] = math.cos(0.2)
    state[..., 6] = math.sin(0.2)
    state[..., 13:] = torch.arange(2 * NUM_J)
    action = torch.arange(float(NUM_J)).expand(4, 2, -1)
    xi = torch.arange(float(XI_DIM + 1 + delay_steps * NUM_J)).expand(4, 2, -1)
    window = {
        "s": state,
        "s1": state.clone(),
        "a": action,
        "xi": xi,
        "mask": torch.ones(4, 2),
    }
    box = torch.tensor([3.0, 0.6, 0.55, 0.94, 0.342, 0.8, 0.4, 0.275, 0.0])
    boxes = box.expand(4, 2, 1, -1)
    mirrored, reflected_boxes = mirror.windows(window, boxes, 0.5)
    torch.testing.assert_close(mirrored["s"][:2, :, 1], -state[:2, :, 1])
    torch.testing.assert_close(mirrored["s"][:2, :, 6], -state[:2, :, 6])
    torch.testing.assert_close(mirrored["a"][:2, :, 1], -action[:2, :, 7])
    torch.testing.assert_close(mirrored["xi"][:2, :, XI_DIM], xi[:2, :, XI_DIM])
    torch.testing.assert_close(
        mirrored["xi"][:2, :, XI_DIM + 2], -xi[:2, :, XI_DIM + 8]
    )
    for field, original in window.items():
        torch.testing.assert_close(mirrored[field][2:], original[2:])
    torch.testing.assert_close(reflected_boxes[2:], boxes[2:])
    points = torch.tensor([[[2.8, 0.55], [3.5, 0.75], [2.8, 1.2]]]).expand(2, -1, -1)
    torch.testing.assert_close(
        box_heights(points, boxes[:2, 0]),
        box_heights(points * torch.tensor([1, -1]), reflected_boxes[:2, 0]),
    )
    again, restored_boxes = mirror.windows(mirrored, reflected_boxes, 0.5)
    for field, original in window.items():
        torch.testing.assert_close(again[field], original)
    torch.testing.assert_close(restored_boxes, boxes)


def test_actor_rate_decays_on_return_or_deadline_and_resumes():
    rate = BlinkActorRate()
    assert rate.factor(0) == 4.0
    assert rate.factor(399) == 4.0
    rate.training_return = 0.75
    assert rate.factor(400) == 4.0
    restored = BlinkActorRate()
    restored.load_state_dict(rate.state_dict())
    assert restored.factor(550) == 2.5
    assert restored.factor(700) == 1.0
    fallback = BlinkActorRate()
    assert fallback.factor(1399) == 4.0
    assert fallback.factor(1400) == 4.0
    assert fallback.factor(1700) == 1.0
