import os

os.environ["ACP_ROBOT"] = "roban"

import mujoco
import torch

from supermimic.acp.blink_contact import BlinkContactWorldModel, update_ema
from supermimic.acp.mdp import XI_DIM
from supermimic.acp.networks import WorldModel
from supermimic.acp.robot import NUM_J, contact_geom_ids, get_spec
from supermimic.rotations import so3_exp


def states():
    generator = torch.Generator().manual_seed(17)
    position = torch.randn(4, 3, generator=generator)
    rotation = so3_exp(torch.randn(4, 3, generator=generator) * 0.2)
    return (
        position,
        rotation,
        torch.randn(4, 3, generator=generator),
        torch.randn(4, 3, generator=generator),
        torch.randn(4, NUM_J, generator=generator),
        torch.randn(4, NUM_J, generator=generator),
    )


def test_structured_targets_integrate_back_to_next_state():
    model = BlinkContactWorldModel(3, hidden=(16, 16))
    start = states()
    end = tuple(value.clone() for value in start)
    end = (
        end[0] + 0.03,
        end[1] @ so3_exp(torch.ones(4, 3) * 0.02),
        end[2] + 0.2,
        end[3] - 0.1,
        end[4] + 0.01,
        end[5] + 0.3,
    )
    actual = model.integrate(start, model.targets(start, end))
    for result, expected in zip(actual, end, strict=True):
        torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-6)
    assert not torch.allclose(model.targets(start, end), WorldModel.targets(start, end))


def test_model_uses_real_scan_input_and_propagates_action_gradient():
    model = BlinkContactWorldModel(3, hidden=(16, 16))
    state = states()
    scan = torch.tensor([[1.0, 2.0, 3.0]]).expand(4, -1)
    action = torch.zeros(4, NUM_J, requires_grad=True)
    context = torch.zeros(4, XI_DIM)
    encoded = model.inputs(state, action, context, scan)
    torch.testing.assert_close(encoded[:, -3:], scan)
    with torch.no_grad():
        model.net[-1].weight.fill_(0.01)
    predicted = model.step(state, action, context, scan)
    sum(value.square().mean() for value in predicted).backward()
    assert action.grad is not None and torch.isfinite(action.grad).all()
    assert action.grad.abs().sum() > 0


def test_ema_updates_parameters_and_copies_buffers():
    live = torch.nn.BatchNorm1d(3)
    average = torch.nn.BatchNorm1d(3)
    with torch.no_grad():
        live.weight.fill_(3)
        live.running_mean.fill_(7)
    update_ema(average, live, 0.5)
    torch.testing.assert_close(average.weight, torch.full((3,), 2.0))
    torch.testing.assert_close(average.running_mean, live.running_mean)


def test_handball_mesh_and_forearm_have_disjoint_contact_ownership():
    model = get_spec().compile()
    groups = contact_geom_ids(model, "roban_knee_caps_v1")
    assert len(groups) == 9
    assert len({geom for group in groups for geom in group}) == sum(map(len, groups))
    for side in ("left", "right"):
        geom = model.geom(f"{side}_hand_collision")
        assert model.geom_type[geom.id] == mujoco.mjtGeom.mjGEOM_MESH
        short = "l" if side == "left" else "r"
        assert model.body(model.geom_bodyid[geom.id]).name == f"{short}_handball"
        target = model.body(f"{side}_hand_contact")
        assert model.body(model.body_parentid[target.id]).name == f"{short}_handball"
