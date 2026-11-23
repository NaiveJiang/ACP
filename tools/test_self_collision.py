from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from supermimic.acp.roban import get_spec
from supermimic.acp.self_collision import SelfCollisionPenalty, hull_vertices


def test_overlap_penalty_has_a_separating_gradient():
    model = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <asset>
          <mesh name="cube" vertex="
            -0.05 -0.05 -0.05  -0.05 -0.05 0.05
            -0.05 0.05 -0.05   -0.05 0.05 0.05
             0.05 -0.05 -0.05   0.05 -0.05 0.05
             0.05 0.05 -0.05    0.05 0.05 0.05"/>
        </asset>
        <worldbody>
          <body name="left"><freejoint/><geom type="mesh" mesh="cube"/></body>
          <body name="right"><freejoint/><geom type="mesh" mesh="cube"/></body>
        </worldbody>
      </mujoco>
    """)
    kin = SimpleNamespace(device="cpu", body_names=["left", "right"])
    penalty = SelfCollisionPenalty(model, kin)
    positions = torch.tensor(
        [[[0, 0, 0], [0.12, 0, 0]], [[0, 0, 0], [0.08, 0, 0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    rotations = torch.eye(3).expand(2, 2, 3, 3)
    depth = penalty.depth(positions, rotations)
    torch.testing.assert_close(depth[0], torch.zeros(()))
    assert depth[1] > penalty.tolerance
    depth.sum().backward()
    assert positions.grad[1, 1, 0] < 0
    assert positions.grad[1, 0, 0] > 0


def test_roban_primitive_soles_match_mesh_bottom_height():
    baseline = get_spec().compile()
    model = get_spec(self_collision=True, primitive_soles=True).compile()
    for side in "lr":
        name = f"leg_{side}6_link"
        foot = model.body(name)
        ids = np.flatnonzero((model.geom_bodyid == foot.id) & (model.geom_contype > 0))
        np.testing.assert_array_equal(
            model.geom_type[ids],
            [mujoco.mjtGeom.mjGEOM_CYLINDER] * 2 + [mujoco.mjtGeom.mjGEOM_BOX],
        )
        for geom in ids:
            vertices = hull_vertices(model, geom) + model.geom_pos[geom]
            np.testing.assert_allclose(vertices[:, 2].min(), -0.035, atol=1e-9)
        np.testing.assert_array_equal(foot.mass, baseline.body(name).mass)
        np.testing.assert_array_equal(foot.inertia, baseline.body(name).inertia)
