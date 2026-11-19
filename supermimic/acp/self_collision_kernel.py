# SPDX-License-Identifier: Apache-2.0
# Adapted from kuavo_amp_mjlab e639e8fa92a2, mjlab/rl/supermimic.
"""CUDA active-feature search; gradients are recomputed in PyTorch."""

# Warp's kernel DSL uses runtime array annotations and overloaded vector math.
# pyright: reportInvalidTypeForm=false, reportArgumentType=false, reportCallIssue=false, reportAssignmentType=false, reportGeneralTypeIssues=false

import warp as wp


@wp.func
def rotation(r: wp.array3d(dtype=float), batch: int, body: int):
    return wp.mat33(
        r[batch, body, 0],
        r[batch, body, 1],
        r[batch, body, 2],
        r[batch, body, 3],
        r[batch, body, 4],
        r[batch, body, 5],
        r[batch, body, 6],
        r[batch, body, 7],
        r[batch, body, 8],
    )


@wp.func
def position(p: wp.array3d(dtype=float), batch: int, body: int):
    return wp.vec3(p[batch, body, 0], p[batch, body, 1], p[batch, body, 2])


@wp.kernel(enable_backward=False)
def active_hull_features(
    positions: wp.array3d(dtype=float),
    rotations: wp.array3d(dtype=float),
    bodies: wp.array(dtype=int),
    pairs: wp.array2d(dtype=int),
    bounds: wp.array2d(dtype=float),
    ranges: wp.array2d(dtype=int),
    points: wp.array2d(dtype=float),
    planes: wp.array2d(dtype=float),
    depths: wp.array2d(dtype=float),
    point_ids: wp.array2d(dtype=int),
    plane_ids: wp.array2d(dtype=int),
):
    batch, pair = wp.tid()
    a, b = pairs[pair, 0], pairs[pair, 1]
    ra, rb = (
        rotation(rotations, batch, bodies[a]),
        rotation(rotations, batch, bodies[b]),
    )
    pa, pb = (
        position(positions, batch, bodies[a]),
        position(positions, batch, bodies[b]),
    )
    ca = pa + ra * wp.vec3(bounds[a, 0], bounds[a, 1], bounds[a, 2])
    cb = pb + rb * wp.vec3(bounds[b, 0], bounds[b, 1], bounds[b, 2])
    depths[batch, pair] = 0.0
    point_ids[batch, pair] = ranges[a, 0]
    plane_ids[batch, pair] = ranges[b, 2]
    # Bounding spheres contain every hull point and can only reject disjoint hulls.
    if wp.length(ca - cb) > bounds[a, 3] + bounds[b, 3] + 0.00001:
        return
    relative_rotation = wp.transpose(rb) * ra
    relative_position = wp.transpose(rb) * (pa - pb)
    best = float(0.0)  # noqa: UP018 - mutable Warp kernel variable
    best_point = ranges[a, 0]
    best_plane = ranges[b, 2]
    for i in range(ranges[a, 0], ranges[a, 1]):
        point = relative_position + relative_rotation * wp.vec3(
            points[i, 0], points[i, 1], points[i, 2]
        )
        signed = float(-1.0e30)  # noqa: UP018 - mutable Warp kernel variable
        plane_id = ranges[b, 2]
        for j in range(ranges[b, 2], ranges[b, 3]):
            distance = (
                wp.dot(point, wp.vec3(planes[j, 0], planes[j, 1], planes[j, 2]))
                + planes[j, 3]
            )
            if distance > signed:
                signed = distance
                plane_id = j
            # A point outside any halfspace cannot be inside this convex hull.
            if signed >= 0.0:
                break
        if -signed > best:
            best = -signed
            best_point = i
            best_plane = plane_id
    depths[batch, pair] = best
    point_ids[batch, pair] = best_point
    plane_ids[batch, pair] = best_plane
