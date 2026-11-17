"""Deterministic primitive sampling for contact labels and differentiable rewards."""
import itertools
import mujoco
import numpy as np


def _axis(lo, hi, spacing):
    return np.linspace(lo, hi, max(1, int(np.ceil((hi - lo) / spacing)) + 1))


def primitive_spheres(kind, size, spacing=.01):
    """Inscribed spheres for boxes/cylinders; physics keeps the original primitives.

    The sampled union is conservative (inside the primitive), including its caps.
    At a 10 mm spacing thin RoBan soles use 5 mm spheres. Edge distances are
    approximate; exact MuJoCo geometry is used to generate training labels.
    """
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        r = min(spacing / 2, float(min(size)))
        pts = np.array(list(itertools.product(*[_axis(-s + r, s - r, spacing) for s in size])))
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius, half = size[:2]
        r = min(spacing / 2, float(radius), float(half))
        inner = radius - r
        xy = np.array(list(itertools.product(_axis(-inner, inner, spacing), repeat=2)))
        xy = xy[np.linalg.norm(xy, axis=-1) <= inner + 1e-10]
        angle = np.linspace(0, 2 * np.pi, max(8, int(np.ceil(2 * np.pi * inner / spacing))), endpoint=False)
        xy = np.vstack((xy, np.c_[inner * np.cos(angle), inner * np.sin(angle)]))
        pts = np.array([(x, y, z) for x, y in xy for z in _axis(-half + r, half - r, spacing)])
    else:
        raise ValueError(f"Unsupported sphere approximation for primitive {kind}")
    return pts, np.full(len(pts), r)


def primitive_surface_points(kind, size, spacing=.01):
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        points = []
        for face in range(3):
            axes = [np.array([-s, s]) if i == face else _axis(-s, s, spacing) for i, s in enumerate(size)]
            points.extend(itertools.product(*axes))
        return np.unique(points, axis=0)
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        r, h = size[:2]
        angles = np.linspace(0, 2 * np.pi, max(32, int(np.ceil(2 * np.pi * r / spacing))), endpoint=False)
        xy = np.array(list(itertools.product(_axis(-r, r, spacing), repeat=2)))
        xy = xy[np.linalg.norm(xy, axis=-1) <= r]
        rim = np.c_[r * np.cos(angles), r * np.sin(angles)]
        return np.vstack(([(x, y, z) for x, y in np.vstack((xy, rim)) for z in (-h, h)],
                          [(x, y, z) for x, y in rim for z in _axis(-h, h, spacing)]))
    if kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE):
        u = np.vstack((np.random.default_rng(0).normal(size=(512, 3)), np.eye(3), -np.eye(3)))
        pts = size[0] * u / np.linalg.norm(u, axis=-1, keepdims=True)
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            pts[:, 2] += np.sign(pts[:, 2]) * size[1]
        return pts
    raise ValueError(f"No surface sampling for primitive {kind}")
