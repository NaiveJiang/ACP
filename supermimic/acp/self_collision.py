# SPDX-License-Identifier: Apache-2.0
# Adapted from kuavo_amp_mjlab e639e8fa92a2, mjlab/rl/supermimic.
"""Differentiable vertex-in-hull and inscribed-sphere self-overlap.

Inscribed proxies avoid penalizing empty space outside the physical convex hull.
Vertex/edge samples cover tips missed by spheres. Pure edge intersections can still
be missed: this is a training signal, not a replacement for the mesh audit.
"""

import mujoco
import numpy as np
import torch
import warp as wp
from scipy.spatial import ConvexHull

from .self_collision_kernel import active_hull_features


def hull_vertices(model, geom):
    kind = model.geom_type[geom]
    sizes = model.geom_size[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        return model.mesh_vert[start : start + count].astype(float)
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return np.array(np.meshgrid(*[[-s, s] for s in sizes])).reshape(3, -1).T
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius, half_height = sizes[:2]
        angle = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        rim = radius * np.column_stack((np.cos(angle), np.sin(angle)))
        return np.concatenate(
            [np.c_[rim, np.full(len(rim), z)] for z in (-half_height, half_height)]
        )
    raise ValueError(f"Unsupported self-collision geometry: {model.geom(geom).name}")


class SelfCollisionPenalty:
    tolerance = 0.001
    normalization = 0.03

    def __init__(self, model, kin):
        body_indices = {name: i for i, name in enumerate(kin.body_names)}
        centers, radii, bodies, geoms = [], [], [], []
        hulls = {}
        for g in range(model.ngeom):
            body = int(model.geom_bodyid[g])
            body_index = body_indices.get(model.body(body).name.removeprefix("robot/"))
            if body_index is None or not (
                model.geom_contype[g] or model.geom_conaffinity[g]
            ):
                continue
            vertices = hull_vertices(model, g)
            hull = ConvexHull(vertices)
            edges = np.unique(
                np.sort(
                    np.concatenate(
                        (
                            hull.simplices[:, [0, 1]],
                            hull.simplices[:, [1, 2]],
                            hull.simplices[:, [2, 0]],
                        )
                    ),
                    axis=1,
                ),
                axis=0,
            )
            samples = np.concatenate((vertices[hull.vertices], vertices[edges].mean(1)))
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[g])
            rotation = rotation.reshape(3, 3)
            normals = hull.equations[:, :3] @ rotation.T
            offsets = hull.equations[:, 3] - normals @ model.geom_pos[g]
            planes = np.unique(
                np.round(np.column_stack((normals, offsets)), 10), axis=0
            )
            hulls[g] = (
                body_index,
                samples @ rotation.T + model.geom_pos[g],
                planes,
            )
            center = vertices[hull.vertices].mean(0)
            axis = int(np.ptp(vertices, axis=0).argmax())
            for fraction in (-0.25, 0.0, 0.25):
                point = center.copy()
                point[axis] += fraction * np.ptp(vertices[:, axis])
                radius = float(
                    -(hull.equations[:, :3] @ point + hull.equations[:, 3]).max()
                )
                if radius <= 0.002:
                    continue
                centers.append(rotation @ point + model.geom_pos[g])
                radii.append(radius)
                bodies.append(body_index)
                geoms.append(g)
        excluded = set(map(int, model.exclude_signature))
        pairs = []
        geom_pairs = set()
        for i, a in enumerate(geoms):
            for j in range(i + 1, len(geoms)):
                b = geoms[j]
                ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
                wa, wb = int(model.body_weldid[ba]), int(model.body_weldid[bb])
                if (
                    wa == wb
                    or wa == model.body_weldid[model.body_parentid[wb]]
                    or (wb == model.body_weldid[model.body_parentid[wa]])
                ):
                    continue
                if (min(ba, bb) << 16) + max(ba, bb) in excluded:
                    continue
                if not (
                    model.geom_contype[a] & model.geom_conaffinity[b]
                    or model.geom_contype[b] & model.geom_conaffinity[a]
                ):
                    continue
                pairs.append((i, j))
                geom_pairs.add((a, b))
        if not pairs:
            raise ValueError("No self-collision sphere pairs found")
        self.centers = torch.tensor(
            np.asarray(centers), dtype=torch.float32, device=kin.device
        )
        self.radii = torch.tensor(radii, dtype=torch.float32, device=kin.device)
        self.bodies = torch.tensor(bodies, device=kin.device)
        self.pairs = torch.tensor(pairs, device=kin.device)
        points, point_bodies, point_ids = [], [], {}
        for g, (body, vertices, _) in hulls.items():
            point_ids[g] = list(range(len(points), len(points) + len(vertices)))
            points.extend(vertices)
            point_bodies.extend([body] * len(vertices))
        self.vertices = torch.tensor(
            np.asarray(points), dtype=torch.float32, device=kin.device
        )
        self.vertex_bodies = torch.tensor(point_bodies, device=kin.device)
        self.queries = []
        for g, (body, _, planes) in hulls.items():
            neighbors = sorted(
                {b if a == g else a for a, b in geom_pairs if g in (a, b)}
            )
            ids = [i for neighbor in neighbors for i in point_ids[neighbor]]
            if ids:
                self.queries.append(
                    (
                        body,
                        torch.tensor(ids, device=kin.device),
                        torch.tensor(
                            planes[:, :3], dtype=torch.float32, device=kin.device
                        ),
                        torch.tensor(
                            planes[:, 3], dtype=torch.float32, device=kin.device
                        ),
                    )
                )
        geometry_ids = {g: i for i, g in enumerate(hulls)}
        ranges, bounds, flat_planes, geometry_bodies = [], [], [], []
        for g, (body, samples, planes) in hulls.items():
            ranges.append(
                [
                    min(point_ids[g]),
                    max(point_ids[g]) + 1,
                    len(flat_planes),
                    len(flat_planes) + len(planes),
                ]
            )
            flat_planes.extend(planes)
            center = samples.mean(0)
            bounds.append([*center, np.linalg.norm(samples - center, axis=1).max()])
            geometry_bodies.append(body)
        directed = sorted(
            (geometry_ids[a], geometry_ids[b])
            for pair in geom_pairs
            for a, b in (pair, pair[::-1])
        )
        self.geometry_bodies = torch.tensor(
            geometry_bodies, dtype=torch.int32, device=kin.device
        )
        self.directed_pairs = torch.tensor(
            directed, dtype=torch.int32, device=kin.device
        )
        self.ranges = torch.tensor(ranges, dtype=torch.int32, device=kin.device)
        self.bounds = torch.tensor(bounds, dtype=torch.float32, device=kin.device)
        self.planes = torch.tensor(
            np.asarray(flat_planes), dtype=torch.float32, device=kin.device
        )
        if self.vertices.is_cuda:
            wp.init()

    @torch.compiler.disable
    def depth(self, positions, rotations):
        shape = positions.shape[:-2]
        positions = positions.reshape(-1, positions.shape[-2], 3)
        rotations = rotations.reshape(-1, rotations.shape[-3], 3, 3)
        return torch.cat(
            [
                self._depth(p, r)
                for p, r in zip(
                    positions.split(2048), rotations.split(2048), strict=True
                )
            ]
        ).reshape(shape)

    def _depth(self, positions, rotations):
        # Select active features without keeping the point-plane intermediates.
        # Recompute the selected features with autograd to train the policy.
        batch = torch.arange(len(positions), device=positions.device)

        def world_points(local, bodies):
            return positions[batch, bodies] + (
                rotations[batch, bodies] @ local[..., None]
            ).squeeze(-1)

        with torch.no_grad():
            centers = positions[:, self.bodies] + (
                rotations[:, self.bodies] @ self.centers[..., None]
            ).squeeze(-1)
            a, b = self.pairs.unbind(-1)
            distance = (centers[:, a] - centers[:, b]).norm(dim=-1)
            best = (self.radii[a] + self.radii[b] - distance).argmax(-1)
            sa, sb = a[best], b[best]
        ca = world_points(self.centers[sa], self.bodies[sa])
        cb = world_points(self.centers[sb], self.bodies[sb])
        depths = [
            (self.radii[sa] + self.radii[sb] - (ca - cb).norm(dim=-1)).clamp_min(0)
        ]
        if positions.is_cuda:
            depths.append(self._cuda_hull_depth(positions, rotations))
            return torch.stack(depths).amax(0)
        with torch.no_grad():
            vertices = positions[:, self.vertex_bodies] + (
                rotations[:, self.vertex_bodies] @ self.vertices[..., None]
            ).squeeze(-1)
        for body, ids, normals, offsets in self.queries:
            with torch.no_grad():
                best_depth = positions.new_full((len(positions),), -torch.inf)
                best_vertex = torch.zeros_like(batch)
                best_plane = torch.zeros_like(batch)
                for chunk in ids.split(128):
                    local = (vertices[:, chunk] - positions[:, body, None]) @ rotations[
                        :, body
                    ]
                    distances = local @ normals.T + offsets
                    signed, plane = distances.max(-1)
                    depth, point = (-signed).max(-1)
                    better = depth > best_depth
                    best_vertex = torch.where(better, chunk[point], best_vertex)
                    best_plane = torch.where(better, plane[batch, point], best_plane)
                    best_depth = torch.maximum(best_depth, depth)
            world = world_points(
                self.vertices[best_vertex], self.vertex_bodies[best_vertex]
            )
            local = (
                (world - positions[:, body])[:, None] @ rotations[:, body]
            ).squeeze(1)
            signed = (local * normals[best_plane]).sum(-1) + offsets[best_plane]
            depths.append((-signed).clamp_min(0))
        return torch.stack(depths).amax(0)

    def _cuda_hull_depth(self, positions, rotations):
        shape = (len(positions), len(self.directed_pairs))
        with torch.no_grad():
            depth = positions.new_empty(shape)
            point_ids = torch.empty(shape, dtype=torch.int32, device=positions.device)
            plane_ids = torch.empty_like(point_ids)
            tensors = (
                positions.contiguous(),
                rotations.flatten(-2).contiguous(),
                self.geometry_bodies,
                self.directed_pairs,
                self.bounds,
                self.ranges,
                self.vertices,
                self.planes,
                depth,
                point_ids,
                plane_ids,
            )
            wp.launch(
                active_hull_features,
                dim=shape,
                inputs=[
                    wp.from_torch(t.detach(), requires_grad=False) for t in tensors
                ],
                stream=wp.stream_from_torch(
                    torch.cuda.current_stream(positions.device)
                ),
            )
            batch = torch.arange(len(positions), device=positions.device)
            best_depth, pair = depth.max(-1)
            vertex = point_ids[batch, pair].long()
            plane = plane_ids[batch, pair].long()
            source = self.vertex_bodies[vertex]
            target = self.geometry_bodies[self.directed_pairs[pair, 1].long()].long()
        world = positions[batch, source] + (
            rotations[batch, source] @ self.vertices[vertex, :, None]
        ).squeeze(-1)
        local = (
            (world - positions[batch, target])[:, None] @ rotations[batch, target]
        ).squeeze(1)
        signed = (local * self.planes[plane, :3]).sum(-1) + self.planes[plane, 3]
        return torch.where(best_depth > 0, (-signed).clamp_min(0), 0.0)
