# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.

"""NVV (voxelized nearest-vertex vectors) utilities.

This module groups the low-level operations over NVV fields: representation
conversions, numba-accelerated bilateral filters, and the priority-watershed
clustering used by ``topology_flow2mesh_QEM``.
"""

import heapq

import numpy as np
import torch
from numba import njit, prange
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# NVV representation conversions (pure torch, no deps)
# ---------------------------------------------------------------------------

eps = 1e-8


def vector2dirnorm(coords=None, feats=None, resolution=None):
    """Split NVV features into ``(direction, magnitude)`` per voxel.

    Args:
        coords: Passed through unchanged.
        feats: ``(N, 3)`` NVV vectors.
        resolution: Unused (kept for signature symmetry with siblings).

    Returns:
        ``(coords, feats_dirnorm)`` where ``feats_dirnorm`` has shape
        ``(N, 4)``: unit-length direction in columns 0–2 and magnitude in
        column 3.
    """
    assert feats is not None
    r = torch.norm(feats, dim=-1, keepdim=True)
    direction = feats / (r + eps)
    feats_transformed = torch.cat([direction, r], dim=-1)
    return coords, feats_transformed


def dirnorm2vector(coords=None, feats=None, resolution=None):
    """Reconstruct NVV vectors from a ``(direction, magnitude)`` representation.

    Inverse of :func:`vector2dirnorm`; re-normalizes direction to unit length
    before multiplying by the scalar magnitude.
    """
    assert feats is not None
    direction = feats[..., 0:3]
    direction = torch.nn.functional.normalize(direction, dim=-1)
    r = feats[..., 3:4]
    vector = direction * (r + eps)
    return coords, vector


def vector2pos(coords, feats, resolution):
    """Convert NVV vectors to absolute target positions in normalized ``[-0.5, 0.5]``.

    The target position of a voxel is its own normalized center plus the NVV
    offset; this helper computes that for every entry.

    Args:
        coords: ``(N, C)`` coordinates; only the last 3 columns are used.
        feats: ``(N, 3)`` NVV vectors.
        resolution: Grid resolution ``R``.
    """
    coords_xyz = coords[:, -3:].float()
    coords_xyz = (coords_xyz + 0.5) / resolution
    coords_xyz -= 0.5
    target_pos = coords_xyz + feats
    return coords, target_pos


def coords2pos(coords, feats, resolution):
    """Return per-voxel centers in normalized ``[-0.5, 0.5]``.

    ``feats`` is accepted but ignored — the signature mirrors
    :func:`vector2pos` so callers can use it as a drop-in for ablation
    (comparing NVV-offset targets against zero-offset ones).
    """
    coords_xyz = coords[:, -3:].float()
    coords_xyz = (coords_xyz + 0.5) / resolution
    coords_xyz -= 0.5
    return coords, coords_xyz


def vector2nid(coords=None, feats=None, resolution=None):
    """Quantize NVV vectors into per-axis neighbor indices in ``{-1, 0, 1}``.

    For each axis, normalizes by the max absolute component and then
    thresholds at ``±0.33`` — voxels whose NVV points predominantly in that
    axis's positive/negative direction snap to ``+1`` / ``-1``; others snap
    to ``0``. Used by the ``neighbor_index_acc`` metric in the NVV VAE
    evaluation.
    """
    assert feats is not None
    vectors = feats
    ax_max = torch.max(torch.abs(vectors), dim=-1, keepdim=True)[0] + eps
    norm_vectors = vectors / ax_max
    new_feats = torch.zeros_like(feats)
    new_feats = torch.where(norm_vectors > 0.33, torch.ones_like(new_feats), new_feats)
    new_feats = torch.where(
        norm_vectors < -0.33, -torch.ones_like(new_feats), new_feats
    )
    return coords, new_feats


# ---------------------------------------------------------------------------
# Geodesic bilateral filtering (numba-accelerated, grid-based BFS)
# ---------------------------------------------------------------------------


@njit(parallel=True, fastmath=True)
def _bilateral_geodesic_grid_kernel(
    coords_idx, lookup_grid, nvv, threshold, sigma_s, sigma_r
):
    """Numba core for the geodesic bilateral filter.

    BFS-walks the 6-connected voxel neighborhood out to distance ``threshold``
    (geodesic, i.e. only through occupied voxels — unoccupied cells in
    ``lookup_grid`` store ``-1`` and are skipped), accumulating a bilateral
    weighted average of NVV values. Weights combine a spatial Gaussian on
    the BFS hop count and a range Gaussian on the NVV difference.
    """
    N = coords_idx.shape[0]
    filtered_nvv = np.zeros_like(nvv)
    s_denom = 1.0 / (2.0 * sigma_s**2)
    r_denom = 1.0 / (2.0 * sigma_r**2)

    adj = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.int32,
    )

    side = 2 * threshold + 1
    local_vol = side**3

    for i in prange(N):
        root_x, root_y, root_z = coords_idx[i]
        v_i = nvv[i]

        visited = np.zeros(local_vol, dtype=np.uint8)
        queue = np.empty((local_vol, 4), dtype=np.int32)
        q_start = 0
        q_end = 0

        queue[q_end] = [0, 0, 0, 0]
        mid_idx = threshold * (side**2) + threshold * side + threshold
        visited[mid_idx] = 1
        q_end += 1

        total_w = 0.0
        weighted_sum = np.zeros(3, dtype=np.float64)

        while q_start < q_end:
            rx, ry, rz, d = queue[q_start]
            q_start += 1

            curr_idx = lookup_grid[root_x + rx, root_y + ry, root_z + rz]
            v_j = nvv[curr_idx]
            d_s_sq = float(d)
            dv = v_i - v_j
            d_r_sq = dv[0] ** 2 + dv[1] ** 2 + dv[2] ** 2
            w = np.exp(-(d_s_sq * s_denom + d_r_sq * r_denom))
            weighted_sum += v_j * w
            total_w += w

            if d < threshold:
                for a in range(6):
                    nrx, nry, nrz = rx + adj[a, 0], ry + adj[a, 1], rz + adj[a, 2]
                    if (
                        abs(nrx) <= threshold
                        and abs(nry) <= threshold
                        and abs(nrz) <= threshold
                    ):
                        l_idx = (
                            (nrx + threshold) * side**2
                            + (nry + threshold) * side
                            + (nrz + threshold)
                        )
                        if visited[l_idx] == 0:
                            if (
                                lookup_grid[root_x + nrx, root_y + nry, root_z + nrz]
                                != -1
                            ):
                                visited[l_idx] = 1
                                queue[q_end] = [nrx, nry, nrz, d + 1]
                                q_end += 1

        if total_w > 1e-12:
            filtered_nvv[i] = weighted_sum / total_w
        else:
            filtered_nvv[i] = v_i

    return filtered_nvv


def filter_nvv_geodesic_fast(coords, nvv, threshold=3, sigma_s=1.0, sigma_r=0.01):
    """Geodesic bilateral filter over an occupied voxel set.

    Snaps the coordinates to an integer lookup grid and runs
    :func:`_bilateral_geodesic_grid_kernel` in parallel. The filter is
    "geodesic" because the neighborhood is a BFS through occupied voxels
    only, so NVV smoothing does not leak across disconnected components.

    Args:
        coords: ``(N, 3)`` voxel coordinates.
        nvv: ``(N, 3)`` NVV vectors.
        threshold: Maximum BFS hop distance considered.
        sigma_s: Spatial-domain Gaussian bandwidth (in hop units).
        sigma_r: Range-domain Gaussian bandwidth (in NVV units).

    Returns:
        Filtered ``(N, 3)`` NVV vectors, in the same order as the input.
    """
    coords_idx = np.round(coords).astype(np.int32)
    sort_idx = np.lexsort((coords_idx[:, 2], coords_idx[:, 1], coords_idx[:, 0]))
    coords_idx = coords_idx[sort_idx]
    nvv_sorted = nvv[sort_idx].astype(np.float64)

    min_b = coords_idx.min(axis=0) - threshold
    coords_idx -= min_b
    grid_shape = coords_idx.max(axis=0) + threshold + 1
    lookup_grid = np.full(grid_shape, -1, dtype=np.int32)
    lookup_grid[coords_idx[:, 0], coords_idx[:, 1], coords_idx[:, 2]] = np.arange(
        len(coords_idx), dtype=np.int32
    )

    filtered_sorted = _bilateral_geodesic_grid_kernel(
        coords_idx, lookup_grid, nvv_sorted, threshold, sigma_s, sigma_r
    )

    unsort_idx = np.empty_like(sort_idx)
    unsort_idx[sort_idx] = np.arange(len(sort_idx))
    return filtered_sorted[unsort_idx]


# ---------------------------------------------------------------------------
# NVV target point watershed algorithm
# ---------------------------------------------------------------------------


def get_target_point_priority_watershed(
    mesh,
    coords,
    nvv,
    root_threshold=0.5,
    resolution=None,
):
    """Cluster mesh vertices into target groups via a priority watershed on NVV targets.

    For each voxel, its "ideal target" is its center + NVV offset. Voxels
    whose NVV is smaller than ``root_threshold`` on every axis are declared
    *roots* (they barely move); a Dijkstra-style watershed then grows each
    root's cluster across the mesh graph by greedily absorbing neighboring
    vertices whose ideal target is closest to the cluster's root position.

    Args:
        mesh: A ``trimesh.Trimesh``.
        coords: ``(N, 3)`` voxel coordinates.
        nvv: ``(N, 3)`` NVV vectors.
        root_threshold: Max absolute NVV component for a voxel to be
            considered a root.
        resolution: If set, ``nvv`` is rescaled by this factor (use when
            ``nvv`` is normalized to ``[-1, 1]`` grid units).

    Returns:
        ``(target_pts_per_vert, refined_root_ids)``:

        * ``target_pts_per_vert``: ``(V, 3)``; the assigned root position
          for each mesh vertex.
        * ``refined_root_ids``: ``(V,)`` int array; the mesh-vertex index of
          each vertex's assigned root.
    """
    nvv_scaled = nvv if resolution is None else nvv * resolution
    voxel_target_pos = coords.astype(float) + 0.5 + nvv_scaled

    mesh_vertices = mesh.vertices
    num_verts = len(mesh_vertices)
    coords_tree = cKDTree(coords.astype(float) + 0.5)
    _, mesh_vert_to_voxel_idx = coords_tree.query(mesh_vertices)
    vert_ideal_targets = voxel_target_pos[mesh_vert_to_voxel_idx]

    mask_small_nvv = np.all(np.abs(nvv_scaled) <= root_threshold, axis=1)
    root_candidate_pts = voxel_target_pos[mask_small_nvv]

    mesh_tree = cKDTree(mesh_vertices)
    _, root_mesh_vert_ids = mesh_tree.query(root_candidate_pts)
    root_mesh_vert_ids = np.unique(root_mesh_vert_ids)

    edges = mesh.edges_unique
    v1 = edges[:, 0]
    v2 = edges[:, 1]
    adj_matrix = csr_matrix(
        (np.ones(len(edges) * 2), (np.concatenate([v1, v2]), np.concatenate([v2, v1]))),
        shape=(num_verts, num_verts),
    )
    indptr = adj_matrix.indptr
    indices = adj_matrix.indices

    refined_root_ids = np.full(num_verts, -1, dtype=int)
    root_positions = vert_ideal_targets[root_mesh_vert_ids]
    all_root_positions = np.zeros((num_verts, 3))
    all_root_positions[root_mesh_vert_ids] = root_positions

    pq = []
    for r_id in root_mesh_vert_ids:
        refined_root_ids[r_id] = r_id
        neighbors = indices[indptr[r_id] : indptr[r_id + 1]]
        for v in neighbors:
            dist = np.linalg.norm(vert_ideal_targets[v] - all_root_positions[r_id])
            heapq.heappush(pq, (dist, v, r_id))

    while pq:
        cost, u, r_id = heapq.heappop(pq)
        if refined_root_ids[u] != -1:
            continue
        refined_root_ids[u] = r_id
        u_root_pos = all_root_positions[r_id]
        for v in indices[indptr[u] : indptr[u + 1]]:
            if refined_root_ids[v] == -1:
                diff = vert_ideal_targets[v] - u_root_pos
                new_dist = np.sqrt(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2)
                heapq.heappush(pq, (new_dist, v, r_id))

    target_pts_per_vert = all_root_positions[refined_root_ids]
    return target_pts_per_vert, refined_root_ids
