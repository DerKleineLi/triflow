# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.

from typing import Tuple

import mcubes
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes, gaussian_filter

# ---------------------------------------------------------------------------
# Meshgrid and device transfer utilities
# ---------------------------------------------------------------------------


def create_meshgrid(resolution=64, normalized=True):
    """Return a ``(3, R, R, R)`` meshgrid of integer or normalized coordinates.

    Args:
        resolution: Grid resolution ``R``.
        normalized: If ``True``, coordinates are mapped to voxel *centers* in
            ``[0, 1]`` (i.e. ``(i + 0.5) / R``). Otherwise integer indices.

    Returns:
        ``np.ndarray`` of shape ``(3, R, R, R)`` with ``(z, y, x)`` along
        axis 0.
    """
    N = resolution
    z, y, x = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    coords = np.stack([z, y, x], axis=0)
    if normalized:
        coords = (coords + 0.5) / N
    return coords


def nested_device_transfer(data, device):
    """Recursively call ``.to(device)`` on every tensor-like leaf of ``data``.

    Dicts, lists, and tuples are walked; any object exposing ``.to(device)``
    is transferred; everything else is passed through unchanged.
    """
    if isinstance(data, dict):
        return {k: nested_device_transfer(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [nested_device_transfer(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(nested_device_transfer(v, device) for v in data)
    elif hasattr(data, "to"):
        return data.to(device)
    else:
        return data


# ---------------------------------------------------------------------------
# Coarse and Fine Sparse Voxel Tensor Operations
# ---------------------------------------------------------------------------


def coord_hash(coords: torch.Tensor) -> torch.Tensor:
    """Hash ``(b, x, y, z)`` integer coordinates into a single int64 for fast matching.

    Each spatial component gets an 11-bit slot (``[0, 2047]``) and the batch
    index fills the remaining high bits. Assumes ``coords`` is ``(N, 4)``
    with the batch index in column 0.
    """
    assert coords.shape[1] == 4, "Coords must be (N, 4) -> (b, x, y, z)"
    b, x, y, z = coords.T
    return (b.long() << 33) | (x.long() << 22) | (y.long() << 11) | z.long()


def find_indices(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Find the index of every query in the keys via sorted search.

    Both ``queries`` and ``keys`` must already be hashed (e.g. via
    :func:`coord_hash`). Returns ``-1`` for queries not found.
    """
    sort_idx = torch.argsort(keys)
    sorted_keys = keys[sort_idx]
    idx_in_sorted = torch.searchsorted(sorted_keys, queries)
    idx_in_sorted = idx_in_sorted.clamp(max=len(keys) - 1)
    matched = sorted_keys[idx_in_sorted] == queries
    original_idx = sort_idx[idx_in_sorted]
    original_idx[~matched] = -1
    return original_idx


def find_coords_indices(
    query_coords: torch.Tensor, key_coords: torch.Tensor
) -> torch.Tensor:
    """Find the index of every query coordinate in the key coordinates.

    Single-batch (``N, 3``) inputs are internally lifted to ``N, 4`` by
    prepending a batch index of 0.

    Args:
        query_coords: ``(N_query, 4)`` or ``(N_query, 3)`` coordinates.
        key_coords: ``(N_key, 4)`` or ``(N_key, 3)`` coordinates.

    Returns:
        Tensor of indices into ``key_coords``. Returns ``-1`` for queries not
        found.
    """
    if query_coords.shape[1] == 3:
        batch_col = torch.zeros(
            (query_coords.shape[0], 1),
            dtype=query_coords.dtype,
            device=query_coords.device,
        )
        query_coords = torch.cat([batch_col, query_coords.long()], dim=1)
    if key_coords.shape[1] == 3:
        batch_col = torch.zeros(
            (key_coords.shape[0], 1), dtype=key_coords.dtype, device=key_coords.device
        )
        key_coords = torch.cat([batch_col, key_coords.long()], dim=1)
    queries_hashed = coord_hash(query_coords)
    keys_hashed = coord_hash(key_coords)
    return find_indices(queries_hashed, keys_hashed)


def fine_coords2coarse_coords(coords_fine: torch.Tensor, ratio: int) -> torch.Tensor:
    """Deduplicate fine coordinates into their parent coarse coordinates.

    Args:
        coords_fine: ``(N, 4)`` batched sparse coords.
        ratio: Integer ``R_fine / R_coarse``.

    Returns:
        ``(M, 4)`` unique coarse coords. Order matches first occurrence in
        ``coords_fine``.
    """
    c_fine = coords_fine
    r = ratio
    c_coarse_raw = c_fine.clone()
    c_coarse_raw[:, 1:] //= r
    hashed_coarse = coord_hash(c_coarse_raw)
    unique_hashes, inverse_indices = torch.unique(hashed_coarse, return_inverse=True)
    perm = torch.arange(
        inverse_indices.size(0),
        dtype=inverse_indices.dtype,
        device=inverse_indices.device,
    )
    unique_indices = torch.zeros_like(unique_hashes, dtype=torch.long).scatter_(
        0, inverse_indices, perm
    )
    c_coarse = c_coarse_raw[unique_indices]
    return c_coarse


def fine_to_coarse(
    feats_fine: torch.Tensor, coords_fine: torch.Tensor, ratio: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Space-to-depth aggregation from a fine to a coarse sparse grid.

    Each ``(r x r x r)`` cube of fine voxels is packed into the feature
    dimension of a single coarse voxel, so a fine tensor of shape
    ``(N_fine, C)`` becomes ``(N_coarse, C * r^3)``.

    Args:
        feats_fine: ``(N_fine, C)`` fine features.
        coords_fine: ``(N_fine, 4)`` fine coordinates with batch index in
            column 0.
        ratio: Integer ``R_fine / R_coarse``.

    Returns:
        ``(feats_coarse, coords_coarse)``.
    """
    f_fine = feats_fine
    c_fine = coords_fine
    N_fine, C = f_fine.shape
    r = ratio
    r3 = r**3

    c_coarse_raw = c_fine.clone()
    c_coarse_raw[:, 1:] //= r
    hashed_coarse = coord_hash(c_coarse_raw)
    unique_hashes, inverse_indices = torch.unique(hashed_coarse, return_inverse=True)
    perm = torch.arange(
        inverse_indices.size(0),
        dtype=inverse_indices.dtype,
        device=inverse_indices.device,
    )
    unique_indices = torch.zeros_like(unique_hashes, dtype=torch.long).scatter_(
        0, inverse_indices, perm
    )
    c_coarse = c_coarse_raw[unique_indices]

    rem = c_fine[:, 1:] % r
    flat_offset = rem[:, 0] * (r * r) + rem[:, 1] * r + rem[:, 2]

    N_coarse = c_coarse.shape[0]
    expanded_C = C * r3
    f_coarse = torch.zeros(
        (N_coarse, expanded_C), dtype=f_fine.dtype, device=f_fine.device
    )
    f_coarse_view = f_coarse.view(N_coarse, r3, C)
    f_coarse_view.index_put_(
        (inverse_indices, flat_offset),
        f_fine,
        accumulate=False,
    )
    f_coarse_final = f_coarse_view.view(N_coarse, expanded_C)
    return f_coarse_final, c_coarse


def coarse_to_fine(
    feats_coarse: torch.Tensor,
    coords_coarse: torch.Tensor,
    coords_fine_query: torch.Tensor,
    ratio: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Depth-to-space expansion from a coarse to a fine sparse grid.

    The inverse of :func:`fine_to_coarse`: given the query fine coordinates,
    retrieve their features by indexing into the packed feature dimension of
    their parent coarse voxel. Fine queries whose parent coarse voxel is
    missing are dropped.

    Args:
        feats_coarse: ``(N_coarse, C * r^3)`` packed coarse features.
        coords_coarse: ``(N_coarse, 4)`` coarse coordinates.
        coords_fine_query: ``(N_query, 4)`` fine coordinates to retrieve.
        ratio: Integer ``R_fine / R_coarse``.

    Returns:
        ``(feats_fine_retrieved, coords_fine_retrieved)``. The returned
        fine-coord tensor is a subset of ``coords_fine_query``.
    """
    f_coarse = feats_coarse
    c_coarse = coords_coarse
    c_fine_query = coords_fine_query
    r = ratio
    r3 = r**3

    c_target_coarse = c_fine_query.clone()
    c_target_coarse[:, 1:] //= r
    keys = coord_hash(c_coarse)
    queries = coord_hash(c_target_coarse)
    coarse_indices = find_indices(queries, keys)
    mask_valid = coarse_indices != -1
    c_fine_retrieved = c_fine_query[mask_valid]
    coarse_indices = coarse_indices[mask_valid]

    if c_fine_retrieved.numel() == 0:
        C_fine_dim = f_coarse.shape[1] // r3
        return torch.empty(
            0, C_fine_dim, dtype=f_coarse.dtype, device=f_coarse.device
        ), torch.empty(0, 4, dtype=c_fine_query.dtype, device=c_fine_query.device)

    rem = c_fine_retrieved[:, 1:] % r
    flat_offset = rem[:, 0] * (r * r) + rem[:, 1] * r + rem[:, 2]
    C_fine_dim = f_coarse.shape[1] // r3
    f_coarse_view = f_coarse.view(-1, r3, C_fine_dim)
    f_fine_retrieved = f_coarse_view[coarse_indices, flat_offset, :]
    return f_fine_retrieved, c_fine_retrieved


def get_coords_coarse2fine(coords_coarse, ratio):
    """Expand each coarse coordinate into all of its ``r^3`` fine-grid children.

    Args:
        coords_coarse: ``(N, 3)`` or ``(N, 4)`` tensor or numpy array. If
            ``(N, 4)``, column 0 is treated as the batch index and is copied
            to every child.
        ratio: Integer ``R_fine / R_coarse``.

    Returns:
        Same container type as the input (tensor or array), shape
        ``(N * r^3, 3)`` or ``(N * r^3, 4)``.
    """
    was_numpy = False
    if isinstance(coords_coarse, np.ndarray):
        was_numpy = True
        coords = torch.from_numpy(coords_coarse)
    else:
        coords = coords_coarse

    if not torch.is_tensor(coords):
        raise ValueError("coords_coarse must be a torch.Tensor or numpy.ndarray")
    if coords.dim() != 2 or coords.shape[1] not in (3, 4):
        raise ValueError("coords_coarse must have shape (N,3) or (N,4)")

    device = coords.device
    coords_long = coords.long()
    was_batched = coords_long.shape[1] == 4
    if was_batched:
        batch_col = coords_long[:, 0]
        coords_xyz = coords_long[:, 1:4]
    else:
        batch_col = None
        coords_xyz = coords_long

    N = coords_xyz.shape[0]
    if N == 0:
        out_shape = (0, 4) if was_batched else (0, 3)
        if was_numpy:
            return np.zeros(out_shape, dtype=np.int64)
        return torch.zeros(out_shape, dtype=torch.long, device=device)

    r = int(ratio)
    r3 = r**3
    a = torch.arange(r, device=device, dtype=torch.long)
    dx, dy, dz = torch.meshgrid(a, a, a, indexing="ij")
    deltas = torch.stack(
        [dx.reshape(-1), dy.reshape(-1), dz.reshape(-1)], dim=1
    )
    scaled = coords_xyz * r
    scaled_exp = scaled.unsqueeze(1)
    deltas_exp = deltas.unsqueeze(0)
    fine_3d = (scaled_exp + deltas_exp).reshape(-1, 3)

    if was_batched:
        batch_rep = batch_col.unsqueeze(1).repeat(1, r3).reshape(-1)
        fine = torch.cat([batch_rep.unsqueeze(1), fine_3d], dim=1)
    else:
        fine = fine_3d

    if was_numpy:
        return fine.cpu().numpy()
    return fine


# ---------------------------------------------------------------------------
# Convert between dense and sparse tensor representations
# ---------------------------------------------------------------------------


def dense2sparse(grid):
    """Convert a dense voxel grid to sparse coordinates + features.

    Channel 0 is treated as a binary occupancy mask (``> 0`` is present),
    and the remaining channels become the per-voxel features.

    Args:
        grid: ``(B, C, D, H, W)`` or ``(C, D, H, W)`` tensor or numpy array.

    Returns:
        ``(coords, feats)``. ``coords`` has shape ``(N, 4)`` or ``(N, 3)``
        matching the input's batched-ness; ``feats`` has shape ``(N, C - 1)``.
    """
    was_numpy = False
    if isinstance(grid, np.ndarray):
        was_numpy = True
        grid = torch.from_numpy(grid).float()

    was_batched = grid.dim() == 5
    if not was_batched:
        grid = grid.unsqueeze(0)

    B, C, D, H, W = grid.shape
    occ = grid[:, 0] > 0
    batch_coords, z, y, x = torch.nonzero(occ, as_tuple=True)
    coords = torch.stack([batch_coords, z, y, x], dim=1).long()

    if coords.numel() == 0:
        feats = torch.empty((0, C - 1), dtype=grid.dtype, device=grid.device).float()
    else:
        feats = grid[batch_coords, 1:, z, y, x]
        feats = feats.contiguous().float()

    if was_numpy:
        coords = coords.cpu().numpy()
        feats = feats.cpu().numpy()

    if not was_batched:
        return coords[:, 1:], feats
    return coords, feats


def sparse2dense(coords, feats, resolution):
    """Convert sparse coordinates + features back to a dense voxel grid.

    Channel 0 of the output is occupancy (``-1`` empty, ``+1`` occupied),
    and the remaining channels hold the per-voxel features. Single-batch
    inputs (``(N, 3)``) produce a ``(C, D, H, W)`` grid; batched inputs
    (``(N, 4)``) produce a ``(B, C, D, H, W)`` grid.

    Args:
        coords: Sparse coordinates, ``(N, 3)`` or ``(N, 4)``.
        feats: Per-voxel features, ``(N, C - 1)``.
        resolution: Either an int ``R`` (cubic) or a triple ``(D, H, W)``.

    Returns:
        Dense grid tensor or array, matching the container type of the input.
    """
    was_numpy = False
    if isinstance(coords, np.ndarray):
        coords = torch.from_numpy(coords).long()
        was_numpy = True
    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats).float()
        was_numpy = True

    was_batched = True
    if coords.dim() != 2:
        raise ValueError("coords must be a 2D tensor with shape (N,3) or (N,4)")

    if coords.shape[1] == 3:
        was_batched = False
        batch_col = torch.zeros(
            (coords.shape[0], 1), dtype=coords.dtype, device=coords.device
        )
        coords_proc = torch.cat([batch_col, coords.long()], dim=1)
    elif coords.shape[1] == 4:
        coords_proc = coords.long()
    else:
        raise ValueError("coords must have shape (N,3) or (N,4)")

    if isinstance(resolution, int):
        D = H = W = resolution
    else:
        try:
            D, H, W = resolution
        except Exception:
            raise ValueError(
                "resolution must be an int or a sequence of three ints (D,H,W)"
            )

    if coords_proc.numel() == 0:
        B = 1 if not was_batched else 0
    else:
        B = coords_proc[:, 0].max().item() + 1

    C_total = feats.shape[1] + 1
    dense = feats.new_zeros((B, C_total, D, H, W))
    dense[:, 0] = -1.0

    for b in range(B):
        slice_mask = coords_proc[:, 0] == b
        b_coords = coords_proc[slice_mask]
        b_feats = feats[slice_mask]
        if b_coords.shape[0] == 0:
            continue
        z = b_coords[:, 1]
        y = b_coords[:, 2]
        x = b_coords[:, 3]
        dense[b, 0, z, y, x] = 1.0
        dense[b, 1:, z, y, x] = b_feats.t()

    if was_numpy:
        dense = dense.cpu().numpy()
    if not was_batched:
        return dense.squeeze(0)
    return dense


# ---------------------------------------------------------------------------
# SDF operations
# ---------------------------------------------------------------------------


def sparse_sdf2dense(coords, sdf, resolution):
    """Convert sparse SDF samples to a dense ``(B, 1, D, H, W)`` grid.

    Voxels not in ``coords`` are filled with ``+pad_sdf`` (``= 2 / R``), and
    voxels *inside* the boundary (via ``binary_fill_holes`` on the
    occupancy mask) are filled with ``-pad_sdf``. The explicit ``coords``
    then overwrite with their actual SDF values.

    Args:
        coords: ``(N, 3)`` or ``(N, 4)`` tensor or numpy array.
        sdf: Per-voxel SDF values of shape ``(N,)`` or ``(N, 1)``.
        resolution: Cubic resolution ``R``.

    Returns:
        Dense SDF grid, ``(B, 1, R, R, R)`` if batched, ``(1, R, R, R)``
        otherwise. Output container type matches the inputs.
    """
    coords_is_tensor = isinstance(coords, torch.Tensor)
    sdf_is_tensor = isinstance(sdf, torch.Tensor)
    out_is_tensor = coords_is_tensor or sdf_is_tensor
    device = (
        coords.device if coords_is_tensor else (sdf.device if sdf_is_tensor else None)
    )

    coords_np = coords.cpu().numpy() if coords_is_tensor else np.asarray(coords)
    sdf_np = sdf.cpu().numpy() if sdf_is_tensor else np.asarray(sdf)
    resolution = int(resolution)

    if coords_np.ndim != 2 or coords_np.shape[1] not in (3, 4):
        raise ValueError("coords must be shape (N,3) or (N,4)")

    was_batched = True
    if coords_np.shape[1] == 3:
        coords_np = np.concatenate(
            [np.zeros((coords_np.shape[0], 1), dtype=int), coords_np.astype(int)],
            axis=1,
        )
        was_batched = False
    else:
        coords_np = coords_np.astype(int)

    pad_sdf = 2.0 / resolution
    D = H = W = resolution
    B = int(coords_np[:, 0].max()) + 1 if coords_np.size > 0 else 1
    dense_sdf_np = np.ones((B, 1, D, H, W), dtype=float) * pad_sdf

    for b in range(B):
        slice_mask = coords_np[:, 0] == b
        b_coords = coords_np[slice_mask]
        if b_coords.shape[0] == 0:
            continue
        b_sdf = sdf_np[slice_mask]
        z = b_coords[:, 1].astype(int)
        y = b_coords[:, 2].astype(int)
        x = b_coords[:, 3].astype(int)
        valid = (z >= 0) & (z < D) & (y >= 0) & (y < H) & (x >= 0) & (x < W)
        if not np.any(valid):
            continue
        zv, yv, xv = z[valid], y[valid], x[valid]
        b_sdf_vals = np.asarray(b_sdf).ravel()[valid]
        interior = np.zeros((D, H, W), dtype=int)
        interior[zv, yv, xv] = 1
        interior = binary_fill_holes(interior).astype(int)
        dense_sdf_np[b, 0, interior == 1] = -pad_sdf
        dense_sdf_np[b, 0, zv, yv, xv] = b_sdf_vals

    ret = dense_sdf_np
    if out_is_tensor:
        dtype = sdf.dtype if sdf_is_tensor else torch.float32
        ret = torch.tensor(ret, dtype=dtype, device=device)
    if not was_batched:
        ret = ret[0]
    return ret


def coords2dense_sdf(coords, resolution, gaussian_sigma=2.5):
    """Build a dense SDF-like grid from sparse occupancy coordinates alone.

    Surface voxels are set to ``0``, interior voxels to ``-pad_sdf``, and
    exterior voxels to ``+pad_sdf``; the result is optionally smoothed by a
    Gaussian blur. Used for visualization / debugging when only occupancy
    is available.

    Args:
        coords: ``(N, 3)`` or ``(N, 4)`` tensor or numpy array.
        resolution: Cubic resolution ``R``.
        gaussian_sigma: Standard deviation of the optional post-smoothing
            Gaussian filter (``0`` disables).

    Returns:
        Dense SDF-like grid, batched or not depending on the input shape.
    """
    coords_is_tensor = isinstance(coords, torch.Tensor)
    out_is_tensor = coords_is_tensor
    device = coords.device if coords_is_tensor else None

    coords_np = coords.cpu().numpy() if coords_is_tensor else np.asarray(coords)
    resolution = int(resolution)
    pad_sdf = 2.0 / resolution

    if coords_np.ndim != 2 or coords_np.shape[1] not in (3, 4):
        raise ValueError("coords must be shape (N,3) or (N,4)")

    was_batched = True
    if coords_np.shape[1] == 3:
        coords_np = np.concatenate(
            [np.zeros((coords_np.shape[0], 1), dtype=int), coords_np.astype(int)],
            axis=1,
        )
        was_batched = False
    else:
        coords_np = coords_np.astype(int)

    D = H = W = resolution
    B = int(coords_np[:, 0].max()) + 1 if coords_np.size > 0 else 1
    dense_sdf_np = np.ones((B, 1, D, H, W), dtype=float) * pad_sdf

    for b in range(B):
        slice_mask = coords_np[:, 0] == b
        b_coords = coords_np[slice_mask]
        if b_coords.shape[0] == 0:
            continue
        z = b_coords[:, 1].astype(int)
        y = b_coords[:, 2].astype(int)
        x = b_coords[:, 3].astype(int)
        valid = (z >= 0) & (z < D) & (y >= 0) & (y < H) & (x >= 0) & (x < W)
        if not np.any(valid):
            continue
        zv, yv, xv = z[valid], y[valid], x[valid]
        interior = np.zeros((D, H, W), dtype=int)
        interior[zv, yv, xv] = 1
        interior = binary_fill_holes(interior).astype(int)
        dense_sdf_np[b, 0, interior == 1] = -pad_sdf
        dense_sdf_np[b, 0, zv, yv, xv] = 0.0
        if gaussian_sigma > 0.0:
            dense_sdf_np[b, 0] = gaussian_filter(
                dense_sdf_np[b, 0], sigma=gaussian_sigma
            )

    ret = dense_sdf_np
    if out_is_tensor:
        ret = torch.tensor(ret, device=device)
    if not was_batched:
        ret = ret[0]
    return ret


def get_mc_mesh(sdf, level=0.0):
    """Extract a triangle mesh from an SDF volume via marching cubes.

    Applies a half-voxel offset so that the output vertices live at voxel
    centers rather than voxel corners, matching the convention used by the
    sparse coordinate utilities in this module.

    Args:
        sdf: 3-D SDF array ``(D, H, W)``. Inside is assumed negative.
        level: Iso-value at which to extract the surface.

    Returns:
        ``(vertices, faces)`` as ``np.ndarray``.
    """
    if isinstance(sdf, torch.Tensor):
        sdf = sdf.detach().cpu().numpy()
    verts, faces = mcubes.marching_cubes(-sdf, -level)
    verts += 0.5
    return verts, faces
