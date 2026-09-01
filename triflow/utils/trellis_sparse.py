# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.

import torch
from trellis.modules.sparse.basic import SparseTensor

from triflow.utils.sparse_voxel import dense2sparse, sparse2dense


def sparse_tensor2dense(sparse_tensor, resolution):
    """Convert a ``trellis`` SparseTensor into a dense 5-D grid.

    Args:
        sparse_tensor: A ``trellis`` SparseTensor whose ``coords`` carry a
            batch index in column 0.
        resolution: Spatial resolution ``R`` of the target dense grid.

    Returns:
        A dense tensor of shape ``(B, C, R, R, R)``.
    """
    coords = sparse_tensor.coords
    feats = sparse_tensor.feats
    dense = sparse2dense(coords, feats, resolution)
    return dense


def sparse2sparse_tensor(coords, feats):
    """Wrap batched sparse ``(coords, feats)`` into a ``trellis`` SparseTensor.

    ``coords`` is expected to have shape ``(N, D+1)`` where the first column
    is the batch index (as produced by ``MeshDataset.collate_coords``). This
    helper additionally:

    * computes the per-batch voxel layout and registers it as a spatial cache
      (required by some downstream ops), and
    * sets the tensor's advertised shape to ``(B, C)`` so shape-dependent ops
      can reason about batch size without scanning ``coords``.

    Identical in structure to
    ``triflow.utils.direct3ds2_sparse.sparse2sparse_tensor`` but constructs a
    ``trellis`` SparseTensor instead of a ``direct3d_s2`` one.

    Args:
        coords: ``int`` tensor of shape ``(N, D+1)`` with the batch index in
            column 0.
        feats: float tensor of shape ``(N, C)``.

    Returns:
        A ``trellis.modules.sparse.basic.SparseTensor``.
    """
    B = coords[:, 0].max().item() + 1

    layout = []
    start = 0
    batch_coords = coords[:, 0]
    for b in range(B):
        num_voxels = (batch_coords == b).sum().item()
        layout.append(slice(start, start + num_voxels))
        start += num_voxels

    sparse_tensor = SparseTensor(coords=coords.int(), feats=feats)
    sparse_tensor._shape = torch.Size([B, feats.shape[1]])
    sparse_tensor.register_spatial_cache("layout", layout)

    return sparse_tensor


def dense2sparse_tensor(grid):
    """Convert a dense ``(B, C, R, R, R)`` grid to a ``trellis`` SparseTensor.

    Non-zero voxels become sparse entries; the resulting tensor carries a
    batch-index column and layout cache just like
    :func:`sparse2sparse_tensor` produces.
    """
    coords, feats = dense2sparse(grid)
    return sparse2sparse_tensor(coords, feats)


def sparse_tensor2sparse(sparse_tensor):
    """Unpack a ``trellis`` SparseTensor into raw ``(coords, feats)`` tensors."""
    coords = sparse_tensor.coords
    feats = sparse_tensor.feats
    return coords, feats
