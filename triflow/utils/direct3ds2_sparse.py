# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.

import torch
from direct3d_s2.modules.sparse import SparseTensor


def sparse2sparse_tensor(coords, feats):
    """Wrap batched sparse ``(coords, feats)`` into a ``direct3d_s2`` SparseTensor.

    ``coords`` is expected to have shape ``(N, D+1)`` where the first column
    is the batch index (as produced by ``MeshDataset.collate_coords``). This
    helper additionally:

    * computes the per-batch voxel layout and registers it as a spatial cache
      (required by some downstream ops), and
    * sets the tensor's advertised shape to ``(B, C)`` so shape-dependent ops
      can reason about batch size without scanning ``coords``.

    Args:
        coords: ``int`` tensor of shape ``(N, D+1)`` with the batch index in
            column 0.
        feats: float tensor of shape ``(N, C)``.

    Returns:
        A ``direct3d_s2.modules.sparse.SparseTensor``.
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
