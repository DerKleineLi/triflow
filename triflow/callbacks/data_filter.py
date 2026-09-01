# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.

class FilterMeshes:
    """Per-mesh filter used at dataset load time to skip bad training samples.

    Called as a predicate ``passed = filter(mesh_name, metadata)`` over each
    entry in ``metadata_512.json``. Any condition set to ``None`` is ignored,
    so the defaults are intentionally permissive. See
    ``configs/dataset/mesh_512.yaml`` for the values used by the release
    training pipeline.

    Args:
        min_faces: Minimum number of decimated faces (inclusive).
        max_faces: Maximum number of decimated faces (inclusive).
        min_ratio_discretization: Minimum ratio of decimated-to-original face
            count. Low values indicate heavy topology collapse during
            voxelization, which usually signals a bad sample.
        min_ratio_sampled_3ptimes: Minimum fraction of faces that were sampled
            by three or more voxels during SDF discretization; a proxy for
            how well the voxel grid covers the mesh surface.
        min_num_coords_fine: Minimum number of fine-resolution occupied voxels.
        max_num_coords_fine: Maximum number of fine-resolution occupied voxels.
        min_num_coords_coarse: Minimum number of coarse-resolution occupied voxels.
        max_num_coords_coarse: Maximum number of coarse-resolution occupied voxels.
        max_ratio_coarse_fine: Maximum ratio of coarse-to-fine voxel counts.
            Large values indicate the coarse grid poorly approximates the fine
            one (e.g. thin shells).
        max_decimation_dist: Maximum allowed max-distance between the decimated
            mesh and the original.
        max_decimation_chamfer: Maximum allowed chamfer distance between the
            decimated mesh and the original.
    """

    def __init__(
        self,
        min_faces=200,
        max_faces=None,
        min_ratio_discretization=1.0,
        min_ratio_sampled_3ptimes=1.0,
        min_num_coords_fine=None,
        max_num_coords_fine=None,
        min_num_coords_coarse=None,
        max_num_coords_coarse=None,
        max_ratio_coarse_fine=None,
        max_decimation_dist=None,
        max_decimation_chamfer=None,
    ):
        self.min_faces = min_faces
        self.max_faces = max_faces
        self.min_ratio_discretization = min_ratio_discretization
        self.min_ratio_sampled_3ptimes = min_ratio_sampled_3ptimes
        self.max_num_coords_fine = max_num_coords_fine
        self.max_num_coords_coarse = max_num_coords_coarse
        self.min_num_coords_fine = min_num_coords_fine
        self.min_num_coords_coarse = min_num_coords_coarse
        self.max_ratio_coarse_fine = max_ratio_coarse_fine
        self.max_decimation_dist = max_decimation_dist
        self.max_decimation_chamfer = max_decimation_chamfer

    def __call__(self, mesh_name, metadata) -> bool:
        """Return ``True`` if the mesh passes all enabled filter conditions."""
        original_num_faces = metadata["original_num_faces"]
        discretized_num_faces = metadata["discretized_num_faces"]
        decimated_num_faces = metadata.get("decimated_num_faces", discretized_num_faces)
        faces_sampled_3ptimes = metadata["faces_sampled_3+times"]
        num_occ_fine = metadata["num_occ_fine"]
        num_occ_coarse = metadata["num_occ_coarse"]
        decimation_max_dist = metadata.get("decimated_max_dist", 0.0)
        decimation_chamfer_dist = metadata.get("decimated_chamfer_dist", 0.0)

        ratio_coarse_fine = (
            num_occ_coarse / num_occ_fine if num_occ_fine > 0 else float("inf")
        )
        ratio_discretization = (
            decimated_num_faces / original_num_faces if original_num_faces > 0 else 0
        )
        ratio_sampled_3ptimes = (
            faces_sampled_3ptimes / decimated_num_faces
            if decimated_num_faces > 0
            else 0
        )

        passed = True
        if self.min_faces is not None:
            passed = passed and (decimated_num_faces >= self.min_faces)
        if self.max_faces is not None:
            passed = passed and (decimated_num_faces <= self.max_faces)
        if self.min_num_coords_fine is not None:
            passed = passed and (num_occ_fine >= self.min_num_coords_fine)
        if self.min_num_coords_coarse is not None:
            passed = passed and (num_occ_coarse >= self.min_num_coords_coarse)
        if self.min_ratio_discretization is not None:
            passed = passed and (ratio_discretization >= self.min_ratio_discretization)
        if self.min_ratio_sampled_3ptimes is not None:
            passed = passed and (
                ratio_sampled_3ptimes >= self.min_ratio_sampled_3ptimes
            )
        if self.max_num_coords_fine is not None:
            passed = passed and (num_occ_fine <= self.max_num_coords_fine)
        if self.max_num_coords_coarse is not None:
            passed = passed and (num_occ_coarse <= self.max_num_coords_coarse)
        if self.max_ratio_coarse_fine is not None:
            passed = passed and (ratio_coarse_fine <= self.max_ratio_coarse_fine)
        if self.max_decimation_dist is not None:
            passed = passed and (decimation_max_dist <= self.max_decimation_dist)
        if self.max_decimation_chamfer is not None:
            passed = passed and (decimation_chamfer_dist <= self.max_decimation_chamfer)
        return passed
