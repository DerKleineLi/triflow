# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from triflow.utils.mesh_processing import process_one_mesh
from triflow.utils.render import (
    render_occ_color_grid,
    render_occ_dir_grid,
    render_sdf_grid,
)
from triflow.utils.sparse_voxel import (
    create_meshgrid,
    get_coords_coarse2fine,
    sparse2dense,
    sparse_sdf2dense,
)


class MeshDataset(Dataset):
    """Voxelized mesh dataset for the SDF VAE / NVV VAE / latent flow training stages.

    Each sample is a mesh, loaded lazily and processed on-the-fly by
    ``process_one_mesh`` into fine-resolution NVV (voxelized nearest-vertex
    vectors) and a coarse-to-fine SDF representation. Optional on-the-fly
    random surface distortion is applied to ``augment_ratio`` of the samples
    as data augmentation.

    Metadata for all candidate meshes is loaded from a single JSON file;
    samples are filtered through ``filter_func`` before being kept.

    Args:
        data_root: Root directory containing a ``meshes/`` subdirectory.
        meta_file: Path to the metadata JSON file (defaults to
            ``<data_root>/metadata.json``).
        filter_func: Optional ``callable(mesh_name, metadata) -> bool``
            predicate used to drop bad samples.
        res_coarse: Coarse voxel grid resolution.
        res_fine: Fine voxel grid resolution.
        pad: Padding (in voxels) added around the mesh bounding box.
        round_verts: Whether to snap mesh vertices to voxel centers before
            processing.
        decimate_length: Target edge length (in voxels) used by the
            preprocessing decimation step.
        vertex_merge_threshold: Minimum edge length below which vertices are
            merged in the final QEM simplification.
        augment_ratio: Fraction of samples (in ``[0, 1]``) on which random
            surface distortion is applied.
        augment_strength: Magnitude of the random distortion.
        visualize_type: Which field is visualized by ``visualize_batch`` —
            ``"nvv"`` renders NVV directions / magnitudes / target positions,
            ``"sdf"`` renders an SDF isosurface.
        overfit: If ``True``, repeats the first sample 1000 times (used for
            debugging).
        dummy: If ``True``, skips all file I/O; the instance is usable only
            for ``collate_fn`` / ``visualize_*`` helpers. Used by
            ``inference.py`` to avoid requiring a dataset at inference time.
    """

    def __init__(
        self,
        data_root="",
        meta_file=None,
        filter_func=None,
        res_coarse=64,
        res_fine=128,
        pad=1.5,
        round_verts=True,
        decimate_length=5.2,
        vertex_merge_threshold=0.0,
        augment_ratio=0.0,
        augment_strength=0.5,
        visualize_type="nvv",
        overfit=False,
        dummy=False,
    ):
        super().__init__()
        self.visualize_type = visualize_type

        if dummy:
            return

        self.data_root = Path(data_root)
        self.meta_file = (
            (self.data_root / "metadata.json") if meta_file is None else Path(meta_file)
        )
        self.filter_func = filter_func
        self.overfit = overfit

        self.res_coarse = res_coarse
        self.res_fine = res_fine
        self.pad = pad
        self.round_verts = round_verts
        self.augment_ratio = augment_ratio
        self.augment_strength = augment_strength
        self.decimate_length = decimate_length
        self.vertex_merge_threshold = vertex_merge_threshold

        with open(self.meta_file, "r") as f:
            self.all_meta = json.load(f)

        self.mesh_dir = self.data_root / "meshes"

        self.mesh_files = []
        self.metadata = {}
        for mesh_name, metadata in self.all_meta.items():
            if self.filter_func is not None and not self.filter_func(
                mesh_name, metadata
            ):
                continue
            mesh_file = self.mesh_dir / metadata["hashed_path"] / f"{mesh_name}.ply"
            self.mesh_files.append(mesh_file)
            self.metadata[mesh_name] = metadata

        if self.overfit:
            self.mesh_files = self.mesh_files[:1] * 1000
        print(f"Found {len(self.mesh_files)} meshes.")

    def __len__(self):
        """Return the number of valid meshes after filtering."""
        return len(self.mesh_files)

    def __getitem__(self, index):
        """Load and preprocess one mesh.

        Returns:
            A dict with three keys:

            * ``data``: tensor dict with ``occ_fine``, ``nvv_fine``,
              ``res_fine``, ``occ_coarse``, ``sdf_coarse2fine``, ``res_coarse``,
              ``face_count``, ``quad_ratio``.
            * ``meta``: dict with the source mesh path.
            * ``mesh``: dict with ``original`` and ``augmented`` trimesh objects
              (used by trainers for visualization only, not model input).
        """
        mesh_path = self.mesh_files[index]

        # Load preprocessed voxel grid
        metadata = self.all_meta[mesh_path.stem]
        remesh_method = metadata.get("remesh_method", "sdf")
        can_augment = (
            metadata["remesh_num_faces"] > 0
            and metadata["remesh_max_dist"] < 1.0
            and remesh_method == "sdf"
        )
        data, original_mesh, augmented_mesh, augment_metadata = process_one_mesh(
            mesh_path,
            res_coarse=self.res_coarse,
            res_fine=self.res_fine,
            pad=self.pad,
            round_verts=self.round_verts,
            decimate_length=self.decimate_length,
            vertex_merge_threshold=self.vertex_merge_threshold,
            augment=can_augment and np.random.rand() < self.augment_ratio,
            augment_strength=self.augment_strength,
            remesh_method=remesh_method,
            cast=False,
            get_metadata=False,
            verbose=False,
        )

        face_count = augment_metadata["decimated_num_faces"]
        quad_ratio = augment_metadata["quad_ratio"]

        data_dict = {
            "occ_fine": torch.from_numpy(data["occ_fine"]).long(),       # (Nf, 3)
            "nvv_fine": torch.from_numpy(data["nvv_fine"]).float(),      # (Nf, 3)
            "res_fine": torch.tensor(data["res_fine"]).long(),
            "occ_coarse": torch.from_numpy(data["occ_coarse"]).long(),   # (Nc, 3)
            "sdf_coarse2fine": torch.from_numpy(data["sdf_coarse2fine"]).float(),  # (Nc, ratio^3)
            "res_coarse": torch.tensor(data["res_coarse"]).long(),
            "face_count": torch.tensor([face_count]).long(),
            "quad_ratio": torch.tensor([quad_ratio]).float(),
        }

        ret = {
            "data": data_dict,
            "meta": {
                "path": str(mesh_path),
            },
            "mesh": {
                "original": original_mesh,
                "augmented": augmented_mesh,
            },
        }

        return ret

    def collate_coords(self, coords_list):
        """Concatenate a list of sparse coord tensors into one batched tensor.

        Prepends a batch-index column to each sample's coords so that samples
        can be distinguished in the concatenated tensor.
        """
        batch_size = len(coords_list)
        all_coords = []
        for b in range(batch_size):
            coords = coords_list[b]
            b_coords = torch.cat(
                [torch.full((coords.shape[0], 1), b, dtype=coords.dtype), coords],
                dim=1,
            )
            all_coords.append(b_coords)
        all_coords = torch.cat(all_coords, dim=0)
        return all_coords

    def collate_feats(self, feats_list):
        """Concatenate a list of sparse feature tensors along dim 0."""
        all_feats = torch.cat(feats_list, dim=0)
        return all_feats

    def collate_fn(self, batch):
        """Collate a list of ``__getitem__`` outputs into a batched dict.

        Sparse ``occ_*`` coords are concatenated with a prepended batch index;
        sparse ``nvv_fine`` / ``sdf_coarse2fine`` features are plain
        concatenated; scalar ``face_count`` / ``quad_ratio`` are stacked;
        ``res_*`` is asserted to be identical across the batch; mesh objects
        are kept as a plain list.
        """
        batch_size = len(batch)
        occ_fine_list = []
        nvv_fine_list = []
        res_fine_list = []
        occ_coarse_list = []
        sdf_coarse2fine_list = []
        res_coarse_list = []
        face_count_list = []
        quad_ratio_list = []
        meta_list = []

        for b in range(batch_size):
            data = batch[b]["data"]
            occ_fine_list.append(data["occ_fine"])
            nvv_fine_list.append(data["nvv_fine"])
            res_fine_list.append(data["res_fine"])
            occ_coarse_list.append(data["occ_coarse"])
            sdf_coarse2fine_list.append(data["sdf_coarse2fine"])
            res_coarse_list.append(data["res_coarse"])
            face_count_list.append(data["face_count"])
            quad_ratio_list.append(data["quad_ratio"])
            meta_list.append(batch[b]["meta"])

        assert all(
            r == res_fine_list[0] for r in res_fine_list
        ), "Mixed res_fine values in a single batch are not supported."
        assert all(
            r == res_coarse_list[0] for r in res_coarse_list
        ), "Mixed res_coarse values in a single batch are not supported."

        occ_fine = self.collate_coords(occ_fine_list)
        nvv_fine = self.collate_feats(nvv_fine_list)
        res_fine = res_fine_list[0]
        occ_coarse = self.collate_coords(occ_coarse_list)
        sdf_coarse2fine = self.collate_feats(sdf_coarse2fine_list)
        res_coarse = res_coarse_list[0]
        face_count = torch.cat(face_count_list, dim=0)
        quad_ratio = torch.cat(quad_ratio_list, dim=0)

        # default collate meta
        meta_dict = default_collate(meta_list)

        mesh_dict = {
            "original": [batch[b]["mesh"]["original"] for b in range(batch_size)],
            "augmented": [batch[b]["mesh"]["augmented"] for b in range(batch_size)],
        }

        ret = {
            "data": {
                "occ_fine": occ_fine,
                "nvv_fine": nvv_fine,
                "res_fine": res_fine,
                "occ_coarse": occ_coarse,
                "sdf_coarse2fine": sdf_coarse2fine,
                "res_coarse": res_coarse,
                "face_count": face_count,
                "quad_ratio": quad_ratio,
            },
            "meta": meta_dict,
            "mesh": mesh_dict,
        }

        return ret

    def to_dir_norm(self, grid):
        """Split a dense NVV grid into ``(grid_dir, grid_norm)``.

        ``grid_dir`` is ``(4, H, W, D)`` = occupancy + unit-length direction;
        ``grid_norm`` is ``(2, H, W, D)`` = occupancy + scalar magnitude.
        """
        if isinstance(grid, torch.Tensor):
            grid = grid.detach().cpu().numpy()
        occ = grid[[0]]
        direction = grid[1:]
        norm = np.linalg.norm(direction, axis=0, keepdims=True)
        dir_normalized = direction / (norm + 1e-8)
        grid_dir = np.concatenate([occ, dir_normalized], axis=0)
        grid_norm = np.concatenate([occ, norm], axis=0)
        return grid_dir, grid_norm

    def to_target_pos(self, grid):
        """Convert a dense NVV grid into an occupancy + target-position grid.

        Computes each voxel's target vertex position by adding the NVV offset
        to the voxel coordinate in normalized ``[0, 1]`` space.
        """
        if isinstance(grid, torch.Tensor):
            grid = grid.detach().cpu().numpy()
        occ = grid[[0]]
        direction = grid[1:]
        res = grid.shape[-1]

        coords = create_meshgrid(resolution=res, normalized=True)  # to [0, 1]
        target_pos = coords + direction  # (3, res, res, res)
        grid_target_pos = np.concatenate([occ, target_pos], axis=0)
        return grid_target_pos

    def visualize_grid(self, grid):
        """Render a 2x2 debug image for a single dense NVV grid.

        Quadrants are: raw occupancy + direction (top-left), direction-only
        (top-right), magnitude (bottom-left), target position (bottom-right).
        """
        color, depth = render_occ_dir_grid(grid)

        grid_dir, grid_norm = self.to_dir_norm(grid)
        grid_norm[1:] *= 5.0
        color_dir, depth_dir = render_occ_dir_grid(grid_dir)
        color_norm, depth_norm = render_occ_color_grid(grid_norm)

        grid_target_pos = self.to_target_pos(grid)
        color_target_pos, depth_target_pos = render_occ_color_grid(grid_target_pos)

        row_dir_norm = np.concatenate([color_dir, color_norm], axis=1)
        for_vec_pos = np.concatenate([color, color_target_pos], axis=1)
        ret_color = np.concatenate([row_dir_norm, for_vec_pos], axis=0)

        return ret_color

    def visualize_batch_nvv(self, batch):
        """Return a list of per-sample NVV debug renders for a collated batch."""
        nvv_fine = batch["data"]["nvv_fine"]  # (N, 3)
        res_fine = batch["data"]["res_fine"].item()
        occ_fine = batch["data"]["occ_fine"]  # (N, 3)
        grid = sparse2dense(occ_fine, nvv_fine, res_fine)  # (B, 4, res, res, res)

        color_list = []
        for b in range(grid.shape[0]):
            grid_b = grid[b].cpu()
            ret_color = self.visualize_grid(grid_b)
            color_list.append(ret_color)

        return color_list

    def visualize_batch_sdf(self, batch):
        """Return a list of per-sample SDF isosurface renders for a collated batch.

        Accepts either a full fine-resolution SDF (``sdf_fine``) or a coarse
        ``sdf_coarse2fine`` that is expanded on-the-fly.
        """
        if "sdf_fine" in batch["data"]:
            sdf_fine = batch["data"]["sdf_fine"].cpu().numpy()  # (N, ratio^3)
            occ_fine = batch["data"]["occ_fine"].cpu().numpy()  # (N, 4)
        else:
            sdf_coarse2fine = batch["data"]["sdf_coarse2fine"]  # (N, ratio^3)
            occ_coarse = batch["data"]["occ_coarse"]  # (N, 4)
            ratio = batch["data"]["res_fine"] // batch["data"]["res_coarse"]

            occ_fine = (
                get_coords_coarse2fine(
                    occ_coarse,
                    ratio=ratio,
                )
                .cpu()
                .numpy()
            )  # (N * ratio^3, 4)
            sdf_fine = sdf_coarse2fine.reshape(-1, 1).cpu().numpy()  # (N * ratio^3, 1)
        sdf_grid = sparse_sdf2dense(
            occ_fine, sdf_fine, int(batch["data"]["res_fine"])
        )  # (B,1,D,H,W)

        color_list = []
        for sdf_b in sdf_grid:
            sdf_color, sdf_depth = render_sdf_grid(sdf_b, level=0.0)
            color_list.append(sdf_color)
        return color_list

    def visualize_batch(self, batch, visualize_type=None):
        """Dispatch to ``visualize_batch_nvv`` or ``visualize_batch_sdf``.

        If ``visualize_type`` is ``None``, the dataset's default
        ``self.visualize_type`` is used.
        """
        if visualize_type is None:
            visualize_type = self.visualize_type
        if visualize_type == "nvv":
            return self.visualize_batch_nvv(batch)
        elif visualize_type == "sdf":
            return self.visualize_batch_sdf(batch)
        else:
            raise ValueError(f"Unsupported visualize_type: {visualize_type}")
