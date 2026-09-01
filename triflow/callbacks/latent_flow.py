# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.

import torch

import triflow.utils.direct3ds2_sparse as direct3ds2_sp_utils
import triflow.utils.trellis_sparse as trellis_sp_utils
from triflow.utils.nvv import (
    coords2pos,
    dirnorm2vector,
    vector2dirnorm,
    vector2pos,
)
from triflow.utils.sparse_voxel import (
    find_coords_indices,
    get_coords_coarse2fine,
)


class GetCondition:
    """Build the conditioning tensor for the latent flow model.

    Encodes the input SDF through a frozen SDF VAE to produce the SDF latent
    consumed by the flow model, and additionally exposes the scalar
    conditioning signals ``face_count`` and ``quad_ratio``.

    Args:
        accelerator: ``accelerate.Accelerator`` used for mixed-precision
            autocast when running the SDF VAE encoder.
        sdf_vae: Frozen SDF VAE. If ``None``, the raw coarse-grid features
            (``occ_coarse`` + ``sdf_coarse2fine``) are wrapped and returned
            unchanged — a debugging / ablation path that skips the encoder.
        sdf_scale: Multiplicative scale applied to SDF values before encoding.
            Must match the scale used when the SDF VAE was trained.
    """

    def __init__(self, accelerator, sdf_vae=None, sdf_scale=128.0):
        self.accelerator = accelerator
        self.sdf_vae = sdf_vae
        self.sdf_scale = sdf_scale

    def get_sdf_latent(self, data):
        """Encode the coarse SDF through the SDF VAE.

        Upsamples the coarse SDF grid to the fine resolution, rescales by
        ``sdf_scale``, masks to the narrow band (``|sdf| <= 1``), and runs
        the frozen VAE encoder under mixed-precision autocast.

        Returns:
            A trellis sparse latent tensor.
        """
        ratio = data["res_fine"] // data["res_coarse"]
        coords, feats = data["occ_coarse"], data["sdf_coarse2fine"]
        feats = feats.clone()

        coords_fine = get_coords_coarse2fine(coords, ratio)
        feats_fine = feats.reshape(-1, 1)
        feats_fine *= self.sdf_scale

        mask = feats_fine[:, 0].abs() <= 1.0
        feats_fine = feats_fine[mask]
        coords_fine = coords_fine[mask]

        model_input = {
            "feats": feats_fine,
            "coords": coords_fine,
        }

        with torch.no_grad(), self.accelerator.autocast():
            latent, _ = self.sdf_vae.encode(model_input, sample_posterior=False)

        return latent

    def __call__(self, batch):
        """Return ``{"sdf_latent": ..., "face_count": (B, 1), "quad_ratio": (B, 1)}``.

        ``sdf_latent`` is a trellis sparse tensor — either the VAE-encoded
        latent or (if ``sdf_vae`` is ``None``) the raw coarse features wrapped
        as a sparse tensor.
        """
        data = batch["data"]
        face_count = data["face_count"].float().unsqueeze(-1)  # (B, 1)
        quad_ratio = data["quad_ratio"].float().unsqueeze(-1)  # (B, 1)

        if self.sdf_vae is not None:
            sdf_latent = self.get_sdf_latent(data)
            sdf_latent = trellis_sp_utils.sparse2sparse_tensor(
                sdf_latent.coords, sdf_latent.feats
            )
        else:
            coords, feats = data["occ_coarse"], data["sdf_coarse2fine"]
            sdf_latent = trellis_sp_utils.sparse2sparse_tensor(coords, feats)

        return {
            "sdf_latent": sdf_latent,
            "face_count": face_count,
            "quad_ratio": quad_ratio,
        }


class PreProcess:
    """Encode NVV data into the flow model's latent space via the NVV VAE encoder.

    Converts NVV vectors into a (direction, magnitude) parameterization,
    applies a square-root transform to the magnitude to equalize its
    distribution, and concatenates auxiliary features (raw vector, target
    position, coord position) for the encoder input. The resulting sparse
    features are then run through the frozen NVV VAE encoder under
    mixed-precision autocast, and the returned latent is wrapped as a
    trellis sparse tensor.

    Args:
        accelerator: ``accelerate.Accelerator`` for autocast.
        nvv_vae: Frozen NVV VAE providing the encoder.
    """

    def __init__(self, accelerator, nvv_vae=None):
        self.accelerator = accelerator
        self.nvv_vae = nvv_vae

    def __call__(self, data):
        """Return the VAE-encoded NVV latent as a trellis sparse tensor."""
        coords, feats = data["occ_fine"], data["nvv_fine"]

        # convert to dirnorm
        coords, dirnorm_feats = vector2dirnorm(coords, feats)
        # convert to target pos
        _, target_pos_feats = vector2pos(coords, feats, data["res_fine"])
        # convert to coords pos
        _, coords_pos_feats = coords2pos(coords, feats, data["res_fine"])

        norm = dirnorm_feats[:, [-1]].clone()
        dirnorm_feats[:, -1].sqrt_()

        feats_merged = torch.cat(
            [dirnorm_feats, norm, feats, target_pos_feats, coords_pos_feats],
            dim=-1,
        )

        model_input = {
            "feats": feats_merged,
            "coords": coords,
        }

        with torch.no_grad(), self.accelerator.autocast():
            latent, _ = self.nvv_vae.encode(model_input, sample_posterior=False)

        latent = trellis_sp_utils.sparse2sparse_tensor(latent.coords, latent.feats)

        return latent


class PostProcess:
    """Decode a latent-space flow sample back to NVV space via the NVV VAE decoder.

    Wraps the flow model's output as a direct3d-s2 sparse tensor, runs the
    frozen NVV VAE decoder conditioned on the fine-resolution occupancy,
    reindexes the decoded features to the fine grid, inverts the square-root
    magnitude transform, and converts the (direction, magnitude)
    representation back to NVV vectors.

    Args:
        accelerator: ``accelerate.Accelerator`` for autocast.
        nvv_vae: Frozen NVV VAE providing the decoder.
    """

    def __init__(self, accelerator, nvv_vae=None):
        self.accelerator = accelerator
        self.nvv_vae = nvv_vae

    def __call__(self, data, model_output):
        """Return a dict with fine-resolution reconstructed NVV vectors."""
        fine_coords = data["occ_fine"]
        model_output = direct3ds2_sp_utils.SparseTensor(
            coords=model_output.coords,
            feats=model_output.feats,
        )

        with torch.no_grad(), self.accelerator.autocast():
            nvv_decoded = self.nvv_vae.decoder(model_output, fine_coords=fine_coords)

        coords, feats = nvv_decoded.coords, nvv_decoded.feats
        feats = feats.clone().to(torch.float32)
        coords_fine = data["occ_fine"]

        indexes = find_coords_indices(coords_fine, coords)
        feats_fine = feats[indexes]

        feats_fine[:, -1].pow_(2)
        coords_fine, vector_feats = dirnorm2vector(coords_fine, feats_fine)

        return {
            "occ_fine": coords_fine,
            "nvv_fine": vector_feats,
            "res_fine": data["res_fine"],
            "res_coarse": data["res_coarse"],
        }
