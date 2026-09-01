# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.

import torch
import torch.nn.functional as F

from triflow.utils.nvv import (
    coords2pos,
    dirnorm2vector,
    vector2dirnorm,
    vector2nid,
    vector2pos,
)
from triflow.utils.sparse_voxel import find_coords_indices


class Metrics:
    """Evaluation metrics for the NVV VAE.

    Compares ground-truth and reconstructed NVV (voxelized nearest-vertex
    vectors) along multiple axes: raw vector L2, direction cosine similarity,
    magnitude L1, target-position distance, and neighbor-index accuracy.
    """

    def __call__(self, gt, sampled) -> dict:
        """Return a dict of NVV reconstruction metrics.

        Args:
            gt: Ground-truth dict with keys ``occ_fine``, ``nvv_fine``, ``res_fine``.
            sampled: Reconstructed dict with the same keys.

        Returns:
            dict[str, float]: scalar metric values keyed by ``vector_l2``,
            ``direction_cosine``, ``magnitude_l1``, ``position_dist``,
            ``neighbor_index_acc``.
        """
        resolution = gt["res_fine"].item()
        coords_gt, feats_gt = gt["occ_fine"], gt["nvv_fine"]
        coords_recon, feats_recon = sampled["occ_fine"], sampled["nvv_fine"]

        # vector l2 loss
        diff = feats_gt - feats_recon
        norm = torch.norm(diff, dim=-1)
        vector_l2 = norm.mean().item()

        # direction cosine similarity
        _, dir_norm_gt = vector2dirnorm(coords_gt, feats_gt)
        _, dir_norm_recon = vector2dirnorm(coords_recon, feats_recon)
        dir_gt = dir_norm_gt[:, :-1]
        dir_recon = dir_norm_recon[:, :-1]
        cos = F.cosine_similarity(dir_gt, dir_recon, dim=-1)
        direction_cosine = cos.mean().item()

        # magnitude l1 loss
        mag_gt = dir_norm_gt[:, -1]
        mag_recon = dir_norm_recon[:, -1]
        mag_l1 = F.l1_loss(mag_gt, mag_recon).item()

        # target position distance
        _, pos_gt = vector2pos(coords_gt, feats_gt, resolution=resolution)
        _, pos_recon = vector2pos(coords_recon, feats_recon, resolution=resolution)
        pos_diff = pos_gt - pos_recon
        pos_dist = torch.norm(pos_diff, dim=-1).mean().item()

        # neighbor index accuracy
        _, ni_gt = vector2nid(coords_gt, feats_gt, resolution=resolution)
        _, ni_recon = vector2nid(coords_recon, feats_recon, resolution=resolution)
        ni_correct = (ni_gt - ni_recon).abs().sum(dim=-1) < 1e-6
        ni_accuracy = ni_correct.float().mean().item()

        return {
            "vector_l2": vector_l2,
            "direction_cosine": direction_cosine,
            "magnitude_l1": mag_l1,
            "position_dist": pos_dist,
            "neighbor_index_acc": ni_accuracy,
        }


class NVVPreProcess:
    """Prepare NVV data for the NVV VAE encoder.

    Converts NVV vectors into a (direction, magnitude) parameterization,
    applies a square-root transform to the magnitude to equalize its
    distribution, and concatenates auxiliary features (raw vector, target
    position, coord position) for the encoder input.
    """

    def __call__(self, data):
        """Return ``{"feats": concatenated features, "coords": sparse coords}``."""
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

        return {
            "feats": feats_merged,
            "coords": coords,
        }


class NVVPostProcess:
    """Invert the NVV VAE encoder preprocessing.

    Reverses the square-root transform on the magnitude and converts
    (direction, magnitude) back to NVV vectors.
    """

    def __call__(self, data, model_output):
        """Return a dict with reconstructed NVV in the original parameterization."""
        coords, feats = model_output.coords, model_output.feats
        feats = feats.clone()
        occ_fine = data["occ_fine"]
        feats_index = find_coords_indices(occ_fine, coords)
        feats = feats[feats_index]

        feats[:, -1].pow_(2)
        occ_fine, vector_feats = dirnorm2vector(occ_fine, feats, data["res_fine"])

        return {
            "occ_fine": occ_fine,
            "nvv_fine": vector_feats,
            "res_fine": data["res_fine"],
            "res_coarse": data["res_coarse"],
        }


class NVVLoss:
    """L1 reconstruction loss on the (direction, magnitude) NVV representation,
    plus an optional KL divergence penalty on the encoder posterior.

    Args:
        kl_weight: Scalar weight applied to the KL loss term (``0`` disables).
    """

    def __init__(self, kl_weight=0.0001):
        self.kl_weight = kl_weight

    def __call__(self, model_output, target, posterior=None):
        """Return ``(total_loss, loss_dict)``.

        The target is sliced to the first 4 channels (direction + magnitude).
        If ``posterior`` is given, its KL is added to the reconstruction loss
        scaled by ``kl_weight``.
        """
        target_coords = target["coords"]
        target_feats = target["feats"][:, :4]

        output_coords = model_output.coords
        output_feats = model_output.feats

        output2target_index = find_coords_indices(target_coords, output_coords)
        output_feats = output_feats[output2target_index]

        loss = torch.mean(torch.abs(output_feats - target_feats))
        loss_dict = {"recon_loss": loss.item()}

        if posterior is not None:
            kl_loss = posterior.kl(dims=(0,)).mean()
            loss_dict["kl_loss"] = kl_loss.item()
            loss = loss + self.kl_weight * kl_loss

        return loss, loss_dict


