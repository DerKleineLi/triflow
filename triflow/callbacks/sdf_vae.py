# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.

import torch

from triflow.utils.sparse_voxel import (
    find_coords_indices,
    get_coords_coarse2fine,
)

# SDF values are multiplied by this before VAE encoding so that a
# [-1/128, 1/128] narrow band maps to [-1, 1].
SDF_SCALE = 128.0


class SDFMetric:
    """Evaluation metric for the SDF VAE.

    Computes reconstruction quality on the narrow band (``|sdf| <= 1`` after
    scaling): MSE, L1, and IoU between ground-truth and reconstructed inside
    regions.
    """

    def __call__(self, gt, recon):
        """Return a dict of ``{sdf_mse, sdf_l1, sdf_iou}`` metrics.

        Args:
            gt: Ground-truth data dict with keys ``occ_coarse``,
                ``sdf_coarse2fine``, ``res_fine``, ``res_coarse``.
            recon: Reconstructed data dict with keys ``sdf_fine``, ``occ_fine``.

        Returns:
            dict[str, float]: scalar metric values.
        """
        ratio = gt["res_fine"] // gt["res_coarse"]
        coords, feats = gt["occ_coarse"], gt["sdf_coarse2fine"]

        gt_coords_fine = get_coords_coarse2fine(coords, ratio)
        gt_feats_fine = feats.reshape(-1, 1)

        pred_feats, pred_coords = recon["sdf_fine"], recon["occ_fine"]

        gt2pred_index = find_coords_indices(pred_coords, gt_coords_fine)
        pred_sdf = pred_feats
        gt_sdf = gt_feats_fine[gt2pred_index]

        mask = gt_sdf.abs() <= 1.0
        pred_sdf = pred_sdf[mask]
        gt_sdf = gt_sdf[mask]

        mse = torch.mean((pred_sdf - gt_sdf) ** 2).item()
        l1 = torch.mean(torch.abs(pred_sdf - gt_sdf)).item()

        gt_inside = gt_sdf < 0
        pred_inside = pred_sdf < 0
        iou = (
            torch.sum((gt_inside & pred_inside).float())
            / torch.sum((gt_inside | pred_inside).float()).item()
        )

        return {
            "sdf_mse": mse,
            "sdf_l1": l1,
            "sdf_iou": iou,
        }


class SDFPreProcess:
    """Prepare SDF data for VAE input.

    Upsamples the coarse SDF grid to the fine resolution, rescales by
    ``SDF_SCALE``, and masks to the narrow band (``|sdf| <= 1``).
    """

    def __call__(self, data):
        """Return ``{"feats": fine SDF values, "coords": fine coordinates}``."""
        ratio = data["res_fine"] // data["res_coarse"]
        coords, feats = data["occ_coarse"], data["sdf_coarse2fine"]
        feats = feats.clone()

        coords_fine = get_coords_coarse2fine(coords, ratio)
        feats_fine = feats.reshape(-1, 1)
        feats_fine *= SDF_SCALE

        mask = feats_fine[:, 0].abs() <= 1.0
        feats_fine = feats_fine[mask]
        coords_fine = coords_fine[mask]

        return {
            "feats": feats_fine,
            "coords": coords_fine,
        }


class SDFPostProcess:
    """Invert the preprocessing applied to SDF data.

    Unscales the reconstructed SDF by ``1 / SDF_SCALE`` so downstream consumers
    see SDF values in the same units as the input data.
    """

    def __call__(self, data, model_output):
        """Return a dict with the reconstructed SDF restored to input units."""
        coords_fine, feats_fine = model_output.coords, model_output.feats
        feats_fine = feats_fine.detach().clone()

        feats_fine /= SDF_SCALE

        return {
            "occ_fine": coords_fine,
            "sdf_fine": feats_fine,
            "res_fine": data["res_fine"],
            "res_coarse": data["res_coarse"],
        }


class SDFLoss:
    """L1 reconstruction loss between predicted and target SDF features.

    Aligns predicted and target features by matching their sparse coordinates.
    """

    def __call__(self, model_output, target):
        """Return ``(loss_tensor, {"recon_loss": float})``."""
        target_coords = target["coords"]
        target_feats = target["feats"]

        output_coords = model_output.coords
        output_feats = model_output.feats

        output2target_index = find_coords_indices(target_coords, output_coords)
        output_feats = output_feats[output2target_index]

        loss = torch.mean(torch.abs(output_feats - target_feats))

        return loss, {"recon_loss": loss.item()}
