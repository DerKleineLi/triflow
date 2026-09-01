# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.
#
# Builds on third_party/TRELLIS (microsoft/TRELLIS, MIT): SLatFlowModel,
# TimestepEmbedder (trellis/models/sparse_structure_flow.py) and
# AbsolutePositionEmbedder (trellis/modules/transformer/blocks.py) are imported
# and composed here. The shape/face-count/quad-ratio conditioning built on top
# of them is our own.

from typing import Optional

import torch
import torch.nn as nn
from trellis.models import SLatFlowModel
from trellis.models.sparse_structure_flow import TimestepEmbedder
from trellis.modules import sparse as sp
from trellis.modules.transformer.blocks import AbsolutePositionEmbedder


class ScalarEmbedder(nn.Module):
    """Scalar conditioning embedder built on top of TRELLIS's ``TimestepEmbedder``.

    Acts as a generic "scalar → hidden vector" projection. Subclasses
    override :meth:`scale` to preprocess the raw scalar into the
    ``[0, 1000]`` range expected by ``TimestepEmbedder``.

    Args:
        hidden_size: Output embedding dimension.
    """

    def __init__(
        self,
        hidden_size,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.scalar_embedder = TimestepEmbedder(hidden_size)

        self.initialize_weights()

    def scale(self, x):
        """Preprocess ``x`` before passing it to the timestep embedder.

        Identity by default; subclasses override this to map their raw
        scalar range into the ``[0, 1000]`` range that the underlying
        ``TimestepEmbedder`` was trained to consume.
        """
        return x

    def forward(self, x):
        """Return an ``(B, hidden_size)`` embedding of the scalar input ``x``."""
        x = self.scale(x)
        x_emb = self.scalar_embedder(x)
        return x_emb

    def initialize_weights(self) -> None:
        """Xavier-init linear layers and normal-init the timestep MLP weights."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.scalar_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.scalar_embedder.mlp[2].weight, std=0.02)


class FaceCountEmbedder(ScalarEmbedder):
    """Logarithmic face-count embedder.

    Applies ``scale_factor * log(face_count + 1)`` where ``scale_factor`` is
    chosen so that ``max_face_count`` maps to ~1000 — the range the
    underlying ``TimestepEmbedder`` was trained with. The log compresses
    the very wide dynamic range of face counts (a few hundred to ~1M) so
    that training is stable across different mesh complexities.

    Args:
        max_face_count: Face count that should map to the top of the
            embedder's input range. Counts beyond this still embed
            correctly but lose resolution.
    """

    def __init__(
        self,
        max_face_count: int = 1e6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_face_count = max_face_count
        scale_factor = 1000.0 / torch.log(
            torch.tensor(self.max_face_count + 1, dtype=torch.float32)
        )
        self.register_buffer("scale_factor", scale_factor)

    def scale(self, x):
        return self.scale_factor * torch.log(x + 1)


class QuadRatioEmbedder(ScalarEmbedder):
    """Quad-ratio embedder.

    Multiplies the input by 1000 so that a ratio in ``[0, 1]`` maps into
    the ``[0, 1000]`` range that ``TimestepEmbedder`` expects.
    """

    def scale(self, x):
        return x * 1000.0


class SparseConditionEmbedder(nn.Module):
    """Per-voxel sparse conditioning embedder.

    Linearly projects the input SDF latent's features to the flow model's
    conditioning channel count and adds a sinusoidal positional embedding
    based on the voxel coordinates. The result is a SparseTensor with the
    same coord layout as the input.

    Args:
        feat_dim: Number of feature channels in the incoming SDF latent.
        hidden_size: Target conditioning embedding dimension.
    """

    def __init__(
        self,
        feat_dim,
        hidden_size,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_size = hidden_size

        self.dtype = torch.float32

        self.cond_proj = sp.SparseLinear(feat_dim, hidden_size)
        self.pos_embedder_cond = AbsolutePositionEmbedder(hidden_size, in_channels=3)

        self.initialize_weights()

    def forward(
        self,
        x: sp.SparseTensor,
    ):
        """Project ``x.feats`` to the conditioning space and add position embeddings."""
        cond = self.cond_proj(x)
        cond = cond + self.pos_embedder_cond(x.coords[:, 1:]).type(self.dtype)
        return cond

    def initialize_weights(self) -> None:
        """Xavier-init linear layers."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)


class ShapeConditionedSlatFlowModel(nn.Module):
    """Flow-matching model for NVV latents conditioned on shape, face count, quad ratio.

    Wraps TRELLIS's ``SLatFlowModel`` with a small conditioning pipeline
    that combines three optional signals into a single per-voxel
    conditioning tensor passed to the flow model:

    * **SDF latent** (sparse, voxelized): encoded via
      :class:`SparseConditionEmbedder`.
    * **Face count** (scalar per sample): encoded via
      :class:`FaceCountEmbedder`.
    * **Quad ratio** (scalar per sample): encoded via
      :class:`QuadRatioEmbedder`.

    When both sparse and dense conditions are used, they are combined with
    a per-voxel linear projection + LayerNorm so the flow model always
    receives a single sparse conditioning tensor. If only one modality is
    enabled, that one is returned directly.

    Args:
        slat_flow_config: Kwargs forwarded to the underlying
            ``SLatFlowModel`` constructor. Must include a
            ``cond_channels`` entry.
        cond_face_count: Enable face-count conditioning.
        cond_quad_ratio: Enable quad-ratio conditioning.
        cond_sdf_latent: Enable SDF-latent conditioning.
        sdf_feat_dim: Feature dimension of the incoming SDF latent (only
            used when ``cond_sdf_latent=True``).
    """

    def __init__(
        self,
        slat_flow_config,
        cond_face_count=False,
        cond_quad_ratio=False,
        cond_sdf_latent=False,
        sdf_feat_dim=8,
    ):
        super().__init__()

        self.flow_model_config = slat_flow_config
        self.cond_face_count = cond_face_count
        self.cond_quad_ratio = cond_quad_ratio
        self.cond_sdf_latent = cond_sdf_latent

        self.flow_model = SLatFlowModel(**slat_flow_config)

        self.out_proj_input_dim = 0
        if self.cond_face_count:
            self.face_count_embedder = FaceCountEmbedder(
                hidden_size=slat_flow_config["cond_channels"]
            )
            self.out_proj_input_dim += slat_flow_config["cond_channels"]
        if self.cond_quad_ratio:
            self.quad_ratio_embedder = QuadRatioEmbedder(
                hidden_size=slat_flow_config["cond_channels"]
            )
            self.out_proj_input_dim += slat_flow_config["cond_channels"]
        if self.cond_sdf_latent:
            self.sdf_latent_embedder = SparseConditionEmbedder(
                feat_dim=sdf_feat_dim,
                hidden_size=slat_flow_config["cond_channels"],
            )
            self.out_proj_input_dim += slat_flow_config["cond_channels"]

        self.has_out_proj = (
            self.cond_sdf_latent
            and self.out_proj_input_dim > slat_flow_config["cond_channels"]
        )

        if self.has_out_proj:
            self.cond_out_proj = nn.Linear(
                self.out_proj_input_dim,
                slat_flow_config["cond_channels"],
            )
            self.cond_out_norm = nn.LayerNorm(slat_flow_config["cond_channels"])

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Xavier-init every linear submodule."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def get_condition(
        self,
        x_t,
        sdf_latent: Optional[sp.SparseTensor] = None,
        face_count: Optional[torch.Tensor] = None,
        quad_ratio: Optional[torch.Tensor] = None,
    ):
        """Combine the enabled conditioning signals into a single tensor.

        Returns one of:

        * The merged sparse tensor (SDF latent features concatenated with
          broadcast dense features, projected through ``cond_out_proj``
          and LayerNorm) — used when both sparse and dense conditions are
          active.
        * The dense tensor alone — used when only face count and/or quad
          ratio are enabled.
        * The sparse tensor alone — used when only the SDF latent is
          enabled.
        * A zero tensor of shape ``(B, 1, cond_channels)`` — used when
          every branch is disabled (unconditional fallback used for
          ablation / debugging).

        Args:
            x_t: Input noisy latent. Only its batch size and device are
                used, to build the unconditional fallback.
            sdf_latent: Sparse SDF latent conditioning tensor. Required if
                ``cond_sdf_latent=True``.
            face_count: Scalar face-count conditioning. Required if
                ``cond_face_count=True``.
            quad_ratio: Scalar quad-ratio conditioning. Required if
                ``cond_quad_ratio=True``.
        """
        # --- SDF latent branch (produces a sparse condition tensor) ---
        sparse_cond = None
        if self.cond_sdf_latent:
            assert (
                sdf_latent is not None
            ), "SDF latent condition is enabled but no SDF latent is provided."
            sdf_cond = self.sdf_latent_embedder(sdf_latent)  # (N, C)
            sparse_cond = sdf_cond

        # --- Scalar branches (produce a dense condition tensor) ---
        dense_cond = None
        if self.cond_face_count:
            assert (
                face_count is not None
            ), "Face count condition is enabled but no face count is provided."
            face_count_cond = self.face_count_embedder(face_count)
            dense_cond = face_count_cond

        if self.cond_quad_ratio:
            assert (
                quad_ratio is not None
            ), "Quad ratio condition is enabled but no quad ratio is provided."
            quad_ratio_cond = self.quad_ratio_embedder(quad_ratio)
            if dense_cond is None:
                dense_cond = quad_ratio_cond
            else:
                dense_cond = torch.cat(
                    [dense_cond, quad_ratio_cond], dim=1
                )  # (B, n, C)

        # --- Combine: merge sparse + dense if both branches fired ---
        if self.has_out_proj:
            assert (
                sparse_cond is not None
            ), "Output projection is enabled but no sparse condition is provided."
            assert (
                dense_cond is not None
            ), "Output projection is enabled but no dense condition is provided."
            concat_cond = [sparse_cond.feats]
            N = sparse_cond.feats.shape[0]

            B, n, C = dense_cond.shape
            dense_cond = dense_cond.view(B, -1)  # (B, n*C)

            batch_indices = sparse_cond.coords[:, 0]  # (N,)
            dense_cond_expanded = dense_cond[batch_indices, :]  # (N, n*C)
            concat_cond.append(dense_cond_expanded)
            concat_cond_tensor = torch.cat(concat_cond, dim=-1).contiguous()  # (N, D)
            projected_cond = self.cond_out_proj(concat_cond_tensor)  # (N, C)
            projected_cond = self.cond_out_norm(projected_cond)
            sparse_cond = sparse_cond.replace(projected_cond)
            return sparse_cond

        # --- Fall-throughs: one branch only, or nothing (unconditional) ---
        if dense_cond is not None:
            return dense_cond
        if sparse_cond is not None:
            return sparse_cond

        empty_cond = torch.zeros(
            (x_t.shape[0], 1, self.flow_model_config["cond_channels"]),
            device=x_t.device,
        )
        return empty_cond

    def forward(self, x_t, t, cond, *args, **kwargs):
        """Build the conditioning tensor and run the underlying flow model.

        Args:
            x_t: Noisy input latent.
            t: Timestep tensor (per-sample).
            cond: Dict forwarded to :meth:`get_condition`. Keys may
                include ``sdf_latent``, ``face_count``, ``quad_ratio``.
            *args, **kwargs: Forwarded to the base flow model.
        """
        condition = self.get_condition(x_t, **cond)
        out = self.flow_model(x_t, t, cond=condition, *args, **kwargs)
        return out
