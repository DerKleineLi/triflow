# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.
#
# Adapted from third_party/Direct3D-S2 (DreamTechAI/Direct3D-S2, MIT):
# SparseNVVEncoder / SparseNVVDecoder / Direct3ds2SparseVAE subclass
# SparseSDFEncoder / SparseSDFDecoder / SparseSDFVAE, and
# SparseReferencedSubdivideBlock3d subclasses SparseSubdivideBlock3d, to carry
# NVV features instead of SDF. See the class docstrings for what changed.

from typing import List, Literal, Optional

import torch
import torch.nn as nn
from direct3d_s2.models.autoencoders.decoder import (
    SparseSDFDecoder,
    SparseSubdivideBlock3d,
)
from direct3d_s2.models.autoencoders.distributions import DiagonalGaussianDistribution
from direct3d_s2.models.autoencoders.encoder import SparseDownBlock3d, SparseSDFEncoder
from direct3d_s2.models.autoencoders.ss_vae import SparseSDFVAE
from direct3d_s2.modules import sparse as sp
from direct3d_s2.modules.sparse import SparseTensor

from triflow.utils.sparse_voxel import find_coords_indices, fine_coords2coarse_coords


class SparseReferencedSubdivide(nn.Module):
    """Upsample a sparse tensor by a factor of 2 on each spatial axis.

    For each input voxel, emits ``2^D`` children that cover its volume in
    the finer grid (``D`` = number of spatial dimensions). Features are
    replicated from the parent voxel. If ``reference_coords`` is supplied,
    the upsampled child voxels are filtered down to the subset that also
    appears in ``reference_coords`` — used when we want the decoder to
    follow a known fine-resolution coordinate layout instead of producing
    the full regular grid.
    """

    def __init__(self):
        super(SparseReferencedSubdivide, self).__init__()

    def forward(self, input: SparseTensor, reference_coords=None) -> SparseTensor:
        """Return an upsampled SparseTensor.

        Args:
            input: Sparse input tensor with coords of shape ``(N, 1 + D)``
                (batch index in column 0).
            reference_coords: Optional target coord set; if given, only
                children whose coord appears in this set are kept.

        Returns:
            A SparseTensor at 2× the input's spatial scale.
        """
        DIM = input.coords.shape[-1] - 1
        # upsample scale=2^DIM
        n_cube = torch.ones([2] * DIM, device=input.device, dtype=torch.int)
        n_coords = torch.nonzero(n_cube)
        n_coords = torch.cat([torch.zeros_like(n_coords[:, :1]), n_coords], dim=-1)
        factor = n_coords.shape[0]
        assert factor == 2**DIM
        new_coords = input.coords.clone()
        new_coords[:, 1:] *= 2
        new_coords = new_coords.unsqueeze(1) + n_coords.unsqueeze(0).to(
            new_coords.dtype
        )

        new_feats = input.feats.unsqueeze(1).expand(
            input.feats.shape[0], factor, *input.feats.shape[1:]
        )

        new_coords = new_coords.flatten(0, 1)
        new_feats = new_feats.flatten(0, 1)
        if reference_coords is not None:
            indices = find_coords_indices(reference_coords, new_coords)
            assert (
                indices >= 0
            ).all(), "Some upsampled coords not found in reference coords."
            new_feats = new_feats[indices]
            new_coords = new_coords[indices]

        out = SparseTensor(new_feats, new_coords, input.shape)
        out._scale = input._scale * 2
        out._spatial_cache = input._spatial_cache
        return out


class SparseReferencedSubdivideBlock3d(SparseSubdivideBlock3d):
    """Upsample-plus-conv block that respects a target coordinate set.

    Like ``SparseSubdivideBlock3d`` but uses
    :class:`SparseReferencedSubdivide` as the upsample step, allowing the
    caller to constrain the output coordinate set via ``reference_coords``.
    Used by :class:`SparseNVVDecoder` so that each upsample level produces
    coords aligned with the target fine grid rather than a regular octree
    expansion.

    Args:
        channels: Input channel count.
        out_channels: Output channel count (defaults to ``channels``).
        use_checkpoint: If ``True``, wraps ``_forward`` in a gradient
            checkpoint to save memory during training.
    """

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__(
            channels=channels,
            out_channels=out_channels,
            use_checkpoint=use_checkpoint,
        )
        self.sub = SparseReferencedSubdivide()

    def _forward(self, x: sp.SparseTensor, reference_coords=None) -> sp.SparseTensor:
        """Apply activation → (referenced) subdivide → output projection."""
        h = self.act_layers(x)
        h = self.sub(h, reference_coords=reference_coords)
        h = self.out_layers(h)
        return h

    def forward(self, x: torch.Tensor, reference_coords=None):
        """Dispatch to ``_forward`` with optional gradient checkpointing."""
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, reference_coords=reference_coords, use_reentrant=False
            )
        else:
            return self._forward(x, reference_coords=reference_coords)


class SparseNVVDecoder(SparseSDFDecoder):
    """Decoder that produces per-voxel NVV vectors on a known target coordinate grid.

    Inherits the SDF decoder's bottleneck + attention blocks and replaces
    the upsample pipeline with a chain of
    :class:`SparseReferencedSubdivideBlock3d` layers, each one upsampling
    the latent by 2× and snapping to the corresponding level of the target
    fine grid. This lets the decoder write features exactly at the voxels
    that the downstream mesh-reconstruction stage needs, instead of having
    to resample from a dense output.

    The channel count is reduced step by step via ``channel_down_factors``;
    the final ``SparseLinear`` projects to ``out_channels`` and the result
    is passed through either a tanh or identity activation.

    Args:
        resolution: Fine-grid spatial resolution.
        model_channels: Inner channel count at the top of the decoder.
        latent_channels: Channel count of the latent input.
        num_blocks: Number of transformer / attention blocks.
        num_heads: Attention heads (forwarded to the parent class).
        num_head_channels: Per-head channel count.
        mlp_ratio: Attention block MLP expansion ratio.
        attn_mode: Attention mode (``"full"``, ``"swin"``, ...).
        window_size: Window size for windowed attention modes.
        pe_mode: Positional encoding mode (``"ape"`` or ``"rope"``).
        use_fp16: Cast parameters to float16 after init.
        use_checkpoint: Enable gradient checkpointing in each block.
        qk_rms_norm: Enable RMSNorm on the QK branch.
        representation_config: Forwarded to the parent class.
        out_channels: Final output channel count.
        chunk_size: Reserved (must be ``1``; chunked decoding is not yet
            implemented for NVV).
        channel_down_factors: Per-upsample divisor applied to
            ``model_channels`` to get each level's channel count.
        out_active_type: Output activation — ``"tanh"`` or ``"identity"``.
    """

    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal[
            "full", "shift_window", "shift_sequence", "shift_order", "swin"
        ] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
        out_channels: int = 1,
        chunk_size: int = 1,
        channel_down_factors: List[int] = [4, 8, 16],
        out_active_type: str = "tanh",
    ):
        super().__init__(
            resolution=resolution,
            model_channels=model_channels,
            latent_channels=latent_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
            representation_config=representation_config,
            out_channels=out_channels,
            chunk_size=chunk_size,
        )

        self.channel_down_factors = channel_down_factors
        self.out_active_type = out_active_type
        self.num_upsamples = len(channel_down_factors)
        assert self.num_upsamples > 0, "At least one upsample block is required."

        upsamples = []
        for i in range(self.num_upsamples):
            in_ch = (
                model_channels // self.channel_down_factors[i - 1]
                if i > 0
                else model_channels
            )
            out_ch = model_channels // self.channel_down_factors[i]
            upsamples.append(
                SparseReferencedSubdivideBlock3d(
                    channels=in_ch,
                    out_channels=out_ch,
                    use_checkpoint=use_checkpoint,
                )
            )

        self.upsample = nn.ModuleList(upsamples)

        self.out_layer = sp.SparseLinear(
            model_channels // self.channel_down_factors[-1], self.out_channels
        )
        if out_active_type == "tanh":
            self.out_active = sp.SparseTanh()
        elif out_active_type == "identity":
            self.out_active = torch.nn.Identity()
        else:
            raise ValueError(f"Unsupported out_active: {out_active_type}")

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        """Zero-init the final output layer so training starts from the identity."""
        super().initialize_weights()
        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(
        self,
        x: sp.SparseTensor,
        fine_coords=None,
        factor: float = None,
        return_feat: bool = False,
    ):
        """Run the bottleneck and the referenced upsample chain.

        If ``fine_coords`` is given, pre-computes the coarser-level target
        coord sets by halving the fine coords repeatedly, then walks the
        upsample stack feeding each block the appropriate level's reference
        coords. This ensures the decoder output's coord set matches
        ``fine_coords`` exactly.

        Args:
            x: Input sparse latent.
            fine_coords: Target fine-resolution coordinates. When set, the
                decoder's output will exactly match this coord set.
            factor: Forwarded to the base decoder.
            return_feat: If ``True``, also return the penultimate feature
                tensor (before the output projection).

        Returns:
            A SparseTensor whose coords align with ``fine_coords`` (or a
            tuple ``(output, penultimate)`` when ``return_feat=True``).
        """
        h = super(SparseSDFDecoder, self).forward(x, factor)

        if fine_coords is not None:
            reference_coords = []
            reference_coords.append(fine_coords)
            current_level_coords = fine_coords
            for _ in range(len(self.upsample) - 1):
                current_level_coords = fine_coords2coarse_coords(
                    current_level_coords, 2
                )
                reference_coords.append(current_level_coords)

        if self.chunk_size <= 1:
            for block in self.upsample:
                ref_coords = reference_coords.pop() if fine_coords is not None else None
                h = block(h, reference_coords=ref_coords)
            h = h.type(x.dtype)

            if return_feat:
                return self.out_active(self.out_layer(h)), h

            h = self.out_layer(h)
            h = self.out_active(h)
            return h
        else:
            raise NotImplementedError("Chunked nvv decoding not implemented yet.")


class SparseNVVEncoder(SparseSDFEncoder):
    """Encoder that maps sparse per-voxel NVV features to a coarse latent.

    Mirrors :class:`SparseNVVDecoder`: starts at a narrow channel count,
    grows through a chain of ``SparseDownBlock3d`` layers, and ends in a
    ``SparseLinear`` that outputs ``2 * latent_channels`` (mean and
    log-variance of the posterior).

    Args are analogous to :class:`SparseNVVDecoder`, with
    ``channel_down_factors`` applied in reverse order to grow from narrow
    to wide.
    """

    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal[
            "full", "shift_window", "shift_sequence", "shift_order", "swin"
        ] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        channel_down_factors: List[int] = [4, 8, 16],
    ):
        super().__init__(
            resolution=resolution,
            latent_channels=latent_channels,
            in_channels=model_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        self.channel_down_factors = channel_down_factors
        self.num_downsamples = len(channel_down_factors)
        assert self.num_downsamples > 0, "At least one downsample block is required."

        self.input_layer1 = sp.SparseLinear(
            in_channels, model_channels // self.channel_down_factors[-1]
        )

        downsamples = []
        for i in range(self.num_downsamples):
            in_ch = model_channels // self.channel_down_factors[-i - 1]
            out_ch = (
                model_channels // self.channel_down_factors[-i - 2]
                if i < self.num_downsamples - 1
                else model_channels
            )
            downsamples.append(
                SparseDownBlock3d(
                    channels=in_ch,
                    out_channels=out_ch,
                    use_checkpoint=use_checkpoint,
                )
            )

        self.downsample = nn.ModuleList(downsamples)

        self.resolution = resolution
        self.out_layer = sp.SparseLinear(model_channels, 2 * latent_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()


class Direct3ds2SparseVAE(SparseSDFVAE):
    """Sparse VAE used for both the SDF and NVV stages of the TriFlow pipeline.

    Wraps the ``direct3d_s2`` ``SparseSDFVAE`` with two toggles:

    * ``use_nvv_encoder``: replaces the default SDF encoder with
      :class:`SparseNVVEncoder`.
    * ``use_nvv_decoder``: replaces the default SDF decoder with
      :class:`SparseNVVDecoder`, which takes ``fine_coords`` at forward
      time to constrain the output coordinate set.

    Both toggles are independent: the SDF VAE stage uses neither (pure
    inherited pipeline), while the NVV VAE stage uses both.

    Args:
        in_channels: Number of input feature channels per voxel.
        embed_dim: Latent channel count.
        resolution: Fine-grid spatial resolution.
        model_channels_encoder: Encoder inner channel count.
        num_blocks_encoder: Encoder transformer block count.
        num_heads_encoder: Encoder attention head count.
        num_head_channels_encoder: Per-head channel count in the encoder.
        model_channels_decoder: Decoder inner channel count.
        num_blocks_decoder: Decoder transformer block count.
        num_heads_decoder: Decoder attention head count.
        num_head_channels_decoder: Per-head channel count in the decoder.
        out_channels: Decoder output channel count.
        use_fp16: Cast parameters to float16 after init.
        use_checkpoint: Enable gradient checkpointing in each block.
        chunk_size: Reserved (see :class:`SparseNVVDecoder`).
        latents_scale: Multiplicative normalization of the latent.
        latents_shift: Additive normalization of the latent.
        out_active: Output activation — ``"tanh"`` or ``"identity"``.
        use_nvv_decoder: If ``True``, swap in :class:`SparseNVVDecoder`.
        decoder_channel_down_factors: Per-level divisors for the NVV
            decoder (ignored when ``use_nvv_decoder=False``).
        use_nvv_encoder: If ``True``, swap in :class:`SparseNVVEncoder`.
        encoder_channel_down_factors: Per-level divisors for the NVV
            encoder (ignored when ``use_nvv_encoder=False``).
        attn_mode: Attention mode forwarded to both encoder and decoder.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        embed_dim: int = 0,
        resolution: int = 64,
        model_channels_encoder: int = 512,
        num_blocks_encoder: int = 4,
        num_heads_encoder: int = 8,
        num_head_channels_encoder: int = 64,
        model_channels_decoder: int = 512,
        num_blocks_decoder: int = 4,
        num_heads_decoder: int = 8,
        num_head_channels_decoder: int = 64,
        out_channels: int = 1,
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        chunk_size: int = 1,
        latents_scale: float = 1.0,
        latents_shift: float = 0.0,
        out_active: str = "tanh",
        use_nvv_decoder: bool = False,
        decoder_channel_down_factors: List[int] = [4, 8, 16],
        use_nvv_encoder: bool = False,
        encoder_channel_down_factors: List[int] = [4, 8, 16],
        attn_mode: Literal[
            "full", "shift_window", "shift_sequence", "shift_order", "swin"
        ] = "swin",
    ):
        super().__init__(
            embed_dim=embed_dim,
            resolution=resolution,
            model_channels_encoder=model_channels_encoder,
            num_blocks_encoder=num_blocks_encoder,
            num_heads_encoder=num_heads_encoder,
            num_head_channels_encoder=num_head_channels_encoder,
            model_channels_decoder=model_channels_decoder,
            num_blocks_decoder=num_blocks_decoder,
            num_heads_decoder=num_heads_decoder,
            num_head_channels_decoder=num_head_channels_decoder,
            out_channels=out_channels,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            chunk_size=chunk_size,
            latents_scale=latents_scale,
            latents_shift=latents_shift,
        )
        self.use_fp16 = use_fp16
        self.out_active = out_active
        self.use_nvv_decoder = use_nvv_decoder
        self.channel_down_factors = decoder_channel_down_factors
        self.in_channels = in_channels
        self.use_nvv_encoder = use_nvv_encoder
        self.encoder_channel_down_factors = encoder_channel_down_factors
        self.attn_mode = attn_mode

        if self.use_nvv_encoder:
            self.encoder = SparseNVVEncoder(
                resolution=resolution,
                in_channels=in_channels,
                model_channels=model_channels_encoder,
                latent_channels=embed_dim,
                num_blocks=num_blocks_encoder,
                num_heads=num_heads_encoder,
                num_head_channels=num_head_channels_encoder,
                use_fp16=use_fp16,
                use_checkpoint=use_checkpoint,
                channel_down_factors=encoder_channel_down_factors,
                attn_mode=attn_mode,
            )
        else:
            # handle in channels
            if in_channels != 1:
                self.encoder.input_layer1 = sp.SparseLinear(
                    in_channels, model_channels_encoder // 16
                )

        if self.use_nvv_decoder:
            self.decoder = SparseNVVDecoder(
                resolution=resolution,
                model_channels=model_channels_decoder,
                latent_channels=embed_dim,
                num_blocks=num_blocks_decoder,
                num_heads=num_heads_decoder,
                num_head_channels=num_head_channels_decoder,
                out_channels=out_channels,
                use_fp16=use_fp16,
                use_checkpoint=use_checkpoint,
                chunk_size=chunk_size,
                channel_down_factors=decoder_channel_down_factors,
                out_active_type=out_active,
                attn_mode=attn_mode,
            )
        else:
            # handle out activation
            if out_active == "tanh":
                pass
            elif out_active == "identity":
                self.decoder.out_active = torch.nn.Identity()
            else:
                raise ValueError(f"Unsupported out_active: {out_active}")

        self.encoder.initialize_weights()

    def forward(self, batch, sample_posterior: bool = True):
        """Encode, sample from the posterior, and decode.

        Args:
            batch: Dict with keys ``feats`` (``(N, in_channels)``) and
                ``coords`` (``(N, 1 + D)`` int with batch index in column 0).
            sample_posterior: If ``True``, the latent is sampled from the
                posterior (training behavior) and ``posterior`` is also
                returned; if ``False``, the latent's mode is used.

        Returns:
            ``reconst_x`` or ``(reconst_x, posterior)``.
        """
        if self.use_fp16:
            batch["feats"] = batch["feats"].half()

        fine_coords = batch["coords"].clone().int()

        z, posterior = self.encode(batch, sample_posterior=sample_posterior)

        if self.use_nvv_decoder:
            reconst_x = self.decoder(z, fine_coords=fine_coords)
        else:
            reconst_x = self.decoder(z)

        if sample_posterior:
            ret = reconst_x, posterior
        else:
            ret = reconst_x
        return ret

    def encode(self, batch, sample_posterior: bool = True):
        """Encode a batch and return ``(latent_sparse_tensor, posterior)``.

        If ``sample_posterior`` is ``True``, the returned sparse tensor
        wraps a sampled latent; otherwise it wraps the posterior mode.
        """
        feats, coords = batch["feats"].clone(), batch["coords"].clone()
        if feats.ndim == 1:
            feats = feats.unsqueeze(-1)
        coords = coords.int()

        x = sp.SparseTensor(feats, coords)
        h = self.encoder(x, batch.get("factor", None))
        posterior = DiagonalGaussianDistribution(h.feats, feat_dim=1)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        z = h.replace(z)

        return z, posterior
