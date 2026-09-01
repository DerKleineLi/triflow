# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.

from triflow.trainers.trellis_slatflow_trainer import TrellisSlatFlowTrainer


class TrellisLatentSlatFlowTrainer(TrellisSlatFlowTrainer):
    """Flow-matching trainer with two frozen VAEs for the latent-flow stage.

    Extends :class:`TrellisSlatFlowTrainer` with a frozen SDF VAE and a
    frozen NVV VAE that the conditioning and pre/post-process callbacks
    can use to encode inputs and decode outputs. Both VAEs are loaded
    from checkpoints, put into eval mode, and have their parameters
    frozen via :meth:`init_frozen_model`.

    The ``pre_process_data``, ``post_process_recon``, and ``get_cond``
    callbacks are partial factories here (not finished callables like in
    the parent class) — this constructor binds them to the accelerator
    and the frozen VAEs before storing them.

    Args:
        sdf_vae: Un-initialized SDF VAE model.
        sdf_vae_ckpt: Path to the SDF VAE checkpoint.
        nvv_vae: Un-initialized NVV VAE model.
        nvv_vae_ckpt: Path to the NVV VAE checkpoint.
        pre_process_data: Partial factory that accepts ``accelerator``
            and ``nvv_vae``, e.g.
            ``triflow.callbacks.latent_flow.PreProcess``.
        get_cond: Partial factory that accepts ``accelerator`` and
            ``sdf_vae``, e.g. ``latent_flow.GetCondition``.
        post_process_recon: Partial factory that accepts ``accelerator``
            and ``nvv_vae``, e.g. ``latent_flow.PostProcess``.
        **kwargs: Forwarded to :class:`TrellisSlatFlowTrainer`.
    """

    def __init__(
        self,
        sdf_vae,
        sdf_vae_ckpt,
        nvv_vae,
        nvv_vae_ckpt,
        pre_process_data=None,
        get_cond=None,
        post_process_recon=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Load SDF VAE
        sdf_vae = self.init_frozen_model(sdf_vae, sdf_vae_ckpt)
        self.sdf_vae = self.register_model(sdf_vae, name="SDF_VAE")
        # Load NVV VAE
        nvv_vae = self.init_frozen_model(nvv_vae, nvv_vae_ckpt)
        self.nvv_vae = self.register_model(nvv_vae, name="NVV_VAE")
        # callbacks
        self.pre_process_data = pre_process_data(
            accelerator=self.accelerator, nvv_vae=self.nvv_vae
        )
        self.post_process_recon = post_process_recon(
            accelerator=self.accelerator, nvv_vae=self.nvv_vae
        )
        self.get_cond = get_cond(accelerator=self.accelerator, sdf_vae=self.sdf_vae)

    def init_frozen_model(self, model, ckpt_path):
        """Load ``ckpt_path`` into ``model`` and freeze all its parameters.

        Wrapper around :meth:`BaseTrainer.init_model` that additionally
        puts the model in eval mode and disables gradients on every
        parameter — used for the SDF / NVV VAEs that are evaluated during
        latent-flow training but never updated.
        """
        super().init_model(model, ckpt_path)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model
