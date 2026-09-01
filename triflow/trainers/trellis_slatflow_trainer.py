# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.
#
# Adapted from third_party/TRELLIS: trellis/trainers/flow_matching/flow_matching.py
# (microsoft/TRELLIS, MIT). The diffuse / get_v / sample_t helpers are bound
# directly from TRELLIS's FlowMatchingTrainer so the noise schedule and target
# velocity match upstream exactly; the surrounding trainer is our own.

import inspect
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from trellis.trainers.flow_matching.flow_matching import FlowMatchingTrainer

from triflow.utils.sampling import FlowEulerSampler
from triflow.trainers.base import BaseTrainer


class TrellisSlatFlowTrainer(BaseTrainer):
    """Trainer for the TRELLIS flow-matching stage on SLat tensors.

    Extends :class:`BaseTrainer` with:

    * a flow-matching forward pass that diffuses a clean batch, computes
      the target velocity, and MSE/L1-matches the model output,
    * conditional sampling through a ``get_cond`` callback,
    * a periodic sample callback that uses a ``FlowEulerSampler`` to run
      full flow-matching inference on the validation set.

    Three helpers from TRELLIS's own ``FlowMatchingTrainer``
    (``diffuse``, ``get_v``, ``sample_t``) are bound directly on
    ``__init__`` so the trainer inherits TRELLIS's exact noise schedule
    and target velocity without subclassing.

    Args:
        t_schedule: TRELLIS timestep schedule config (forwarded to
            ``sample_t``).
        sigma_min: Minimum flow-matching noise sigma; passed to the Euler
            sampler.
        pre_process_data: Callable applied to each batch to build the
            sparse tensor fed to the flow model.
        get_cond: Callable that builds the conditioning tensor from a
            batch. When ``None``, an unconditional zero tensor of shape
            ``(B, 1, 768)`` is used (ablation path).
        post_process_recon: Callable applied to the reconstructed sample
            before metrics / visualization.
        loss_func: ``"mse"``, ``"l1"``, or a custom callable
            ``(x, model_output, target) -> loss_tensor``.
        metrics_func: Callable returning a scalar metric dict during
            sampling.
        sample_every: Global-step interval for running :meth:`sample`.
        inference_step_size: Reserved; currently unused by the Euler
            sampler path (which uses a fixed 50-step schedule).
        **kwargs: Forwarded to :class:`BaseTrainer`.
    """

    def __init__(
        self,
        t_schedule,
        sigma_min: float = 1e-5,
        pre_process_data=None,
        get_cond=None,
        post_process_recon=None,
        loss_func="mse",
        metrics_func=None,
        sample_every=5000,
        inference_step_size=0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.t_schedule = t_schedule
        self.sigma_min = sigma_min
        self.sample_every = sample_every
        self.inference_step_size = inference_step_size
        self.pre_process_data = pre_process_data
        self.get_cond = get_cond
        self.post_process_recon = post_process_recon
        self.loss_func = loss_func
        self.metrics_func = metrics_func

        # Register sample callback
        self.register_callback(self.sample, self.sample_every)

        # Borrow helpers from FlowMatchingTrainer so we inherit TRELLIS's
        # exact noise schedule / target velocity without subclassing.
        self.diffuse = FlowMatchingTrainer.diffuse.__get__(self, self.__class__)
        self.get_v = FlowMatchingTrainer.get_v.__get__(self, self.__class__)
        self.sample_t = FlowMatchingTrainer.sample_t.__get__(self, self.__class__)

    def get_sampler(self, **kwargs):
        """Return a fresh ``FlowEulerSampler`` with this trainer's ``sigma_min``."""
        return FlowEulerSampler(self.sigma_min)

    def run_one_step(self, batch):
        """Run one flow-matching training step and return the MSE / L1 loss.

        Builds the clean sparse tensor, samples noise and timesteps,
        diffuses, computes the target velocity, runs the model, and
        matches model output to target velocity. The ``t * 1000.0``
        scaling inside the model call maps TRELLIS-convention timesteps
        in ``[0, 1]`` into the ``[0, 1000]`` range the flow model's
        timestep embedder was trained for.
        """
        x = batch["data"]
        if self.get_cond is not None:
            c = self.get_cond(batch)
        else:
            c = torch.zeros((x.shape[0], 1, 768), device=x.device)  # dummy cond

        # 1. build sparse tensor
        if self.pre_process_data is not None:
            clean_images = self.pre_process_data(x)
        else:
            clean_images = x

        # 2. do diffusion's forward
        # Sample noise that we'll add to the images
        noise = clean_images.replace(torch.randn_like(clean_images.feats))
        t = self.sample_t(clean_images.shape[0]).to(clean_images.device).float()

        x_t = self.diffuse(clean_images, t, noise=noise)
        u_t = self.get_v(clean_images, noise, t)
        if self.model.training:
            model_output = self.model(x_t, t * 1000.0, cond=c)
        else:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            model_output = unwrapped_model(x_t, t * 1000.0, cond=c)

        if self.loss_func == "mse":
            loss = F.mse_loss(model_output.feats, u_t.feats)
        elif self.loss_func == "l1":
            loss = F.l1_loss(model_output.feats, u_t.feats)
        elif callable(self.loss_func):
            loss = self.loss_func(x, model_output, u_t)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_func}")

        return loss

    @torch.no_grad()
    def sample(self, num_samples=None, output_file=None):
        """Run full flow-matching inference on the validation set and log samples.

        Uses a fixed-seed generator for noise so visualized samples are
        reproducible across steps. Each batch is:

        1. pre-processed to sparse form,
        2. noise-initialized,
        3. conditioned via ``get_cond``,
        4. sampled via :meth:`sample_pipeline` (50 Euler steps),
        5. optionally post-processed,
        6. scored via ``metrics_func``,
        7. rendered and uploaded.
        """
        num_samples = num_samples if num_samples is not None else self.num_samples
        output_file = (
            output_file
            if output_file is not None
            else os.path.join(
                self.run_dir, "samples", f"samples_step{self.global_step:06d}.pt"
            )
        )

        self.model.eval()
        unet = self.accelerator.unwrap_model(self.model)  # unwrap WrappedModel
        generator = torch.Generator(device=self.accelerator.device).manual_seed(0)

        samples = []
        for batch in tqdm(self.val_loader_unshared, desc="Sampling"):
            x = batch["data"]

            # 1. build sparse tensor
            if self.pre_process_data is not None:
                clean_images = self.pre_process_data(x)
            else:
                clean_images = x

            noise = clean_images.replace(
                torch.randn(
                    clean_images.feats.size(),
                    dtype=clean_images.feats.dtype,
                    layout=clean_images.feats.layout,
                    device=clean_images.feats.device,
                    generator=generator,
                )
            )

            if self.get_cond is not None:
                c = self.get_cond(batch)
            else:
                c = torch.zeros((noise.shape[0], 1, 768), device=noise.device)

            recon = self.sample_pipeline(unet, noise, c)

            if self.post_process_recon is not None:
                # check howmany arguments the function has
                sig = inspect.signature(self.post_process_recon)
                if len(sig.parameters) == 1:
                    recon = self.post_process_recon(recon)
                else:
                    recon = self.post_process_recon(x, recon)

            if self.metrics_func is not None:
                metrics_dict = self.metrics_func(x, recon)
                for k, v in metrics_dict.items():
                    self.logger.log_scalars(**{f"sample_{k}": v})

            samples.append(recon)
            if len(samples) * self.batch_size_per_gpu >= num_samples:
                break

        batches = [{"data": s} for s in samples]
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        torch.save(batches, output_file)
        for batch in batches:
            color_list = self.dataset.visualize_batch(batch)
            for color in color_list:
                self.logger.log_images(
                    sampled=torch.from_numpy(color.copy()).permute(2, 0, 1)
                )

        self.model.train()
        self.upload_log()

    def sample_pipeline(self, velocity_model, noise, cond):
        """Run the flow-matching Euler sampler end-to-end.

        Args:
            velocity_model: The unwrapped flow model.
            noise: Starting noise sparse tensor.
            cond: Conditioning tensor (dense or sparse, produced by
                ``get_cond``).

        Returns:
            The final sampled sparse tensor (50 Euler steps).
        """
        sampler = self.get_sampler()

        res = sampler.sample(
            velocity_model,
            noise=noise,
            cond=cond,
            steps=50,
            verbose=False,
        )

        return res.samples
