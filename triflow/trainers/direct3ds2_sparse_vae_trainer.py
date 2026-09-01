# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.

import os

import torch
from tqdm import tqdm

from triflow.trainers.base import BaseTrainer


class Direct3ds2SparseVAETrainer(BaseTrainer):
    """Trainer for the SDF and NVV VAE stages (``Direct3ds2SparseVAE``).

    Extends :class:`BaseTrainer` with:

    * pluggable ``pre_process_data`` / ``post_process_recon`` callbacks
      (e.g. the SDF or NVV helpers from ``triflow.callbacks``),
    * a ``loss_func`` that takes the model output, target, and optional
      posterior and returns ``(loss, loss_dict)``,
    * a ``metrics_func`` evaluated during sampling, and
    * a periodic sample callback that runs the VAE on the validation set
      and uploads the reconstructions to WandB.

    Args:
        pre_process_data: Callable applied to each batch before the model.
        post_process_recon: Callable applied to the reconstruction before
            visualization / metrics.
        loss_func: Callable returning ``(loss_tensor, loss_dict)``.
        metrics_func: Callable returning a scalar metric dict.
        no_variation: If ``True``, the encoder uses its posterior mode
            instead of sampling — used for deterministic VAE evaluation
            and inference.
        sample_every: Global-step interval for running :meth:`sample`.
        **kwargs: Forwarded to :class:`BaseTrainer`.
    """

    def __init__(
        self,
        pre_process_data=None,
        post_process_recon=None,
        loss_func=None,
        metrics_func=None,
        no_variation=False,
        sample_every=5000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_every = sample_every
        self.pre_process_data = pre_process_data
        self.post_process_recon = post_process_recon
        self.loss_func = loss_func
        self.metrics_func = metrics_func
        self.no_variation = no_variation

        # Register sample callback
        self.register_callback(self.sample, self.sample_every)

    def run_one_step(self, batch):
        """Run the VAE forward + loss for one batch.

        Args:
            batch: Collated training batch; ``batch["data"]`` is passed
                through ``pre_process_data`` (if set) before the model.

        Returns:
            The scalar loss tensor. Auxiliary loss components are pushed
            to the step logger under ``train_`` or ``val_`` prefixes
            depending on model mode.
        """
        x = batch["data"]

        if self.pre_process_data is not None:
            model_input = self.pre_process_data(x)
        else:
            model_input = x

        sample_posterior = not self.no_variation
        if self.model.training:
            model_output = self.model(model_input, sample_posterior=sample_posterior)
        else:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            model_output = unwrapped_model(
                model_input, sample_posterior=sample_posterior
            )

        if sample_posterior:
            model_output, posterior = model_output

        if callable(self.loss_func):
            if sample_posterior:
                loss, loss_dict = self.loss_func(model_output, model_input, posterior)
            else:
                loss, loss_dict = self.loss_func(model_output, model_input)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_func}")

        prefix = "train_" if self.model.training else "val_"
        for k, v in loss_dict.items():
            self.logger.log_scalars(**{f"{prefix}{k}": v})

        return loss

    @torch.no_grad()
    def sample(self, num_samples=None, output_file=None):
        """Run the VAE in eval mode over the validation loader and log samples.

        Uses the encoder's posterior mode (no sampling), applies
        ``post_process_recon`` if set, evaluates metrics via
        ``metrics_func``, and dumps a preview render per sample via
        whichever visualization method the dataset exposes.

        Args:
            num_samples: Number of samples to log. Defaults to
                ``self.num_samples``.
            output_file: Target ``.pt`` file for the raw reconstructions.
                Defaults to ``<run_dir>/samples/samples_step<N>.pt``.
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

        samples = []
        for batch in tqdm(self.val_loader_unshared, desc="Sampling"):
            x = batch["data"]

            if self.pre_process_data is not None:
                model_input = self.pre_process_data(x)
            else:
                model_input = x

            recon = unet(model_input, sample_posterior=False)

            if self.post_process_recon is not None:
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
