# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.

import os
import pickle
from abc import ABC, abstractmethod
from datetime import timedelta

import hydra
import torch
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    set_seed,
)
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from triflow.utils.training import StepLogger, print_run_summary


class BaseTrainer(ABC):
    """Abstract trainer base class handling boilerplate for all TriFlow stages.

    Handles dataset splitting, Accelerate setup, optimizer / scheduler /
    warmup, checkpointing, per-step callbacks, validation, and WandB
    logging. Subclasses implement :meth:`run_one_step` (and typically also
    override :meth:`sample` for stage-specific visualization).

    Args:
        model: The main model being trained.
        dataset: A dataset whose ``__getitem__`` returns a training sample;
            if it has a ``collate_fn`` attribute, that's used as the
            DataLoader collate function.
        optimizer: A partial-initialized ``torch.optim.Optimizer`` factory.
            Called as ``optimizer(params=...)`` by ``__init__``.
        lr_scheduler: A partial-initialized LR scheduler factory. Called
            as ``lr_scheduler(optimizer=...)``.
        batch_sampler: Optional partial-initialized
            ``FixedSizeBalancedBatchSampler`` factory. When set, replaces
            the default random-shuffle loader.
        lr_warmup: Dict with ``warmup_steps`` / ``start`` / ``end``
            controlling a linear LR warmup.
        model_ckpt: Optional path to a safetensors / ``.pth`` file to load
            matching weights from at startup (non-strict load).
        resume_from: Optional path to an Accelerate checkpoint directory to
            resume from.
        val_ratio: Fraction of the dataset held out for validation.
        num_workers: DataLoader worker count.
        batch_size_per_gpu: Micro-batch size per GPU.
        gradient_accumulation_steps: Number of micro-batches between
            optimizer steps.
        max_grad_norm: Maximum allowed gradient norm (``None`` disables
            clipping).
        max_iterations: Total number of optimizer steps to run.
        log_every: Global-step interval for pushing the step log.
        eval_every: Global-step interval for running validation.
        save_every: Global-step interval for saving a checkpoint.
        num_samples: Number of dataset samples to materialize and log as
            previews at the start of training.
        seed: Accelerate / torch seed.
        precision: Accelerate mixed-precision mode (``"no"``, ``"fp16"``,
            ``"bf16"``).
        log_with: Accelerate tracker backend (``"wandb"``, etc.).
        project_name: Name shown in WandB.
    """

    def __init__(
        self,
        model,
        dataset,
        optimizer,
        lr_scheduler,
        batch_sampler=None,
        lr_warmup={
            "warmup_steps": 0,
            "start": 1e-6,
            "end": 1e-4,
        },
        model_ckpt=None,
        resume_from=None,
        val_ratio=0.05,
        num_workers=4,
        batch_size_per_gpu=16,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        max_iterations=100000,
        log_every=100,
        eval_every=1000,
        save_every=10000,
        num_samples=16,
        seed=42,
        precision="no",
        log_with="wandb",
        project_name="triflow",
    ):
        # Store parameters
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer(
            params=[p for p in self.model.parameters() if p.requires_grad]
        )
        self.lr_scheduler = lr_scheduler(optimizer=self.optimizer)
        self.lr_warmup = lr_warmup
        self.model_ckpt = model_ckpt
        self.resume_from = resume_from
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.batch_size_per_gpu = batch_size_per_gpu
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.max_iterations = max_iterations
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.num_samples = num_samples
        self.seed = seed
        self.precision = precision
        self.log_with = log_with
        self.project_name = project_name

        # runtime variables
        self.global_step = 0
        self.weight_dtype = torch.float32
        if self.precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # prepare dataset
        if self.val_ratio < 1 / len(self.dataset):
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.dataset,
                [len(self.dataset) - 1, 1],
                generator=torch.Generator().manual_seed(42),
            )
        else:
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.dataset,
                [1 - self.val_ratio, self.val_ratio],
                generator=torch.Generator().manual_seed(42),
            )

        if batch_sampler is not None:
            train_metadata = {}
            meta_keys = list(self.dataset.metadata.keys())
            meta_values = list(self.dataset.metadata.values())
            for idx in self.train_dataset.indices:
                train_metadata[meta_keys[idx]] = meta_values[idx]
            self.batch_sampler = batch_sampler(
                metadata=train_metadata,
                batch_size=self.batch_size_per_gpu,
            )
            self.train_loader_unshared = DataLoader(
                self.train_dataset,
                batch_sampler=self.batch_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True,
                collate_fn=(
                    self.dataset.collate_fn
                    if hasattr(self.dataset, "collate_fn")
                    else None
                ),
            )
        else:
            self.train_loader_unshared = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size_per_gpu,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
                collate_fn=(
                    self.dataset.collate_fn
                    if hasattr(self.dataset, "collate_fn")
                    else None
                ),
                shuffle=True,
            )
        self.val_loader_unshared = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=self.batch_size_per_gpu,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
            collate_fn=(
                self.dataset.collate_fn if hasattr(self.dataset, "collate_fn") else None
            ),
        )

        # setup accelerator
        self.global_config = hydra.compose(
            config_name=hydra.core.hydra_config.HydraConfig.get().job.config_name,
            overrides=hydra.core.hydra_config.HydraConfig.get().overrides["task"],
        )
        self.run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        timeout = InitProcessGroupKwargs(timeout=timedelta(days=4))
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.precision,
            log_with=self.log_with,
            project_dir=self.run_dir,
            kwargs_handlers=[kwargs, timeout],
        )
        self.accelerator.init_trackers(
            project_name=self.project_name,
            config=OmegaConf.to_container(self.global_config),
            init_kwargs={"wandb": {"name": self.global_config.output_folder}},
        )
        set_seed(self.seed, device_specific=True)

        if self.model_ckpt is not None:
            self.init_model(self.model, self.model_ckpt)

        self.model = self.register_model(model, name="Model")
        (
            self.optimizer,
            self.train_loader,
        ) = self.accelerator.prepare(
            self.optimizer,
            self.train_loader_unshared,
        )

        self.val_loader_unshared = self.accelerator.prepare_data_loader(
            self.val_loader_unshared, device_placement=True, slice_fn_for_dispatch=None
        )
        self.val_loader_unshared._index_sampler.num_processes = 1

        # logger
        self.logger = StepLogger()

        # training callbacks
        self.callbacks = []
        self.register_callback(self.upload_log, self.log_every)
        self.register_callback(self.validate, self.eval_every)
        self.register_callback(self.save_checkpoint, self.save_every)

    def init_model(self, model, ckpt_path):
        """Load matching weights from ``ckpt_path`` into ``model`` (non-strict).

        Shape-mismatched keys are silently skipped so that partial
        checkpoints (e.g. a shared backbone) can be reused across model
        variants. Returns the mutated model for chaining.
        """
        if ckpt_path is not None:
            if ckpt_path.endswith(".safetensors"):
                state_dict = load_file(ckpt_path, device="cpu")
            else:
                state_dict = torch.load(ckpt_path, map_location="cpu")
            model_state_dict = model.state_dict()
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if k in model_state_dict and v.size() == model_state_dict[k].size()
            }
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded {len(state_dict)} layers from {ckpt_path}")
        return model

    def register_model(self, model, name="Model"):
        """Register ``model`` for logging / parameter counting and prepare it with Accelerate.

        Call this for every model the trainer drives.
        """
        if not hasattr(self, "models"):
            self.models = {}
        self.models[name] = model
        return self.accelerator.prepare(model)

    @abstractmethod
    def run_one_step(self, batch):
        """
        Perform a single training step.

        Args:
            batch (Tensor): A batch of input data.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        pass

    def register_callback(self, function, step_interval):
        """Schedule ``function`` to be called every ``step_interval`` global steps.

        Callbacks run on the main process only, after gradient sync, at
        the end of a training step.
        """
        self.callbacks.append((function, step_interval))

    def save_checkpoint(self):
        """Save the full Accelerate training state to ``checkpoints/step_<n>/``."""
        self.accelerator.save_state(
            os.path.join(self.run_dir, "checkpoints", f"step_{self.global_step}")
        )

    def load_checkpoint(self, checkpoint_path):
        """Load model weights from ``checkpoint_path/model.safetensors`` and restore
        ``global_step`` from the checkpoint directory name.
        """
        model_ckpt_path = os.path.join(checkpoint_path, "model.safetensors")
        state_dict = load_file(model_ckpt_path, device="cpu")
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(state_dict)
        self.global_step = int(checkpoint_path.split("_")[-1])

    def upload_log(self):
        """Flush the step logger, dump to ``logs.pkl``, and forward to the accelerator."""
        logs = self.logger.flush()
        with open(os.path.join(self.run_dir, "logs.pkl"), "wb") as f:
            pickle.dump(logs, f)
        self.accelerator.log(logs, step=self.global_step)

    def lr_warmup_update(self):
        """Linearly ramp the optimizer's learning rate during the warmup phase."""
        if self.lr_warmup is not None:
            warmup_steps = self.lr_warmup["warmup_steps"]
            start_lr = self.lr_warmup["start"]
            end_lr = self.lr_warmup["end"]

            if self.global_step < warmup_steps:
                lr_scale = start_lr + (end_lr - start_lr) * (
                    self.global_step / warmup_steps
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr_scale

    def train(self):
        """Run the full training loop until ``max_iterations`` is reached.

        Handles resume, dataset sample logging, progress bar, micro-batch
        / gradient-accumulation bookkeeping, autocast forward / backward,
        gradient clipping, optimizer and scheduler steps, scheduled
        callbacks, and the final checkpoint. Any exception raised by
        :meth:`run_one_step` is caught, logged with the offending batch
        metadata, checkpointed, and re-raised.
        """
        if self.resume_from is not None:
            self.load_checkpoint(self.resume_from)
            print(f"Resumed from checkpoint: {self.resume_from}")

        # log before training
        if self.accelerator.is_main_process:
            self.log_dataset_samples()
            self.log_training_statistics()

        self.model.train()
        progress_bar = tqdm(
            range(self.max_iterations),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.update(self.global_step)
        epoch = 0
        print(f"Starting training from global step {self.global_step}")

        len_train_loader = len(self.train_loader)
        while self.global_step < self.max_iterations:
            micro_batch_count = 0
            for i, batch in enumerate(self.train_loader):
                remaining_micro_batches = len_train_loader - i
                if (
                    micro_batch_count % self.gradient_accumulation_steps == 0
                    and remaining_micro_batches < self.gradient_accumulation_steps
                ):
                    break

                if self.global_step >= self.max_iterations:
                    break

                with self.accelerator.accumulate(self.model):
                    try:
                        with self.accelerator.autocast():
                            loss = self.run_one_step(batch)
                        self.accelerator.backward(loss)
                    except Exception as e:
                        print(
                            f"[Rank {self.accelerator.process_index}]Error during training: {e}"
                        )
                        print(batch["meta"])
                        self.save_checkpoint()
                        raise e

                    if (
                        self.accelerator.sync_gradients
                        and self.max_grad_norm is not None
                    ):
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        if torch.isnan(grad_norm):
                            print(
                                f"[Rank {self.accelerator.process_index}]NaN gradient norm."
                            )
                            print(batch["meta"]["path"])
                        self.logger.log_scalars(grad_norm=grad_norm.mean().item())

                    if self.accelerator.sync_gradients:
                        self.lr_scheduler.step()
                        self.lr_warmup_update()

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.logger.log_scalars(
                        lr=self.optimizer.param_groups[0]["lr"], train_loss=loss.item()
                    )

                self.accelerator.wait_for_everyone()
                if self.accelerator.sync_gradients and self.accelerator.is_main_process:

                    for func, interval in self.callbacks:
                        if self.global_step % interval == 0:
                            func()

                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    progress_bar.update(1)
                micro_batch_count += 1
            epoch += 1
            self.accelerator.wait_for_everyone()

        self.accelerator.save_state(os.path.join(self.run_dir, "checkpoints", "final"))
        self.accelerator.end_training()

    @torch.no_grad()
    def validate(self):
        """Run :meth:`run_one_step` over the validation loader and log ``val_loss``."""
        self.model.eval()

        for batch in tqdm(self.val_loader_unshared, desc="Validation"):
            loss = self.run_one_step(batch)
            self.logger.log_scalars(val_loss=loss.item())
        logs = self.logger.flush()
        self.accelerator.log(logs, step=self.global_step)

        self.model.train()

    def get_steps_per_epoch(self):
        """Return the number of optimizer steps per epoch.

        Accounts for batch size, number of GPUs, and gradient accumulation.
        """
        train_size = len(self.train_dataset)
        batch_size_per_gpu = self.batch_size_per_gpu
        num_gpus = self.accelerator.num_processes
        gradient_accumulation_steps = self.gradient_accumulation_steps
        batch_size = batch_size_per_gpu * num_gpus * gradient_accumulation_steps
        steps_per_epoch = train_size // batch_size
        return steps_per_epoch

    def get_epoch(self):
        """Return the current epoch index derived from the global step."""
        epoch = self.global_step // self.get_steps_per_epoch()
        return epoch

    def log_training_statistics(self):
        """Print a rich summary panel and push training stats to WandB.

        Includes dataset sizes, effective batch size, steps per epoch, and
        per-model parameter counts.
        """
        dataset_size = len(self.dataset)
        train_size = len(self.train_dataset)
        val_size = len(self.val_dataset)
        batch_size_per_gpu = self.batch_size_per_gpu
        num_gpus = self.accelerator.num_processes
        gradient_accumulation_steps = self.gradient_accumulation_steps
        batch_size = batch_size_per_gpu * num_gpus * gradient_accumulation_steps
        steps_per_epoch = len(self.train_loader) // gradient_accumulation_steps
        num_parameters = 0
        num_trainable_parameters = 0
        models_metadata = {}
        for name, model in self.models.items():
            models_metadata[name] = {
                "model_str": str(model).strip(),
                "num_parameters": 0,
                "num_trainable_parameters": 0,
            }
            for param in model.parameters():
                models_metadata[name]["num_parameters"] += param.numel()
                if param.requires_grad:
                    models_metadata[name]["num_trainable_parameters"] += param.numel()
            num_parameters += models_metadata[name]["num_parameters"]
            num_trainable_parameters += models_metadata[name][
                "num_trainable_parameters"
            ]

        stats = {
            "Dataset size": dataset_size,
            "Training set size": train_size,
            "Validation set size": val_size,
            "Batch size per GPU": batch_size_per_gpu,
            "Number of GPUs": num_gpus,
            "Gradient accumulation steps": gradient_accumulation_steps,
            "Effective batch size": batch_size,
            "Steps per epoch": steps_per_epoch,
            "Total model parameters": num_parameters,
            "Trainable model parameters": num_trainable_parameters,
        }

        wandb_tracker = self.accelerator.get_tracker("wandb").tracker
        wandb_tracker.config.update({"Training Statistics": stats})

        print_run_summary(self.global_config, models_metadata, stats)

    def log_dataset_samples(self, num_samples=None, output_file=None):
        """Dump a handful of dataset samples to disk and upload renders to WandB.

        Used once at the start of training as a sanity check. If the
        dataset defines ``visualize_sample`` or ``visualize_batch``, the
        samples are also rendered into preview images.

        Args:
            num_samples: Number of samples to log. Defaults to
                ``self.num_samples`` (set by :meth:`__init__`).
            output_file: Target ``.pt`` file. Defaults to
                ``<run_dir>/samples/data_samples.pt``.
        """
        if not self.accelerator.is_main_process:
            return
        num_samples = num_samples if num_samples is not None else self.num_samples
        output_file = (
            output_file
            if output_file is not None
            else os.path.join(self.run_dir, "samples", f"data_samples.pt")
        )

        batches = []
        sample_count = 0
        for batch in self.val_loader_unshared:
            batches.append(batch)
            sample_count += self.batch_size_per_gpu
            if sample_count >= num_samples:
                break
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        torch.save(batches, output_file)
        for batch in batches:
            color_list = self.dataset.visualize_batch(batch)
            for color in color_list:
                self.logger.log_images(
                    dataset=torch.from_numpy(color.copy()).permute(2, 0, 1)
                )
        self.upload_log()
