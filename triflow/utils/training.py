# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.

from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import torch
import torchvision.utils as vutils
import wandb
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Step Logger (WandB)
# ---------------------------------------------------------------------------


class StepLogger:
    """Accumulate scalars and images across training micro-steps and flush to W&B.

    Scalars are averaged over the accumulated window on ``flush``; images are
    collected, stacked into a grid with ``torchvision.utils.make_grid``, and
    wrapped in a ``wandb.Image``. The returned dict is what the trainer hands
    off to ``wandb.log``.
    """

    def __init__(self):
        self._scalars = defaultdict(list)
        self._images = defaultdict(list)

    def log_scalars(self, **kwargs):
        """Append one value per keyword to the per-key running list."""
        for key, value in kwargs.items():
            self._scalars[key].append(value)

    def log_images(self, **kwargs):
        """Append one image (or a batch of images) per keyword.

        Each value must be a ``torch.Tensor`` or ``np.ndarray`` of shape
        ``(C, H, W)`` or ``(B, C, H, W)``.
        """
        for key, value in kwargs.items():
            if not isinstance(value, (torch.Tensor, np.ndarray)):
                raise TypeError(
                    f"log_images only accepts torch.Tensor or np.ndarray, got {type(value)}"
                )
            self._images[key].append(value)

    def flush(self):
        """Aggregate pending logs and clear internal buffers.

        Scalars are averaged over the buffered window. Images are assembled
        into a single grid per key and wrapped in ``wandb.Image``. Returns a
        ``dict`` ready to be passed to ``wandb.log``.
        """
        logs = {}
        for key, values in self._scalars.items():
            if values:
                logs[key] = sum(values) / len(values)
        for key, imgs_list in self._images.items():
            all_imgs = []
            for imgs in imgs_list:
                if imgs.ndim == 3:
                    all_imgs.append(imgs)
                elif imgs.ndim == 4:
                    all_imgs.extend([imgs[i] for i in range(imgs.size(0))])
            if all_imgs:
                grid = vutils.make_grid(all_imgs, nrow=int(len(all_imgs) ** 0.5))
                logs[key] = wandb.Image(grid)
        self._scalars.clear()
        self._images.clear()
        return logs


# ---------------------------------------------------------------------------
# Rich console printing
# ---------------------------------------------------------------------------


def format_value(v):
    """Format a number with thousands separators; other types fall back to ``str``."""
    if isinstance(v, (int, float)):
        return f"{v:,}"
    return str(v)


def add_config_rows(table: Table, config: dict, indent: int = 0):
    """Recursively append config entries to a rich ``Table``, indenting nested dicts."""
    indent_str = " " * (indent * 2)
    for k, v in config.items():
        if isinstance(v, Mapping):
            table.add_row(f"{indent_str}[bold]{k}[/bold]", "")
            add_config_rows(table, v, indent + 1)
        else:
            table.add_row(f"{indent_str}{k}", format_value(v))


def make_config_panel(config: dict) -> Panel:
    """Build a rich panel displaying the full training config as a two-column table."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    add_config_rows(table, config)
    return Panel(table, title="[bold blue]Config[/bold blue]", expand=False)


def make_model_panel(model_str, title="Model") -> Panel:
    """Wrap a ``repr(model)``-style string in a titled rich panel."""
    return Panel.fit(model_str, title=f"[bold blue]{title}[/bold blue]")


def make_models_panel(models_metadata: dict, title="Models") -> Panel:
    """Build a rich panel summarizing every model: module repr + parameter counts.

    Args:
        models_metadata: Dict keyed by model name, each value a dict with
            ``model_str``, ``num_parameters``, ``num_trainable_parameters``.
    """
    model_panels = [
        make_model_panel(meta["model_str"], title=name)
        for name, meta in models_metadata.items()
    ]
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model Name", style="bold cyan")
    table.add_column("Total Params", justify="right")
    table.add_column("Trainable Params", justify="right")
    for name, meta in models_metadata.items():
        table.add_row(
            name,
            format_value(meta["num_parameters"]),
            format_value(meta["num_trainable_parameters"]),
        )
    combined = Columns([*model_panels, table])
    return Panel.fit(combined, title=f"[bold blue]{title}[/bold blue]")


def make_stats_panel(stats: dict) -> Panel:
    """Build a rich panel summarizing arbitrary training statistics."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    for k, v in stats.items():
        table.add_row(k, format_value(v))
    return Panel(
        table, title="[bold blue]Training Statistics[/bold blue]", expand=False
    )


def print_run_summary(config: dict, models_metadata: dict, stats: dict):
    """Print the config, model, and training-stats panels to the rich console.

    Used by ``BaseTrainer`` at the start of training to give the user a
    human-readable snapshot of the run.
    """
    console.print(make_config_panel(config))
    console.print(make_models_panel(models_metadata))
    console.print(make_stats_panel(stats))
