# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.

import os
import shutil
import sys

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


def backup_experiment():
    """Back up the CLI command and source code into the Hydra run dir.

    Hydra saves the resolved config automatically; this helper additionally
    writes ``command.txt`` (the exact argv used to launch the run) and copies
    the current ``triflow/`` package plus root ``.py`` files to
    ``<run_dir>/.hydra/code/`` so that the run is reproducible even if the
    source tree is later edited.
    """
    run_dir = os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, ".hydra"
    )  # Hydra changes CWD into the run dir automatically

    # Save exact command
    command_path = os.path.join(run_dir, "command.txt")
    with open(command_path, "w") as f:
        f.write(" ".join(sys.argv))

    # Back up code
    src_backup = os.path.join(run_dir, "code")
    if not os.path.exists(src_backup):
        os.makedirs(src_backup)
        src_root = hydra.utils.get_original_cwd()

        # Copy root .py files
        for f in os.listdir(src_root):
            if f.endswith(".py"):
                shutil.copy2(os.path.join(src_root, f), src_backup)

        # Copy triflow package
        shutil.copytree(
            os.path.join(src_root, "triflow"),
            os.path.join(src_backup, "triflow"),
            ignore=shutil.ignore_patterns("__pycache__"),
        )


@hydra.main(config_path="configs", config_name="direct3ds2_sparse_sdf_vae_512", version_base=None)
def main(cfg: DictConfig):
    """Train a TriFlow stage from a Hydra config.

    The config name is resolved from the ``--config-name`` CLI flag (see
    ``configs/direct3ds2_sparse_sdf_vae_512.yaml`` for the default and README
    for the three-stage training recipe). This function instantiates the
    trainer (which transitively instantiates the model, dataset, optimizer,
    and callbacks), writes a backup of the experiment to the Hydra run dir,
    and then calls ``trainer.train()``.
    """
    # Instantiate trainer (model and dataset are nested under trainer config)
    trainer = instantiate(cfg.trainer)

    # Backup experiment info
    if trainer.accelerator.is_main_process:
        backup_experiment()

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
