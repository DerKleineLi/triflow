# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.

import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from hydra import compose, initialize
from hydra.utils import instantiate
from safetensors.torch import load_file

from triflow.utils.sampling import FlowEulerSampler
from triflow.utils.mesh_reconstruction import topology_flow2mesh_QEM
from triflow.utils.mesh_processing import process_one_mesh, robust_remesh
from triflow.utils.sparse_voxel import nested_device_transfer


def run_inference(
    input_dir: str,
    output_dir: str,
    rank: int,
    world_size: int,
    face_count=4000,
    qem_threshold=12.0,
    quad_ratio=0.95,
):
    """Run TriFlow mesh-to-mesh topology-aware remeshing on a directory of .obj files.

    For each input mesh the pipeline: (1) encodes the SDF through the SDF VAE to
    obtain a conditioning latent, (2) samples an NVV latent with flow matching,
    (3) decodes to a sparse NVV field via the NVV VAE, and (4) extracts a final
    triangle mesh via watershed clustering + constrained QEM simplification.

    Here ``NVV`` denotes voxelized nearest-vertex vectors (the discretized form
    of the nearest-vertex vector field described in the paper).

    Checkpoints are loaded from ``checkpoints/`` (see README). Meshes already
    present in ``output_dir`` are skipped. Distributed processing is supported
    by splitting the file list modulo ``world_size``.

    Hyperparameters of the post-processing stage (``nvv_smooth_kwargs``,
    ``root_threshold``, ``merge_threshold``, ``target_position_weight``, and
    the flow sampler's ``steps`` / ``sigma_min``) are tuned constants and kept
    inline below. See ``topology_flow2mesh_QEM`` and the paper for their
    meaning and tuning.

    Args:
        input_dir: Directory containing input ``.obj`` files.
        output_dir: Directory where output meshes are written. Created if missing.
        rank: Worker rank for distributed inference, in ``[0, world_size)``.
        world_size: Total number of distributed workers.
        face_count: Target face count for the output mesh. If ``None``, the
            input mesh's decimated face count is used.
        qem_threshold: Maximum allowed quadric error during QEM simplification.
        quad_ratio: Target quad ratio in ``[0, 1]``, passed as a conditioning
            signal to the flow model. If ``None``, the input mesh's computed
            ratio is used.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_paths = list(input_dir.glob("*.obj"))
    mesh_paths = [p for i, p in enumerate(mesh_paths) if i % world_size == rank]

    # initialize model
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="trellis_slatflow_latent_512")

    dataset = instantiate(cfg.trainer.dataset, dummy=True)

    ckpt_dir = Path("checkpoints")

    model = instantiate(cfg.trainer.model)
    state_dict = load_file(ckpt_dir / "flow_model.safetensors", device="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()

    sdf_vae = instantiate(cfg.trainer.sdf_vae)
    state_dict_vae = load_file(ckpt_dir / "sdf_vae.safetensors", device="cpu")
    sdf_vae.load_state_dict(state_dict_vae)
    sdf_vae.eval()
    sdf_vae.cuda()

    nvv_vae = instantiate(cfg.trainer.nvv_vae)
    state_dict_nvv = load_file(ckpt_dir / "nvv_vae.safetensors", device="cpu")
    nvv_vae.load_state_dict(state_dict_nvv)
    nvv_vae.eval()
    nvv_vae.cuda()

    accelerator = Accelerator(
        mixed_precision="fp16",
    )

    pre_process_data = instantiate(cfg.trainer.pre_process_data)
    post_process_recon = instantiate(cfg.trainer.post_process_recon)
    get_cond = instantiate(cfg.trainer.get_cond)

    pre_process_data = pre_process_data(accelerator, nvv_vae)
    post_process_recon = post_process_recon(accelerator, nvv_vae)
    get_cond = get_cond(accelerator, sdf_vae)

    # process data
    total_time = 0.0
    count = 0
    for i in range(len(mesh_paths)):
        mesh_path = mesh_paths[i]
        mesh_name = mesh_path.stem
        output_stem = f"{mesh_name}"
        output_file = output_dir / f"{output_stem}.obj"
        if output_file.exists():
            print(f"({i}/{len(mesh_paths)}) {output_file} already exists, skipping...")
            continue

        print(f"({i}/{len(mesh_paths)}) Processing {mesh_path}...")

        # import data
        results, trimesh_mesh, augmented_mesh, metadata = process_one_mesh(
            mesh_path,
            res_fine=512,
            pad=1.5,
            round_verts=False,
            decimate_length=1.0,
            vertex_merge_threshold=0.0,
            augment=False,
            augment_strength=1.0,
            augment_density=False,
            cast=False,
            get_metadata=True,
        )

        if face_count is not None:
            results["face_count"] = face_count
        else:
            results["face_count"] = metadata["decimated_num_faces"]
        if quad_ratio is not None:
            results["quad_ratio"] = quad_ratio
        else:
            results["quad_ratio"] = metadata["quad_ratio"]

        data = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                data[k] = torch.tensor(v)
                if data[k].dtype == torch.float64:
                    data[k] = data[k].half()
            else:
                data[k] = torch.tensor([v])
        meta = {"path": str(mesh_path)}
        mesh = {
            "original": trimesh_mesh,
            "augmented": augmented_mesh,
        }
        sample = {"data": data, "meta": meta, "mesh": mesh}

        batch = dataset.collate_fn([sample])
        batch = nested_device_transfer(batch, accelerator.device)
        data = batch["data"]

        clean_images = pre_process_data(data)
        noise = clean_images.replace(torch.randn_like(clean_images.feats))
        sampler = FlowEulerSampler(1e-5)
        c = get_cond(batch)

        mesh_gt = batch["mesh"]["augmented"][0]
        remeshed_mesh, _ = robust_remesh(
            mesh_gt,
            remesh_method="adaptive",
            allow_collapse=False,
            get_metadata=False,
            verbose=False,
        )

        # post process args
        nvv_smooth_kwargs = {
            "radius": 3.5,
            "threshold": 6,
            "sigma_s": 1.0,
            "sigma_r": 1.0,
        }
        root_threshold = 0.5
        merge_threshold = 1.0

        start_time = time.time()
        with torch.no_grad():
            with accelerator.autocast():
                res = sampler.sample(
                    model,
                    noise=noise,
                    cond=c,
                    steps=50,
                    verbose=True,
                )
        recon = res.samples
        recon = post_process_recon(data, recon)

        coords = recon["occ_fine"][:, 1:].cpu().numpy()
        nvv = recon["nvv_fine"].cpu().numpy()
        resolution = int(recon["res_fine"])

        mesh_qem = topology_flow2mesh_QEM(
            remeshed_mesh,
            coords,
            nvv,
            resolution,
            nvv_smooth_kwargs=nvv_smooth_kwargs,
            root_threshold=root_threshold,
            merge_threshold=merge_threshold,
            target_face_count=results["face_count"],
            max_quadratic_error=qem_threshold,
            target_position_weight=0.1,
            verbose=True,
            debug_output=None,
        )
        end_time = time.time()
        total_time += end_time - start_time
        count += 1

        mesh_qem.export(output_dir / f"{output_stem}.obj")

    if count == 0:
        print(
            f"No meshes were processed. Check that '{input_dir}' contains .obj files "
            f"directly (the search is not recursive, so files in subdirectories are "
            f"ignored), and that '{output_dir}' does not already contain the results."
        )
    else:
        print(
            f"total time: {total_time}s, count: {count}, average time: {total_time / count}s"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--face_count", type=int, default=4000)
    parser.add_argument("--qem_threshold", type=float, default=12.0)
    parser.add_argument("--quad_ratio", type=float, default=0.95)
    args = parser.parse_args()

    run_inference(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        rank=args.rank,
        world_size=args.world_size,
        face_count=args.face_count if args.face_count > 0 else None,
        qem_threshold=args.qem_threshold,
        quad_ratio=args.quad_ratio if args.quad_ratio > 0 else None,
    )
