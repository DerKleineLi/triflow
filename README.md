<p align="center">
  <h3 align="center"><strong>TriFlow: Generating Artist-Like 3D Mesh Topology <br> via Nearest-Vertex Vector Fields</strong></h3>

<p align="center">
    <a href="https://niessnerlab.org/members/haoxuan_li/profile.html">Haoxuan Li</a><sup>1</sup>,
    <a href="https://ziyaerkoc.com/">Ziya Erko&ccedil;</a><sup>1</sup>,
    <a href="https://de.linkedin.com/in/dsirigatti">Daniele Sirigatti</a><sup>2</sup>,
    <a href="https://de.linkedin.com/in/vladislav-rosov">Vladislav Rosov</a><sup>2</sup>,
    <a href="https://craigleili.github.io/">Lei Li</a><sup>3</sup>,
    <a href="https://www.3dunderstanding.org/index.html">Angela Dai</a><sup>1</sup>,
    <a href="https://www.niessnerlab.org/">Matthias Nie&szlig;ner</a><sup>1</sup>
    <br>
    <sup>1</sup>Technical University of Munich,
    <sup>2</sup>AUDI AG,
    <sup>3</sup>University of Virginia
</p>

<div align="center">

<a href='https://arxiv.org/abs/2606.20131'><img src='https://img.shields.io/badge/arXiv-2606.20131-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href='https://derkleineli.github.io/triflow/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/lihcxr/TriFlow"><img src="https://img.shields.io/badge/Weights-HuggingFace-yellow?logo=huggingface"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://www.youtube.com/watch?v=Nl57QcuKkeA"><img src="https://img.shields.io/badge/Video-YouTube-red?logo=youtube"></a>

</div>

<!-- Optional: copy teaser.png from the project page repo into assets/ and uncomment.
<p align="center">
    <img src="assets/teaser.png" width="100%">
</p> -->

## Contents
- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Project Structure](#project-structure)
- [License](#license)
- [Acknowledgement](#acknowledgement)
- [BibTeX](#bibtex)

## Overview

TriFlow is a generative approach for producing compact 3D meshes with artist-like triangle topology directly from input geometry conditions such as signed distance fields. Our key insight is to represent mesh topology as a **Nearest-Vertex Vector Field (NVF)** defined over the surface, where each point encodes its association to the nearest triangle vertex in the local barycentric frame. We train a latent flow-matching model to synthesize this field, enabling topology generation conditioned on the input geometry. To extract a coherent mesh, we cluster surface regions using the generated NVF and guide a constrained **Quadric Error Metric (QEM)** mesh simplification with topology-aware optimization.

The codebase implements a three-stage pipeline trained sequentially:

1. **SDF VAE** — encodes the input SDF into a sparse voxel latent.
2. **NVF VAE** — encodes the NVF into the same sparse voxel latent space.
3. **Latent Flow Matching** — generates NVF latents conditioned on SDF latents, face count, and quad ratio.

At inference time, the pipeline encodes the input mesh's SDF, samples an NVF latent via flow matching, and extracts the final mesh with watershed clustering + constrained QEM.

## Environment Setup

```bash
git clone --recursive https://github.com/DerKleineLi/triflow.git
cd triflow
bash setup.sh
source .env
```

`setup.sh` creates a `triflow` conda environment (Python 3.10, CUDA 11.8), initialises
the submodules under `third_party/`, installs the dependencies, and writes the `.env`
file that puts `third_party/TRELLIS` and `third_party/Direct3D-S2` on `PYTHONPATH`.

### Pretrained weights

The pretrained weights are hosted on Hugging Face at
[lihcxr/TriFlow](https://huggingface.co/lihcxr/TriFlow). Download the three files
into `checkpoints/` (the directory ships empty):

```bash
hf download lihcxr/TriFlow \
    flow_model.safetensors sdf_vae.safetensors nvv_vae.safetensors \
    --local-dir checkpoints
```

The `hf` command ships with `huggingface_hub`, which is installed as a
dependency of `transformers` by `setup.sh`. Equivalently, from Python:

```python
from huggingface_hub import hf_hub_download

for name in ["flow_model.safetensors", "sdf_vae.safetensors", "nvv_vae.safetensors"]:
    hf_hub_download("lihcxr/TriFlow", name, local_dir="checkpoints")
```

Either way you should end up with:

```
checkpoints/
    flow_model.safetensors
    sdf_vae.safetensors
    nvv_vae.safetensors
```

`inference.py` loads them from `checkpoints/` by default.

## Data Preparation

Two small samples ship with the repository so the pipeline can be exercised without
downloading anything:

* `data/dummy_input/` — a handful of raw `.obj` meshes, split into
  `single_component/` and `multi_component/` subdirectories, for testing dataset
  preparation.
* `data/dummy_dataset/` — the prepared dataset those meshes produce, for testing
  training and inference.

Expected dataset layout:

```
data/my_dataset/
    meshes/<hashed_path>/<mesh_id>.ply
    metadata_512.json
```

To build a dataset from a directory of raw meshes:

```bash
python -m triflow.datasets.prepare_dataset /path/to/raw/meshes --output_dir data/my_dataset
```

To try it on the bundled samples:

```bash
python -m triflow.datasets.prepare_dataset data/dummy_input --output_dir data/my_dataset
```

Distributed (4 workers):

```bash
for rank in 0 1 2 3; do
    python -m triflow.datasets.prepare_dataset /path/to/raw/meshes \
        --output_dir data/my_dataset --world_size 4 --rank $rank &
done
wait
python -m triflow.datasets.prepare_dataset /path/to/raw/meshes \
    --output_dir data/my_dataset --merge_only
```

## Training

`trainer.model_ckpt` sets the weights each stage is initialised from. It defaults to
the corresponding file under `checkpoints/`, so pass `trainer.model_ckpt=null` to train
a stage from scratch, or point it at your own checkpoint to fine-tune.

Stage 1 — SDF VAE:

```bash
accelerate launch --multi_gpu train.py \
    --config-name=direct3ds2_sparse_sdf_vae_512 \
    trainer.dataset.data_root=data/my_dataset \
    trainer.dataset.meta_file=data/my_dataset/metadata_512.json \
    trainer.model_ckpt=null
```

Stage 2 — NVF VAE:

```bash
accelerate launch --multi_gpu train.py \
    --config-name=direct3ds2_sparse_nvv_vae_512 \
    trainer.dataset.data_root=data/my_dataset \
    trainer.dataset.meta_file=data/my_dataset/metadata_512.json \
    trainer.model_ckpt=null
```

Stage 3 — Latent Flow Matching:

```bash
accelerate launch --multi_gpu train.py \
    --config-name=trellis_slatflow_latent_512 \
    trainer.dataset.data_root=data/my_dataset \
    trainer.dataset.meta_file=data/my_dataset/metadata_512.json \
    trainer.model_ckpt=null \
    trainer.sdf_vae_ckpt=<path_to_sdf_vae_checkpoint> \
    trainer.nvv_vae_ckpt=<path_to_nvv_vae_checkpoint>
```

Override hyperparameters from the command line:

```bash
train.py --config-name=direct3ds2_sparse_sdf_vae_512 \
    trainer.batch_size_per_gpu=2 \
    trainer.optimizer.lr=5e-5
```

Resume from a checkpoint:

```bash
train.py --config-name=direct3ds2_sparse_sdf_vae_512 \
    +trainer.resume_from=outputs/train/<run_dir>/checkpoints/step_50000
```

Outputs are saved to `outputs/train/<config_name>/<timestamp>/`.

## Inference

`-i` is searched for `.obj` files non-recursively, so point it at a directory that
contains the meshes directly rather than at a parent of per-category subdirectories.
To try it on the bundled samples:

```bash
python inference.py \
    -i data/dummy_input/single_component \
    -o /path/to/output/meshes \
    --face_count 4000 \
    --qem_threshold 12.0 \
    --quad_ratio 0.95
```

| Argument | Default | Description |
|---|---|---|
| `-i, --input_dir` | (required) | Directory containing input `.obj` files |
| `-o, --output_dir` | (required) | Directory for output meshes |
| `--face_count` | `4000` | Target face count (0 = use input mesh count) |
| `--qem_threshold` | `12.0` | QEM error threshold for mesh simplification |
| `--quad_ratio` | `0.95` | Target quad ratio (0 = use input mesh ratio) |
| `--rank` | `0` | Worker rank for distributed inference |
| `--world_size` | `1` | Number of distributed workers |

Checkpoints are loaded from `checkpoints/` by default.

## Project Structure

```
triflow/
├── checkpoints/                      # Pretrained model weights
├── configs/                          # Hydra configuration files
│   ├── direct3ds2_sparse_sdf_vae_512.yaml   # Stage 1: SDF VAE
│   ├── direct3ds2_sparse_nvv_vae_512.yaml   # Stage 2: NVF VAE
│   ├── trellis_slatflow_latent_512.yaml     # Stage 3: Latent Flow
│   ├── dataset/                      # Dataset configs
│   ├── model/                        # Model architecture configs
│   └── trainer/                      # Trainer configs
├── triflow/                          # Main source package
│   ├── callbacks/                    # Training callbacks (loss, metrics, pre/post-processing)
│   ├── datasets/                     # Dataset, batch sampler, and data preparation
│   ├── models/                       # Model architectures
│   │   ├── direct3ds2_sparse_vae.py         # Sparse VAE (SDF & NVF)
│   │   └── trellis_shape_conditioned_slat_flow.py  # Conditioned flow model
│   ├── trainers/                     # Training loops
│   │   ├── base.py                          # Base trainer with checkpointing & logging
│   │   ├── direct3ds2_sparse_vae_trainer.py # VAE trainer
│   │   ├── trellis_slatflow_trainer.py      # Flow matching trainer
│   │   └── trellis_latent_slatflow_trainer.py  # Latent flow trainer (with frozen VAEs)
│   ├── utils/                        # Utilities
│   │   ├── mesh_processing.py               # Mesh voxelization & NVF computation
│   │   ├── mesh_reconstruction.py           # QEM-based mesh reconstruction from NVF
│   │   ├── sampling.py                      # Flow Euler sampler
│   │   └── ...
├── third_party/                      # Git submodules
│   ├── TRELLIS/                      # Microsoft TRELLIS
│   ├── Direct3D-S2/                  # DreamTech Direct3D-S2
│   └── pyfqmr-triflow/               # Customized fork of pyfqmr (editable install)
├── data/                             # Datasets (dummy included for testing)
├── setup.sh                          # Environment setup script
├── train.py                          # Training entry point
├── inference.py                      # Inference entry point
└── requirements.txt                  # Python dependencies
```

## License

The code in this repository is released under the
[Automotive Development Public Non-Commercial License v1.0](LICENSE) (ADPNCL).
As the name says, this license permits non-commercial use only; please read it
in full before using TriFlow.

### Third-party dependencies

The third-party projects under `third_party/` are included as git submodules — no
third-party source is redistributed by this repository. They carry their own
licenses: [TRELLIS](https://github.com/microsoft/TRELLIS) (MIT),
[Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2) (MIT) and our fork of
[pyfqmr](https://github.com/Kramer84/pyfqmr-Fast-Quadric-Mesh-Reduction) (MIT).

Please note that the mesh-processing pipeline additionally depends on
[MeshLib](https://github.com/MeshInspector/MeshLib) (`meshlib` in
`requirements.txt`), which is **not distributed under an open-source license**.
MeshLib is installed from PyPI by the user and no MeshLib code is bundled here,
but you should review MeshLib's own license terms before using TriFlow — in
particular for any commercial use, which their free tier does not cover.

## Acknowledgement

This project builds on the following excellent works:

* [TRELLIS](https://github.com/microsoft/TRELLIS) — Sparse structured latent representations for 3D generation
* [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2) — Scalable sparse 3D generation
* [pyfqmr](https://github.com/Kramer84/pyfqmr-Fast-Quadric-Mesh-Reduction) — Fast quadric mesh reduction

This work was funded by AUDI AG.

## BibTeX

```bibtex
@inproceedings{li2026triflow,
  title = {TriFlow: Generating Artist-Like 3D Mesh Topology via Nearest-Vertex Vector Fields},
  author = {Li, Haoxuan and Erko{\c{c}}, Ziya and Sirigatti, Daniele and Rosov, Vladislav and Li, Lei and Dai, Angela and Nie{\ss}ner, Matthias},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year = {2026},
}
```
