#!/usr/bin/env python3

# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.

"""
prepare_dataset.py — End-to-end data preprocessing for TriFlow.

Takes a directory of mesh files (obj, glb, fbx, ply, ...), and produces the
dataset directory expected by MeshDataset:

    <output_dir>/
        meshes/<rel_path>/<component>.ply
        metadata_512.json

Pipeline:
    1. Component splitting  — load each mesh/scene, split into connected
       components, filter for manifold quality, deduplicate, and export PLY.
    2. Metadata computation — for every component PLY, run process_one_mesh()
       to compute occupancy, NVV, SDF, quad-ratio, etc.
    3. Metadata merge       — collect per-sample JSON into a single metadata
       file that MeshDataset can load.

Supports distributed processing via --world_size / --rank.

Usage:
    # Single process (small dataset):
    python -m triflow.datasets.prepare_dataset /path/to/meshes --output_dir data/my_dataset

    # Distributed (e.g. 4 workers):
    for rank in 0 1 2 3; do
        python -m triflow.datasets.prepare_dataset /path/to/meshes \
            --output_dir data/my_dataset --world_size 4 --rank $rank &
    done
    wait
    # Merge metadata from all ranks:
    python -m triflow.datasets.prepare_dataset /path/to/meshes \
        --output_dir data/my_dataset --merge_only
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm

from triflow.utils.mesh_processing import pack_trimesh

# ---------------------------------------------------------------------------
# Step 1: Component splitting
# ---------------------------------------------------------------------------

MESH_EXTENSIONS = {
    ".obj", ".glb", ".gltf", ".fbx", ".ply", ".stl", ".off", ".dae",
}
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB


def sanitize_filename(name: str) -> str:
    """Return a filename-safe ASCII version of ``name``.

    Non-ASCII characters are dropped; any remaining character that isn't a
    letter, digit, underscore, or hyphen is replaced with an underscore.
    """
    name_ascii = name.encode("ascii", errors="ignore").decode()
    return re.sub(r"[^A-Za-z0-9_\-]", "_", name_ascii)


def is_closed_manifold(mesh: trimesh.Trimesh) -> bool:
    """Return ``True`` iff ``mesh`` has no non-manifold edges (each edge shared
    by at most 2 faces) and is watertight.
    """
    if mesh.faces.size == 0:
        return False
    faces = mesh.faces
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges_sorted = np.sort(edges, axis=1)
    edges_sorted = np.ascontiguousarray(edges_sorted)
    _, counts = np.unique(
        edges_sorted.view([("", edges_sorted.dtype)] * 2), return_counts=True
    )
    if np.any(counts > 2):
        return False
    return mesh.is_watertight


def compute_component_signature(mesh: trimesh.Trimesh) -> str:
    """Return a stable sha256 hash for a mesh component.

    Derived from its vertex, edge, and face counts. Used to deduplicate
    identical components across input files (e.g. when the same mesh is
    referenced from multiple scenes).
    """
    nv = mesh.vertices.shape[0]
    ne = mesh.edges_unique.shape[0]
    nf = mesh.faces.shape[0]
    arr = np.array([nv, ne, nf], dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def split_components(input_path: Path, input_dir: Path, out_dir: Path, min_faces=16, max_faces=32768):
    """Load a mesh file, split into components, filter, deduplicate, and save.

    Accepts ``.fbx`` (via ``fbxloader``, if installed) and anything
    ``trimesh.load`` understands as a Scene or Trimesh. Each component that
    is watertight-manifold and whose face count is in
    ``[min_faces, max_faces]`` is exported as a PLY under
    ``<out_dir>/<relative parent path>/``. Components with identical
    signatures are deduplicated. The component index in the output filename
    is the raw per-mesh index and is not guaranteed to be contiguous after
    filtering.

    Returns:
        tuple[int, dict]: the number of components saved and a per-component
        metadata dict keyed by the output file stem.
    """
    try:
        if input_path.suffix.lower() == ".fbx":
            try:
                from fbxloader import FBXLoader
                fbx = FBXLoader(str(input_path))
                loaded = fbx.export_trimesh()
            except ImportError:
                print(f"  fbxloader not installed, skipping {input_path}", file=sys.stderr)
                return 0, {}
        else:
            loaded = trimesh.load(
                str(input_path), force="scene", process=False, skip_materials=True
            )
    except Exception as e:
        print(f"  Error loading {input_path}: {e}", file=sys.stderr)
        return 0, {}

    meshes = []
    if isinstance(loaded, trimesh.Scene):
        for name, geom in loaded.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append((f"{input_path.stem}_{name}", geom.copy()))
    elif isinstance(loaded, trimesh.Trimesh):
        meshes.append((input_path.stem, loaded.copy()))
    else:
        return 0, {}

    rel_path = input_path.relative_to(input_dir).parent
    comp_dir = out_dir / rel_path
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Build a unique prefix from the subdirectory path to avoid name collisions
    # when different subdirs contain files with the same name.
    if str(rel_path) != ".":
        dir_prefix = sanitize_filename(str(rel_path)) + "__"
    else:
        dir_prefix = ""

    saved = 0
    metadata = {}
    seen_hashes = set()
    global_idx = 0

    for base_name, mesh in meshes:
        pack_trimesh(mesh)
        comps = mesh.split(only_watertight=False, repair=True)

        for comp in comps:
            global_idx += 1

            if len(comp.vertices) == 0 or len(comp.faces) < min_faces or len(comp.faces) > max_faces:
                continue

            if not is_closed_manifold(comp):
                continue

            h = compute_component_signature(comp)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            file_stem = f"{dir_prefix}{sanitize_filename(base_name)}_component_{global_idx:03d}"
            out_file = comp_dir / f"{file_stem}.ply"

            try:
                comp.export(str(out_file), file_type="ply")
                saved += 1
                metadata[file_stem] = {
                    "vertices": int(comp.vertices.shape[0]),
                    "faces": int(comp.faces.shape[0]),
                    "edges": int(comp.edges_unique.shape[0]),
                }
            except Exception as e:
                print(f"  Failed to export {out_file}: {e}", file=sys.stderr)

    return saved, metadata


def run_component_splitting(input_dir: Path, components_dir: Path, world_size=1, rank=0):
    """Recursively walk ``input_dir`` and split every supported mesh file.

    Work is sharded across distributed workers by ``i % world_size == rank``,
    so each worker processes a disjoint subset of files. Files larger than
    ``MAX_FILE_SIZE`` are skipped with a warning.
    """
    all_paths = sorted([
        p for p in input_dir.rglob("*") if p.suffix.lower() in MESH_EXTENSIONS
    ])
    all_files = []
    for p in all_paths:
        if p.stat().st_size > MAX_FILE_SIZE:
            print(
                f"  Skipping {p}: file size {p.stat().st_size / 1e6:.1f} MB "
                f"exceeds MAX_FILE_SIZE ({MAX_FILE_SIZE / 1e6:.0f} MB)",
                file=sys.stderr,
            )
            continue
        all_files.append(p)
    my_files = [p for i, p in enumerate(all_files) if i % world_size == rank]

    print(f"[Rank {rank}] Found {len(all_files)} mesh files, processing {len(my_files)}")

    total_saved = 0
    for path in tqdm(my_files, desc=f"Rank {rank} — splitting", ncols=100):
        saved, _ = split_components(path, input_dir, components_dir)
        total_saved += saved

    print(f"[Rank {rank}] Saved {total_saved} components to {components_dir}")


# ---------------------------------------------------------------------------
# Step 2: Metadata computation
# ---------------------------------------------------------------------------

def run_metadata_computation(
    components_dir: Path, output_dir: Path, res_coarse=64, res_fine=512,
    world_size=1, rank=0,
):
    """Run ``process_one_mesh`` on each component PLY and save per-sample metadata.

    Each sample's metadata (occupancy / NVV / SDF / quad-ratio / etc.) is
    written as an individual JSON file under ``<output_dir>/metadata/``, and
    the PLY itself is copied to ``<output_dir>/meshes/``. Samples that already
    have a metadata JSON are skipped, making reruns idempotent.
    """
    from triflow.utils.mesh_processing import process_one_mesh

    meshes_dir = output_dir / "meshes"
    metadata_dir = output_dir / "metadata"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    all_plys = sorted(components_dir.rglob("*.ply"))
    my_plys = [p for i, p in enumerate(all_plys) if i % world_size == rank]

    print(f"[Rank {rank}] Computing metadata for {len(my_plys)}/{len(all_plys)} components")

    for ply_path in tqdm(my_plys, desc=f"Rank {rank} — metadata", ncols=100):
        mesh_id = ply_path.stem
        rel = ply_path.relative_to(components_dir).parent
        hashed_path = str(rel)

        # Output paths
        mesh_out = meshes_dir / rel / f"{mesh_id}.ply"
        meta_out = metadata_dir / rel / f"{mesh_id}.json"

        if meta_out.exists():
            continue  # already processed

        try:
            _, _, _, metadata = process_one_mesh(
                ply_path,
                res_coarse=res_coarse,
                res_fine=res_fine,
                round_verts=False,
                decimate_length=3.5,
                vertex_merge_threshold=0.0,
                augment=False,
                get_metadata=True,
                verbose=False,
            )

            # Copy mesh to output
            mesh_out.parent.mkdir(parents=True, exist_ok=True)
            if not mesh_out.exists():
                shutil.copy2(ply_path, mesh_out)

            # Save metadata
            metadata["hashed_path"] = hashed_path
            meta_out.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_out, "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            print(f"  Failed processing {ply_path}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Step 3: Metadata merge
# ---------------------------------------------------------------------------

def merge_metadata(output_dir: Path, res_fine=512):
    """Collect per-sample JSON files into a single ``metadata_{res_fine}.json``.

    This is what ``MeshDataset`` loads at training time.
    """
    metadata_dir = output_dir / "metadata"
    merged = {}

    json_files = sorted(metadata_dir.rglob("*.json"))
    print(f"Merging {len(json_files)} metadata files...")

    for jf in tqdm(json_files, desc="Merging", ncols=100):
        mesh_id = jf.stem
        with open(jf, "r") as f:
            merged[mesh_id] = json.load(f)

    out_file = output_dir / f"metadata_{res_fine}.json"
    with open(out_file, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Saved merged metadata ({len(merged)} samples) to {out_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry point. See the module docstring for usage."""
    parser = argparse.ArgumentParser(
        description="Prepare a mesh dataset for TriFlow training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_dir", type=str,
        help="Directory containing input mesh files (obj, glb, fbx, ply, ...).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/my_dataset",
        help="Output dataset directory (default: data/my_dataset).",
    )
    parser.add_argument(
        "--res_coarse", type=int, default=64,
        help="Coarse voxel resolution (default: 64).",
    )
    parser.add_argument(
        "--res_fine", type=int, default=512,
        help="Fine voxel resolution (default: 512).",
    )
    parser.add_argument(
        "--world_size", type=int, default=1,
        help="Number of distributed workers (default: 1).",
    )
    parser.add_argument(
        "--rank", type=int, default=0,
        help="Rank of this worker (default: 0).",
    )
    parser.add_argument(
        "--skip_split", action="store_true",
        help="Skip component splitting (use if components already exist).",
    )
    parser.add_argument(
        "--skip_metadata", action="store_true",
        help="Skip metadata computation (use if metadata already computed).",
    )
    parser.add_argument(
        "--merge_only", action="store_true",
        help="Only merge per-sample metadata into a single JSON file.",
    )
    parser.add_argument(
        "--min_faces", type=int, default=16,
        help="Minimum number of faces per component (default: 16).",
    )
    parser.add_argument(
        "--max_faces", type=int, default=32768,
        help="Maximum number of faces per component (default: 32768).",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    components_dir = output_dir / "components"

    if args.merge_only:
        merge_metadata(output_dir, res_fine=args.res_fine)
        return

    # Step 1: Component splitting
    if not args.skip_split:
        print(f"\n{'='*60}")
        print(f"Step 1: Component splitting")
        print(f"  Input:  {input_dir}")
        print(f"  Output: {components_dir}")
        print(f"{'='*60}\n")
        run_component_splitting(
            input_dir, components_dir,
            world_size=args.world_size, rank=args.rank,
        )

    # Step 2: Metadata computation
    if not args.skip_metadata:
        print(f"\n{'='*60}")
        print(f"Step 2: Metadata computation")
        print(f"  Input:  {components_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Resolution: {args.res_fine} (fine), {args.res_coarse} (coarse)")
        print(f"{'='*60}\n")
        run_metadata_computation(
            components_dir, output_dir,
            res_coarse=args.res_coarse, res_fine=args.res_fine,
            world_size=args.world_size, rank=args.rank,
        )

    # Step 3: Merge metadata (only rank 0 or single process)
    if args.rank == 0 and args.world_size == 1:
        print(f"\n{'='*60}")
        print(f"Step 3: Merging metadata")
        print(f"{'='*60}\n")
        merge_metadata(output_dir, res_fine=args.res_fine)
    elif args.rank == 0:
        print(
            f"\nAll ranks done. Run with --merge_only to merge metadata:\n"
            f"  python -m triflow.datasets.prepare_dataset {args.input_dir} "
            f"--output_dir {args.output_dir} --merge_only\n"
        )


if __name__ == "__main__":
    main()
