# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.

"""
Mesh reconstruction from NVV (voxelized nearest-vertex vectors).

Converts sparse voxel NVV predictions into triangle meshes using
QEM (Quadric Error Metric) simplification with priority watershed.
"""

import ctypes
import logging
import time

import meshlib.mrmeshnumpy as mrmeshnumpy
import meshlib.mrmeshpy as mrmesh
import numpy as np
import pyfqmr_triflow
import trimesh
from scipy.spatial import cKDTree

from triflow.utils.mesh_processing import pack_trimesh
from triflow.utils.nvv import (
    filter_nvv_geodesic_fast,
    get_target_point_priority_watershed,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def query_nearest_val(tree, points, values):
    """Look up per-point nearest-neighbor values from a pre-built KD-tree.

    Args:
        tree: A ``scipy.spatial.cKDTree`` built over some reference points.
        points: Query points of shape ``(M, D)``.
        values: Either an array of shape ``(N, ...)`` (per-reference values)
            or a list of such arrays (in which case the same index lookup is
            applied to each).

    Returns:
        An array (or list of arrays) indexed by the nearest-neighbor indices,
        with length ``M``.
    """
    dists, idxs = tree.query(points)
    if isinstance(values, list):
        val = [v[idxs] for v in values]
    else:
        val = values[idxs]
    return val


def trimesh2mrmesh(trimesh_mesh):
    """Convert a ``trimesh.Trimesh`` to a ``meshlib.mrmeshpy.Mesh``."""
    mesh_mrmesh = mrmeshnumpy.meshFromFacesVerts(
        trimesh_mesh.faces,
        trimesh_mesh.vertices,
    )
    return mesh_mrmesh


def extract_point_proj_coords(result):
    """Extract ``(x, y, z)`` coordinates from a ``std_vector_MeshProjectionResult``.

    Reads the underlying C++ struct fields directly via ctypes to avoid a
    per-element Python loop. The returned array has shape ``(N, 3)``.
    """
    n = len(result)
    elem_size = (
        mrmesh.std_vector_MeshProjectionResult.element_type_byte_size
    )  # 32 bytes
    stride_int32 = elem_size // 4  # 8 int32 units per element

    # --- Memory Offsets ---
    # PointOnFace starts at offset 0 of MeshProjectionResult
    proj_off = mrmesh.MeshProjectionResult._offsetof_proj

    # Coordinates start at offset 4 of PointOnFace
    point_base_off = proj_off + mrmesh.PointOnFace._offsetof_point  # 4

    x_abs = point_base_off + mrmesh.Vector3f._offsetof_x  # 4
    y_abs = point_base_off + mrmesh.Vector3f._offsetof_y  # 8
    z_abs = point_base_off + mrmesh.Vector3f._offsetof_z  # 12

    base = result.data_pointer()

    # ---- Read memory as raw int32s for slicing ----
    total_ints = n * stride_int32
    array_type = ctypes.c_int32 * total_ints
    raw = np.ctypeslib.as_array(array_type.from_address(base))

    # ---- Extract X, Y, Z (interpreting raw bits as float32) ----
    x_vals = raw[x_abs // 4 : total_ints : stride_int32].view(np.float32)
    y_vals = raw[y_abs // 4 : total_ints : stride_int32].view(np.float32)
    z_vals = raw[z_abs // 4 : total_ints : stride_int32].view(np.float32)

    # ---- Pack into (N, 3) array ----
    points = np.stack([x_vals, y_vals, z_vals], axis=1)

    return points


def project_points_onto_mesh(mrmesh_mesh, points):
    """Project query ``points`` onto the closest location on ``mrmesh_mesh``.

    Thin wrapper around ``meshlib``'s ``PointsToMeshProjector``. The returned
    value is a raw C++ ``std_vector_MeshProjectionResult`` — use
    ``extract_point_proj_coords`` to pull out just the projected positions.

    The distance cutoffs (``up_dist_limit = 10000.0 = 100²``, ``low_dist_limit = 0``)
    are large enough that all queries succeed in practice.
    """
    Points2MeshProjector = mrmesh.PointsToMeshProjector()
    Points2MeshProjector.updateMeshData(mrmesh_mesh)
    query_mrmesh = mrmeshnumpy.fromNumpyArray(points)
    result = mrmesh.std_vector_MeshProjectionResult()
    objxf = mrmesh.AffineXf3f()
    refobjxf = mrmesh.AffineXf3f()
    up_dist_limit = 10000.0  # (100^2)
    low_dist_limit = 0.0
    Points2MeshProjector.findProjections(
        result, query_mrmesh, objxf, refobjxf, up_dist_limit, low_dist_limit
    )
    return result


# ----------------------------------------------------------------------------
# QEM mesh reconstruction
# ----------------------------------------------------------------------------


def topology_flow2mesh_QEM(
    dense_mesh,
    coords,
    nvv,
    resolution,
    nvv_smooth_kwargs={
        "radius": 1.5,
        "threshold": 2,
        "sigma_s": 1.0,
        "sigma_r": 1,
    },
    root_threshold=0.5,
    target_face_count=1500,
    max_quadratic_error=1e6,
    merge_threshold=-1.0,
    target_position_weight=0.1,
    verbose=False,
    debug_output=None,
):
    """Convert a predicted NVV field into a triangle mesh via constrained QEM.

    Given a dense source mesh and the predicted NVV (voxelized nearest-vertex
    vectors) on a sparse coordinate grid, this function:

    1. Bilaterally smooths the NVV targets and projects them back onto the
       source mesh surface.
    2. Runs a priority watershed clustering on the filtered targets to pick
       a small set of unique vertex positions.
    3. Assembles a per-vertex quadratic matrix that penalizes deviation from
       each target position, with a weight proportional to the number of
       source voxels that voted for that target.
    4. Runs pyfqmr's QEM mesh simplification with the custom quadratics
       injected, producing the final output mesh.

    This is the core of the "flow-to-mesh" stage of the inference pipeline;
    see the paper's Method section for the watershed + constrained-QEM
    derivation.

    Args:
        dense_mesh: High-resolution source ``trimesh.Trimesh`` that QEM
            simplifies (typically from marching cubes on the input SDF).
        coords: Sparse fine-grid voxel coordinates of shape ``(N, 3)``.
        nvv: Predicted NVV vectors of shape ``(N, 3)`` (one per voxel),
            expressed in normalized ``[-1, 1]`` grid units.
        resolution: Fine-grid resolution ``R``.
        nvv_smooth_kwargs: Kwargs forwarded to ``filter_nvv_geodesic_fast``
            for the bilateral smoothing step.
        root_threshold: Threshold used by the priority watershed to decide
            which targets are "root" vertices.
        target_face_count: Target face count handed to pyfqmr.
        max_quadratic_error: Maximum quadric error allowed during
            simplification (controls how aggressively edges are collapsed).
        merge_threshold: Minimum edge length below which vertices are
            merged. Negative values disable the merge.
        target_position_weight: Scalar weight applied to the target-position
            quadratic term, before being multiplied by the per-vertex voter
            count.
        verbose: If ``True``, print per-stage timings.
        debug_output: If set, path to a directory where intermediate NVV /
            target / cluster meshes are exported as OBJ files for inspection.

    Returns:
        A ``trimesh.Trimesh`` — the simplified output mesh.
    """
    logging.getLogger("pyfqmr_triflow").setLevel(
        logging.DEBUG if verbose else logging.WARNING
    )

    def quadratic_vert_target_pose(vertices, target_poses, weights=1.0):
        """Build per-vertex quadratic matrices penalizing deviation from target positions.

        Each row of the returned ``(N, 10)`` array encodes the upper triangle
        of a 4x4 symmetric quadric matrix in pyfqmr's order
        ``[m11, m12, m13, m14, m22, m23, m24, m33, m34, m44]`` such that the
        cost of placing a vertex at ``(x, y, z)`` equals
        ``weights * ||(x, y, z) - target_pose||^2``.
        """
        custom_quadratics = np.zeros((len(vertices), 10), dtype=np.float64)

        xp, yp, zp = target_poses[:, 0], target_poses[:, 1], target_poses[:, 2]

        # m11, m22, m33 — diagonal of the quadratic term
        custom_quadratics[:, 0] = weights  # m11
        custom_quadratics[:, 4] = weights  # m22
        custom_quadratics[:, 7] = weights  # m33

        # m14, m24, m34 — linear cross terms with the bias row
        custom_quadratics[:, 3] = -weights * xp  # m14
        custom_quadratics[:, 6] = -weights * yp  # m24
        custom_quadratics[:, 8] = -weights * zp  # m34

        # m44 = w * (xp^2 + yp^2 + zp^2) — constant term
        target_sq_norms = np.sum(target_poses**2, axis=1)
        custom_quadratics[:, 9] = weights * target_sq_norms
        return custom_quadratics

    t0 = time.time()

    mesh_simplifier = pyfqmr_triflow.Simplify()
    mesh_simplifier.setMesh(dense_mesh.vertices, dense_mesh.faces)

    # --- Stage 1: bilateral smooth the NVV field and project targets to the mesh surface ---
    nvv_scaled = nvv * resolution
    target_pos = coords.astype(float) + 0.5 + nvv_scaled
    t_nvv_start = time.time()
    target_pos = filter_nvv_geodesic_fast(
        coords,
        target_pos,
        threshold=nvv_smooth_kwargs.get("threshold", 3),
        sigma_s=nvv_smooth_kwargs.get("sigma_s", 1.0),
        sigma_r=nvv_smooth_kwargs.get("sigma_r", 1),
    )
    if verbose:
        print(
            f"[flow2mesh] NVV bilateral filter done: {time.time() - t_nvv_start:.2f}s"
        )

    original_mrmesh_mesh = trimesh2mrmesh(dense_mesh)
    result = project_points_onto_mesh(original_mrmesh_mesh, target_pos)
    target_pos = extract_point_proj_coords(result)

    nvv_filtered = target_pos - (coords.astype(float) + 0.5)

    mesh_vertices = dense_mesh.vertices
    coords_tree = cKDTree(coords.astype(float) + 0.5)

    # --- Stage 2: priority watershed clustering of targets on the mesh graph ---
    t_watershed_start = time.time()
    vert_target_poses, unique_vert_ids = get_target_point_priority_watershed(
        dense_mesh,
        coords,
        nvv_filtered,
        root_threshold=root_threshold,
    )
    if verbose:
        print(
            f"[flow2mesh] Watershed target constraint done: {time.time() - t_watershed_start:.2f}s"
        )

    # --- Stage 3: build per-vertex quadratics with weight ∝ voter count ---
    unique_ids, counts = np.unique(unique_vert_ids, return_counts=True)
    quadratic_weights = np.zeros(len(mesh_vertices))
    quadratic_weights[unique_ids] = counts * target_position_weight

    custom_quadratics = quadratic_vert_target_pose(
        mesh_vertices, vert_target_poses, weights=quadratic_weights
    )

    # --- Stage 4: QEM mesh simplification with the injected quadratics ---
    t_simplify_start = time.time()
    mesh_simplifier.simplify_mesh(
        target_count=target_face_count,
        aggressiveness=7,
        max_iterations=200,
        preserve_border=False,
        verbose=verbose,
        lossless=False,
        threshold_lossless=max_quadratic_error,
        custom_quadratics=custom_quadratics,
        target_vertex_ids=unique_vert_ids,
        min_edge_length=merge_threshold,
    )
    vertices, faces, normals = mesh_simplifier.getMesh()
    if verbose:
        print(
            f"[flow2mesh] QEM simplification done: {time.time() - t_simplify_start:.2f}s"
        )
    out = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )
    pack_trimesh(out)
    out.fix_normals()

    # --- (Optional) debug output: export 4 OBJ files for visual inspection ---
    #   nvv.obj                    — dense mesh, vertex-colored by NVV direction
    #                                 (unit vectors mapped to RGB around mid-gray).
    #   target_poses.obj           — dense mesh, vertex-colored by watershed
    #                                 target position (voxel coords -> RGB).
    #   flow_targets.obj           — point cloud of the unique target positions.
    #   flow_groups_vertices.obj   — dense mesh, vertex-colored by watershed
    #                                 group label (hashed to pseudo-random RGB).
    if debug_output is not None:
        palette = np.array(
            [
                [0xC9, 0x68, 0x47],
                [0xBC, 0x9D, 0x4B],
                [0x78, 0xB1, 0x52],
                [0x5C, 0xB6, 0x96],
                [0x70, 0x98, 0xD0],
                [0xA3, 0x6E, 0xC7],
                [0xCD, 0x5F, 0x86],
            ],
            dtype=np.uint8,
        )
        div_to_gray = np.linalg.norm(palette.astype(np.float32) - 128.0, axis=1).mean()

        def labels_to_colors(labels):
            """Deterministic hash from integer labels to pseudo-random RGB triples.

            Uses three coprime multipliers so neighboring labels get visually
            distinct colors; good enough for debug visualization of cluster ids.
            """
            labels = labels.astype(np.uint32)
            r = (labels * 37) % 255
            g = (labels * 59) % 255
            b = (labels * 83) % 255
            return np.stack([r, g, b], axis=1).astype(np.uint8)

        verts = dense_mesh.vertices
        colors = query_nearest_val(coords_tree, verts, nvv_scaled)
        colors = colors / (np.linalg.norm(colors, axis=1, keepdims=True) + 1e-8)
        colors = colors * div_to_gray + 128.0
        colors = np.clip(colors, 0, 255).astype(np.uint8)

        debug_mesh = trimesh.Trimesh(
            vertices=verts,
            faces=dense_mesh.faces,
            vertex_colors=colors,
            process=False,
        )
        debug_mesh.export(f"{debug_output}/nvv.obj")

        colors = vert_target_poses / resolution * 255.0
        colors = np.clip(colors, 0, 255).astype(np.uint8)
        debug_mesh = trimesh.Trimesh(
            vertices=dense_mesh.vertices,
            faces=dense_mesh.faces,
            vertex_colors=colors,
            process=False,
        )
        debug_mesh.export(f"{debug_output}/target_poses.obj")

        unique_targets = np.unique(vert_target_poses.round(decimals=5), axis=0)
        target_cloud = trimesh.Trimesh(
            vertices=unique_targets,
            faces=None,
            process=False,
        )
        target_cloud.export(f"{debug_output}/flow_targets.obj")

        colors = labels_to_colors(unique_vert_ids)

        debug_mesh = trimesh.Trimesh(
            vertices=dense_mesh.vertices,
            faces=dense_mesh.faces,
            vertex_colors=colors,
            process=False,
        )

        debug_mesh.export(f"{debug_output}/flow_groups_vertices.obj")

    return out
