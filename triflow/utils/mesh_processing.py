# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.

import ctypes
import time

import meshiki
import meshlib.mrmeshnumpy as mrmeshnumpy
import meshlib.mrmeshpy as mrmesh
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

from triflow.utils.sparse_voxel import get_coords_coarse2fine


def pack_trimesh(mesh):
    """Clean up a trimesh in place: merge duplicate verts, drop degenerate
    and unreferenced geometry. Returns the same mesh for chaining.
    """
    trimesh.grouping.merge_vertices(
        mesh, merge_tex=True, merge_norm=True, digits_vertex=5
    )
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def extract_point_proj_results(result):
    """Extract face indices and barycentrics from a ``std_vector_MeshProjectionResult``.

    Reads the underlying C++ struct fields directly via ctypes to avoid a
    per-element Python loop. The returned barycentric coordinates satisfy
    ``bary[:, 0] + bary[:, 1] + bary[:, 2] = 1``, with column ``i`` equal
    to 1 when the projection lands on triangle vertex ``i``.

    Args:
        result: A ``mrmesh.std_vector_MeshProjectionResult``.

    Returns:
        ``(face_ids, bary)`` where ``face_ids`` has shape ``(N,)`` and
        ``bary`` has shape ``(N, 3)``.
    """
    n = len(result)
    elem_size = (
        mrmesh.std_vector_MeshProjectionResult.element_type_byte_size
    )  # 32 bytes
    stride_int32 = elem_size // 4  # 8 int32 units per element

    # --- Offsets ---
    face_off = (
        mrmesh.MeshProjectionResult._offsetof_proj + mrmesh.PointOnFace._offsetof_face
    )  # likely 0

    mtp_off = mrmesh.MeshProjectionResult._offsetof_mtp  # 16
    bary_off = mrmesh.MeshTriPoint._offsetof_bary  # 4
    a_off = mrmesh.TriPointf._offsetof_a  # 0
    b_off = mrmesh.TriPointf._offsetof_b  # 4

    a_abs = mtp_off + bary_off + a_off  # 20
    b_abs = mtp_off + bary_off + b_off  # 24

    base = result.data_pointer()

    total_ints = n * stride_int32
    array_type = ctypes.c_int32 * total_ints
    raw = np.ctypeslib.as_array(array_type.from_address(base))

    face_ids = raw[face_off // 4 : total_ints : stride_int32]

    a_vals = raw[a_abs // 4 : total_ints : stride_int32].view(np.float32)
    b_vals = raw[b_abs // 4 : total_ints : stride_int32].view(np.float32)

    c_vals = 1.0 - a_vals - b_vals

    bary = np.empty((n, 3), dtype=np.float32)
    bary[:, 1] = a_vals  # a == 1 -> vertex 1
    bary[:, 2] = b_vals  # b == 1 -> vertex 2
    bary[:, 0] = c_vals  # c == 1 -> vertex 0

    return face_ids, bary


def merge_close_vertices(vertices, threshold=2.0):
    """Snap groups of vertices within ``threshold`` distance to a single point.

    Uses ``trimesh.grouping.group_distance`` to find the groups and replaces
    every vertex in a group with the group's representative. The mesh's face
    connectivity is left untouched; this is typically followed by a
    :func:`pack_trimesh` call to drop the resulting degenerate faces.

    Args:
        vertices: ``(N, 3)`` array of positions.
        threshold: Maximum distance for two vertices to be merged.

    Returns:
        A new ``(N, 3)`` array with merged positions.
    """
    unique, groups_dist = trimesh.grouping.group_distance(vertices, distance=threshold)
    verts = vertices.copy()
    for u, g in zip(unique, groups_dist):
        verts[g] = u
    return verts


def decimate_mrmesh(mesh, min_edge_length=2.0):
    """Collapse short edges of an mrmesh in place.

    Runs ``mrmesh.decimateMesh`` with a pre-collapse callback that only
    allows collapses on edges shorter than ``min_edge_length``. This is used
    after voxel-scale discretization to merge voxel-adjacent vertices that
    would otherwise produce tiny faces.

    Args:
        mesh: A ``mrmesh.Mesh`` (modified in place).
        min_edge_length: Minimum allowed edge length before collapse.
    """
    mesh.packOptimally()

    settings = mrmesh.DecimateSettings()

    settings.tinyEdgeLength = min_edge_length
    settings.maxError = float("inf")
    settings.maxEdgeLen = min_edge_length * 5.0
    settings.optimizeVertexPos = False

    def pre_collapse(edge_id, _):
        org, dst = mesh.topology.org(edge_id), mesh.topology.dest(edge_id)
        p0 = mesh.points[org]
        p1 = mesh.points[dst]
        length = (p0 - p1).length()
        return length < min_edge_length

    settings.preCollapse = pre_collapse

    mrmesh.decimateMesh(mesh, settings)

    mesh.pack()


def discretize_mesh(
    mesh,
    resolution,
    pad_space,
    merge_threshold=2.0,
    round_verts=False,
    verbose=True,
):
    """Scale and translate a mesh into an integer voxel grid.

    The mesh's bounding box is rescaled so its longest side fits within
    ``resolution - 2 * pad_space`` voxels, then centered at
    ``(resolution / 2, resolution / 2, resolution / 2)``.

    Args:
        mesh: ``mrmesh.Mesh`` to discretize (modified in place).
        resolution: Target grid resolution.
        pad_space: Number of voxels of padding to leave around the bounding
            box.
        merge_threshold: If ``> 0``, nearby vertices are snapped together
            via :func:`merge_close_vertices` after discretization.
        round_verts: If ``True``, round each vertex coordinate to the nearest
            voxel center ``(i + 0.5)``.
        verbose: Print progress.

    Returns:
        ``(mesh, metadata)`` where ``metadata`` contains ``scale_factor``.
    """
    metadata = {}
    if verbose:
        print(
            f"Discretizing mesh to resolution {resolution} with padding {pad_space}..."
        )

    bbox = mesh.computeBoundingBox()
    size = bbox.max - bbox.min
    max_dim = max(size.x, size.y, size.z)

    if max_dim == 0:
        scale_factor = 1.0
    else:
        scale_factor = (resolution - 2 * pad_space) / max_dim
    metadata["scale_factor"] = scale_factor
    center = (bbox.min + bbox.max) / 2
    translation_to_origin = mrmesh.Vector3f(0, 0, 0) - center

    scale_val = scale_factor
    col_x = mrmesh.Vector3f(scale_val, 0.0, 0.0)
    col_y = mrmesh.Vector3f(0.0, scale_val, 0.0)
    col_z = mrmesh.Vector3f(0.0, 0.0, scale_val)

    scale_mtx = mrmesh.Matrix3f(col_x, col_y, col_z)

    xform = mrmesh.AffineXf3f.linear(scale_mtx)
    xform = xform * mrmesh.AffineXf3f.translation(translation_to_origin)

    final_offset = mrmesh.Vector3f(resolution / 2, resolution / 2, resolution / 2)
    xform = mrmesh.AffineXf3f.translation(final_offset) * xform

    mesh.transform(xform)
    mesh.invalidateCaches()

    points_np = mrmeshnumpy.getNumpyVerts(mesh)
    faces_np = mrmeshnumpy.getNumpyFaces(mesh.topology)

    if merge_threshold > 0.0:
        points_np = merge_close_vertices(points_np, threshold=merge_threshold)

    if round_verts:
        points_np = np.round(points_np - 0.5).astype(np.float32) + 0.5

    rounded_trimesh = trimesh.Trimesh(vertices=points_np, faces=faces_np)
    rounded_trimesh = pack_trimesh(rounded_trimesh)

    mesh = mrmeshnumpy.meshFromFacesVerts(
        rounded_trimesh.faces, rounded_trimesh.vertices
    )

    mesh.pack()

    return mesh, metadata


def get_precise_occupancy(mesh, resolution, verbose=True):
    """Find the voxels that intersect the mesh surface using Open3D.

    Converts the mrmesh into an Open3D ``TriangleMesh`` and uses
    ``VoxelGrid.create_from_triangle_mesh_within_bounds`` to get
    surface-intersecting voxel indices. This is exact surface rasterization,
    not a signed-distance or conservative-hull approximation.

    Args:
        mesh: ``mrmesh.Mesh``.
        resolution: Cubic grid resolution ``R``.
        verbose: Print progress.

    Returns:
        ``np.ndarray`` of shape ``(N, 3)`` int32 voxel indices in
        ``[0, R)``.
    """
    if verbose:
        print(f"Computing precise occupancy (Open3D) for resolution {resolution}...")
    t0 = time.time()

    faces_np = mrmeshnumpy.getNumpyFaces(mesh.topology)
    verts_np = mrmeshnumpy.getNumpyVerts(mesh)

    if faces_np.size == 0:
        if verbose:
            print("  Mesh is empty. No occupancy found.")
        return np.array([], dtype=np.int32).reshape(0, 3)

    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts_np.astype(np.float64)),
        o3d.utility.Vector3iVector(faces_np),
    )

    voxel_size = 1.0
    minb, maxb = (0.0, 0.0, 0.0), (resolution, resolution, resolution)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        o3d_mesh, voxel_size, minb, maxb
    )

    voxels = voxel_grid.get_voxels()

    if not voxels:
        if verbose:
            print("  Open3D Voxelization found no occupied voxels.")
        return np.array([], dtype=np.int32).reshape(0, 3)

    indices = [voxel.grid_index for voxel in voxels]
    occupied_indices = np.array(indices, dtype=np.int32)

    mask = (
        (occupied_indices[:, 0] >= 0)
        & (occupied_indices[:, 0] < resolution)
        & (occupied_indices[:, 1] >= 0)
        & (occupied_indices[:, 1] < resolution)
        & (occupied_indices[:, 2] >= 0)
        & (occupied_indices[:, 2] < resolution)
    )

    occupied_indices = occupied_indices[mask]

    if verbose:
        print(
            f"  Precise occupancy (Open3D): {len(occupied_indices)} voxels. Time: {time.time()-t0:.2f}s"
        )
    return occupied_indices


def compute_sparse_sdf(mesh, occupied_indices, resolution, verbose=True):
    """Compute SDF values at voxel centers for a given set of occupied indices.

    Uses mrmesh's ``findSignedDistances`` for the actual query and normalizes
    the result by ``resolution`` so the SDF is expressed in normalized grid
    units.

    Args:
        mesh: ``mrmesh.Mesh``.
        occupied_indices: ``(N, 3)`` int array of voxel indices.
        resolution: Cubic grid resolution (used for normalization).
        verbose: Print progress.

    Returns:
        ``(N, 1)`` float array of normalized SDF values.
    """
    if verbose:
        print(
            f"Computing sparse SDF (direct query) for {len(occupied_indices)} voxels..."
        )
    t0 = time.time()

    voxel_centers = occupied_indices.astype(np.float32) + 0.5
    testPoints_mrmesh = mrmeshnumpy.fromNumpyArray(voxel_centers)
    signed_distances_mrmesh = mrmesh.findSignedDistances(mesh, testPoints_mrmesh)
    sdf_values = np.array(signed_distances_mrmesh.vec)
    sdf_values = sdf_values.reshape(-1, 1)
    sdf_values /= resolution

    if verbose:
        print(f"  Sparse SDF calculation complete. Time: {time.time()-t0:.2f}s")
    return sdf_values


def compute_sparse_direction(
    mesh,
    occupied_indices,
    resolution,
    augmented=False,
    prev_verts=None,
    post_verts=None,
    get_metadata=True,
    verbose=True,
):
    """Compute the NVV (nearest-vertex vector) at each occupied voxel center.

    For each voxel, finds the closest point on the mesh surface, looks up
    which triangle vertex that point is nearest to (via the largest
    barycentric coordinate), and returns the offset from the voxel center
    to that vertex, normalized by ``resolution``.

    When ``augmented`` is ``True``, the query point used for the projection
    is re-mapped from the augmented mesh back onto the pre-augmentation
    mesh: the nearest post-augmentation vertex is found, and its
    pre-augmentation counterpart is used as the actual query position. This
    keeps the NVV field consistent with the unaugmented topology while
    sampling at the augmented voxel grid.

    Args:
        mesh: ``mrmesh.Mesh`` used for the projection (the unaugmented one
            when ``augmented=True``).
        occupied_indices: ``(N, 3)`` int voxel indices.
        resolution: Cubic grid resolution (used to normalize).
        augmented: Whether to remap query points through the prev/post
            vertex pair.
        prev_verts: Pre-augmentation vertex positions (``(V, 3)``), required
            when ``augmented=True``.
        post_verts: Post-augmentation vertex positions (``(V, 3)``), required
            when ``augmented=True``.
        get_metadata: If ``True``, also return face-sampling statistics in
            the returned metadata dict.
        verbose: Print progress.

    Returns:
        ``(directions, metadata)`` where ``directions`` has shape ``(N, 3)``
        and is normalized to grid-voxel units.
    """
    if verbose:
        print(f"Computing sparse direction for {len(occupied_indices)} voxels...")
    t0 = time.time()

    metadata = {}

    voxel_centers = occupied_indices.astype(np.float32) + 0.5
    if augmented:
        tree_post = cKDTree(post_verts)
        _, idxs = tree_post.query(voxel_centers, k=1)
        query_points = prev_verts[idxs]
    else:
        query_points = voxel_centers

    Points2MeshProjector = mrmesh.PointsToMeshProjector()
    Points2MeshProjector.updateMeshData(mesh)
    voxel_centers_mrmesh = mrmeshnumpy.fromNumpyArray(query_points)
    result = mrmesh.std_vector_MeshProjectionResult()
    objxf = mrmesh.AffineXf3f()
    refobjxf = mrmesh.AffineXf3f()
    up_dist_limit = 10000.0
    low_dist_limit = 0.0
    Points2MeshProjector.findProjections(
        result, voxel_centers_mrmesh, objxf, refobjxf, up_dist_limit, low_dist_limit
    )
    if verbose:
        print(f"  Projections computed in {time.time()-t0:.2f}s")

    face_ids, bary_coords = extract_point_proj_results(result)

    verts_np = mrmeshnumpy.getNumpyVerts(mesh)
    faces_np = mrmeshnumpy.getNumpyFaces(mesh.topology)

    if get_metadata:
        triangle_id_counts = np.bincount(face_ids, minlength=len(faces_np))
        metadata["faces_sampled"] = np.sum(triangle_id_counts > 0).item()
        metadata["faces_sampled_3+times"] = np.sum(triangle_id_counts >= 3).item()
        metadata["faces_samples_0.25_quantile"] = np.quantile(
            triangle_id_counts, 0.25
        ).item()
        metadata["faces_samples_0.5_quantile"] = np.quantile(
            triangle_id_counts, 0.5
        ).item()
        metadata["faces_samples_0.75_quantile"] = np.quantile(
            triangle_id_counts, 0.75
        ).item()

    triangles = verts_np[faces_np]
    selected_triangles = triangles[face_ids]

    max_idx = np.argmax(bary_coords, axis=1)
    selected_vertices = selected_triangles[np.arange(len(max_idx)), max_idx]
    directions = selected_vertices - voxel_centers

    directions /= resolution

    if verbose:
        print(f"  Sparse direction calculation complete. Time: {time.time()-t0:.2f}s")
    return directions, metadata


def adaptive_remesh(
    trimesh_mesh,
    target_edge_length=2.0,
    get_metadata=True,
    allow_collapse=True,
    verbose=True,
):
    """Remesh a trimesh to a target uniform edge length using mrmesh's adaptive remesher.

    Args:
        trimesh_mesh: A ``trimesh.Trimesh`` to remesh.
        target_edge_length: Desired uniform edge length of the output.
        get_metadata: If ``True``, also compute chamfer / max distance from
            the input and put them in the returned metadata dict.
        allow_collapse: If ``False``, a pre-collapse hook blocks every
            collapse, preserving the original vertex count.
        verbose: Print progress.

    Returns:
        ``(remeshed_trimesh, metadata)``.
    """
    t0 = time.time()
    metadata = {}

    mesh = mrmeshnumpy.meshFromFacesVerts(
        trimesh_mesh.faces,
        trimesh_mesh.vertices,
    )

    settings = mrmesh.RemeshSettings()
    settings.targetEdgeLen = target_edge_length
    settings.useCurvature = False

    def pre_collapse(edge_id, new_pos):
        return False

    if not allow_collapse:
        settings.preCollapse = pre_collapse

    mrmesh.remesh(mesh, settings)

    verts_np = mrmeshnumpy.getNumpyVerts(mesh)
    faces_np = mrmeshnumpy.getNumpyFaces(mesh.topology)
    remeshed_trimesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)
    remeshed_trimesh = pack_trimesh(remeshed_trimesh)

    if get_metadata:
        remeshed_mrmesh = mrmeshnumpy.meshFromFacesVerts(
            remeshed_trimesh.faces, remeshed_trimesh.vertices
        )
        orig_points = trimesh_mesh.sample(100000)
        testPoints_mrmesh = mrmeshnumpy.fromNumpyArray(orig_points)
        signed_distances_mrmesh = mrmesh.findSignedDistances(remeshed_mrmesh, testPoints_mrmesh)
        sdf_values = np.array(signed_distances_mrmesh.vec)
        chamfer_dist = np.mean(np.abs(sdf_values))
        max_dist = np.max(np.abs(sdf_values))

        metadata["remesh_chamfer_dist"] = chamfer_dist
        metadata["remesh_max_dist"] = max_dist
        metadata["remesh_num_vertices"] = len(remeshed_trimesh.vertices)
        metadata["remesh_num_faces"] = len(remeshed_trimesh.faces)

    if verbose:
        print(
            f"  Remeshed mesh: {len(remeshed_trimesh.vertices)} vertices, {len(remeshed_trimesh.faces)} faces."
        )
        if get_metadata:
            print(
                f"  Chamfer distance to original mesh: {chamfer_dist:.4f}, max dist: {max_dist:.4f}"
            )
        print(f"  Adaptive Remeshing time: {time.time()-t0:.2f}s")

    return remeshed_trimesh, metadata


def fill_hole_mrmesh(mesh):
    """Close every hole in an mrmesh in place using mrmesh's universal-metric filler.

    Needed before SDF-based remeshing so that the voxelized distance field
    does not leak through open surfaces and produce a thickened shell.
    """
    hole_edges = mesh.topology.findHoleRepresentiveEdges()
    for e in hole_edges:
        params = mrmesh.FillHoleParams()
        params.metric = mrmesh.getUniversalMetric(mesh)
        mrmesh.fillHole(mesh, e, params)


def sdf_remesh(trimesh_mesh, voxel_size=1.0, get_metadata=True, verbose=True):
    """Remesh by voxelizing the SDF and re-extracting an iso-surface.

    This is more robust to poorly-triangulated or self-intersecting input
    meshes than :func:`adaptive_remesh` because it goes through a signed
    distance volume in between. Holes are filled first via
    :func:`fill_hole_mrmesh` to keep the SDF well-defined.

    Args:
        trimesh_mesh: A ``trimesh.Trimesh``.
        voxel_size: Voxel size used to build the intermediate SDF volume.
            Smaller values preserve more detail at the cost of memory.
        get_metadata: If ``True``, also compute chamfer / max distance from
            the input.
        verbose: Print progress.

    Returns:
        ``(remeshed_trimesh, metadata)``.
    """
    t0 = time.time()
    metadata = {}

    mesh = mrmeshnumpy.meshFromFacesVerts(trimesh_mesh.faces, trimesh_mesh.vertices)
    fill_hole_mrmesh(mesh)

    params = mrmesh.MeshToVolumeParams()
    params.surfaceOffset = 3
    params.type = mrmesh.MeshToVolumeParams.Type.Signed
    params.voxelSize = mrmesh.Vector3f.diagonal(voxel_size)
    voxelsShift = mrmesh.AffineXf3f()
    params.outXf = voxelsShift
    vdbVolume = mrmesh.meshToDistanceVdbVolume(mesh, params)

    gSettings = mrmesh.GridToMeshSettings()
    gSettings.voxelSize = params.voxelSize
    gSettings.isoValue = 0.0
    remeshed_mrmesh = mrmesh.gridToMesh(vdbVolume.data, gSettings)
    remeshed_mrmesh.transform(voxelsShift)

    verts_np = mrmeshnumpy.getNumpyVerts(remeshed_mrmesh)
    faces_np = mrmeshnumpy.getNumpyFaces(remeshed_mrmesh.topology)
    remeshed_trimesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)
    remeshed_trimesh = pack_trimesh(remeshed_trimesh)

    if get_metadata:
        orig_points = trimesh_mesh.sample(100000)
        testPoints_mrmesh = mrmeshnumpy.fromNumpyArray(orig_points)
        signed_distances_mrmesh = mrmesh.findSignedDistances(remeshed_mrmesh, testPoints_mrmesh)
        sdf_values = np.array(signed_distances_mrmesh.vec)
        chamfer_dist = np.mean(np.abs(sdf_values))
        max_dist = np.max(np.abs(sdf_values))

        metadata["remesh_chamfer_dist"] = chamfer_dist
        metadata["remesh_max_dist"] = max_dist
        metadata["remesh_num_vertices"] = len(remeshed_trimesh.vertices)
        metadata["remesh_num_faces"] = len(remeshed_trimesh.faces)

    if verbose:
        print(
            f"  Remeshed mesh: {len(remeshed_trimesh.vertices)} vertices, {len(remeshed_trimesh.faces)} faces."
        )
        if get_metadata:
            print(
                f"  Chamfer distance to original mesh: {chamfer_dist:.4f}, max dist: {max_dist:.4f}"
            )
        print(f"  SDF Remeshing time: {time.time()-t0:.2f}s")

    return remeshed_trimesh, metadata


def robust_remesh(
    original_mesh,
    remesh_voxel_size=1.0,
    remesh_method="sdf",
    allow_collapse=True,
    get_metadata=True,
    verbose=True,
):
    """Remesh a trimesh, preferring SDF remeshing and falling back to adaptive.

    Tries :func:`sdf_remesh` first (more robust for irregular inputs). If it
    throws, returns an empty mesh, or produces a mesh that deviates too far
    from the original (``remesh_max_dist > 2.0``), falls back to
    :func:`adaptive_remesh` with the same effective voxel size.

    Args:
        original_mesh: Input ``trimesh.Trimesh``.
        remesh_voxel_size: Voxel size for ``sdf_remesh`` (and doubled as the
            target edge length for the adaptive fallback).
        remesh_method: ``"sdf"`` or ``"adaptive"``. ``"adaptive"`` skips the
            SDF path entirely.
        allow_collapse: Passed to the adaptive remesher; see its docstring.
        get_metadata: If ``True``, compute chamfer / max-distance metadata.
        verbose: Print progress and (if the SDF remesh raised) the exception
            that triggered the fallback.

    Returns:
        ``(remeshed_trimesh, metadata)``; ``metadata["remesh_method"]`` says
        which branch produced the result.
    """
    fallback_to_adaptive = False
    mesh = None
    metadata = {}

    if remesh_method == "sdf":
        try:
            mesh, remesh_metadata = sdf_remesh(
                original_mesh,
                voxel_size=remesh_voxel_size,
                get_metadata=get_metadata,
                verbose=verbose,
            )
            metadata.update(remesh_metadata)
            if get_metadata:
                metadata["remesh_method"] = "sdf"
            fallback_to_adaptive = (
                remesh_metadata["remesh_num_faces"] == 0
                or remesh_metadata["remesh_max_dist"] > 2.0
            )
        except Exception as e:
            fallback_to_adaptive = True
            if verbose:
                print(f"  SDF remeshing raised ({e}); will fall back to adaptive.")

    if verbose and fallback_to_adaptive:
        print("  Falling back to adaptive remeshing due to SDF remeshing issues.")

    if remesh_method == "adaptive" or fallback_to_adaptive:
        mesh, remesh_metadata = adaptive_remesh(
            original_mesh,
            target_edge_length=remesh_voxel_size * 2.0,
            allow_collapse=allow_collapse,
            get_metadata=get_metadata,
            verbose=verbose,
        )
        metadata.update(remesh_metadata)
        if get_metadata:
            metadata["remesh_method"] = "adaptive"

    return mesh, metadata


def _wendland_c2(r):
    """Wendland C² radial basis function ``(1 - r)^4 * (4 r + 1)``.

    Evaluated at ``r`` (typically a distance normalized to a support
    radius). Values at or beyond ``r = 1`` are zero. Used by
    :func:`_interpolate_displacement` as the smooth falloff weight.
    """
    mask = r < 1.0
    out = np.zeros_like(r)
    rm = r[mask]
    out[mask] = (1 - rm) ** 4 * (4 * rm + 1)
    return out


def _interpolate_displacement(vertices, ctrl_pos, ctrl_disp, radius):
    """Scatter a set of control-point displacements onto ``vertices``.

    Each control point ``ctrl_pos[i]`` contributes its displacement
    ``ctrl_disp[i]`` to every vertex within ``radius``, weighted by the
    Wendland C² falloff on the normalized distance.
    """
    disp = np.zeros_like(vertices)
    tree = cKDTree(vertices)
    for p, d in zip(ctrl_pos, ctrl_disp):
        idx = tree.query_ball_point(p, radius)
        if not idx:
            continue
        v = vertices[idx] - p
        r = np.linalg.norm(v, axis=1) / radius
        w = _wendland_c2(r)
        disp[idx] += w[:, None] * d
    return disp


def _static_falloff(vertices, static_pos, radius):
    """Per-vertex ``1 - exp(-d² / (2 r²))`` falloff against the nearest static point.

    Produces a scalar in ``[0, 1]`` per vertex: zero exactly on a static
    point and rising smoothly to 1 far away. Used by :func:`augment_mesh`
    to keep the augmented mesh anchored at the original mesh's vertices.
    """
    tree = cKDTree(static_pos)
    dists, _ = tree.query(vertices, k=1)
    mask = 1.0 - np.exp(-(dists**2) / (2 * radius**2))
    return mask[:, None]


def augment_mesh(
    original_mesh: trimesh.Trimesh,
    mesh: trimesh.Trimesh = None,
    remesh_method="sdf",
    remesh_voxel_size=1.0,
    num_bumps_low=4,
    num_bumps_high=8,
    low_freq_strength=5,
    high_freq_strength=1,
    static_radius=5,
    low_radius=5,
    high_radius=1,
    get_metadata=True,
    verbose=True,
):
    """Apply random sub-triangle distortions to a remeshed mesh for data augmentation.

    Builds two sets of control points via farthest-point sampling: a small
    "low-frequency" set (big Wendland bumps) and a larger "high-frequency"
    set (small bumps). Random displacements are scattered onto the mesh
    vertices through the Wendland kernel, then attenuated by
    :func:`_static_falloff` against the original (pre-augmentation)
    vertices so the augmentation does not drift the mesh away from its
    source anchors.

    Args:
        original_mesh: The reference ``trimesh.Trimesh`` used as static
            anchors.
        mesh: Optional pre-remeshed ``trimesh.Trimesh``. If ``None``, the
            input is first remeshed via :func:`robust_remesh`.
        remesh_method: Which remeshing backend to use when ``mesh`` is
            ``None``; forwarded to :func:`robust_remesh`.
        remesh_voxel_size: Voxel size for the initial remesh.
        num_bumps_low: Number of low-frequency (large) control points.
        num_bumps_high: Number of high-frequency (small) control points.
        low_freq_strength: Standard deviation of the low-frequency
            displacements (in mesh units).
        high_freq_strength: Standard deviation of the high-frequency
            displacements.
        static_radius: Falloff radius for the static-point mask.
        low_radius: Support radius of the Wendland kernel for low-frequency
            bumps.
        high_radius: Support radius of the Wendland kernel for
            high-frequency bumps.
        get_metadata: Passed to :func:`robust_remesh` when a remesh is
            performed internally.
        verbose: Print per-stage timings.

    Returns:
        ``(augmented_trimesh, metadata)``.
    """
    t0 = time.time()
    metadata = {}
    original_verts = np.asarray(original_mesh.vertices)
    if mesh is None:
        mesh, remesh_metadata = robust_remesh(
            original_mesh,
            remesh_voxel_size=remesh_voxel_size,
            remesh_method=remesh_method,
            get_metadata=get_metadata,
            verbose=verbose,
        )
        metadata.update(remesh_metadata)
        if verbose:
            print(f"  Mesh augmentation time for remesh: {time.time()-t0:.2f}s")

    vertices = np.asarray(mesh.vertices)
    static_pos = original_verts

    vertices_subsampled = np.random.choice(
        len(vertices), size=min(len(vertices), 16384), replace=False
    )
    vertices_subsampled = vertices[vertices_subsampled]
    low_pos = meshiki.fps(vertices_subsampled, num_bumps_low, backend="kdline")
    high_pos = meshiki.fps(vertices_subsampled, num_bumps_high, backend="kdline")

    low_disp = np.random.randn(num_bumps_low, 3) * low_freq_strength
    high_disp = np.random.randn(num_bumps_high, 3) * high_freq_strength

    if verbose:
        print(f"  Mesh augmentation time for fps: {time.time()-t0:.2f}s")

    disp_low = _interpolate_displacement(vertices, low_pos, low_disp, low_radius)
    disp_high = _interpolate_displacement(vertices, high_pos, high_disp, high_radius)
    disp = disp_low + disp_high

    if verbose:
        print(f"  Mesh augmentation time for interpolate: {time.time()-t0:.2f}s")

    mask = _static_falloff(vertices, static_pos, radius=static_radius)
    disp *= mask

    if verbose:
        print(f"  Mesh augmentation time for falloff: {time.time()-t0:.2f}s")

    new_vertices = vertices + disp

    if verbose:
        print(f"  Mesh augmentation time: {time.time()-t0:.2f}s")

    return (
        trimesh.Trimesh(
            vertices=new_vertices,
            faces=np.asarray(mesh.faces),
            process=False,
        ),
        metadata,
    )


def _compute_fine_nvv(
    mesh,
    augmented_mrmesh,
    res_fine,
    augment,
    remeshed_mesh,
    augmented_mesh,
    get_metadata,
    verbose,
):
    """Compute fine-resolution occupancy + NVV for one processed mesh.

    The occupancy mask is taken from the (possibly augmented) mrmesh, while
    the NVV direction field is queried against the pre-augmentation
    ``mesh`` — :func:`compute_sparse_direction` remaps each augmented voxel
    back through the pre/post vertex pair when ``augment`` is ``True``.

    Returns:
        ``(occ_fine, dir_fine, dir_metadata)``.
    """
    if verbose:
        print(f"\n--- Processing Fine Resolution ({res_fine}) ---")
    occ_fine = get_precise_occupancy(augmented_mrmesh, res_fine, verbose=verbose)
    dir_fine, dir_metadata = compute_sparse_direction(
        mesh,
        occ_fine,
        res_fine,
        augmented=augment,
        prev_verts=remeshed_mesh.vertices if augment else None,
        post_verts=augmented_mesh.vertices if augment else None,
        get_metadata=get_metadata,
        verbose=verbose,
    )
    return occ_fine, dir_fine, dir_metadata


def _compute_coarse_sdf(augmented_mrmesh, res_fine, res_coarse, ratio, verbose):
    """Compute coarse occupancy and its packed coarse-to-fine SDF payload.

    Builds a downscaled copy of ``augmented_mrmesh`` (so one coarse voxel
    occupies a unit cube), finds the coarse occupancy mask, expands every
    coarse voxel to its ``r^3`` fine children, and queries the SDF at each
    child center.

    Returns:
        ``(occ_coarse, sdf_coarse)`` where ``sdf_coarse`` has shape
        ``(N, r^3)``; both are empty arrays when no coarse voxel is
        occupied.
    """
    if verbose:
        print(f"\n--- Processing Coarse Resolution ({res_coarse}) ---")

    # Downscale the augmented mesh so one coarse voxel occupies a unit cube.
    mesh_coarse = mrmesh.Mesh(augmented_mrmesh)
    scale_down_val = 1.0 / ratio
    col_x_down = mrmesh.Vector3f(scale_down_val, 0.0, 0.0)
    col_y_down = mrmesh.Vector3f(0.0, scale_down_val, 0.0)
    col_z_down = mrmesh.Vector3f(0.0, 0.0, scale_down_val)
    scale_down_mtx = mrmesh.Matrix3f(col_x_down, col_y_down, col_z_down)
    scale_down = mrmesh.AffineXf3f.linear(scale_down_mtx)
    mesh_coarse.transform(scale_down)
    mesh_coarse.invalidateCaches()

    occ_coarse = get_precise_occupancy(mesh_coarse, res_coarse, verbose=verbose)
    N = len(occ_coarse)
    ratio_cubed = int(ratio) ** 3

    if verbose:
        print("Generating Coarse-to-Fine SDF mapping...")

    if N > 0:
        occ_coarse2fine = get_coords_coarse2fine(occ_coarse, ratio)
        if verbose:
            print(f"  Mapped {N} coarse voxels to {len(occ_coarse2fine)} fine voxels.")
        sdf_fine = compute_sparse_sdf(
            augmented_mrmesh,
            occ_coarse2fine,
            res_fine,
            verbose=verbose,
        )
        sdf_coarse = sdf_fine.reshape(N, ratio_cubed)
    else:
        if verbose:
            print("  No coarse voxels occupied. Skipping SDF calculation.")
        sdf_coarse = np.array([], dtype=np.float32).reshape(0, ratio_cubed)

    return occ_coarse, sdf_coarse


def process_one_mesh(
    input_file,
    res_coarse=64,
    res_fine=128,
    pad=1.5,
    round_verts=True,
    decimate_length=8.0,
    vertex_merge_threshold=2.0,
    augment=False,
    augment_density=True,
    augment_strength=1.0,
    remesh_method="sdf",
    cast=True,
    get_metadata=True,
    verbose=True,
):
    """End-to-end preprocessing of a single mesh file for training or inference.

    Loads a mesh, scales and discretizes it into a fine voxel grid, and
    computes the sparse payload used throughout the codebase:

    * ``occ_fine`` — fine-resolution occupancy mask.
    * ``nvv_fine`` — voxelized nearest-vertex vectors at each occupied fine
      voxel.
    * ``occ_coarse`` — coarse-resolution occupancy mask.
    * ``sdf_coarse2fine`` — per coarse voxel, ``r^3`` SDF values at its
      fine children (packed into one row).

    Optionally performs training-time data augmentation: random subdivision
    density increase, random scale/translation in the voxel grid, and
    sub-triangle ARAP-style random distortions via :func:`augment_mesh`.

    Args:
        input_file: Path to the input mesh (any format ``trimesh.load``
            understands; the ``.ply`` files produced by
            ``prepare_dataset`` are the usual input).
        res_coarse: Coarse grid resolution.
        res_fine: Fine grid resolution. Must be an integer multiple of
            ``res_coarse``.
        pad: Number of coarse voxels of padding around the bounding box.
        round_verts: If ``True``, snap discretized vertices to voxel
            centers.
        decimate_length: Minimum edge length threshold for post-
            discretization decimation. ``0`` disables decimation.
        vertex_merge_threshold: Minimum vertex distance for the merge step
            in :func:`discretize_mesh`.
        augment: If ``True``, applies random scale / translation and
            sub-triangle distortions.
        augment_density: If ``True``, also randomly subdivides the input
            mesh up to a face-count cap before discretization (only
            relevant under ``augment=True``).
        augment_strength: Scalar multiplier for the distortion amplitude.
        remesh_method: ``"sdf"`` or ``"adaptive"``; passed through to
            :func:`robust_remesh`.
        cast: If ``True``, cast the output arrays to compact dtypes
            (float16 / uint{8,16,32}) before returning to save memory
            during on-disk caching.
        get_metadata: If ``True``, populate the metadata dict with chamfer,
            max-distance, face-sampling, and remesh statistics.
        verbose: Print per-stage progress.

    Returns:
        ``(results, trimesh_mesh, augmented_mesh, metadata)``:

        * ``results``: dict of sparse tensors (``occ_fine``, ``nvv_fine``,
          ``res_fine``, ``occ_coarse``, ``sdf_coarse2fine``, ``res_coarse``).
        * ``trimesh_mesh``: the decimated discretized mesh as a
          ``trimesh.Trimesh``.
        * ``augmented_mesh``: the same mesh *after* the optional
          sub-triangle distortions (or ``trimesh_mesh`` when
          ``augment=False``).
        * ``metadata``: dict of per-sample statistics.
    """
    t0 = time.time()

    ratio = int(res_fine / res_coarse)
    results = {}
    metadata = {}

    if verbose:
        print(f"Loading {input_file}...")
    try:
        orig_trimesh_mesh = trimesh.load(input_file, force="mesh", process=False)
    except Exception as e:
        if verbose:
            print(f"Error loading mesh {input_file}: {e}")
        raise e

    if augment and augment_density:
        # Cap the subdivided face count at ~32k so downstream preprocessing
        # stays within a bounded memory budget even for initially-coarse inputs.
        MAX_FACES = 32000
        face_mux = MAX_FACES / len(orig_trimesh_mesh.faces)
        if face_mux > 4.0:
            max_subdiv = np.log(face_mux) / np.log(4)
            max_subdiv = np.floor(max_subdiv).astype(int)
            subdiv = np.random.randint(0, max_subdiv + 1)
        else:
            subdiv = 0
        for _ in range(subdiv):
            orig_trimesh_mesh = orig_trimesh_mesh.subdivide()

    metadata["original_num_vertices"] = len(orig_trimesh_mesh.vertices)
    metadata["original_num_faces"] = len(orig_trimesh_mesh.faces)
    metadata["original_num_edges"] = len(orig_trimesh_mesh.edges)

    mesh = mrmeshnumpy.meshFromFacesVerts(
        orig_trimesh_mesh.faces, orig_trimesh_mesh.vertices
    )

    mesh, metadata_discretize = discretize_mesh(
        mesh,
        res_fine,
        pad * ratio,
        merge_threshold=vertex_merge_threshold,
        round_verts=round_verts,
        verbose=verbose,
    )
    metadata.update(metadata_discretize)

    verts_np = mrmeshnumpy.getNumpyVerts(mesh)
    faces_np = mrmeshnumpy.getNumpyFaces(mesh.topology)
    discretized_mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)

    if decimate_length > 0.0:
        decimate_mrmesh(mesh, min_edge_length=decimate_length)

    verts_np = mrmeshnumpy.getNumpyVerts(mesh)
    faces_np = mrmeshnumpy.getNumpyFaces(mesh.topology)
    trimesh_mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)

    metadata["discretized_num_vertices"] = len(discretized_mesh.vertices)
    metadata["discretized_num_faces"] = len(discretized_mesh.faces)
    metadata["discretized_num_edges"] = len(discretized_mesh.edges)

    metadata["decimated_num_vertices"] = len(trimesh_mesh.vertices)
    metadata["decimated_num_faces"] = len(trimesh_mesh.faces)
    metadata["decimated_num_edges"] = len(trimesh_mesh.edges)

    if get_metadata:
        orig_points = discretized_mesh.sample(100000)
        testPoints_mrmesh = mrmeshnumpy.fromNumpyArray(orig_points)
        signed_distances_mrmesh = mrmesh.findSignedDistances(mesh, testPoints_mrmesh)
        sdf_values = np.array(signed_distances_mrmesh.vec)
        chamfer_dist = np.mean(np.abs(sdf_values))
        max_dist = np.max(np.abs(sdf_values))

        metadata["decimated_chamfer_dist"] = chamfer_dist
        metadata["decimated_max_dist"] = max_dist

    if augment:
        remeshed_mesh, remesh_metadata = robust_remesh(
            trimesh_mesh,
            remesh_voxel_size=1,
            remesh_method=remesh_method,
            get_metadata=get_metadata,
            verbose=verbose,
        )
        metadata.update(remesh_metadata)
        augmented_mesh, aug_metadata = augment_mesh(
            trimesh_mesh,
            mesh=remeshed_mesh,
            remesh_voxel_size=1,
            num_bumps_low=32,
            num_bumps_high=64,
            low_freq_strength=0.5 * augment_strength,
            high_freq_strength=1 * augment_strength,
            static_radius=res_fine // 8,
            low_radius=res_fine // 2,
            high_radius=res_fine // 8,
            get_metadata=get_metadata,
            verbose=verbose,
        )
        metadata.update(aug_metadata)
        augmented_mrmesh = mrmeshnumpy.meshFromFacesVerts(
            augmented_mesh.faces, augmented_mesh.vertices
        )
    else:
        if get_metadata:
            remeshed_mesh, remesh_metadata = robust_remesh(
                trimesh_mesh,
                remesh_voxel_size=1,
                remesh_method=remesh_method,
                get_metadata=get_metadata,
                verbose=verbose,
            )
            metadata.update(remesh_metadata)
        augmented_mesh = trimesh_mesh
        augmented_mrmesh = mesh

    # --- Fine-resolution occupancy + NVV ---
    occ_fine, dir_fine, dir_metadata = _compute_fine_nvv(
        mesh,
        augmented_mrmesh,
        res_fine,
        augment,
        remeshed_mesh if augment else None,
        augmented_mesh,
        get_metadata,
        verbose,
    )
    metadata.update(dir_metadata)

    results["occ_fine"] = occ_fine
    results["nvv_fine"] = dir_fine
    results["res_fine"] = res_fine

    # --- Coarse-resolution occupancy + packed SDF payload ---
    occ_coarse, sdf_coarse = _compute_coarse_sdf(
        augmented_mrmesh, res_fine, res_coarse, ratio, verbose
    )

    results["occ_coarse"] = occ_coarse
    results["sdf_coarse2fine"] = sdf_coarse
    results["res_coarse"] = res_coarse

    if cast:
        results["nvv_fine"] = results["nvv_fine"].astype(np.float16)
        results["sdf_coarse2fine"] = results["sdf_coarse2fine"].astype(np.float16)

        if res_coarse <= 2**8:
            results["occ_coarse"] = results["occ_coarse"].astype(np.uint8)
        elif res_coarse <= 2**16:
            results["occ_coarse"] = results["occ_coarse"].astype(np.uint16)
        else:
            results["occ_coarse"] = results["occ_coarse"].astype(np.uint32)

        if res_fine <= 2**8:
            results["occ_fine"] = results["occ_fine"].astype(np.uint8)
        elif res_fine <= 2**16:
            results["occ_fine"] = results["occ_fine"].astype(np.uint16)
        else:
            results["occ_fine"] = results["occ_fine"].astype(np.uint32)

    metadata["num_occ_coarse"] = len(results["occ_coarse"])
    metadata["num_occ_fine"] = len(results["occ_fine"])

    quad_ratio = compute_quad_ratio(trimesh_mesh)
    metadata["quad_ratio"] = quad_ratio

    if verbose:
        print(f"Done processing mesh in {time.time() - t0:.2f} seconds.")

    return results, trimesh_mesh, augmented_mesh, metadata


def compute_quad_ratio(mesh: trimesh.Trimesh):
    """Return the fraction of faces that can be paired into quads.

    Runs ``meshiki``'s quadrangulation pass on a copy of the mesh and
    reports its ``quad_ratio`` — a scalar in ``[0, 1]`` that is close to 1
    for meshes with largely quadrilateral topology and 0 for purely
    triangle-dominated ones. Used as a conditioning signal for the flow
    model.
    """
    meshiki_mesh = meshiki.Mesh(mesh.vertices, mesh.faces)
    meshiki_mesh.quadrangulate(thresh_bihedral=5, thresh_convex=185)
    return meshiki_mesh.quad_ratio
