# Copyright (c) 2026 Haoxuan Li.
# Licensed under the Automotive Development Public Non-Commercial License v1.0.
# See LICENSE for details.

import os

import mcubes
import numpy as np
import pyrender
import torch
import trimesh
from matplotlib import cm
from pyrender.constants import RenderFlags

os.environ["PYOPENGL_PLATFORM"] = "egl"


def look_at(eye, target, up):
    """Build a 4x4 camera-to-world pose matrix for pyrender.

    Args:
        eye: ``(3,)`` array_like camera position.
        target: ``(3,)`` array_like point the camera looks at.
        up: ``(3,)`` array_like approximate up direction.

    Returns:
        A ``(4, 4)`` ndarray camera-to-world transform.
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    # Camera axes
    forward = eye - target
    forward /= np.linalg.norm(forward)

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    up = np.cross(forward, right)

    # Build pose matrix
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = forward
    pose[:3, 3] = eye

    return pose


def render_mesh(obj: trimesh.Trimesh, eye=np.array([2, 1.4, -2]), with_wireframe=False):
    """Render a trimesh with a fixed camera and headlight.

    The mesh is first centered and rescaled into the ``[-1, 1]`` cube so
    calls to this helper are invariant to the mesh's original scale. Uses
    an 800x800 offscreen pyrender context with a directional light attached
    to the camera.

    Args:
        obj: A ``trimesh.Trimesh`` to render (modified in place to normalize).
        eye: Camera eye position in world coordinates.
        with_wireframe: If ``True``, overlay wireframe edges on the render.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(color, depth)`` images of shape
        ``(800, 800, 3)`` and ``(800, 800)``.
    """
    assert isinstance(obj, trimesh.Trimesh), "Input must be a trimesh.Trimesh object."
    # normalize mesh into [-1, 1] cube
    verts = np.asarray(obj.vertices)
    if len(verts) > 0:
        bbox_max, bbox_min = np.amax(verts, axis=0), np.amin(verts, axis=0)
        center = (bbox_max + bbox_min) / 2.0
        scale = 1.0 / np.max(bbox_max - bbox_min)
        verts = (verts - center[None, :]) * scale
        obj.vertices = verts

    mesh = pyrender.Mesh.from_trimesh(
        obj,
        smooth=False,
    )

    for p in mesh.primitives:
        p.material.doubleSided = True

    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 6.0, aspectRatio=1.0)
    target = np.array([0, 0, 0])
    up = np.array([0, 1, 0])

    camera_pose = look_at(eye, target, up)
    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=1e3)
    scene.add(light, pose=camera_pose)
    renderer = pyrender.OffscreenRenderer(800, 800)
    color, depth = renderer.render(
        scene, flags=RenderFlags.ALL_WIREFRAME if with_wireframe else RenderFlags.NONE
    )
    renderer.delete()
    return color, depth


def render_occ_grid(voxel_grid, cmap=None, eye=np.array([2, 1.4, -2])):
    """Render a multi-channel occupancy grid with one color per channel.

    Later channels take precedence: a voxel is drawn with channel ``c``'s
    color only if it is occupied in channel ``c`` and not in any later
    channel. Used to visualize joint face/edge/vertex masks where labels
    overlap.

    Args:
        voxel_grid: Shape ``(C, R, R, R)``, values ``-1`` / ``+1``. If
            ``C=1`` the multi-channel logic degenerates to a single-mask
            render.
        cmap: Optional channel -> RGB color mapping. Defaults to
            gray/blue/red for ``C <= 3``, or matplotlib ``tab10`` / ``hsv``
            otherwise.
        eye: Camera eye position.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(color, depth)``.
    """

    if isinstance(voxel_grid, torch.Tensor):
        voxel_grid = voxel_grid.detach().cpu().numpy()

    # Number of channels
    C = voxel_grid.shape[0]
    res = voxel_grid.shape[1]

    # Binary occupancy per channel
    occ = voxel_grid > 0.0

    # Make channels exclusive: later channels win
    for c in range(C - 1):
        occ[c] &= ~np.any(occ[c + 1 :], axis=0)

    meshes = []
    # Pick distinct colors from colormap (tab10 for up to 10, else hsv)
    if cmap is None:
        if C <= 3:
            cmap = {
                0: (128, 128, 128),  # face -> gray
                1: (0, 0, 255),  # edge -> blue
                2: (255, 0, 0),  # vertex -> red
            }
        else:
            cmap = cm.get_cmap("tab10" if C <= 10 else "hsv", C)

    for c in range(C):
        if np.any(occ[c]):
            vg = trimesh.voxel.VoxelGrid(occ[c])
            mesh_c = vg.as_boxes()
            # Assign color
            if isinstance(cmap, dict):
                color = np.array(cmap[c])
            else:
                color = (np.array(cmap(c)[:3]) * 255).astype(np.uint8)
            mesh_c.visual.vertex_colors = np.tile(color, (len(mesh_c.vertices), 1))
            meshes.append(mesh_c)

    # Merge meshes
    mesh = trimesh.util.concatenate(meshes)

    return render_mesh(mesh, eye=eye)


def render_sdf_grid(sdf_grid, level=0.0, is_object_centered=True):
    """Render an SDF volume by extracting an iso-surface with marching cubes.

    Args:
        sdf_grid: ``(1, R, R, R)`` array of signed distances. Inside is
            assumed negative.
        level: Iso-value at which to extract the surface.
        is_object_centered: If ``True``, pads the SDF with a value of ``+1``
            before marching cubes so that objects touching the grid boundary
            still produce closed meshes.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(color, depth)``.
    """

    if isinstance(sdf_grid, torch.Tensor):
        sdf_grid = sdf_grid.detach().cpu().numpy()

    sdf = sdf_grid[0]
    if is_object_centered:
        sdf = np.pad(sdf, 1, mode="constant", constant_values=1.0)

    verts, faces = mcubes.marching_cubes(
        -sdf, -level
    )  # ? mcubes assumes inside to be positive
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return render_mesh(mesh)


def render_occ_dir_grid(voxel_grid, eye=np.array([2, 1.4, -2])):
    """Render an occupancy grid colored by a per-voxel direction.

    The first channel is a binary occupancy mask; channels 1-3 are a
    direction vector per voxel. The direction is rescaled and mapped to
    RGB (with ``(x, y, z) -> (R, G, B)``) before being baked into the voxel
    boxes' vertex colors. Used to visualize NVV fields.

    Args:
        voxel_grid: Shape ``(4, R, R, R)``; channel 0 occupancy, channels
            1-3 direction.
        eye: Camera eye position.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(color, depth)``.
    """

    if isinstance(voxel_grid, torch.Tensor):
        voxel_grid = voxel_grid.detach().cpu().numpy()

    # Number of channels
    C = voxel_grid.shape[0]
    res = voxel_grid.shape[1]

    # Binary occupancy per channel
    occ = voxel_grid[0] > 0
    color = voxel_grid[1:]  # (3, res, res, res)
    color = color * 3
    color = (color + 1) / 2  # Normalize to [0, 1]
    color = (color * 255).clip(0, 255).astype(np.uint8)  # Scale to [0, 255]
    color = np.transpose(color, (1, 2, 3, 0))  # (res, res, res, 3)

    # Create voxel grid from occupancy
    vg = trimesh.voxel.VoxelGrid(occ)
    mesh = vg.as_boxes(colors=color)

    return render_mesh(mesh, eye=eye)


def render_occ_color_grid(voxel_grid, eye=np.array([2, 1.4, -2])):
    """Render an occupancy grid colored by a per-voxel scalar or RGB value.

    Channel 0 is a binary occupancy mask. The remaining channels are
    interpreted as per-voxel color: either a single channel (broadcast to
    grayscale) or three channels ``(R, G, B)`` in ``[0, 1]``.

    Args:
        voxel_grid: Shape ``(2, R, R, R)`` or ``(4, R, R, R)`` — channel 0
            occupancy, channel 1 grayscale or channels 1-3 RGB.
        eye: Camera eye position.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(color, depth)``.
    """

    if isinstance(voxel_grid, torch.Tensor):
        voxel_grid = voxel_grid.detach().cpu().numpy()

    # Number of channels
    C = voxel_grid.shape[0]
    res = voxel_grid.shape[1]

    # Binary occupancy per channel
    occ = voxel_grid[0] > 0
    color = voxel_grid[1:]  # (3, res, res, res)
    if color.shape[0] == 1:
        color = color.repeat(3, axis=0)
    color = (color * 255).clip(0, 255).astype(np.uint8)  # Scale to [0, 255]
    color = np.transpose(color, (1, 2, 3, 0))  # (res, res, res, 3)

    # Create voxel grid from occupancy
    vg = trimesh.voxel.VoxelGrid(occ)
    mesh = vg.as_boxes(colors=color)

    return render_mesh(mesh, eye=eye)
