"""Build a target SDF from a single masked RGB-D observation.

This is the ``image -> point cloud -> target SDF`` pathway.  The upstream
``image -> point cloud`` step happens outside this repo; here we start from a
saved organised point cloud (``H x W x >=3``) plus a boolean object ``mask`` and
produce supervision for the optimiser:

* surface points (object + a flat "footprint" on the table plane), SDF = 0;
* optionally, near-surface *band* samples with signed distances so the optimiser
  also sees the inside/outside of the object, not just the zero level set.

The observed object cloud is then aligned (similarity ICP) to the current CSG
model's surface so that optimisation happens in the model's normalised frame.

NOTE (handoff): the original experiments used a private ``sunglasses_data/``
cloud that is no longer available.  ``build_target_from_files`` is kept for when
new data is collected; the runnable demo instead uses
``fitcsg.synthetic.sample_target_from_tree`` which needs no external files.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from .alignment import similarity_icp
from .csg import Node, evaluate
from .grid import create_grid, surface_points


def sample_band_around_object(
    object_points: Tensor,
    table_z: float,
    num_sample_points: int = 5000,
    radius: float = 0.15,
    num_perturbations_per_point: int = 15,
    footprint_threshold: float = 0.01,
) -> Tuple[Tensor, Tensor]:
    """Sample a near-surface band around the object with signed distances.

    Points outside the object's xy footprint are treated as outside (positive
    sign); points inside the footprint are signed by comparing their height to
    the nearest observed surface point.
    """
    device = object_points.device
    idx = torch.randint(0, object_points.shape[0], (num_sample_points,), device=device)
    sampled = object_points[idx]

    n = num_sample_points * num_perturbations_per_point
    perturb = torch.randn(n, 3, device=device)
    perturb = perturb / perturb.norm(dim=1, keepdim=True)
    perturb *= torch.rand(n, 1, device=device) * radius

    band = sampled.repeat_interleave(num_perturbations_per_point, dim=0) + perturb
    band = band[band[:, 2] > table_z]

    dists = torch.cdist(band, object_points)
    min_dist, closest = dists.min(dim=1)

    xy_dists = torch.cdist(band[:, :2], object_points[:, :2])
    inside_footprint = xy_dists.min(dim=1).values <= footprint_threshold

    signs = torch.ones(band.shape[0], device=device)
    closest_z = object_points[closest, 2]
    signs[inside_footprint] = torch.where(
        band[inside_footprint, 2] > closest_z[inside_footprint], 1.0, -1.0
    )
    return band, signs * min_dist


def sample_box_below_footprint(
    footprint_points: Tensor,
    table_z: float,
    radius: float = 0.05,
    margin: float = 0.05,
    num_sample_points: int = 1000,
) -> Tuple[Tensor, Tensor]:
    """Sample a slab below the table plane (always outside -> positive SDF)."""
    device = footprint_points.device
    min_xy = footprint_points[:, :2].min(dim=0).values - margin
    max_xy = footprint_points[:, :2].max(dim=0).values + margin

    sx = torch.rand(num_sample_points, device=device) * (max_xy[0] - min_xy[0]) + min_xy[0]
    sy = torch.rand(num_sample_points, device=device) * (max_xy[1] - min_xy[1]) + min_xy[1]
    sz = torch.rand(num_sample_points, device=device) * radius + (table_z - radius)
    sampled = torch.stack([sx, sy, sz], dim=1)

    min_dist = torch.cdist(sampled, footprint_points).min(dim=1).values
    return sampled, min_dist


def build_target(
    object_points: Tensor,
    tree: Node,
    grid_size: int = 100,
    surface_only: bool = True,
    table_z: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Turn an observed object cloud into aligned SDF supervision.

    Args:
        object_points: ``(P, 3)`` observed surface points of the object.
        tree: current CSG model (used only to get a surface for ICP alignment).
        grid_size: resolution of the grid used to extract the model surface.
        surface_only: if True, supervise only the zero level set; otherwise add
            signed band samples.
        table_z: height of the supporting plane (the footprint level).
            TODO (roadmap #4): this assumes a single object resting on a known
            plane. For cluttered scenes, segment/identify the target object
            (optionally via a language prompt) before building the target.

    Returns:
        ``(sdf_points, sdf_values, R, scale, shift)``.  The transform maps the
        target into the model frame; invert it to push predictions back out.
    """
    device = object_points.device

    object_sdf = torch.zeros(object_points.shape[0], device=device)
    footprint = torch.cat(
        [object_points[:, :2], torch.full((object_points.shape[0], 1), table_z, device=device)],
        dim=1,
    )
    footprint_sdf = torch.zeros(footprint.shape[0], device=device)

    if surface_only:
        sdf_points = torch.cat([object_points, footprint], dim=0)
        sdf_values = torch.cat([object_sdf, footprint_sdf], dim=0)
    else:
        band_p, band_v = sample_band_around_object(object_points, table_z)
        below_p, below_v = sample_box_below_footprint(footprint, table_z)
        sdf_points = torch.cat([object_points, footprint, band_p, below_p], dim=0)
        sdf_values = torch.cat([object_sdf, footprint_sdf, band_v, below_v], dim=0)

    # Extract the current model surface and align the target onto it.
    grid = create_grid(grid_size, device=device)
    model_sdf, _ = evaluate(tree, grid)
    model_surface = surface_points(grid, model_sdf)

    R, scale, shift = similarity_icp(model_surface, object_points)
    sdf_points = (sdf_points * scale) @ R.T + shift
    sdf_values = sdf_values * scale
    return sdf_points, sdf_values, R, scale, shift


def build_target_from_files(
    pc_path: str,
    mask_path: str,
    tree: Node,
    device="cpu",
    grid_size: int = 100,
    surface_only: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Load an organised point cloud + object mask from ``.npy`` files.

    ``pc`` is expected to be ``(H, W, >=3)`` and ``mask`` ``(H, W)`` boolean.
    """
    # TODO (roadmap #1): this real-data path is implemented but UNTESTED -- the
    # original capture (sunglasses_data/*.npy) is gone. Validate once new data is
    # collected. Also note: a single masked view gives only the *visible*
    # surface, so the band/footprint sign heuristics below are crude (TODO:
    # proper TSDF fusion / visibility-aware signs / normals).
    pc = torch.tensor(np.load(pc_path), dtype=torch.float32, device=device)
    mask = torch.tensor(np.load(mask_path), dtype=torch.bool, device=device)
    object_points = pc[:, :, :3][mask]
    return build_target(object_points, tree, grid_size, surface_only)
