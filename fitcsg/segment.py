"""Assign observed point-cloud points to the CSG parts of a fitted model.

Once a CSG hypothesis has been fitted to an object, a useful downstream product
(e.g. for part-based grasping) is a labelling of the *observed* points by which
primitive part they belong to -- the body of a mug vs. its handle, the blade of
a knife vs. its handle, and so on.

The strategy mirrors the original research code: rasterise the fitted solid on a
dense grid, colour each interior point by the primitive that "wins" the CSG
operations there (subtracted cavities therefore never claim points), then label
every observed point by its nearest coloured solid point.

This module is intentionally simple and dependency-free (pure torch). It is
*independent* of how the points were obtained (synthetic or real).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor

from .csg import Node, evaluate, iter_leaves
from .grid import create_grid, surface_points


def _solid_reference(
    tree: Node, leaves: List, grid_res: int, bound: float, device
) -> Tuple[Tensor, Tensor]:
    """Return interior solid points and their owning-leaf index."""
    grid = create_grid(grid_res, bound=bound, device=device)
    sdf, colors = evaluate(tree, grid, with_colors=True)
    inside = sdf.flatten() <= 0
    pts = grid[inside]
    cols = colors[inside]

    leaf_colors = torch.tensor([leaf.color for leaf in leaves], device=device, dtype=pts.dtype)
    # Each interior colour matches exactly one leaf colour; nearest-match is robust.
    idx = torch.cdist(cols, leaf_colors).argmin(dim=1)
    return pts, idx


def _nearest_label(points: Tensor, ref_pts: Tensor, ref_idx: Tensor, chunk: int = 4096) -> Tensor:
    """Label each point by the leaf index of its nearest reference point."""
    labels = torch.empty(points.shape[0], dtype=torch.long, device=points.device)
    for start in range(0, points.shape[0], chunk):
        sl = slice(start, start + chunk)
        nn = torch.cdist(points[sl], ref_pts).argmin(dim=1)
        labels[sl] = ref_idx[nn]
    return labels


def segment_point_cloud(
    tree: Node,
    points: Tensor,
    *,
    grid_res: int = 96,
    bound: float = 1.0,
    max_reference: int = 40000,
    device="cpu",
) -> Tuple[Dict[str, Tensor], Tensor, List[str]]:
    """Label observed ``points`` by the fitted CSG part they belong to.

    Args:
        tree: a fitted CSG tree (its leaf params define the solid).
        points: ``(N, 3)`` observed points, in the same normalised frame as the
            fit.
        grid_res: resolution of the rasterised reference solid.
        bound: half-extent of the sampling grid (match the fit's coordinate box).
        max_reference: cap on reference points (subsampled for speed/memory).

    Returns:
        ``(parts, labels, names)`` where ``parts`` maps each leaf name to its
        assigned ``(M, 3)`` points, ``labels`` is the per-point leaf index, and
        ``names`` lists leaf names in index order.
    """
    points = points.to(device)
    leaves = list(iter_leaves(tree))
    names = [leaf.name for leaf in leaves]

    ref_pts, ref_idx = _solid_reference(tree, leaves, grid_res, bound, device)
    if ref_pts.numel() == 0:
        raise ValueError("Fitted solid is empty; cannot segment (check the fit).")
    if ref_pts.shape[0] > max_reference:
        sel = torch.randperm(ref_pts.shape[0], device=device)[:max_reference]
        ref_pts, ref_idx = ref_pts[sel], ref_idx[sel]

    labels = _nearest_label(points, ref_pts, ref_idx)
    parts = {name: points[labels == i] for i, name in enumerate(names)}
    return parts, labels, names
