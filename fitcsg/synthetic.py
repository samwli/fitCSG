"""Generate self-contained target SDFs (no external data required).

The original experiments fit to a real, masked RGB-D point cloud
(``sunglasses_data/``) that is no longer available.  To keep the optimisation
pathway runnable as a demo / sanity check, we sample a target directly from a
known "ground-truth" CSG tree:

* dense grid samples give signed distances everywhere (good supervision);
* the recovered fit can be compared against the GT parameters.

This exercises exactly the same ``CSG -> SDF`` and optimisation code paths that
real data would, minus the (separate) point-cloud capture step.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from .csg import Node, evaluate
from .grid import create_grid


def sample_target_from_tree(
    tree: Node,
    grid_size: int = 48,
    num_points: int = 20000,
    bound: float = 1.0,
    near_surface_fraction: float = 0.5,
    device="cpu",
) -> Tuple[Tensor, Tensor]:
    """Sample ``(points, sdf_values)`` from a ground-truth CSG tree.

    A mix of uniform volume samples and near-surface samples is returned so the
    optimiser sees both the zero level set and the inside/outside signs.
    """
    grid = create_grid(grid_size, bound=bound, device=device)
    grid_sdf, _ = evaluate(tree, grid)

    n_surface = int(num_points * near_surface_fraction)
    n_uniform = num_points - n_surface

    # Uniform volume samples.
    uniform = (torch.rand(n_uniform, 3, device=device) * 2 - 1) * bound

    # Near-surface samples: take the grid cells closest to the level set and
    # jitter them slightly.
    order = grid_sdf.abs().argsort()
    near = grid[order[: max(n_surface, 1)]]
    if near.shape[0] < n_surface:
        reps = (n_surface // near.shape[0]) + 1
        near = near.repeat(reps, 1)[:n_surface]
    near = near + torch.randn_like(near) * 0.02

    points = torch.cat([uniform, near], dim=0)
    values, _ = evaluate(tree, points)
    return points, values
