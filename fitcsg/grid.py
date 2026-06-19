"""Sampling helpers for evaluating / visualising SDFs."""

from __future__ import annotations

import torch
from torch import Tensor


def create_grid(num_points: int = 100, bound: float = 1.0, device="cpu") -> Tensor:
    """Return a dense ``(num_points**3, 3)`` grid of points in ``[-bound, bound]^3``."""
    axis = torch.linspace(-bound, bound, num_points, device=device)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack([x.ravel(), y.ravel(), z.ravel()], dim=-1)


def surface_points(points: Tensor, sdf: Tensor, level: float = 0.0) -> Tensor:
    """Return the subset of ``points`` whose SDF is below ``level`` (the solid)."""
    return points[sdf.flatten() <= level]
