"""Rigid + scale alignment (similarity ICP).

This is used by the point-cloud -> target-SDF pathway to bring the *observed*
object point cloud into the same normalised frame as the CSG model before
optimisation.  See the handoff notes: the object's canonical orientation is
currently assumed (we only solve a similarity transform here, not a full coarse
pose search), which is one of the known shortcomings to revisit.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def similarity_icp(
    source: Tensor,
    target: Tensor,
    max_iters: int = 50,
    tolerance: float = 1e-6,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Align ``source`` to ``target`` with an iterative closest point loop.

    Solves for a similarity transform (rotation ``R``, isotropic ``scale`` and
    translation ``T``) such that ``scale * R @ source + T ~= target`` using the
    closed-form Umeyama/Kabsch step on nearest-neighbour correspondences.

    Args:
        source: ``(M, 3)`` point cloud to be transformed.
        target: ``(N, 3)`` reference point cloud.
        max_iters: maximum ICP iterations.
        tolerance: stop once ``R`` and ``T`` change by less than this.

    Returns:
        ``(R, scale, T)`` with ``R`` ``(3, 3)``, ``scale`` scalar tensor,
        ``T`` ``(3,)``.
    """
    # TODO (roadmap #3): this is a *local* solver and needs a decent initial
    # pose; it diverges under large rotation offsets. Replace with / precede by
    # real coarse alignment (global registration, symmetry-aware pose search, or
    # a learned/LLM pose prior) before CSG refinement.
    device = source.device
    R = torch.eye(3, device=device)
    T = torch.zeros(3, device=device)
    scale = torch.ones((), device=device)

    src = source.clone()
    for _ in range(max_iters):
        # Nearest neighbour in target for each source point.
        dists = torch.cdist(src, target)
        nn = target[torch.argmin(dists, dim=1)]

        src_c = src.mean(dim=0)
        nn_c = nn.mean(dim=0)
        src_centered = src - src_c
        nn_centered = nn - nn_c

        scale = nn_centered.norm(dim=1).mean() / (src_centered.norm(dim=1).mean() + 1e-12)

        H = (scale * src_centered).T @ nn_centered
        U, _, Vt = torch.linalg.svd(H)
        R_new = Vt.T @ U.T
        if torch.linalg.det(R_new) < 0:  # fix reflection
            Vt = Vt.clone()
            Vt[-1, :] *= -1
            R_new = Vt.T @ U.T

        T_new = nn_c - (R_new @ (src_c * scale))

        if torch.norm(T - T_new) < tolerance and torch.norm(R - R_new) < tolerance:
            R, T = R_new, T_new
            break
        R, T = R_new, T_new
        src = (scale * source) @ R.T + T

    return R, scale, T
