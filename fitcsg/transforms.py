"""Rigid transforms shared by every SDF primitive.

Convention (chosen during the 2026 refactor to make the codebase consistent):

* A primitive is authored in its own *canonical/local* frame, centred at the
  origin and aligned to the +Z axis where an axis is meaningful (cylinder,
  cone).
* ``center`` is the primitive origin expressed in world coordinates.
* ``rotation`` is an Euler-angle triple ``(rx, ry, rz)`` **in degrees** applied
  in XYZ order. The rotation matrix maps *local -> world*::

      R = Rz @ Ry @ Rx
      p_world = R @ p_local + center

To evaluate an SDF we need the inverse map *world -> local*. Because ``R`` is a
rotation (an isometry) the local-frame distance is exactly the world-frame
distance, so we can transform the query points into the local frame, evaluate
the canonical SDF, and the result is still a valid signed distance::

      p_local = R^T @ (p_world - center)

Everything here is differentiable in ``center`` and ``rotation`` so the angles
can be optimised directly.
"""

from __future__ import annotations

import torch
from torch import Tensor


def euler_deg_to_matrix(angles_deg: Tensor) -> Tensor:
    """Build a 3x3 rotation matrix from XYZ Euler angles given in degrees.

    Args:
        angles_deg: ``(3,)`` tensor ``(rx, ry, rz)`` in degrees.

    Returns:
        ``(3, 3)`` rotation matrix ``R = Rz @ Ry @ Rx`` (local -> world).
    """
    a = angles_deg * (torch.pi / 180.0)
    cx, cy, cz = torch.cos(a[0]), torch.cos(a[1]), torch.cos(a[2])
    sx, sy, sz = torch.sin(a[0]), torch.sin(a[1]), torch.sin(a[2])

    # Stack the individual axis rotations and compose. We build them with
    # torch.stack so the graph stays differentiable w.r.t. the angles.
    one = torch.ones_like(cx)
    zero = torch.zeros_like(cx)

    rx = torch.stack([
        torch.stack([one, zero, zero]),
        torch.stack([zero, cx, -sx]),
        torch.stack([zero, sx, cx]),
    ])
    ry = torch.stack([
        torch.stack([cy, zero, sy]),
        torch.stack([zero, one, zero]),
        torch.stack([-sy, zero, cy]),
    ])
    rz = torch.stack([
        torch.stack([cz, -sz, zero]),
        torch.stack([sz, cz, zero]),
        torch.stack([zero, zero, one]),
    ])
    return rz @ ry @ rx


def world_to_local(points: Tensor, center: Tensor, rotation_deg: Tensor) -> Tensor:
    """Transform world-space query points into a primitive's local frame.

    Args:
        points: ``(N, 3)`` query points in world coordinates.
        center: ``(3,)`` primitive origin in world coordinates.
        rotation_deg: ``(3,)`` XYZ Euler angles in degrees.

    Returns:
        ``(N, 3)`` points expressed in the primitive's local frame.
    """
    r = euler_deg_to_matrix(rotation_deg)
    # p_local = R^T (p - c); for row-vectors this is (p - c) @ R.
    return (points - center) @ r
