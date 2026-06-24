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


def axis_to_euler_deg(axis) -> Tensor:
    """Convert a direction vector into XYZ Euler angles (degrees).

    The legacy / LLM schema describes an axis-aligned primitive (cylinder, cone,
    capsule) by the *direction* its long axis should point, rather than by a
    rotation. Our canonical primitives are aligned to local ``+Z``, so this
    returns the minimal rotation that maps ``+Z`` onto ``axis``, expressed in the
    same XYZ-degrees convention as :func:`euler_deg_to_matrix`.

    Args:
        axis: length-3 direction (need not be normalised).

    Returns:
        ``(3,)`` tensor ``(rx, ry, rz)`` in degrees. A zero/degenerate axis maps
        to no rotation.
    """
    a = torch.as_tensor(axis, dtype=torch.float32).flatten()
    n = a.norm()
    if n < 1e-8:
        return torch.zeros(3)
    a = a / n
    z = torch.tensor([0.0, 0.0, 1.0])
    c = float(torch.dot(a, z))  # cos(angle) between +Z and axis
    if c > 1 - 1e-8:  # already +Z
        return torch.zeros(3)
    if c < -1 + 1e-8:  # antiparallel: 180 deg about X
        return torch.tensor([180.0, 0.0, 0.0])
    # Rodrigues rotation taking +Z -> a, then extract XYZ Euler from the matrix.
    v = torch.linalg.cross(z, a)
    s = v.norm()
    k = v / s
    kx = torch.tensor([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ])
    r = torch.eye(3) + kx + kx @ kx * ((1 - c) / (s * s))
    return _matrix_to_euler_deg(r)


def _matrix_to_euler_deg(r: Tensor) -> Tensor:
    """Inverse of :func:`euler_deg_to_matrix` for ``R = Rz @ Ry @ Rx``."""
    sy = torch.clamp(-r[2, 0], -1.0, 1.0)
    ry = torch.asin(sy)
    if torch.abs(r[2, 0]) < 1 - 1e-6:
        rx = torch.atan2(r[2, 1], r[2, 2])
        rz = torch.atan2(r[1, 0], r[0, 0])
    else:  # gimbal lock
        rx = torch.atan2(-r[1, 2], r[1, 1])
        rz = torch.zeros(())
    return torch.stack([rx, ry, rz]) * (180.0 / torch.pi)


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
