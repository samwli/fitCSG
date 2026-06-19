"""Signed-distance functions for the CSG leaf primitives.

Each primitive is defined by a *canonical* SDF that lives at the origin and is
aligned to the +Z axis.  The public :func:`sdf` wrapper first transforms the
query points into the primitive's local frame (see :mod:`fitcsg.transforms`)
and then evaluates the canonical SDF, which automatically gives correct
``center`` + ``rotation`` handling for every shape.

Parameter schema (all leaves share ``center`` and ``rotation``)::

    center   : (3,)  world-space origin
    rotation : (3,)  XYZ Euler angles in DEGREES (see transforms.py)

Shape-specific size parameters:

    sphere    : radius (scalar)
    ellipsoid : size   (3,)  -- semi-axis full lengths (diameters)
    box       : size   (3,)  -- full side lengths
    cylinder  : radius (scalar), height (scalar)   axis = local +Z
    cone      : radius (scalar), height (scalar)   axis = local +Z, apex at +Z
    torus     : radius (scalar, major), tube (scalar, minor)  axis = local +Z
    capsule   : radius (scalar), height (scalar)   segment along local +Z

Notes / design choices made during the refactor:

* The old code treated the orientation field as an ``axis`` direction for
  cylinder/cone and silently ignored it for the box/ellipsoid.  Everything is
  now a single, consistent Euler ``rotation`` and is applied to *all* shapes.
* Size-like parameters are passed through ``abs`` inside the SDF so the
  optimiser cannot drive them negative (which previously produced inside-out
  shapes).  This keeps gradients well-defined without hard constraints.
* The box and cylinder SDFs are *exact*.  The ellipsoid uses the standard
  Inigo-Quilez first-order approximation (a true ellipsoid SDF has no closed
  form).  The cone is the exact capped-cone SDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import torch
from torch import Tensor

from .transforms import world_to_local


def _dot2(v: Tensor) -> Tensor:
    """Squared L2 norm along the last dim."""
    return (v * v).sum(dim=-1)


# --------------------------------------------------------------------------- #
# Canonical SDFs (origin-centred, +Z aligned). ``p`` is ``(N, 3)`` local.
# --------------------------------------------------------------------------- #
def _sphere(p: Tensor, params: Dict[str, Tensor]) -> Tensor:
    radius = params["radius"].abs()
    return p.norm(dim=-1) - radius


def _ellipsoid(p: Tensor, params: Dict[str, Tensor]) -> Tensor:
    # Semi-axes; ``size`` stores full lengths so divide by 2.
    r = params["size"].abs() / 2 + 1e-8
    k0 = (p / r).norm(dim=-1)
    k1 = (p / (r * r)).norm(dim=-1) + 1e-8
    return k0 * (k0 - 1.0) / k1


def _box(p: Tensor, params: Dict[str, Tensor]) -> Tensor:
    half = params["size"].abs() / 2
    q = p.abs() - half
    outside = torch.clamp(q, min=0.0).norm(dim=-1)
    inside = torch.clamp(q.max(dim=-1).values, max=0.0)
    return outside + inside


def _cylinder(p: Tensor, params: Dict[str, Tensor]) -> Tensor:
    radius = params["radius"].abs()
    half_h = params["height"].abs() / 2
    d_r = p[:, :2].norm(dim=-1) - radius
    d_z = p[:, 2].abs() - half_h
    d = torch.stack([d_r, d_z], dim=-1)
    outside = torch.clamp(d, min=0.0).norm(dim=-1)
    inside = torch.clamp(d.max(dim=-1).values, max=0.0)
    return outside + inside


def _cone(p: Tensor, params: Dict[str, Tensor]) -> Tensor:
    # Exact capped cone (Inigo Quilez), axis = +Z, base at z=-h/2 with radius
    # ``radius``, apex at z=+h/2 (top radius 0).
    r1 = params["radius"].abs()  # base radius (bottom)
    r2 = torch.zeros_like(r1)  # apex radius (a true cone)
    h = params["height"].abs() / 2  # half-height

    q = torch.stack([p[:, :2].norm(dim=-1), p[:, 2]], dim=-1)  # (N, 2)
    k1 = torch.stack([r2, h])  # (2,)
    k2 = torch.stack([r2 - r1, 2 * h])  # (2,)

    rsel = torch.where(q[:, 1] < 0, r1, r2)
    ca = torch.stack([q[:, 0] - torch.minimum(q[:, 0], rsel), q[:, 1].abs() - h], dim=-1)
    t = torch.clamp((((k1 - q) * k2).sum(dim=-1)) / _dot2(k2), 0.0, 1.0)
    cb = q - k1 + k2 * t.unsqueeze(-1)
    s = torch.where((cb[:, 0] < 0) & (ca[:, 1] < 0), -1.0, 1.0)
    return s * torch.sqrt(torch.minimum(_dot2(ca), _dot2(cb)) + 1e-12)


def _torus(p: Tensor, params: Dict[str, Tensor]) -> Tensor:
    major = params["radius"].abs()  # distance from centre to tube centre
    minor = params["tube"].abs()  # tube radius
    q = torch.stack([p[:, :2].norm(dim=-1) - major, p[:, 2]], dim=-1)
    return q.norm(dim=-1) - minor


def _capsule(p: Tensor, params: Dict[str, Tensor]) -> Tensor:
    radius = params["radius"].abs()
    half_h = params["height"].abs() / 2
    pz = p.clone()
    pz[:, 2] = pz[:, 2] - torch.clamp(p[:, 2], -half_h, half_h)
    return pz.norm(dim=-1) - radius


@dataclass
class Primitive:
    """Static description of a primitive shape type."""

    name: str
    canonical_sdf: Callable[[Tensor, Dict[str, Tensor]], Tensor]
    # Scalar params (stored as shape ``(1,)`` tensors) and vector params.
    scalar_params: List[str] = field(default_factory=list)
    vector_params: List[str] = field(default_factory=list)

    def sdf(self, points: Tensor, params: Dict[str, Tensor]) -> Tensor:
        """Evaluate the world-space SDF for this primitive."""
        local = world_to_local(points, params["center"], params["rotation"])
        return self.canonical_sdf(local, params)

    @property
    def param_keys(self) -> List[str]:
        return ["center", "rotation", *self.scalar_params, *self.vector_params]


# Registry of all available primitives, keyed by lowercase shape name.
PRIMITIVES: Dict[str, Primitive] = {
    "sphere": Primitive("sphere", _sphere, scalar_params=["radius"]),
    "ellipsoid": Primitive("ellipsoid", _ellipsoid, vector_params=["size"]),
    "box": Primitive("box", _box, vector_params=["size"]),
    "cylinder": Primitive("cylinder", _cylinder, scalar_params=["radius", "height"]),
    "cone": Primitive("cone", _cone, scalar_params=["radius", "height"]),
    "torus": Primitive("torus", _torus, scalar_params=["radius", "tube"]),
    "capsule": Primitive("capsule", _capsule, scalar_params=["radius", "height"]),
}


def get_primitive(shape: str) -> Primitive:
    """Look up a primitive by name (case-insensitive)."""
    key = shape.lower()
    if key not in PRIMITIVES:
        raise KeyError(
            f"Unknown primitive '{shape}'. Available: {sorted(PRIMITIVES)}"
        )
    return PRIMITIVES[key]
