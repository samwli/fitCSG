"""CSG tree representation, (de)serialisation and SDF evaluation.

A CSG model is a binary tree:

* **Leaf** nodes are primitive shapes (see :mod:`fitcsg.primitives`).
* **Op** nodes combine their two children with ``union`` / ``intersection`` /
  ``subtraction``.

JSON schema (the canonical, post-refactor format)::

    # internal node
    {"op": "union", "left": <node>, "right": <node>, "smooth": 0.0}

    # leaf node
    {"shape": "ellipsoid", "name": "lens_l",
     "params": {"center": [...], "rotation": [...], "size": [...]}}

For convenience the parser also accepts the *legacy* keys that the original
code used (``operation``/``type``/``sizes``/``axis``) so old hand-written trees
still load; new code should emit the canonical schema via :func:`tree_to_dict`.

``smooth`` (optional, default 0) selects the smooth-min/-max blend radius. With
``smooth == 0`` the exact hard CSG operators are used; ``smooth > 0`` gives
differentiable blends that tend to behave better early in optimisation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor

from .primitives import get_primitive
from .transforms import axis_to_euler_deg

# Stable, distinguishable colours used purely for visualisation.
PALETTE: List[Tuple[float, float, float]] = [
    (0.12, 0.47, 0.71),  # blue
    (0.84, 0.15, 0.16),  # red
    (0.17, 0.63, 0.17),  # green
    (0.58, 0.40, 0.74),  # purple
    (1.00, 0.50, 0.05),  # orange
    (0.09, 0.75, 0.81),  # cyan
    (0.89, 0.47, 0.76),  # pink
    (0.55, 0.34, 0.29),  # brown
]

# Legacy / LLM leaf "type" names -> canonical primitive names. The numeric
# suffix (e.g. "Cylinder0") is stripped before this lookup.
_SHAPE_ALIASES = {"prism": "box", "cuboid": "box", "rectangularprism": "box"}
# Legacy param key -> canonical param key (``axis`` is handled separately
# because it is a direction vector, not an Euler rotation).
_LEGACY_PARAM_ALIASES = {"sizes": "size"}
# Shapes for which a legacy ``axis`` direction is meaningful and should be
# turned into a ``rotation``; for everything else ``axis`` is ignored.
_AXIS_SHAPES = {"cylinder", "cone", "capsule"}


def _to_tensor(value, dtype=torch.float32) -> Tensor:
    t = torch.as_tensor(value, dtype=dtype)
    return t


@dataclass
class Leaf:
    """A primitive shape instance with its (optimisable) parameters."""

    name: str
    shape: str
    params: Dict[str, Tensor]
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)

    def sdf(self, points: Tensor) -> Tensor:
        return get_primitive(self.shape).sdf(points, self.params)


@dataclass
class Op:
    """An internal node applying a boolean operator to two children."""

    op: str
    left: "Node"
    right: "Node"
    smooth: float = 0.0


Node = Union[Leaf, Op]


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _canonical_shape(data: Dict) -> str:
    """Resolve the canonical primitive name from a leaf dict.

    Accepts the canonical ``shape`` key or the legacy ``type`` (``"Cylinder0"``)
    and applies the alias table (``Prism`` -> ``box``).
    """
    raw = data["shape"] if "shape" in data else data["type"].rstrip("0123456789")
    raw = raw.lower()
    return _SHAPE_ALIASES.get(raw, raw)


def _normalise_params(raw: Dict, shape: str) -> Dict[str, Tensor]:
    """Convert a raw param dict (canonical *or* legacy) to canonical tensors.

    * ``sizes`` -> ``size``.
    * legacy ``axis`` (a direction vector) -> ``rotation`` (Euler degrees) for
      axis-aligned shapes, and is dropped for the rest where it was meaningless.
    * a missing ``rotation`` defaults to no rotation.
    """
    params: Dict[str, Tensor] = {}
    axis = None
    for key, value in raw.items():
        if key == "axis":
            axis = value
            continue
        params[_LEGACY_PARAM_ALIASES.get(key, key)] = _to_tensor(value)

    if "rotation" not in params:
        if axis is not None and shape in _AXIS_SHAPES:
            params["rotation"] = axis_to_euler_deg(axis)
        else:
            params["rotation"] = torch.zeros(3)
    if "center" not in params:
        params["center"] = torch.zeros(3)
    return params


def _parse_node(data: Dict, counter: Dict[str, int]) -> Node:
    is_leaf = "shape" in data or "type" in data
    if is_leaf:
        shape = _canonical_shape(data)
        idx = counter.get(shape, 0)
        counter[shape] = idx + 1
        # Prefer an explicit name, then the legacy semantic "part" label
        # (e.g. "Handle"), then the raw type, then a generated fallback.
        name = data.get("name") or data.get("part") or data.get("type") or f"{shape}_{idx}"
        params = _normalise_params(data["params"], shape)
        # Unique colour per leaf (global leaf order), so colours never collide --
        # this matters for colour-based part segmentation (see fitcsg.segment).
        global_idx = sum(counter.values()) - 1
        color = PALETTE[global_idx % len(PALETTE)]
        return Leaf(name=name, shape=shape, params=params, color=color)

    op = (data.get("op") or data["operation"]).lower()
    smooth = float(data.get("smooth", 0.0))
    return Op(
        op=op,
        left=_parse_node(data["left"], counter),
        right=_parse_node(data["right"], counter),
        smooth=smooth,
    )


def parse_tree(source: Union[str, Dict]) -> Node:
    """Parse a CSG tree from a JSON path, a JSON string, or a dict.

    TODO (roadmap #5): the tree *topology* is fixed input here (hand-authored
    JSON). The intended workflow is to have an LLM propose both the topology and
    sensible normalised initial parameters, which then feed straight into this
    parser.
    """
    if isinstance(source, str):
        if source.strip().startswith("{"):
            data = json.loads(source)
        else:
            with open(source) as f:
                data = json.load(f)
    else:
        data = source
    return _parse_node(data, counter={})


def iter_leaves(node: Node) -> Iterator[Leaf]:
    """Yield every leaf in the tree (depth-first)."""
    if isinstance(node, Leaf):
        yield node
    else:
        yield from iter_leaves(node.left)
        yield from iter_leaves(node.right)


def tree_to_dict(node: Node) -> Dict:
    """Serialise a tree back to the canonical JSON-able dict format."""
    if isinstance(node, Leaf):
        return {
            "shape": node.shape,
            "name": node.name,
            "params": {k: v.detach().cpu().tolist() for k, v in node.params.items()},
        }
    out = {"op": node.op, "left": tree_to_dict(node.left), "right": tree_to_dict(node.right)}
    if node.smooth:
        out["smooth"] = node.smooth
    return out


# --------------------------------------------------------------------------- #
# CSG operators
# --------------------------------------------------------------------------- #
def _smin(a: Tensor, b: Tensor, k: float) -> Tensor:
    """Polynomial smooth minimum (k -> 0 recovers ``min``)."""
    h = torch.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return torch.lerp(b, a, h) - k * h * (1.0 - h)


def _smax(a: Tensor, b: Tensor, k: float) -> Tensor:
    return -_smin(-a, -b, k)


def combine(op: str, left: Tensor, right: Tensor, smooth: float = 0.0) -> Tensor:
    """Apply a CSG operator to two SDF tensors."""
    if op == "union":
        return torch.minimum(left, right) if smooth <= 0 else _smin(left, right, smooth)
    if op == "intersection":
        return torch.maximum(left, right) if smooth <= 0 else _smax(left, right, smooth)
    if op == "subtraction":
        return torch.maximum(left, -right) if smooth <= 0 else _smax(left, -right, smooth)
    raise ValueError(f"Unknown CSG operation '{op}'")


def evaluate(
    node: Node, points: Tensor, with_colors: bool = False
) -> Tuple[Tensor, Optional[Tensor]]:
    """Evaluate the tree SDF at ``points``.

    Args:
        node: tree root.
        points: ``(N, 3)`` query points.
        with_colors: also return a per-point ``(N, 3)`` colour, taken from the
            primitive that "wins" the boolean op at each point (handy for viz).

    Returns:
        ``(sdf, colors)`` where ``colors`` is ``None`` when ``with_colors`` is
        False.
    """
    if isinstance(node, Leaf):
        d = node.sdf(points)
        colors = None
        if with_colors:
            colors = torch.tensor(node.color, device=points.device, dtype=points.dtype)
            colors = colors.repeat(points.shape[0], 1)
        return d, colors

    left, left_c = evaluate(node.left, points, with_colors)
    right, right_c = evaluate(node.right, points, with_colors)
    combined = combine(node.op, left, right, node.smooth)

    if not with_colors:
        return combined, None

    # Colour comes from whichever child determined the (hard) result.
    if node.op == "union":
        mask = left < right
    elif node.op == "intersection":
        mask = left > right
    else:  # subtraction
        mask = left > -right
    colors = torch.where(mask.unsqueeze(1), left_c, right_c)
    return combined, colors


def collect_parameters(node: Node, device=None) -> Dict[str, Tensor]:
    """Return a flat ``{f"{leaf}.{param}": Parameter}`` dict for the optimiser.

    The returned tensors are the *same objects* stored on the leaves (wrapped
    as ``nn.Parameter`` in-place), so optimising them updates the tree.
    """
    out: Dict[str, Tensor] = {}
    for leaf in iter_leaves(node):
        for key, value in leaf.params.items():
            param = torch.nn.Parameter(value.to(device) if device else value)
            leaf.params[key] = param
            out[f"{leaf.name}.{key}"] = param
    return out
