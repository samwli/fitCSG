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

_LEGACY_OPS = {"union": "union", "intersection": "intersection", "subtraction": "subtraction"}
_LEGACY_PARAM_ALIASES = {"sizes": "size", "axis": "rotation"}


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
def _normalise_params(raw: Dict) -> Dict[str, Tensor]:
    params: Dict[str, Tensor] = {}
    for key, value in raw.items():
        key = _LEGACY_PARAM_ALIASES.get(key, key)
        params[key] = _to_tensor(value)
    return params


def _parse_node(data: Dict, counter: Dict[str, int]) -> Node:
    is_leaf = "shape" in data or "type" in data
    if is_leaf:
        # Legacy "type" looked like "Ellipsoid0"; strip a trailing index and
        # lowercase to get the shape name.
        if "shape" in data:
            shape = data["shape"].lower()
        else:
            shape = data["type"].rstrip("0123456789").lower()
        idx = counter.get(shape, 0)
        counter[shape] = idx + 1
        name = data.get("name") or data.get("type") or f"{shape}_{idx}"
        params = _normalise_params(data["params"])
        color = PALETTE[(idx + sum(counter.values())) % len(PALETTE)]
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
