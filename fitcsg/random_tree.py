"""Random CSG tree generation (useful for smoke-tests / data bootstrapping).

Rewritten during the refactor: the old module imported a non-existent
``sdf_zero`` and a ``create_grid`` from the wrong place, so it could not run.
This version emits the canonical JSON schema directly and validates operations
against an actual SDF evaluation so the result is a single, non-empty solid.

The hand-crafted trees in ``examples/`` are still the primary workflow; random
trees are mostly handy for stress-testing the SDF / optimisation code.
"""

from __future__ import annotations

import json
import random
from typing import Dict, List

import torch

from .csg import combine, evaluate, parse_tree
from .grid import create_grid
from .primitives import PRIMITIVES

_DEFAULT_SHAPES = ["sphere", "ellipsoid", "box", "cylinder", "cone"]


def _random_params(shape: str) -> Dict:
    spec = PRIMITIVES[shape]
    params: Dict[str, List[float]] = {
        "center": [random.uniform(-0.4, 0.4) for _ in range(3)],
        "rotation": [random.uniform(0, 360) for _ in range(3)],
    }
    for key in spec.vector_params:
        params[key] = [random.uniform(0.3, 1.0) for _ in range(3)]
    for key in spec.scalar_params:
        if key == "tube":
            params[key] = random.uniform(0.1, 0.3)
        elif key == "height":
            params[key] = random.uniform(0.4, 1.2)
        else:  # radius
            params[key] = random.uniform(0.2, 0.6)
    return params


def _random_leaf(shape: str, idx: int) -> Dict:
    return {"shape": shape, "name": f"{shape}_{idx}", "params": _random_params(shape)}


def random_csg_tree(depth: int, shapes: List[str] = None) -> Dict:
    """Build a random balanced CSG tree (canonical dict form) of the given depth."""
    shapes = shapes or _DEFAULT_SHAPES
    grid = create_grid(40)

    counts: Dict[str, int] = {}
    nodes: List[Dict] = []
    for _ in range(2 ** depth):
        shape = random.choice(shapes)
        counts[shape] = counts.get(shape, 0) + 1
        nodes.append(_random_leaf(shape, counts[shape] - 1))

    while len(nodes) > 1:
        nxt: List[Dict] = []
        for i in range(0, len(nodes), 2):
            left = nodes[i]
            right = nodes[i + 1] if i + 1 < len(nodes) else nodes[i]
            op = _choose_operation(left, right, grid)
            nxt.append({"op": op, "left": left, "right": right})
        nodes = nxt
    return nodes[0]


def _choose_operation(left: Dict, right: Dict, grid) -> str:
    """Pick an operation that yields a non-empty solid (prefer union)."""
    l_sdf, _ = evaluate(parse_tree(left), grid)
    r_sdf, _ = evaluate(parse_tree(right), grid)
    candidates = ["union"]
    if (combine("intersection", l_sdf, r_sdf) < 0).any():
        candidates.append("intersection")
    if (combine("subtraction", l_sdf, r_sdf) < 0).any():
        candidates.append("subtraction")
    return random.choice(candidates)


def save_csg_tree(tree: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(tree, f, indent=4)
