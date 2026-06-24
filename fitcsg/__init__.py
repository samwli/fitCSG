"""fitCSG: fit Constructive-Solid-Geometry SDF trees to 3D objects.

Public API re-exports for convenience::

    from fitcsg import parse_tree, evaluate, create_grid, fit

See ``README.md`` for the intended workflow, conventions, known limitations and
the open TODO roadmap.
"""

from .csg import (
    Leaf,
    Op,
    collect_parameters,
    combine,
    evaluate,
    iter_leaves,
    parse_tree,
    tree_to_dict,
)
from .grid import create_grid, surface_points
from .optimize import FitResult, fit, fit_with_restarts, randomize_leaf_params
from .primitives import PRIMITIVES, get_primitive
from .segment import segment_point_cloud
from .synthetic import sample_target_from_tree

__all__ = [
    "Leaf",
    "Op",
    "parse_tree",
    "tree_to_dict",
    "evaluate",
    "combine",
    "iter_leaves",
    "collect_parameters",
    "create_grid",
    "surface_points",
    "fit",
    "fit_with_restarts",
    "randomize_leaf_params",
    "FitResult",
    "PRIMITIVES",
    "get_primitive",
    "sample_target_from_tree",
    "segment_point_cloud",
]
