"""End-to-end check of the paper pipeline on a point cloud (Sec. III-B/C).

Exercises ``target.build_target`` -- the custom ground-truth SDF procedure
(surface + signed band samples) described in CSGGrasp Sec. III-B -- and confirms
the full chain runs: point cloud -> target SDF -> CSG optimisation -> part
segmentation.
"""

import torch

from fitcsg.csg import evaluate, parse_tree
from fitcsg.grid import create_grid, surface_points
from fitcsg.optimize import fit
from fitcsg.segment import segment_point_cloud
from fitcsg.target import build_target


def _object_cloud_from(tree, grid_size=48):
    grid = create_grid(grid_size)
    sdf, _ = evaluate(tree, grid)
    return surface_points(grid, sdf)


def test_pointcloud_to_target_to_fit_to_segment():
    torch.manual_seed(0)
    # A "lifted 3D object point cloud" P, standing in for the depth back-projection.
    instance = parse_tree("examples/mug.json")
    object_points = _object_cloud_from(instance)
    assert object_points.shape[0] > 0

    # Module B: custom ground-truth SDF (surface + signed band samples).
    hypothesis = parse_tree("examples/mug_init.json")
    sdf_points, sdf_values, R, scale, shift = build_target(
        object_points, hypothesis, grid_size=48, surface_only=False
    )
    assert torch.isfinite(sdf_points).all() and torch.isfinite(sdf_values).all()
    assert (sdf_values == 0).any() and (sdf_values != 0).any()  # surface + band

    # Module C: optimise the hypothesis onto the target SDF (staged + regularised).
    result = fit(
        hypothesis, sdf_points, sdf_values,
        num_steps=150, lr=1e-2, reg_weight=1e-3, coarse_to_fine=True, verbose=False,
    )
    assert result.history[-1] < result.history[0]

    # Segmentation: label the observed cloud by fitted part.
    parts, labels, names = segment_point_cloud(result.tree, object_points, grid_res=48)
    assert labels.shape[0] == object_points.shape[0]
    assert sum(p.shape[0] for p in parts.values()) == object_points.shape[0]
