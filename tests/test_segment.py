import torch

from fitcsg.csg import parse_tree
from fitcsg.segment import segment_point_cloud
from fitcsg.synthetic import sample_target_from_tree


def test_segmentation_assigns_all_points_to_parts():
    tree = parse_tree("examples/mug.json")
    # Use the on-surface points of the same model as the "observed" cloud.
    points, _ = sample_target_from_tree(tree, grid_size=48, num_points=4000)

    parts, labels, names = segment_point_cloud(tree, points, grid_res=64)

    # Every observed point gets exactly one part label.
    assert labels.shape[0] == points.shape[0]
    assert sum(p.shape[0] for p in parts.values()) == points.shape[0]
    # The mug has three parts; the body and handle should both claim some points.
    assert set(names) == {"body", "cavity", "handle"}
    assert parts["body"].shape[0] > 0
    assert parts["handle"].shape[0] > 0
