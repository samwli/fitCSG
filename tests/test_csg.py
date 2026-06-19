import torch

from fitcsg.csg import combine, evaluate, iter_leaves, parse_tree, tree_to_dict
from fitcsg.grid import create_grid


def _two_spheres(op):
    return {
        "op": op,
        "left": {"shape": "sphere", "name": "a", "params": {"center": [-0.3, 0, 0], "rotation": [0, 0, 0], "radius": 0.5}},
        "right": {"shape": "sphere", "name": "b", "params": {"center": [0.3, 0, 0], "rotation": [0, 0, 0], "radius": 0.5}},
    }


def test_combine_signs():
    a = torch.tensor([-1.0, 1.0])
    b = torch.tensor([0.5, -0.5])
    assert torch.allclose(combine("union", a, b), torch.minimum(a, b))
    assert torch.allclose(combine("intersection", a, b), torch.maximum(a, b))
    assert torch.allclose(combine("subtraction", a, b), torch.maximum(a, -b))


def test_csg_op_solid_volumes():
    grid = create_grid(48)
    union = (evaluate(parse_tree(_two_spheres("union")), grid)[0] <= 0).sum()
    inter = (evaluate(parse_tree(_two_spheres("intersection")), grid)[0] <= 0).sum()
    # Two overlapping spheres: union strictly larger than intersection.
    assert union > inter > 0


def test_parse_roundtrip():
    tree = parse_tree(_two_spheres("union"))
    again = parse_tree(tree_to_dict(tree))
    assert [l.shape for l in iter_leaves(tree)] == [l.shape for l in iter_leaves(again)]


def test_legacy_schema_loads():
    legacy = {
        "operation": "union",
        "left": {"type": "Sphere0", "params": {"center": [0, 0, 0], "rotation": [0, 0, 0], "radius": 0.5}},
        "right": {"type": "Box0", "params": {"center": [0.2, 0, 0], "sizes": [0.5, 0.5, 0.5], "axis": [0, 0, 0]}},
    }
    tree = parse_tree(legacy)
    shapes = sorted(l.shape for l in iter_leaves(tree))
    assert shapes == ["box", "sphere"]
    # legacy "sizes"/"axis" mapped to canonical "size"/"rotation".
    box_leaf = [l for l in iter_leaves(tree) if l.shape == "box"][0]
    assert "size" in box_leaf.params and "rotation" in box_leaf.params


def test_colors_shape():
    grid = create_grid(16)
    sdf, colors = evaluate(parse_tree(_two_spheres("union")), grid, with_colors=True)
    assert colors.shape == (grid.shape[0], 3)
