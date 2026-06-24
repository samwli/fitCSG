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


def test_llm_legacy_prism_axis_and_part():
    # Schema exactly as the LLM hypothesis generator emits it.
    legacy = {
        "operation": "Union",
        "left": {
            "type": "Prism0",
            "params": {"center": [0, 0, 0], "sizes": [0.6, 0.05, 0.02], "axis": [1, 0, 0]},
            "part": "Blade",
        },
        "right": {
            "type": "Cylinder0",
            "params": {"center": [0, 0, 0.2], "radius": 0.05, "height": 0.3, "axis": [0, 0, 1]},
            "part": "Handle",
        },
    }
    tree = parse_tree(legacy)
    leaves = {l.name: l for l in iter_leaves(tree)}
    # "part" labels become leaf names.
    assert set(leaves) == {"Blade", "Handle"}
    # Prism -> box; box ignores the (meaningless) axis and gets zero rotation.
    assert leaves["Blade"].shape == "box"
    assert torch.allclose(leaves["Blade"].params["rotation"], torch.zeros(3))
    # Vertical cylinder axis [0,0,1] -> identity rotation.
    assert leaves["Handle"].shape == "cylinder"
    assert torch.allclose(leaves["Handle"].params["rotation"], torch.zeros(3), atol=1e-4)


def test_legacy_axis_direction_becomes_rotation():
    # A cylinder pointing along +X should rotate canonical +Z onto +X.
    from fitcsg.transforms import euler_deg_to_matrix

    leaf = {"type": "Cylinder0", "params": {"center": [0, 0, 0], "radius": 0.1, "height": 0.5, "axis": [1, 0, 0]}}
    tree = parse_tree(leaf)
    rot = next(iter_leaves(tree)).params["rotation"]
    mapped_z = euler_deg_to_matrix(rot) @ torch.tensor([0.0, 0.0, 1.0])
    assert torch.allclose(mapped_z, torch.tensor([1.0, 0.0, 0.0]), atol=1e-4)
