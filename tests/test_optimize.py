import torch

from fitcsg.csg import iter_leaves, parse_tree
from fitcsg.optimize import fit, randomize_leaf_params
from fitcsg.synthetic import sample_target_from_tree


def test_synthetic_fit_reduces_loss():
    torch.manual_seed(0)
    gt = parse_tree("examples/mug.json")
    points, values = sample_target_from_tree(gt, grid_size=32, num_points=3000)

    tree = parse_tree("examples/mug.json")
    randomize_leaf_params(tree, position=0.3, log_scale=0.3)

    result = fit(tree, points, values, num_steps=150, lr=1e-2, verbose=False)
    # The fit should make clear progress from its (randomised) starting point.
    assert result.history[-1] < 0.5 * result.history[0]
    assert result.loss < 1e-3


def test_coarse_to_fine_and_reg_run():
    # Smoke test: the coarse-to-fine schedule + init regularisation run and the
    # loss still decreases.
    torch.manual_seed(0)
    gt = parse_tree("examples/mug.json")
    points, values = sample_target_from_tree(gt, grid_size=32, num_points=3000)

    tree = parse_tree("examples/mug.json")
    randomize_leaf_params(tree, position=0.2, log_scale=0.2)
    result = fit(
        tree,
        points,
        values,
        num_steps=120,
        lr=1e-2,
        reg_weight=1e-3,
        coarse_to_fine=True,
        verbose=False,
    )
    assert result.history[-1] < result.history[0]


def test_init_regularisation_limits_drift():
    # With a large reg weight and an already-good init, params should barely move.
    torch.manual_seed(0)
    gt = parse_tree("examples/mug.json")
    points, values = sample_target_from_tree(gt, grid_size=32, num_points=3000)

    tree = parse_tree("examples/mug.json")
    before = [leaf.params["center"].clone() for leaf in iter_leaves(tree)]
    fit(tree, points, values, num_steps=80, lr=1e-2, reg_weight=10.0, verbose=False)
    after = [leaf.params["center"].detach() for leaf in iter_leaves(tree)]
    drift = max(float((a - b).abs().max()) for a, b in zip(after, before))
    assert drift < 0.2
