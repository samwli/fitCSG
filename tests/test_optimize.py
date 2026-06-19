import torch

from fitcsg.csg import parse_tree
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
