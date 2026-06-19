import random

from fitcsg.csg import evaluate, parse_tree
from fitcsg.grid import create_grid
from fitcsg.random_tree import random_csg_tree


def test_random_tree_is_parseable_and_nonempty():
    random.seed(0)
    tree = parse_tree(random_csg_tree(depth=2))
    sdf, _ = evaluate(tree, create_grid(32))
    assert (sdf <= 0).sum() > 0
