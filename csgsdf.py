import json
import argparse

import torch

from csg_utils import construct_sdf, get_tree
from viz import plot_sdf, visualize_csg_tree


def create_grid(num_points=50, device='cpu'):
    x = torch.linspace(-2, 2, num_points, device=device)
    y = torch.linspace(-2, 2, num_points, device=device)
    z = torch.linspace(-2, 2, num_points, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    points = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], dim=-1)
    return points


def main(random_tree=False, depth=None):
    if random_tree and depth is None:
        raise ValueError("Depth must be specified when random_tree is True.")
    
    if random_tree:
        from random_tree_utils import random_csg_tree, save_csg_tree
        csg_tree = random_csg_tree(depth)
        csg_tree = save_csg_tree(csg_tree, 'random_csg_tree.json')
    else:
        csg_tree = json.load(open('csg_tree.json'))

    if csg_tree:
        graph = visualize_csg_tree(csg_tree)
        graph.render('csg_tree', view=True)

        grid_side = 50
        points = create_grid(grid_side)
        gt_tree, gt_params = get_tree(csg_tree, True)
        final_sdf, colors = construct_sdf(gt_tree, gt_params, points)
        final_sdf = final_sdf.reshape((grid_side, grid_side, grid_side))
        torch.save(final_sdf, 'final_sdf.pt')
        plot_sdf(final_sdf, colors, "Final CSG Shape SDF")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSG Tree Visualization")
    parser.add_argument("--random_tree", action="store_true", help="Generate a random CSG tree.")
    parser.add_argument("--depth", type=int, help="Depth of the random CSG tree.")

    args = parser.parse_args()
    
    main(random_tree=args.random_tree, depth=args.depth)