import json
import argparse

import torch

from csg_utils import construct_sdf, get_tree, create_grid
from viz import plot_sdf, visualize_csg_tree


def main(grid_size, random_tree, depth, render_graph, tree_path):
    if random_tree and depth is None:
        raise ValueError("Depth must be specified when random_tree is True.")
    
    if random_tree:
        from random_tree_utils import random_csg_tree, save_csg_tree
        csg_tree = random_csg_tree(depth)
        csg_tree = save_csg_tree(csg_tree, tree_path)
    else:
        csg_tree = json.load(open(tree_path))

    if render_graph:
        graph = visualize_csg_tree(csg_tree)
        graph.render(tree_path[: -len(".json")], view=True)

    points = create_grid(grid_size)
    gt_tree, gt_params = get_tree(csg_tree)
    final_sdf, colors = construct_sdf(gt_tree, gt_params, points, True)
    
    # mask = (-0.01 <= final_sdf.flatten()) & (final_sdf.flatten() <= 0.01)
    mask = final_sdf.flatten() <= 0
    shape_points = points[mask]
    colors = colors[mask]

    plot_sdf(shape_points, colors, "Final CSG Shape SDF")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSG Tree Visualization")
    parser.add_argument("--grid_size", type=int, default=100, help="Size of the grid for visualization.")
    parser.add_argument("--random_tree", action="store_true", help="Generate a random CSG tree.")
    parser.add_argument("--depth", type=int, help="Depth of the random CSG tree.")
    parser.add_argument("--render_graph", action="store_true", help="Render the CSG tree graph.")
    parser.add_argument("--tree_path", type=str, default="csg_tree.json", help="Path read/write csg json trees.")

    args = parser.parse_args()
    
    main(grid_size=args.grid_size, random_tree=args.random_tree, depth=args.depth, render_graph=args.render_graph, tree_path=args.tree_path)