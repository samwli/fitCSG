"""Visualise a CSG tree: render its graph and/or scatter-plot its SDF surface.

Examples:
    python scripts/visualize_tree.py --tree examples/sunglasses.json --save out.png
    python scripts/visualize_tree.py --random --depth 3 --save random.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fitcsg import create_grid, evaluate, parse_tree, surface_points  # noqa: E402
from fitcsg.visualize import plot_sdf, visualize_tree  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="CSG tree / SDF visualisation")
    parser.add_argument("--tree", default="examples/sunglasses.json", help="CSG tree JSON path")
    parser.add_argument("--random", action="store_true", help="Generate a random tree instead")
    parser.add_argument("--depth", type=int, default=3, help="Depth for --random")
    parser.add_argument("--grid_size", type=int, default=128, help="SDF grid resolution")
    parser.add_argument("--render_graph", action="store_true", help="Render the Graphviz diagram")
    parser.add_argument("--save", default=None, help="Save the SDF scatter to this path")
    parser.add_argument("--show", action="store_true", help="Display interactively")
    args = parser.parse_args()

    if args.random:
        from fitcsg.random_tree import random_csg_tree

        tree = parse_tree(random_csg_tree(args.depth))
    else:
        tree = parse_tree(args.tree)

    if args.render_graph:
        graph = visualize_tree(tree)
        graph.render("csg_tree", view=args.show, cleanup=True)

    grid = create_grid(args.grid_size)
    sdf, colors = evaluate(tree, grid, with_colors=True)
    mask = sdf.flatten() <= 0
    plot_sdf(grid[mask], colors[mask], "CSG SDF", show=args.show, save_path=args.save)
    if args.save:
        print(f"Saved SDF visualisation to {args.save}")


if __name__ == "__main__":
    main()
