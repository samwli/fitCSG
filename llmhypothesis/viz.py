"""Visualise an LLM-generated CSG hypothesis.

Thin wrapper around the maintained renderer in the ``fitcsg`` package so the
generator and the fitter share one *correct* visualiser. The previous
standalone plotly version mishandled the actual schema -- it matched the shape
type ``"Cylinder"`` exactly (real types are ``"Cylinder0"``) and read
``params['rotation']`` (the schema uses ``axis``), so it produced empty figures
and crashed. Going through ``fitcsg.parse_tree`` fixes both, since the parser
normalises the legacy/LLM schema.

Usage:
    python viz.py csg_hypotheses/csg_mug_1.json            # saves a PNG next to it
    python viz.py csg_hypotheses/csg_mug_1.json --show     # interactive window
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fitcsg import create_grid, evaluate, parse_tree  # noqa: E402
from fitcsg.visualize import plot_sdf  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Visualise a CSG hypothesis JSON")
    parser.add_argument("tree", help="path to a CSG hypothesis JSON")
    parser.add_argument("--grid_size", type=int, default=128, help="SDF grid resolution")
    parser.add_argument("--save", default=None, help="output PNG (default: alongside the JSON)")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    args = parser.parse_args()

    tree = parse_tree(args.tree)
    grid = create_grid(args.grid_size)
    sdf, colors = evaluate(tree, grid, with_colors=True)
    mask = sdf.flatten() <= 0

    save_path = args.save
    if save_path is None and not args.show:
        save_path = os.path.splitext(args.tree)[0] + ".png"

    plot_sdf(grid[mask], colors[mask], title=os.path.basename(args.tree), show=args.show, save_path=save_path)
    if save_path:
        print(f"saved {save_path}")


if __name__ == "__main__":
    main()
