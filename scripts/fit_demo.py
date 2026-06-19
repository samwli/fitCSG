"""End-to-end, data-free demo: watch a CSG tree fit a point cloud.

This is the visual smoke test for the whole pipeline. It:

1. loads a ground-truth CSG tree (``--tree``) and samples a target point cloud
   + target SDF from it (no external data needed);
2. randomises the leaf parameters to get a deliberately-wrong starting tree;
3. optimises the leaf parameters back toward the target, rendering a frame of
   ``target cloud (grey) + current CSG surface (colour)`` every few steps;
4. stitches the frames into ``<outdir>/fit.gif`` (the GIF is the deliverable;
   the per-step PNG frames are temporary and removed unless ``--keep_frames``).

Example:
    python scripts/fit_demo.py --tree examples/mug.json --num_steps 500 --outdir demo_out
"""

import argparse
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fitcsg import create_grid, evaluate, parse_tree  # noqa: E402
from fitcsg.optimize import fit, randomize_leaf_params  # noqa: E402
from fitcsg.synthetic import sample_target_from_tree  # noqa: E402
from fitcsg.visualize import frames_to_gif, plot_fit_frame  # noqa: E402


def surface_for_display(tree, grid):
    sdf, colors = evaluate(tree, grid, with_colors=True)
    mask = sdf <= 0
    return grid[mask], colors[mask]


def main():
    parser = argparse.ArgumentParser(description="Animated CSG-fits-pointcloud demo")
    parser.add_argument("--tree", default="examples/mug.json")
    parser.add_argument("--num_steps", type=int, default=600)
    parser.add_argument("--frame_every", type=int, default=20)
    parser.add_argument("--grid_size", type=int, default=64)
    parser.add_argument("--num_target_points", type=int, default=8000)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", default="demo_out")
    parser.add_argument("--gif_name", default="fit.gif")
    parser.add_argument("--keep_frames", action="store_true", help="keep the per-step PNGs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    # Frames are an implementation detail of GIF assembly: write them to a temp
    # dir and discard afterwards unless the user asks to keep them.
    frames_dir = os.path.join(args.outdir, "frames") if args.keep_frames else tempfile.mkdtemp()

    gt_tree = parse_tree(args.tree)
    grid = create_grid(args.grid_size, device=args.device)

    # Target SDF (for the loss) and target surface cloud (for display).
    target_points, target_values = sample_target_from_tree(
        gt_tree, num_points=args.num_target_points, device=args.device
    )
    target_display, _ = surface_for_display(gt_tree, grid)

    # Deliberately-wrong starting tree.
    tree = parse_tree(args.tree)
    randomize_leaf_params(tree, position=0.4, log_scale=0.4)

    frame_paths = []

    def callback(step, current_tree, loss):
        pred_xyz, pred_rgb = surface_for_display(current_tree, grid)
        path = os.path.join(frames_dir, f"frame_{len(frame_paths):04d}.png")
        plot_fit_frame(target_display, pred_xyz, pred_rgb, path, step=step, loss=loss)
        frame_paths.append(path)
        print(f"  frame {len(frame_paths):3d}  step {step:5d}  loss {loss:.5f}  surf_pts {pred_xyz.shape[0]}")

    print(f"fitting {args.tree} on {args.device} ...")
    callback(0, tree, float("nan"))  # initial (random) state
    fit(
        tree,
        target_points,
        target_values,
        optimizer=args.optimizer,
        lr=args.lr,
        num_steps=args.num_steps,
        device=args.device,
        log_every=args.frame_every,
        step_callback=callback,
        verbose=False,
    )

    gif_path = os.path.join(args.outdir, args.gif_name)
    frames_to_gif(frame_paths, gif_path, fps=6.0)
    print(f"\nwrote animation to {gif_path}")
    if args.keep_frames:
        print(f"kept {len(frame_paths)} frames in {frames_dir}")
    else:
        shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
