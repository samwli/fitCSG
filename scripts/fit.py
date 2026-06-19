"""Fit CSG leaf parameters to a target SDF.

Two target sources are supported:

* ``--target synthetic`` (default, **runnable with no external data**): the
  loaded tree is treated as ground truth, a target SDF is sampled from it, the
  leaf parameters are then randomised, and the optimiser tries to recover them.
  This is the end-to-end smoke test for the CSG->SDF and optimisation paths.

* ``--target files``: load a real observation (``--pc`` organised point cloud +
  ``--mask`` object mask ``.npy``) and fit the tree to it.  This needs data that
  is not shipped with the repo (see HANDOFF.md).

Examples:
    python scripts/fit.py --tree examples/sunglasses.json --num_steps 1500
    python scripts/fit.py --tree examples/mug.json --target synthetic --restarts 4
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fitcsg import evaluate, parse_tree, sample_target_from_tree, tree_to_dict  # noqa: E402
from fitcsg.optimize import fit, fit_with_restarts, randomize_leaf_params  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Fit a CSG tree to a target SDF")
    parser.add_argument("--tree", default="examples/sunglasses.json")
    parser.add_argument("--target", choices=["synthetic", "files"], default="synthetic")
    parser.add_argument("--pc", default=None, help="organised point cloud .npy (target=files)")
    parser.add_argument("--mask", default=None, help="object mask .npy (target=files)")
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "sgd", "rmsprop"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_steps", type=int, default=1500)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--truncation", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="fitted_tree.json", help="where to save the fitted tree")
    args = parser.parse_args()

    gt_tree = parse_tree(args.tree)

    if args.target == "synthetic":
        target_points, target_values = sample_target_from_tree(gt_tree, device=args.device)
        # Start optimisation from a randomised copy so the demo is non-trivial.
        tree = parse_tree(args.tree)
        randomize_leaf_params(tree)
    else:
        from fitcsg.target import build_target_from_files

        if not args.pc or not args.mask:
            parser.error("--target files requires --pc and --mask")
        tree = parse_tree(args.tree)
        target_points, target_values, *_ = build_target_from_files(
            args.pc, args.mask, tree, device=args.device, surface_only=False
        )

    fit_kwargs = dict(
        optimizer=args.optimizer,
        lr=args.lr,
        num_steps=args.num_steps,
        truncation=args.truncation,
        device=args.device,
    )
    if args.restarts > 1:
        result = fit_with_restarts(tree, target_points, target_values, num_restarts=args.restarts, **fit_kwargs)
    else:
        result = fit(tree, target_points, target_values, **fit_kwargs)

    print(f"\nfinal loss: {result.loss:.6f}")
    with open(args.out, "w") as f:
        json.dump(tree_to_dict(result.tree), f, indent=4)
    print(f"saved fitted tree to {args.out}")


if __name__ == "__main__":
    main()
