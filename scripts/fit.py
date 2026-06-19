"""Fit a CSG hypothesis's leaf parameters to a target object.

Workflow (see README "How it works"): a *hypothesis* tree (shapes + initial
params, ideally proposed by an LLM in a roughly unit-cube space) is optimised so
its SDF matches a *target* derived from an observed object. We assume the
hypothesis and target are already coarsely aligned (pose estimation is a TODO).

The starting hypothesis is ``--init_tree`` if given, otherwise the parameters of
``--tree`` are randomised (a harder stress test rather than a realistic start).

Two target sources are supported:

* ``--target synthetic`` (default, **runnable with no external data**): a target
  SDF is sampled from ``--tree`` (treated as the ground-truth instance). With an
  ``--init_tree`` this is the realistic "fit an abstract hypothesis onto the
  instance" case; without one it randomises ``--tree``'s params and tries to
  recover them (a recovery sanity check).

* ``--target files``: load a real observation (``--pc`` organised point cloud +
  ``--mask`` object mask ``.npy``) and fit ``--init_tree``/``--tree`` to it.
  This needs data that is not shipped with the repo (see README TODOs).

Examples:
    # realistic: abstract hypothesis -> synthetic instance
    python scripts/fit.py --tree examples/mug.json --init_tree examples/mug_init.json
    # recovery sanity check: randomise and recover
    python scripts/fit.py --tree examples/mug.json --restarts 4
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
    parser = argparse.ArgumentParser(description="Fit a CSG hypothesis to a target object")
    parser.add_argument("--tree", default="examples/mug.json", help="the target instance")
    parser.add_argument(
        "--init_tree",
        default=None,
        help="starting hypothesis (same topology as --tree); if omitted, "
        "--tree's params are randomised instead",
    )
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

    # Starting hypothesis: a separate init tree (realistic) or randomised params.
    if args.init_tree:
        tree = parse_tree(args.init_tree)
    else:
        tree = parse_tree(args.tree)
        randomize_leaf_params(tree)

    if args.target == "synthetic":
        target_points, target_values = sample_target_from_tree(gt_tree, device=args.device)
    else:
        from fitcsg.target import build_target_from_files

        if not args.pc or not args.mask:
            parser.error("--target files requires --pc and --mask")
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
