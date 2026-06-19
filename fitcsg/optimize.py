"""Fit CSG leaf parameters to a target SDF.

Improvements over the original ``fit_csg.py``:

* robust **Huber** loss with optional **SDF truncation** (TSDF-style) so the
  fit concentrates on the near-surface region instead of being dominated by
  far-away points;
* a **cosine LR schedule**;
* **multiple random restarts**, keeping the best result (the optimisation is
  highly non-convex, so this matters a lot in practice);
* positivity of size/radius is handled inside the primitives (``abs``), so the
  optimiser is unconstrained and can use any of the supported optimisers.

The tree topology is fixed; only the continuous leaf parameters move.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .csg import Node, collect_parameters, evaluate, iter_leaves

_OPTIMIZERS: Dict[str, Callable] = {
    "adam": lambda p, lr: torch.optim.Adam(p, lr=lr or 5e-3),
    "adamw": lambda p, lr: torch.optim.AdamW(p, lr=lr or 5e-3, weight_decay=1e-2),
    "sgd": lambda p, lr: torch.optim.SGD(p, lr=lr or 1e-2, momentum=0.9),
    "rmsprop": lambda p, lr: torch.optim.RMSprop(p, lr=lr or 1e-3),
}


@dataclass
class FitResult:
    tree: Node
    loss: float
    history: List[float]


def _loss(pred: Tensor, target: Tensor, truncation: Optional[float]) -> Tensor:
    # TODO (roadmap #2): this is a truncated-Huber match on SDF *values* at fixed
    # target points. For a better best-fit, move toward a surface objective
    # (bidirectional/Chamfer distance, Eikonal/normal regularisation) and/or
    # jointly optimise the alignment so the fit tracks the observed surface.
    if truncation is not None:
        pred = pred.clamp(-truncation, truncation)
        target = target.clamp(-truncation, truncation)
    return F.huber_loss(pred, target, delta=0.05)


def randomize_leaf_params(tree: Node, position: float = 0.3, log_scale: float = 0.3) -> None:
    """Jitter leaf parameters in-place to seed a fresh restart.

    ``center``/``rotation`` get additive noise; size-like params get
    multiplicative log-normal noise so they stay positive and scale-aware.
    """
    with torch.no_grad():
        for leaf in iter_leaves(tree):
            for key, value in leaf.params.items():
                if key == "center":
                    value.add_(torch.randn_like(value) * position)
                elif key == "rotation":
                    value.add_(torch.randn_like(value) * 30.0)  # degrees
                else:
                    value.mul_(torch.exp(torch.randn_like(value) * log_scale))


def fit(
    tree: Node,
    target_points: Tensor,
    target_values: Tensor,
    *,
    optimizer: str = "adam",
    lr: Optional[float] = None,
    num_steps: int = 2000,
    truncation: Optional[float] = 0.1,
    device="cpu",
    log_every: int = 200,
    step_callback: Optional[Callable[[int, Node, float], None]] = None,
    verbose: bool = True,
) -> FitResult:
    """Optimise a single tree against ``(target_points, target_values)``."""
    target_points = target_points.to(device)
    target_values = target_values.to(device)

    params = list(collect_parameters(tree, device).values())
    opt = _OPTIMIZERS[optimizer](params, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps)

    history: List[float] = []
    for step in range(num_steps):
        opt.zero_grad()
        pred, _ = evaluate(tree, target_points)
        loss = _loss(pred, target_values, truncation)
        loss.backward()
        opt.step()
        sched.step()

        history.append(loss.item())
        if verbose and step % log_every == 0:
            print(f"  step {step:5d}  loss {loss.item():.6f}  lr {sched.get_last_lr()[0]:.2e}")
        if step_callback is not None and step % log_every == 0:
            step_callback(step, tree, loss.item())

    return FitResult(tree=tree, loss=history[-1], history=history)


def fit_with_restarts(
    tree: Node,
    target_points: Tensor,
    target_values: Tensor,
    *,
    num_restarts: int = 4,
    randomize_first: bool = False,
    verbose: bool = True,
    **fit_kwargs,
) -> FitResult:
    """Run :func:`fit` from several random inits and keep the best.

    The optimisation surface is riddled with local minima, so restarts are the
    cheapest reliability win.
    TODO (roadmap #6): these runs are independent and embarrassingly parallel --
    run them across processes/GPUs instead of sequentially; also consider a
    second-order optimiser (LBFGS / Levenberg-Marquardt) per restart.
    """
    best: Optional[FitResult] = None
    for r in range(num_restarts):
        candidate = copy.deepcopy(tree)
        if r > 0 or randomize_first:
            randomize_leaf_params(candidate)
        if verbose:
            print(f"restart {r + 1}/{num_restarts}")
        result = fit(candidate, target_points, target_values, verbose=verbose, **fit_kwargs)
        if best is None or result.loss < best.loss:
            best = result
    return best
