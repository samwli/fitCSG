"""Fit CSG leaf parameters to a target SDF.

Improvements over the original ``fit_csg.py``:

* robust **Huber** loss with optional **SDF truncation** (TSDF-style) so the
  fit concentrates on the near-surface region instead of being dominated by
  far-away points;
* a **cosine LR schedule**;
* **multiple random restarts**, keeping the best result (the optimisation is
  highly non-convex, so this matters a lot in practice);
* positivity of size/radius is handled inside the primitives (``abs``), so the
  optimiser is unconstrained and can use any of the supported optimisers;
* **regularisation toward the initial hypothesis** (``reg_weight``): because the
  LLM proposes a *meaningful* abstract shape, we can penalise drifting far from
  it. This was ``reg_beta`` in the original research code and keeps the fit from
  collapsing into a degenerate but low-loss configuration;
* an optional **coarse-to-fine** schedule (``coarse_to_fine``): first move only
  the part *centers* (place the pieces), then unfreeze every parameter to refine
  sizes/rotations. This mirrors the two-phase ``all_params`` trick from the
  original code and is far more stable than optimising everything at once.

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
    reg_weight: float = 0.0,
    coarse_to_fine: bool = False,
    coarse_frac: float = 0.5,
    device="cpu",
    log_every: int = 200,
    step_callback: Optional[Callable[[int, Node, float], None]] = None,
    verbose: bool = True,
) -> FitResult:
    """Optimise a single tree against ``(target_points, target_values)``.

    Args:
        reg_weight: strength of the L2 pull back toward the *initial* parameter
            values (the hypothesis we started from). ``0`` disables it.
        coarse_to_fine: if True, only the part ``center`` parameters are updated
            for the first ``coarse_frac`` of steps; the rest unfreeze afterwards.
        coarse_frac: fraction of ``num_steps`` spent in the centers-only phase.
    """
    target_points = target_points.to(device)
    target_values = target_values.to(device)

    named = collect_parameters(tree, device)
    names = list(named.keys())
    params = list(named.values())
    # Snapshot the starting point for the regularisation term.
    initial = [p.detach().clone() for p in params]
    # Parameters frozen during the coarse phase = everything that is not a center.
    coarse_steps = int(coarse_frac * num_steps) if coarse_to_fine else 0
    non_center = [not n.endswith(".center") for n in names]

    opt = _OPTIMIZERS[optimizer](params, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps)

    history: List[float] = []
    for step in range(num_steps):
        opt.zero_grad()
        pred, _ = evaluate(tree, target_points)
        loss = _loss(pred, target_values, truncation)
        total = loss
        if reg_weight > 0:
            reg = sum(F.mse_loss(p, p0) for p, p0 in zip(params, initial))
            total = total + reg_weight * reg
        total.backward()
        if step < coarse_steps:
            # Centers-only phase: drop gradients of frozen (non-center) params.
            for p, frozen in zip(params, non_center):
                if frozen and p.grad is not None:
                    p.grad.zero_()
        opt.step()
        sched.step()

        history.append(loss.item())
        if verbose and step % log_every == 0:
            phase = "coarse" if step < coarse_steps else "fine"
            print(
                f"  step {step:5d}  loss {loss.item():.6f}  "
                f"lr {sched.get_last_lr()[0]:.2e}  [{phase}]"
            )
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
