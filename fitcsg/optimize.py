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
* an optional **multi-stage** schedule (``coarse_to_fine``): following the paper
  (CSGGrasp Sec. III-C.3), parameters unfreeze in three stages over the run --
  **centers first** (positional alignment), then **scales** (size/radius/height/
  tube; refine proportions), then **rotations** (tune orientation). This staged
  schedule "prevents overfitting and collapse" and is far more stable than
  optimising everything at once.

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

# Multi-stage schedule order (CSGGrasp Sec. III-C.3): centers, then scales, then
# rotations. Any unlisted (size-like) param defaults to the middle "scales" stage.
_PARAM_STAGE: Dict[str, int] = {
    "center": 0,
    "size": 1,
    "radius": 1,
    "height": 1,
    "tube": 1,
    "rotation": 2,
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
    device="cpu",
    log_every: int = 200,
    step_callback: Optional[Callable[[int, Node, float], None]] = None,
    verbose: bool = True,
) -> FitResult:
    """Optimise a single tree against ``(target_points, target_values)``.

    Args:
        reg_weight: strength of the L2 pull back toward the *initial* parameter
            values (the hypothesis we started from). ``0`` disables it.
        coarse_to_fine: if True, parameters unfreeze in three stages over the
            run -- centers, then scales, then rotations (CSGGrasp Sec. III-C.3).
    """
    target_points = target_points.to(device)
    target_values = target_values.to(device)

    named = collect_parameters(tree, device)
    names = list(named.keys())
    params = list(named.values())
    # Snapshot the starting point for the regularisation term.
    initial = [p.detach().clone() for p in params]
    # Stage index per parameter for the multi-stage schedule:
    # 0 = center (first), 1 = scale-like, 2 = rotation (last).
    param_stage = [_PARAM_STAGE.get(n.rsplit(".", 1)[-1], 1) for n in names]
    num_stages = 3

    opt = _OPTIMIZERS[optimizer](params, lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps)

    history: List[float] = []
    for step in range(num_steps):
        opt.zero_grad()
        pred, _ = evaluate(tree, target_points)
        loss = _loss(pred, target_values, truncation)
        total = loss
        if reg_weight > 0:
            # Pull params back toward the (meaningful) starting hypothesis so the
            # fit refines it rather than drifting to a degenerate low-loss state.
            reg = sum(F.mse_loss(p, p0) for p, p0 in zip(params, initial))
            total = total + reg_weight * reg
        total.backward()
        active_stage = num_stages - 1
        if coarse_to_fine:
            # Progressively unfreeze: a param trains once its stage is active.
            active_stage = min(num_stages - 1, (step * num_stages) // max(num_steps, 1))
            for p, stage in zip(params, param_stage):
                if stage > active_stage and p.grad is not None:
                    p.grad.zero_()
        opt.step()
        sched.step()

        history.append(loss.item())
        if verbose and step % log_every == 0:
            stage_name = ("centers", "scales", "rotations")[active_stage]
            print(
                f"  step {step:5d}  loss {loss.item():.6f}  "
                f"lr {sched.get_last_lr()[0]:.2e}  [{stage_name}]"
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
