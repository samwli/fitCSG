"""Visualisation helpers: Graphviz CSG diagrams and matplotlib SDF scatter."""

from __future__ import annotations

import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe; callers can switch backends if interactive
import matplotlib.pyplot as plt  # noqa: E402
from torch import Tensor  # noqa: E402

from .csg import Leaf, Node, Op  # noqa: E402


def visualize_tree(node: Node, graph=None):
    """Render the CSG tree to a Graphviz ``Digraph`` (requires ``graphviz``)."""
    from graphviz import Digraph

    if graph is None:
        graph = Digraph()

    if isinstance(node, Leaf):
        lines = [node.shape]
        for key, value in node.params.items():
            vals = value.detach().cpu().flatten().tolist()
            lines.append(f"{key}: " + ", ".join(f"{v:.2f}" for v in vals))
        graph.node(str(id(node)), label="\n".join(lines))
    else:
        graph.node(str(id(node)), label=node.op[0].upper())
        for child in (node.left, node.right):
            graph.edge(str(id(node)), str(id(child)))
            visualize_tree(child, graph)
    return graph


def plot_fit_frame(
    target_xyz: Tensor,
    pred_xyz: Tensor,
    pred_rgb: Tensor,
    save_path: str,
    step: Optional[int] = None,
    loss: Optional[float] = None,
    elev: float = 20.0,
    azim: float = -60.0,
    bound: float = 1.0,
):
    """Render one optimisation frame: target point cloud + current CSG surface.

    The target observation is drawn as faint grey points; the current predicted
    CSG solid surface is drawn in colour on top so progress is easy to see.
    Axis limits and the camera are fixed so frames can be stitched into a GIF.
    """
    import numpy as np

    tgt = target_xyz.detach().cpu().numpy()
    pred = pred_xyz.detach().cpu().numpy()
    rgb = pred_rgb.detach().cpu().numpy()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(tgt[:, 0], tgt[:, 1], tgt[:, 2], c="0.55", s=4, alpha=0.4, label="target cloud")
    if pred.shape[0] > 0:
        ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c=rgb, s=6, alpha=0.85, label="CSG fit")
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_zlim(-bound, bound)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    title = "CSG fit"
    if step is not None:
        title += f"  |  step {step}"
    if loss is not None:
        title += f"  |  loss {loss:.4f}"
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=110)
    plt.close(fig)


def frames_to_gif(frame_paths, out_path: str, fps: float = 6.0):
    """Stitch a list of saved PNG frames into an animated GIF (uses Pillow)."""
    from PIL import Image

    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    if not frames:
        raise ValueError("no frames to write")
    duration_ms = int(1000.0 / fps)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def plot_sdf(
    points: Tensor,
    colors: Tensor,
    title: str = "SDF",
    show: bool = True,
    save_path: Optional[str] = None,
    bound: float = 1.0,
):
    """Scatter-plot the solid region of an SDF (points with sdf <= 0)."""
    xyz = points.detach().cpu().numpy()
    rgb = colors.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_zlim(-bound, bound)
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(title)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
