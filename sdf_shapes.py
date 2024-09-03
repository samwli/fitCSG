import torch
from matplotlib import colors as mcolors


def hex_to_rgb_normalized(hex_color):
    """Convert a hex color string to a normalized RGB tuple."""
    rgb = mcolors.hex2color(hex_color)  # Converts to (R, G, B) tuple with values in [0, 1]
    return torch.tensor(rgb, dtype=torch.float32)


vibrant_colors = [
    'blue', 'red', 'green', 'yellow', 'purple', 'deepskyblue', 'darkorange', 'mediumseagreen'
]
filtered_colors = {key: hex_to_rgb_normalized(mcolors.CSS4_COLORS[key]) for key in vibrant_colors}


def sdf_ellipsoid(params, color, points):
    center, sizes = params[:3], params[3:6]
    distances = torch.linalg.norm((points - center) / sizes, dim=1) - 1
    colors = color.repeat(points.shape[0], 1)
    return distances, colors


def sdf_prism(params, color, points):
    center, sizes = params[:3], params[3:6]
    rel_pos = points - center
    outside = torch.abs(rel_pos) - sizes
    distances = torch.maximum(outside, torch.zeros_like(outside)).sum(dim=1) + torch.minimum(torch.max(outside, dim=1).values, torch.zeros_like(points[:, 0]))
    colors = color.repeat(points.shape[0], 1)
    
    return distances, colors


def sdf_zero(params, color, points):
    distances = torch.ones(points.shape[0])
    colors = color.repeat(points.shape[0], 1)
    return distances, colors