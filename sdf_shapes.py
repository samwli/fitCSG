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
    center, sizes, _ = params['center'], params['sizes'], params['rotation']
    distances = torch.linalg.norm((points - center) / sizes, dim=1) - 1
    colors = color.repeat(points.shape[0], 1) if color is not None else None

    return distances, colors


def sdf_prism(params, color, points):
    center, sizes, _ = params['center'], params['sizes'], params['rotation']
    rel_pos = points - center
    outside = torch.abs(rel_pos) - sizes
    distances = torch.maximum(outside, torch.zeros_like(outside)).sum(dim=1) + torch.minimum(torch.max(outside, dim=1).values, torch.zeros_like(points[:, 0]))
    colors = color.repeat(points.shape[0], 1) if color is not None else None

    return distances, colors


def sdf_zero(params, color, points):
    distances = torch.ones(points.shape[0])
    colors = color.repeat(points.shape[0], 1) if color is not None else None
    
    return distances, colors


def sdf_cylinder(params, color, points):
    # Cylinder parameters: [center_x, center_y, center_z, radius, height, axis_x, axis_y, axis_z]
    center, radius, height, axis = params['center'], params['radius'], params['height'], params['rotation']
    axis = axis / torch.linalg.norm(axis)  # Normalize the axis vector
    axis = torch.tensor([0, 0, 1])
    # Vector from the center of the cylinder to each point
    center_to_point = points - center

    # Project the center_to_point onto the axis to find the projected height
    projected_height_length = torch.sum(center_to_point * axis, dim=1).unsqueeze(1)  # Scalar projection
    projected_height = projected_height_length * axis  # Vector projection

    # Calculate the radial component (perpendicular to the axis)
    radial_component = center_to_point - projected_height
    radial_distance = torch.linalg.norm(radial_component, dim=1) - radius  # Distance from the surface radially

    # Calculate the distance in the height direction
    height_distance = torch.abs(projected_height_length.squeeze()) - (height / 2)

    # Determine whether the points are inside the cylinder
    inside_cylinder = (radial_distance <= 0) & (torch.abs(projected_height_length.squeeze()) <= (height / 2))

    # Combine distances to return the signed distance field
    distances = torch.where(inside_cylinder, torch.min(radial_distance, height_distance), torch.max(radial_distance, height_distance))

    # Handle colors if provided
    colors = color.repeat(points.shape[0], 1) if color is not None else None

    return distances, colors


def sdf_cone(params, color, points):
    # Cylinder parameters: [center_x, center_y, center_z, radius, height, axis_x, axis_y, axis_z]
    center, radius, height, axis = params['center'], params['radius'], params['height'], params['rotation']
    axis = axis / torch.linalg.norm(axis)  # Normalize the axis vector
    axis = torch.tensor([0, 0, 1])
 
    # Vector from the center of the cylinder to each point

    center_to_point = points - center

    # Project the center_to_point onto the axis to find the projected height
    projected_height_length = torch.sum(center_to_point * axis, dim=1).unsqueeze(1)  # Scalar projection
    projected_height = projected_height_length * axis  # Vector projection

    # Calculate the radial component (perpendicular to the axis)
    radial_component = center_to_point - projected_height
    distance_from_base = (height / 2) - projected_height_length.squeeze()

    effective_radius = (distance_from_base / height) * radius
    radial_distance = torch.linalg.norm(radial_component, dim=1) - effective_radius  # Distance from the surface radially

    # Calculate the distance in the height direction
    height_distance = torch.abs(projected_height_length.squeeze()) - (height / 2)

    # Determine whether the points are inside the cylinder
    inside_cylinder = (radial_distance <= 0) & (torch.abs(projected_height_length.squeeze()) <= (height / 2))

    # Combine distances to return the signed distance field
    distances = torch.where(inside_cylinder, torch.min(radial_distance, height_distance), torch.max(radial_distance, height_distance))

    # Handle colors if provided
    colors = color.repeat(points.shape[0], 1) if color is not None else None

    return distances, colors