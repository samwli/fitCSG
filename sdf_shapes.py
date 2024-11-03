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
    colors = color.repeat(points.shape[0], 1) if color is not None else None
    
    return distances, colors


def sdf_prism(params, color, points):
    center, sizes = params[:3], params[3:6]
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
    center, radius, height, axis = params[:3], params[3], params[4], params[5:8]
    axis = axis / torch.linalg.norm(axis)  # Normalize the axis vector

    # Vector from center to points
    center_to_point = points - center
    # Project the center_to_point onto the axis to find the height component
    projected_height = torch.sum(center_to_point * axis, dim=1).unsqueeze(1) * axis
    radial_component = center_to_point - projected_height
    
    # Calculate the signed distance to the cylinder
    distances = torch.linalg.norm(radial_component, dim=1) - radius
    distances = torch.max(distances, torch.abs(projected_height.norm(dim=1) - height / 2))

    colors = color.repeat(points.shape[0], 1) if color is not None else None

    return distances, colors


def sdf_cone(params, color, points):
    # Cone parameters: [center_x, center_y, center_z, radius, height, axis_x, axis_y, axis_z]
    center, radius, height, axis = params[:3], params[3], params[4], params[5:8]
    axis = axis / torch.linalg.norm(axis)  # Normalize the axis vector

    # Vector from the cone's apex to the points
    apex_to_point = points - center

    # Project apex_to_point onto the axis to find the height component
    projected_height = torch.sum(apex_to_point * axis, dim=1, keepdim=True) * axis
    radial_component = apex_to_point - projected_height

    # Calculate the angle of the cone
    cone_angle = torch.atan(radius / height)

    # Compute distances based on projection
    distance_to_axis = torch.linalg.norm(radial_component, dim=1)
    projected_height_length = torch.linalg.norm(projected_height, dim=1)

    # Compute the SDF for the cone
    distance = torch.max(
        distance_to_axis - projected_height_length * torch.tan(cone_angle),
        -projected_height_length + height
    )
    distance = torch.where(projected_height_length > height, projected_height_length - height, distance)

    colors = color.repeat(points.shape[0], 1) if color is not None else None

    return distance, colors
