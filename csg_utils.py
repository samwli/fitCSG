import random
import json

import torch
from scipy import ndimage
import numpy as np

from sdf_shapes import sdf_ellipsoid, sdf_prism, filtered_colors


def create_grid(num_points=100, device='cpu'):
    x = torch.linspace(-20, 20, num_points, device=device)
    y = torch.linspace(-20, 20, num_points, device=device)
    z = torch.linspace(-20, 20, num_points, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    points = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], dim=-1)
    return points


def get_tree(tree):
    """Get tree structure and params (true or random)"""
    if isinstance(tree, str):
        with open(tree, 'rb') as f:
            tree = json.load(open(tree))
    
    leaf_params = {}
    tree_outline = process_tree(tree, leaf_params)
    return tree_outline, leaf_params


def process_tree(tree, leaf_params):
    """Recursively process the tree to return the outline and leaf params."""
    if 'type' in tree:  # It's a leaf node with parameters
        leaf_name = tree['type']
        
        center = torch.tensor(tree['params']['center'], dtype=torch.float32)
        sizes = torch.tensor(tree['params']['sizes'], dtype=torch.float32)
        rotation = torch.tensor(tree['params']['rotation'], dtype=torch.float32)
        leaf_params[leaf_name] = torch.cat([center, sizes, rotation])
        
        return leaf_name  # Return the leaf name as a node in the tree
    else:  # It's an internal node
        # Recursively handle the left and right children
        new_tree = {'operation': tree['operation']}
        if 'left' in tree:
            new_tree['left'] = process_tree(tree['left'], leaf_params)
        if 'right' in tree:
            new_tree['right'] = process_tree(tree['right'], leaf_params)
        return new_tree


# Define a function that takes the tree structure and leaf parameters to construct the SDF
def construct_sdf(tree, leaf_params, points, get_colors=False):
    if isinstance(tree, str):  # Leaf node case
        if get_colors:
            color = random.choice(list(filtered_colors.values()))
        else:
            color = None
            
        if 'Ellip' in tree:
            return sdf_ellipsoid(leaf_params[tree], color, points)
        elif 'Prism' in tree:
            return sdf_prism(leaf_params[tree], color, points)
    else:  # Recursive case for operations
        left_sdf, left_color = construct_sdf(tree['left'], leaf_params, points, get_colors)
        right_sdf, right_color = construct_sdf(tree['right'], leaf_params, points, get_colors)

        operation = tree['operation'].lower()

        if operation == 'union':
            final_sdf = torch.minimum(left_sdf, right_sdf)
            if get_colors:
                color_mask = left_sdf < right_sdf
        elif operation == 'intersection':
            final_sdf = torch.maximum(left_sdf, right_sdf)
            if get_colors:
                color_mask = left_sdf > right_sdf
        elif operation == 'subtraction':
            final_sdf = torch.maximum(left_sdf, -right_sdf)
            if get_colors:
                color_mask = left_sdf > -right_sdf
        
        if get_colors:
            left_color = left_color.to(final_sdf.device)
            right_color = right_color.to(final_sdf.device)
            colors = torch.where(color_mask.unsqueeze(1), left_color, right_color)
            return final_sdf, colors
        else:
            return final_sdf, None
        

def create_footprint_mask(xy_surface, resolution=100):
    """
    Create a 2D mask for the footprint of the object based on the xy-coordinates of the surface points.
    
    Args:
    - xy_surface: Tensor of surface points in the xy-plane (N, 2).
    - resolution: The resolution of the grid for the mask.
    
    Returns:
    - mask: 2D boolean mask representing the footprint of the object.
    - x_bins: The x-axis grid boundaries used to bin points.
    - y_bins: The y-axis grid boundaries used to bin points.
    """
    # Get the min and max values for x and y
    x_min, x_max = xy_surface[:, 0].min(), xy_surface[:, 0].max()
    y_min, y_max = xy_surface[:, 1].min(), xy_surface[:, 1].max()

    # Create bins along x and y based on the resolution
    x_bins = torch.linspace(x_min, x_max, resolution).to(xy_surface.device)
    y_bins = torch.linspace(y_min, y_max, resolution).to(xy_surface.device)

    # Initialize an empty mask (False by default)
    mask = torch.zeros((resolution, resolution), dtype=torch.bool)

    # Iterate through the surface points and mark the corresponding cells in the mask as True
    for point in xy_surface:
        x_idx = torch.searchsorted(x_bins, point[0]) - 1  # Get the x bin index
        y_idx = torch.searchsorted(y_bins, point[1]) - 1  # Get the y bin index

        # Set the mask cell to True if within bounds
        if 0 <= x_idx < resolution and 0 <= y_idx < resolution:
            mask[x_idx, y_idx] = True

    return mask.to(xy_surface.device), x_bins, y_bins


def vectorized_is_inside_footprint(band_points, mask, x_bins, y_bins):
    """
    Vectorized check for whether points' xy-coordinates are inside the 2D mask footprint.

    Args:
    - band_points: Tensor of points (M, 3).
    - mask: The 2D footprint mask (resolution x resolution).
    - x_bins: The x-axis bin edges.
    - y_bins: The y-axis bin edges.

    Returns:
    - inside_footprint: Boolean Tensor of shape (M,) indicating whether each point is inside the footprint.
    """
    # Extract the xy-coordinates of the band points
    x_coords = band_points[:, 0].contiguous()
    y_coords = band_points[:, 1].contiguous()

    # Find the corresponding mask indices for each point
    x_indices = torch.searchsorted(x_bins, x_coords) - 1
    y_indices = torch.searchsorted(y_bins, y_coords) - 1

    # Check if the indices are within bounds
    valid_x = (x_indices >= 0) & (x_indices < len(x_bins))
    valid_y = (y_indices >= 0) & (y_indices < len(y_bins))
    valid_points = valid_x & valid_y  # Only consider points within the bounds of the mask

    # Get the mask values for the valid points
    inside_footprint = torch.zeros(band_points.shape[0], dtype=torch.bool, device=band_points.device)
    inside_footprint[valid_points] = mask[x_indices[valid_points], y_indices[valid_points]]

    return inside_footprint
        

def sample_points_and_compute_sdf(xyz, num_sample_points=1000, radius=0.1, num_perturbations_per_point=10, resolution=100):
    """
    Sample points from the surface, perturb them to form a band, and compute signed distances (SDF).
    
    Args:
    - xyz: Tensor of surface points (N, 3).
    - num_sample_points: Number of surface points to sample from xyz.
    - radius: Radius within which to perturb the sampled points.
    - num_perturbations_per_point: Number of points to generate around each sampled surface point.
    - resolution: Resolution of the 2D mask footprint grid.
    
    Returns:
    - band_points: Tensor of perturbed points around the surface (M, 3).
    - sdf_values: Tensor of signed distance values for each perturbed point (M,).
    """
    # Step 1: Randomly sample points from the surface
    sampled_indices = torch.randint(0, xyz.shape[0], (num_sample_points,))
    sampled_points = xyz[sampled_indices]  # Sampled points from the surface

    # Step 2: Generate perturbations around the sampled points
    perturbations = torch.randn(num_sample_points * num_perturbations_per_point, 3)  # Generate random directions
    perturbations = perturbations / torch.norm(perturbations, dim=1, keepdim=True)  # Normalize to unit vectors
    perturbations *= torch.rand(num_sample_points * num_perturbations_per_point, 1) * radius  # Scale perturbations

    # Repeat sampled points to apply perturbations
    sampled_points_repeated = sampled_points.repeat_interleave(num_perturbations_per_point, dim=0)

    # Generate band points by adding perturbations
    band_points = sampled_points_repeated + perturbations.to(xyz.device)

    # Step 3: Create a 2D footprint mask based on the xy-plane of the surface points
    xy_surface = xyz[:, :2]  # Extract xy-coordinates from the surface points
    mask, x_bins, y_bins = create_footprint_mask(xy_surface, resolution=resolution)

    # Step 4: Compute Euclidean distances to the surface points
    dists = torch.cdist(band_points, xyz)  # (M, N) where M is band points and N is surface points

    # Find the minimum distance for each band point (SDF magnitude)
    min_distances, closest_indices = torch.min(dists, dim=1)  # Get the closest surface point for each band point

    # Step 5: Vectorized check if points are inside the footprint
    inside_footprint = vectorized_is_inside_footprint(band_points, mask, x_bins, y_bins)

    # Get the z-values of the band points and their closest surface points
    band_z = band_points[:, 2]
    closest_surface_z = xyz[closest_indices, 2]

    # Step 6: Determine the SDF signs vectorized
    sdf_signs = torch.ones(band_points.shape[0], device=band_points.device)  # Initialize all signs as positive (outside)
    
    # For points inside the footprint, compare z-values to determine the sign
    sdf_signs[inside_footprint] = torch.where(band_z[inside_footprint] > closest_surface_z[inside_footprint], 1.0, -1.0)

    # Step 7: Combine sign and distance to get the signed distance values
    sdf_values = sdf_signs * min_distances  # Apply sign to distances

    return band_points, sdf_values

        
def get_gt(gt_path, device, tree_outline, leaf_params, grid_size, surfance_only=True):
    xyz = torch.tensor(np.load(gt_path), dtype=torch.float32).to(device)
    xyz_sdf_values = torch.zeros(xyz.shape[0], dtype=torch.float32).to(device)
    
    if surfance_only:
        sdf_points = xyz.clone()
        sdf_values = xyz_sdf_values.clone()
    else:
        band_points, band_sdf_values = sample_points_and_compute_sdf(xyz, num_sample_points=len(xyz), radius=0.01, num_perturbations_per_point=5)
        sdf_points = torch.cat([xyz, band_points], dim=0)
        sdf_values = torch.cat([xyz_sdf_values, band_sdf_values], dim=0)
    
    # np.save('sampled_points.npy', sdf_points.cpu().numpy())
    # np.save('sampled_values.npy', sdf_values.cpu().numpy())
    # breakpoint()
    
    points = create_grid(grid_size).to(device)
    predicted_sdf, _ = construct_sdf(tree_outline, leaf_params, points, False)
    mask = predicted_sdf <= 0
    pred_xyz = points[mask]
    
    # scale
    pred_min = pred_xyz.min(dim=0)[0]
    pred_max = pred_xyz.max(dim=0)[0]
    pred_extent = torch.norm(pred_max - pred_min)  
    
    gt_min = xyz.min(dim=0)[0]
    gt_max = xyz.max(dim=0)[0]
    gt_extent = torch.norm(gt_max - gt_min)  
    
    scale = pred_extent / gt_extent
    xyz *= scale
    sdf_points *= scale
    
    # rotation
    R = torch.tensor([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32).to(xyz.device)

    xyz_homogeneous = torch.cat([xyz, torch.ones((xyz.shape[0], 1), device=xyz.device)], dim=1)  # (N, 4)
    xyz_homogeneous = xyz_homogeneous @ R
    xyz = xyz_homogeneous[:, :3]
    
    sdf_points_homogeneous = torch.cat([sdf_points, torch.ones((sdf_points.shape[0], 1), device=sdf_points.device)], dim=1)  # (N, 4)
    sdf_points_homogeneous = sdf_points_homogeneous @ R
    sdf_points = sdf_points_homogeneous[:, :3]

    # shift
    shift = torch.mean(xyz, dim=0) - torch.mean(pred_xyz, dim=0)
    xyz -= shift 
    sdf_points -= shift
    
    return sdf_points, sdf_values, R, scale, shift