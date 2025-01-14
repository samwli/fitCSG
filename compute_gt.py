import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from csg_utils import construct_sdf, create_grid


def icp(object_points, predicted_points, max_iters=50, tolerance=1e-6):
    """
    Align ground truth (object_points) to prediction (predicted_points) using ICP.
    
    Args:
        object_points (torch.Tensor): Ground truth point cloud (M x 3).
        predicted_points (torch.Tensor): Predicted point cloud (N x 3).
        max_iters (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance.
    
    Returns:
        R (torch.Tensor): Rotation matrix (3x3).
        scale (float): Scaling factor.
        T (torch.Tensor): Translation vector (3,).
    """
    # Initialize transformation parameters
    R = torch.eye(3, device=object_points.device)
    T = torch.zeros(3, device=object_points.device)
    scale = 1.0
    
    # Start with the ground truth (object points)
    obj_points = object_points.clone()
    pred_points = predicted_points.clone()
    
    for i in range(max_iters):
        # Find closest points
        distances = torch.cdist(obj_points, pred_points)  # Pairwise distance matrix
        closest_indices = torch.argmin(distances, dim=1)  # Closest predicted point for each object point
        closest_points = pred_points[closest_indices]
        
        # Compute centroids
        obj_centroid = obj_points.mean(dim=0)
        pred_centroid = closest_points.mean(dim=0)
        
        # Center the points
        obj_centered = obj_points - obj_centroid
        pred_centered = closest_points - pred_centroid
        
        # Compute scale
        obj_norm = torch.norm(obj_centered, dim=1).mean()
        pred_norm = torch.norm(pred_centered, dim=1).mean()
        scale = pred_norm / obj_norm
        
        obj_centered *= scale  # Apply scaling to centered object points
        
        # Compute rotation using Singular Value Decomposition (SVD)
        H = obj_centered.T @ pred_centered  # Covariance matrix
        U, _, Vt = torch.linalg.svd(H)
        R_new = Vt.T @ U.T
        
        # Correct reflection if necessary
        if torch.linalg.det(R_new) < 0:
            Vt[-1, :] *= -1
            R_new = Vt.T @ U.T
        
        # Compute final translation to map object to predicted
        T_new = pred_centroid - (R_new @ (obj_centroid * scale))
        
        # Check for convergence
        if (
            torch.norm(T - T_new) < tolerance and
            torch.norm(R - R_new) < tolerance
        ):
            break
        
        # Update transformation parameters
        R = R_new
        T = T_new
    
    return R, scale, T


def sample_band_around_object(object_points, table_z, footprint_points, num_sample_points=1000, radius=0.1, num_perturbations_per_point=10):
    """
    Sample points around the object, using the footprint to determine SDF signs.
    Points outside the footprint get positive sign, and for points within, sign is based on z comparison.
    """
    threshold = (object_points[:, 2].min() + table_z) / 2
    
    # Randomly sample points from the object
    sampled_indices = torch.randint(0, object_points.shape[0], (num_sample_points,))
    sampled_points = object_points[sampled_indices]

    # Generate random perturbations
    perturbations = torch.randn(num_sample_points * num_perturbations_per_point, 3)
    perturbations = perturbations / torch.norm(perturbations, dim=1, keepdim=True)  # Normalize
    perturbations *= torch.rand(num_sample_points * num_perturbations_per_point, 1) * radius  # Scale perturbations

    # Apply perturbations to sampled points
    band_points = sampled_points.repeat_interleave(num_perturbations_per_point, dim=0) + perturbations.to(object_points.device)
    band_points = band_points[band_points[:, 2] > table_z]

    # Compute distances to object points
    dists = torch.cdist(band_points, object_points)
    min_distances, closest_indices = torch.min(dists, dim=1)

    band_xy = band_points[:, :2] 
    object_xy = object_points[:, :2]
    
    footprint_threshold = 0.01
    xy_dists = torch.cdist(band_xy, object_xy)
    min_xy_distances, _ = torch.min(xy_dists, dim=1)
    inside_footprint = min_xy_distances <= footprint_threshold

    # Get z-values of band points and their closest object points
    band_z = band_points[:, 2]
    closest_surface_z = object_points[closest_indices, 2]

    # Determine SDF signs: positive outside footprint, z-based sign inside footprint
    sdf_signs = torch.ones(band_points.shape[0], device=band_points.device)  # Default to positive
    sdf_signs[inside_footprint] = torch.where(band_z[inside_footprint] > closest_surface_z[inside_footprint], 1.0, -1.0)

    # Apply signs to minimum distances
    sdf_values = sdf_signs * min_distances

    return band_points, sdf_values

# def sample_bounding_box_around_object(object_points, table_z, footprint_points, margin=0.1, num_sample_points=1000):
#     """
#     Sample points within a 3D bounding box around the object. Points are sampled from a volume
#     where the min z-value is determined by the threshold `(object_points[:, 2].min() + table_z) / 2`.
#     """
#     # Step 1: Set the threshold (z-level) for the bounding box
#     threshold = (object_points[:, 2].min() + table_z) / 2

#     # Step 2: Compute the axis-aligned bounding box (AABB) for the object
#     min_xyz = object_points.min(dim=0)[0]  # (x_min, y_min, z_min)
#     max_xyz = object_points.max(dim=0)[0]  # (x_max, y_max, z_max)
#     min_xyz -= margin
#     max_xyz += margin

#     # Adjust the z-range to start from the threshold
#     min_xyz[2] = threshold

#     # Step 3: Randomly sample points within the 3D bounding box
#     sample_x = torch.rand(num_sample_points, device=object_points.device) * (max_xyz[0] - min_xyz[0]) + min_xyz[0]
#     sample_y = torch.rand(num_sample_points, device=object_points.device) * (max_xyz[1] - min_xyz[1]) + min_xyz[1]
#     sample_z = torch.rand(num_sample_points, device=object_points.device) * (max_xyz[2] - min_xyz[2]) + min_xyz[2]

#     sampled_points = torch.stack([sample_x, sample_y, sample_z], dim=1)

#     # Step 4: Compute distances to object points
#     dists = torch.cdist(sampled_points, object_points)
#     min_distances, closest_indices = torch.min(dists, dim=1)

#     sampled_xy = sampled_points[:, :2] 
#     object_xy = object_points[:, :2]
    
#     footprint_threshold = 0.01
#     xy_dists = torch.cdist(sampled_xy, object_xy)
#     min_xy_distances, _ = torch.min(xy_dists, dim=1)
#     inside_footprint = min_xy_distances <= footprint_threshold

#     # Step 8: Get z-values of sampled points and their closest object points
#     sampled_z = sampled_points[:, 2]
#     closest_surface_z = object_points[closest_indices, 2]

#     # Step 9: Determine SDF signs: positive outside footprint, z-based sign inside footprint
#     sdf_signs = torch.ones(sampled_points.shape[0], device=sampled_points.device)  # Default to positive
#     sdf_signs[inside_footprint] = torch.where(sampled_z[inside_footprint] > closest_surface_z[inside_footprint], 1.0, -1.0)

#     # Step 10: Apply signs to minimum distances
#     sdf_values = sdf_signs * min_distances

#     return sampled_points, sdf_values


def sample_box_below_footprint(footprint_points, table_z, radius=0.1, margin=0.05, num_sample_points=1000):
    """
    Sample points within a 3D bounding box below the footprint.
    Points are sampled below the footprint's z-level (table_z), within an xy bounding box with an added margin.

    Args:
    - footprint_points: Tensor of footprint points (N, 3).
    - table_z: Z-coordinate of the footprint/table surface.
    - radius: The z-depth below the footprint for sampling.
    - margin: Margin to add to the xy bounding box.
    - num_sample_points: Number of points to sample in the bounding box.

    Returns:
    - sampled_points: Tensor of sampled points (M, 3).
    - sdf_values: Tensor of positive signed distance values for the sampled points (M,).
    """
    # Step 1: Compute the bounding box (AABB) for the footprint in the xy-plane
    min_xy = footprint_points[:, :2].min(dim=0)[0] - margin  # (x_min, y_min)
    max_xy = footprint_points[:, :2].max(dim=0)[0] + margin  # (x_max, y_max)

    # Step 2: Set the z-range to be strictly less than the table_z (footprint level)
    min_z = table_z - radius
    max_z = table_z  # Ensure we sample below the footprint

    # Step 3: Randomly sample points within the bounding box below the footprint
    sample_x = torch.rand(num_sample_points, device=footprint_points.device) * (max_xy[0] - min_xy[0]) + min_xy[0]
    sample_y = torch.rand(num_sample_points, device=footprint_points.device) * (max_xy[1] - min_xy[1]) + min_xy[1]
    sample_z = torch.rand(num_sample_points, device=footprint_points.device) * (max_z - min_z) + min_z  # Sample below the footprint

    sampled_points = torch.stack([sample_x, sample_y, sample_z], dim=1)

    # Step 4: Compute distances to the footprint points
    dists = torch.cdist(sampled_points, footprint_points)
    min_distances, _ = torch.min(dists, dim=1)

    # Step 5: Assign positive SDF values as all points are below the footprint (outside)
    sdf_values = min_distances

    return sampled_points, sdf_values


def get_gt(pc1_path, mask_path, device, tree_outline, leaf_params, grid_size, surface_only=True):
    """
    Compute object and footprint points from the point clouds and mask, with or without sampling bands.
    """
    # Load point clouds and mask
    pc1 = torch.tensor(np.load(pc1_path), dtype=torch.float32).to(device)
    mask = torch.tensor(np.load(mask_path), dtype=torch.bool).to(device)
    H, W = mask.shape

    # Extract object points using the mask
    object_points = pc1[:, :, :3][mask]

    # Find the bounding box around the object in 2D
    rows, cols = torch.where(mask)
    x_min, x_max = cols.min().item(), cols.max().item()
    y_min, y_max = rows.min().item(), rows.max().item()

    bbox_margin = 0.01  # Small margin for cropping
    x_min = max(0, x_min - int(bbox_margin * W))
    x_max = min(W, x_max + int(bbox_margin * W))
    y_min = max(0, y_min - int(bbox_margin * H))
    y_max = min(H, y_max + int(bbox_margin * H))

    # Crop region for table estimation
    cropped_pc1 = pc1[y_min:y_max, x_min:x_max, :3]
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    # Get the table points and compute table_z
    outside_points =cropped_pc1[~cropped_mask]
    table_z = outside_points[:, 2].min()
    table_z = 0
    
    # Object SDF values (initialized to 0 since they are on the surface)
    object_sdf_values = torch.zeros(object_points.shape[0], dtype=torch.float32).to(device)

    # Compute footprint points and their SDF values (also 0)
    xy_object = object_points[:, :2]  # XY-plane for footprint
    footprint_points = torch.cat([xy_object, torch.full((xy_object.shape[0], 1), table_z, device=device)], dim=1)
    # sampled_indices = torch.randint(0, footprint_points.shape[0], (500,))
    # footprint_points = footprint_points[sampled_indices]
    footprint_sdf_values = torch.zeros(footprint_points.shape[0], dtype=torch.float32).to(device)

    # If surface_only is True, return the surface points directly
    if surface_only:
        sdf_points = torch.cat([object_points, footprint_points], dim=0)
        sdf_values = torch.cat([object_sdf_values, footprint_sdf_values], dim=0)
    else:
        # Otherwise, compute the bands around the object and footprint
        band_points_object, band_sdf_object = sample_band_around_object(object_points, table_z, footprint_points, num_sample_points=5000, radius=0.15, num_perturbations_per_point=15)
        # band_points_object, band_sdf_object = sample_bounding_box_around_object(object_points, table_z, footprint_points, margin=0.1, num_sample_points=50000)
        band_points_footprint, band_sdf_footprint = sample_box_below_footprint(footprint_points, table_z, radius=0.05, margin=0.05, num_sample_points=1000)
        sdf_points = torch.cat([object_points, footprint_points, band_points_object, band_points_footprint], dim=0)
        sdf_values = torch.cat([object_sdf_values, footprint_sdf_values, band_sdf_object, band_sdf_footprint], dim=0)
    
    points = create_grid(grid_size).to(device)
    predicted_sdf, _ = construct_sdf(tree_outline, leaf_params, points, False)
    mask = predicted_sdf <= 0
    pred_xyz = points[mask]
    
    R, scale, shift = icp(pred_xyz, object_points)
    sdf_points = (sdf_points * scale) @ R.T + shift
    sdf_values *= scale

    return sdf_points, sdf_values, R, scale, shift
