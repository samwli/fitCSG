import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from csg_utils import construct_sdf, create_grid


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


def get_gt(pc1_path, pc2_path, mask_path, device, tree_outline, leaf_params, grid_size, surface_only=True):
    """
    Compute object and footprint points from the point clouds and mask, with or without sampling bands.
    """
    # Load point clouds and mask
    pc1 = torch.tensor(np.load(pc1_path), dtype=torch.float32).to(device)
    pc2 = torch.tensor(np.load(pc2_path), dtype=torch.float32).to(device)
    mask = torch.tensor(np.load(mask_path), dtype=torch.bool).to(device)
    H, W = mask.shape

    # Extract object points using the mask
    pc1_object_points = pc1[:, :, :3][mask]  
    pc2_object_points = pc2[:, :, :3][mask]
    object_points = torch.cat([pc1_object_points, pc2_object_points], dim=0)

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
    cropped_pc2 = pc2[y_min:y_max, x_min:x_max, :3]
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    # Get the table points and compute table_z
    outside_points = torch.cat([cropped_pc1[~cropped_mask], cropped_pc2[~cropped_mask]], dim=0)
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
    
    np.save('sdf_points.npy', sdf_points.cpu().numpy())
    np.save('sdf_values.npy', sdf_values.cpu().numpy())
    
    points = create_grid(grid_size).to(device)
    predicted_sdf, _ = construct_sdf(tree_outline, leaf_params, points, False)
    mask = predicted_sdf <= 0
    pred_xyz = points[mask]
    
    # scale
    pred_min = pred_xyz.min(dim=0)[0]
    pred_max = pred_xyz.max(dim=0)[0]
    pred_extent = torch.norm(pred_max - pred_min)  
    
    gt_min = object_points.min(dim=0)[0]
    gt_max = object_points.max(dim=0)[0]
    gt_extent = torch.norm(gt_max - gt_min)  
    
    scale = pred_extent / gt_extent
    object_points *= scale
    sdf_points *= scale
    sdf_values *= scale
    
    # rotation
    # R = torch.tensor([
    #     [1, 0, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, -1, 0, 0],
    #     [0, 0, 0, 1]
    # ], dtype=torch.float32).to(object_points.device)
    
    R = torch.tensor([
        [-1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  0,  1]
    ], dtype=torch.float32).to(object_points.device)


    object_points_homogeneous = torch.cat([object_points, torch.ones((object_points.shape[0], 1), device=object_points.device)], dim=1)  # (N, 4)
    object_points_homogeneous = object_points_homogeneous @ R
    object_points = object_points_homogeneous[:, :3]
    
    sdf_points_homogeneous = torch.cat([sdf_points, torch.ones((sdf_points.shape[0], 1), device=sdf_points.device)], dim=1)  # (N, 4)
    sdf_points_homogeneous = sdf_points_homogeneous @ R
    sdf_points = sdf_points_homogeneous[:, :3]

    # shift
    shift = torch.mean(object_points, dim=0) - torch.mean(pred_xyz, dim=0)
    object_points -= shift 
    sdf_points -= shift

    return sdf_points, sdf_values, R, scale, shift
