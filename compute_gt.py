import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import open3d as o3d 
from csg_utils import construct_sdf, create_grid
import PIL
import os


def resize_mask(mask_path, size=512):
    binary_mask = np.array(PIL.Image.open(mask_path).convert('RGB'))[:, :, 2] > 0
    binary_mask_img = PIL.Image.fromarray(binary_mask.astype(np.uint8) * 255)
    S = max(binary_mask_img.size)
    new_size = tuple(int(round(x * size / S)) for x in binary_mask_img.size)
    resized_mask = binary_mask_img.resize(new_size, PIL.Image.BILINEAR)
    resized_binary_mask_np = np.array(resized_mask) > 0
    return resized_binary_mask_np


def compute_transformation(degrees):
    rotation_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0]
    ])

    theta = np.radians(degrees)  # Convert degrees to radians
    rotation_x_90 = np.array([
        [1, 0,              0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])
    
    theta = np.radians(90)  # Convert degrees to radians
    rotation_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

    # Construct the new pose
    target_pose = np.eye(4)
    target_pose[:3, :3] = rotation_z @ rotation_x_90 @ rotation_matrix

    return target_pose


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


def get_alignment(device, mask, pc1, tree_outline, leaf_params):
    rel_pose = torch.tensor(compute_transformation(0)).float().to(device)
    H, W = mask.shape

    pc_xyz = pc1[:, :, :3].reshape(-1, 3)
    ones = torch.ones((pc_xyz.shape[0], 1), dtype=torch.float32).to(device)
    homogeneous_points = torch.cat([pc_xyz, ones], dim=1)
    transformed_points = (rel_pose @ homogeneous_points.T).T
    pc_xyz = transformed_points[:, :3]
    
    H, W = pc1.shape[0], pc1.shape[1]
    pc_xyz = pc_xyz.reshape(H, W, 3)
    pc1[:, :, :3] = pc_xyz

    # Extract object points using the mask
    object_points = pc1[:, :, :3][mask]
    
    points = create_grid(200).to(device)
    predicted_sdf, _ = construct_sdf(tree_outline, leaf_params, points, False)
    mask = predicted_sdf <= 0
    pred_xyz = points[mask]
    
    pred_centroid = pred_xyz.mean(dim=0)
    object_centroid = object_points.mean(dim=0)
    
    # Center the points
    pred_centered = pred_xyz - pred_centroid
    object_centered = object_points - object_centroid
    
    # Compute scales
    pred_scale = torch.sqrt((pred_centered ** 2).sum(dim=1).mean())
    object_scale = torch.sqrt((object_centered ** 2).sum(dim=1).mean())
    
    # Compute scale and shift to align object_points to pred_xyz
    scale = pred_scale / object_scale
    shift = pred_centroid - object_centroid + torch.tensor([0.1, 0, 0]).to(device)
    
    # pred = pred_xyz.cpu().numpy()
    # gt = object_points.cpu().numpy()
    # gt = scale.cpu().numpy() * (gt + shift.cpu().numpy())
    # np.save('points.npy', np.vstack([pred, gt]))
    # breakpoint()
    
    rel_pose[:3, 3] += shift

    return scale, rel_pose


def get_sdf_results(mask, pc1, device, surface_only, pose_to_table):
    if pose_to_table is None:
        pose_to_table = torch.tensor(compute_transformation(0)).float().to(device)
    H, W = mask.shape
    
    pc_xyz = pc1[:, :, :3].reshape(-1, 3)
    ones = torch.ones((pc_xyz.shape[0], 1), dtype=torch.float32).to(device)
    homogeneous_points = torch.cat([pc_xyz, ones], dim=1)
    transformed_points = (pose_to_table @ homogeneous_points.T).T
    pc_xyz = transformed_points[:, :3]
    pc_xyz = pc_xyz.reshape(H, W, 3)
    pc1[:, :, :3] = pc_xyz
    
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
        band_points_object, band_sdf_object = sample_band_around_object(object_points, table_z, footprint_points, num_sample_points=5000, radius=0.3, num_perturbations_per_point=15)
        # band_points_object, band_sdf_object = sample_bounding_box_around_object(object_points, table_z, footprint_points, margin=0.1, num_sample_points=50000)
        band_points_footprint, band_sdf_footprint = sample_box_below_footprint(footprint_points, table_z, radius=0.05, margin=0.05, num_sample_points=1000)
        sdf_points = torch.cat([object_points, footprint_points, band_points_object, band_points_footprint], dim=0)
        sdf_values = torch.cat([object_sdf_values, footprint_sdf_values, band_sdf_object, band_sdf_footprint], dim=0)

    ones = torch.ones((sdf_points.shape[0], 1), dtype=torch.float32).to(device)
    homogeneous_points = torch.cat([sdf_points, ones], dim=1)
    transformed_points = (torch.linalg.inv(pose_to_table) @ homogeneous_points.T).T
    sdf_points = transformed_points[:, :3]

    return sdf_points, sdf_values, pose_to_table


def get_gt(pc1_path, mask_path, device, tree_outline, leaf_params, grid_size, surface_only=True):
    """
    Compute object and footprint points from the point clouds and mask, with or without sampling bands.
    """
    # Load point clouds and mask
    pc1 = torch.tensor(np.load(pc1_path), dtype=torch.float32).to(device)
    mask = torch.tensor(resize_mask(mask_path)).bool().to(device)
    
    obj = pc1_path.split('/')[1].split('_')[0]
    
    # if the path exists, load the dict
    params_exists = os.path.exists(f'custom_dataset/{obj}_params.npz')
    if params_exists:
        params = np.load(f'custom_dataset/{obj}_params.npz')
        scale, rel_pose, pose_to_table = torch.tensor(params['scale']).to(device), torch.tensor(params['rel_pose']).to(device), torch.tensor(params['pose_to_table']).to(device)
        rel_pose = torch.eye(4).to(device)
    else:
        scale, rel_pose = get_alignment(device, mask.clone(), pc1.clone(), tree_outline, leaf_params)
        pose_to_table = None

    # determine GT SDF by transforming to world frame
    sdf_points, sdf_values, pose_to_table = get_sdf_results(mask.clone(), pc1.clone(), device, surface_only, pose_to_table)
    
    if not params_exists:
        params = {'scale': scale.cpu().numpy(), 'rel_pose': rel_pose.cpu().numpy(), 'pose_to_table': pose_to_table.cpu().numpy()}
        np.savez(f'custom_dataset/{obj}_params.npz', **params)
    
    # Align computed SDF using the relative pose
    ones = torch.ones((sdf_points.shape[0], 1), dtype=torch.float32).to(device)
    homogeneous_points = torch.cat([sdf_points, ones], dim=1)
    transformed_points = (rel_pose @ homogeneous_points.T).T
    sdf_points = transformed_points[:, :3]
    
    sdf_points *= scale
    sdf_values *= scale
    
    # np.save('sdf_points.npy', sdf_points.cpu().numpy())
    # np.save('sdf_values.npy', sdf_values.cpu().numpy())
    # breakpoint()
    
    return sdf_points, sdf_values, rel_pose, scale, pose_to_table
