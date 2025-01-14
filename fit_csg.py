import os
import json
import argparse
import torch
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from csg_utils import construct_sdf, get_tree, create_grid
from compute_gt import get_gt, resize_mask
from viz import plot_sdf
from multiprocessing import Process, Manager
from tqdm import tqdm

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')


def extract_params_as_tensor(param_dict):
    """
    Extract all parameters from a nested dictionary into a single flattened tensor.
    The order of extraction must be consistent.
    """
    return torch.cat([
        param.flatten() for obj in param_dict.values() for param in obj.values()
    ])
    

# Function to perform the optimization
def optimize(input_dir, output_name, opt, grid_size, viz_outputs, num_steps, log_steps, tree_path, obj):    
    print(f'Using device: {device}')
    
    # Load the gt tree and parameters
    tree_outline, leaf_params = get_tree(os.path.join(input_dir, tree_path))

    if viz_outputs:
        grid_points = create_grid(grid_size).to(device)
        
    params_to_optimize = []
    for shape_key in leaf_params.keys():
        for param_key in leaf_params[shape_key].keys():
            # Convert to `torch.nn.Parameter` and move to the device
            param = torch.nn.Parameter(leaf_params[shape_key][param_key].to(device))
            leaf_params[shape_key][param_key] = param  # Update the dictionary with the parameter
            params_to_optimize.append(param)  # Collect the parameter for optimization

    # Choose optimizer
    if opt == 'adam':
        learning_rate = 0.0001
        betas = (0.9, 0.999)   
        eps = 1e-8  
        optimizer = optim.Adam(params_to_optimize, lr=learning_rate, betas=betas, eps=eps)
    elif opt == 'sgd':
        learning_rate = 0.01  
        momentum = 0.9
        weight_decay = 0.0001 
        optimizer = optim.SGD(params_to_optimize, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif opt == 'rmsprop':
        learning_rate = 0.001
        alpha = 0.99  
        eps = 1e-8  
        optimizer = optim.RMSprop(params_to_optimize, lr=learning_rate, alpha=alpha, eps=eps)
    elif opt == 'adamw':
        learning_rate = 0.001 
        weight_decay = 0.01
        optimizer = optim.AdamW(params_to_optimize, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError('Optimizer not supported.')    
    
    pc_path = f"custom_dataset/{obj}_pc.npy"
    mask_path = f"custom_dataset/{obj}_mask.png"
    sdf_points, sdf_values, rel_pose, scale, pose_to_table = get_gt(pc_path, mask_path, device, tree_outline, leaf_params, grid_size, False)
    
    # Convert a dictionary of Parameters into a single fixed tensor
    initial_params = extract_params_as_tensor(leaf_params).detach()
    reg_beta = 2
    
    for step in range(num_steps):
        optimizer.zero_grad()
        all_params = False if step < num_steps * 0.5 else True
        predicted_sdf, _ = construct_sdf(tree_outline, leaf_params, sdf_points, all_params)
        loss = torch.nn.functional.mse_loss(predicted_sdf, sdf_values)

        current_params = extract_params_as_tensor(leaf_params)
        reg_term = torch.nn.functional.mse_loss(current_params, initial_params)
        
        total_loss = loss + reg_term * reg_beta
        
        total_loss.backward()
        
        # Manually adjust learning rate
        if all_params:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate / 4  # Set to desired learning rate
        
        optimizer.step()
        
        if step % log_steps == 0:
            print(f'Step {step}, Loss: {loss.item()}')
            if viz_outputs:
                save_path = os.path.join(input_dir, f'{output_name}')
                shape, colors = construct_sdf(tree_outline, leaf_params, grid_points, all_params, True)
                mask = shape.flatten() <= 0
                shape_points = grid_points[mask]
                colors = colors[mask]
                plot_sdf(shape_points, colors, "Predicted SDF", viz=False, step=step, save_path=save_path)

    return tree_outline, leaf_params, rel_pose, scale, pose_to_table


# Function to start multiple processes or run single-threaded based on num_processes
def main(input_dir, output_name, opt, grid_size, viz_outputs, num_steps, log_steps, tree_path):
    obj = tree_path.split('/')[-1].split('_')[1]
    tree_outline, leaf_params, rel_pose, scale, pose_to_table = optimize(input_dir, output_name, opt, grid_size, viz_outputs, num_steps, log_steps, tree_path, obj)
    grid_points = create_grid(500).to(device)
    sdf_values, colors = construct_sdf(tree_outline, leaf_params, grid_points, False, True)
    mask = sdf_values.flatten() <= 0
    pred_xyz = grid_points[mask]
    colors = colors[mask]
    
    pred_xyz /= scale
    
    # apply inverse rel pose
    ones = torch.ones((pred_xyz.shape[0], 1), dtype=torch.float32).to(device)
    homogeneous_points = torch.cat([pred_xyz, ones], dim=1)
    transformed_points = (torch.linalg.inv(rel_pose) @ homogeneous_points.T).T
    pred_xyz = transformed_points[:, :3]
    
    # # apply to table
    # ones = torch.ones((pred_xyz.shape[0], 1), dtype=torch.float32).to(device)
    # homogeneous_points = torch.cat([pred_xyz, ones], dim=1)
    # transformed_points = (pose_to_table @ homogeneous_points.T).T
    # pred_xyz = transformed_points[:, :3]
    
    pc = np.concatenate([np.array(pred_xyz.cpu()), np.array(colors.cpu())], axis=1)
    np.save(f'outputs/{obj}_init_pc.npy', pc)
    breakpoint()
    
    unique_colors, indices = torch.unique(colors, dim=0, return_inverse=True)
    shape_sets = {i: pred_xyz[indices == i] for i in range(len(unique_colors))}
    np.savez(f"outputs/{obj}_shape_sets.npz", **{str(k): v.cpu().numpy() for k, v in shape_sets.items()})
    
    pc = np.load(f'custom_dataset/{obj}_pc.npy')
    mask = resize_mask(f'custom_dataset/{obj}_mask.png')
    xyz = torch.tensor(pc[:, :, :3][mask]).to(device)
    point_sets = {k: [] for k, v in shape_sets.items()}
    
    for key, set_xyz in shape_sets.items():
        # downsample set_xyz
        shape_sets[key] = set_xyz[torch.randperm(set_xyz.shape[0])[:1000]]
    
    # Loop through each point in new_xyz
    for point in tqdm(xyz):
        min_distance = float("inf")
        closest_set = None

        # Find the closest set for the current point
        for key, set_xyz in shape_sets.items():
            distances = torch.linalg.norm(set_xyz - point, dim=1)
            min_dist = distances.min()
            
            if min_dist < min_distance:
                min_distance = min_dist
                closest_set = key
        point_sets[closest_set].append(point.cpu().numpy()) 
        
    point_sets = {str(k): np.array(v) for k, v in point_sets.items()}
    np.savez(f"outputs/{obj}_point_sets.npz", **point_sets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit CSG parameters to a given SDF.")
    parser.add_argument('-i', '--input_dir', default='.', help='Input directory containing the tree json')
    parser.add_argument('-o', '--output_name', default='viz', help='Output directory for visualizations.')
    parser.add_argument('-opt', default='adam', help='optimizer')
    parser.add_argument("--grid_size", type=int, default=200, help="Size of the grid for visualization.")
    parser.add_argument("--viz_outputs", action="store_true", help="Render the CSG tree graph.")
    parser.add_argument("--num_steps", type=int, default=0, help="Number of optimization steps.")
    parser.add_argument("--log_steps", type=int, default=500, help="Log every n steps.")
    parser.add_argument("--tree_path", type=str, default="example_csg_tree.json", help="Path read/write csg json trees.")
    
    args = parser.parse_args()
    
    main(args.input_dir, args.output_name, args.opt, args.grid_size, args.viz_outputs, args.num_steps, args.log_steps, args.tree_path)
