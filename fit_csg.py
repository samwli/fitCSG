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
from compute_gt import get_gt
from viz import plot_sdf
from multiprocessing import Process, Manager

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Function to perform the optimization
def optimize(input_dir, output_name, opt, grid_size, viz_outputs, num_steps, log_steps, tree_path):    
    print(f'Using device: {device}')
    
    # Load the gt tree and parameters
    tree_outline, leaf_params = get_tree(os.path.join(input_dir, tree_path))

    if viz_outputs:
        grid_points = create_grid(grid_size).to(device)
        
    for param_key in leaf_params.keys():
        leaf_params[param_key] = torch.nn.Parameter(leaf_params[param_key].to(device))

    # Choose optimizer
    if opt == 'adam':
        learning_rate = 0.0005
        betas = (0.9, 0.999)   
        eps = 1e-8  
        optimizer = optim.Adam(leaf_params.values(), lr=learning_rate, betas=betas, eps=eps)
    elif opt == 'sgd':
        learning_rate = 0.01  
        momentum = 0.9
        weight_decay = 0.0001 
        optimizer = optim.SGD(leaf_params.values(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif opt == 'rmsprop':
        learning_rate = 0.001
        alpha = 0.99  
        eps = 1e-8  
        optimizer = optim.RMSprop(leaf_params.values(), lr=learning_rate, alpha=alpha, eps=eps)
    elif opt == 'adamw':
        learning_rate = 0.001 
        weight_decay = 0.01
        optimizer = optim.AdamW(leaf_params.values(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError('Optimizer not supported.')    
    
    pc1_path = "pc1.npy"
    pc2_path = "pc2.npy"
    mask_path = "pc_mask.npy"
    sdf_points, sdf_values, R, scale, shift = get_gt(pc1_path, pc2_path, mask_path, device, tree_outline, leaf_params, grid_size, False)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        predicted_sdf, _ = construct_sdf(tree_outline, leaf_params, sdf_points, False)
        loss = torch.nn.functional.mse_loss(predicted_sdf, sdf_values)
        loss.backward()
        optimizer.step()
        
        if step % log_steps == 0:
            print(f'Step {step}, Loss: {loss.item()}')
            if viz_outputs:
                save_path = os.path.join(input_dir, f'{output_name}')
                shape, colors = construct_sdf(tree_outline, leaf_params, grid_points, True)
                mask = shape.flatten() <= 0
                shape_points = grid_points[mask]
                colors = colors[mask]
                breakpoint()
                plot_sdf(shape_points, colors, "Predicted SDF", viz=False, step=step, save_path=save_path)

    return tree_outline, leaf_params, R, scale, shift


# Function to start multiple processes or run single-threaded based on num_processes
def main(input_dir, output_name, opt, grid_size, viz_outputs, num_steps, log_steps, tree_path):
    # Run single-threaded if num_processes is None or 1
    tree_outline, leaf_params, R, scale, shift = optimize(input_dir, output_name, opt, grid_size, viz_outputs, num_steps, log_steps, tree_path)
    grid_points = create_grid(500).to(device)
    sdf_values, colors = construct_sdf(tree_outline, leaf_params, grid_points, True)
    mask = sdf_values.flatten() <= 0
    pred_xyz = grid_points[mask]
    colors = colors[mask]

    # inverse shift
    pred_xyz += shift
    
    # inverse rotation
    R_inv = R.transpose(0, 1)
    pred_xyz = torch.cat([pred_xyz, torch.ones((pred_xyz.shape[0], 1), device=pred_xyz.device)], dim=1)  # (N, 4)
    pred_xyz = pred_xyz @ R_inv
    pred_xyz = pred_xyz[:, :3]
    
    # inverse scale
    pred_xyz /= scale
    
    pc = np.concatenate([np.array(pred_xyz.cpu()), np.array(colors.cpu())], axis=1)
    np.save('predicted_screwdriver_pc.npy', pc)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit CSG parameters to a given SDF.")
    parser.add_argument('-i', '--input_dir', default='.', help='Input directory containing the tree json')
    parser.add_argument('-o', '--output_name', default='viz', help='Output directory for visualizations.')
    parser.add_argument('-opt', default='adam', help='optimizer')
    parser.add_argument("--grid_size", type=int, default=100, help="Size of the grid for visualization.")
    parser.add_argument("--viz_outputs", action="store_true", help="Render the CSG tree graph.")
    parser.add_argument("--num_steps", type=int, default=5001, help="Number of optimization steps.")
    parser.add_argument("--log_steps", type=int, default=500, help="Log every n steps.")
    parser.add_argument("--tree_path", type=str, default="example_csg_tree.json", help="Path read/write csg json trees.")
    
    args = parser.parse_args()
    
    main(args.input_dir, args.output_name, args.opt, args.grid_size, args.viz_outputs, args.num_steps, args.log_steps, args.tree_path)
