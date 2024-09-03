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
from viz import plot_sdf


def main(input_dir, output_name, opt, grid_size, viz_outputs, num_steps, log_steps, tree_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load the gt tree and parameters
    tree_outline, gt_params = get_tree(os.path.join(input_dir, tree_path), True)
    _, leaf_params = get_tree(os.path.join(input_dir, tree_path))

    for param_key in gt_params.keys():
        gt_params[param_key].data = gt_params[param_key].data.to(device)
        leaf_params[param_key].data = leaf_params[param_key].data.to(device)
    
    if opt == 'adam':
        learning_rate = 0.001
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
    
    points = create_grid(grid_size, device)
    ground_truth_sdf_tensor, colors = construct_sdf(tree_outline, gt_params, points)
    ground_truth_sdf_flat = ground_truth_sdf_tensor.view(-1).to(device)
    
    if viz_outputs:
        save_path = os.path.join(input_dir, output_name)
        plot_sdf(ground_truth_sdf_tensor, colors, "Ground Truth SDF", viz=False, step='_gt', save_path=save_path)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        predicted_sdf, colors = construct_sdf(tree_outline, leaf_params, points)
        predicted_sdf_flat = predicted_sdf.view(-1)
        loss = torch.nn.functional.mse_loss(predicted_sdf_flat, ground_truth_sdf_flat)
        loss.backward()
        optimizer.step()
        
        if step % log_steps == 0:
            print(f'Step {step}, Loss: {loss.item()}')
            if viz_outputs:
                plot_sdf(predicted_sdf, colors, "Predicted SDF", viz=False, step=step, save_path=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit CSG parameters to a given SDF.")
    parser.add_argument('-i', '--input_dir', required=True, help='Input directory containing the tree json')
    parser.add_argument('-o', '--output_name', required=True, help='Output directory for visualizations.')
    parser.add_argument('-opt', required=True, help='optimizer')
    parser.add_argument("--grid_size", type=int, default=50, help="Size of the grid for visualization.")
    parser.add_argument("--viz_outputs", action="store_true", help="Render the CSG tree graph.")
    parser.add_argument("--num_steps", type=int, default=5001, help="Number of optimization steps.")
    parser.add_argument("--log_steps", type=int, default=500, help="Log every n steps.")
    parser.add_argument("--tree_path", type=str, default="csg_tree.json", help="Path read/write csg json trees.")
    

    args = parser.parse_args()
    
    main(args.input_dir, args.output_name, args.opt, args.grid_size, args.viz_outputs, args.num_steps, args.log_steps, args.tree_path)