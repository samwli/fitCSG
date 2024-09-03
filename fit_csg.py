import os
import json
import argparse

import torch
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from csg_utils import construct_sdf, get_tree
from viz import plot_sdf

def parse_args():
    parser = argparse.ArgumentParser(description="Fit CSG parameters to a given SDF.")
    parser.add_argument('-i', '--input_dir', required=True, help='Input directory containing SDF and CSG outline files.')
    parser.add_argument('-o', '--output_name', required=True, help='Output directory for visualizations.')
    parser.add_argument('-opt', required=True, help='optimizer')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Define the grid of points
    grid_side = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    x = torch.linspace(-2, 2, grid_side, device=device)
    y = torch.linspace(-2, 2, grid_side, device=device)
    z = torch.linspace(-2, 2, grid_side, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    points = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], dim=-1).to(device)

    # Load the gt tree and parameters
    gt_tree, gt_params = get_tree(os.path.join(args.input_dir, 'csg_tree.json'), True)
    for param in gt_params.values():
        param.data = param.data.to(device)

    ground_truth_sdf_tensor, colors = construct_sdf(gt_tree, gt_params, points)
    ground_truth_sdf_tensor = ground_truth_sdf_tensor.to(device)
    
    save_path = os.path.join(args.input_dir, args.output_name)
    plot_sdf(ground_truth_sdf_tensor, colors, "Ground Truth SDF", viz=False, step='_gt', save_path=save_path)
    ground_truth_sdf_flat = ground_truth_sdf_tensor.view(-1)
    
    # Load the CSG tree outline and initialize random parameters
    csg_tree_outline, leaf_params = get_tree(os.path.join(args.input_dir, 'csg_tree.json'))
    for param in leaf_params.values():
        param.data = param.data.to(device)
    
    if args.opt == 'adam':
        learning_rate = 0.001
        betas = (0.9, 0.999)   
        eps = 1e-8  
        optimizer = optim.Adam(leaf_params.values(), lr=learning_rate, betas=betas, eps=eps)
    elif args.opt == 'sgd':
        learning_rate = 0.01  
        momentum = 0.9
        weight_decay = 0.0001 
        optimizer = optim.SGD(leaf_params.values(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        learning_rate = 0.001
        alpha = 0.99  
        eps = 1e-8  
        optimizer = optim.RMSprop(leaf_params.values(), lr=learning_rate, alpha=alpha, eps=eps)
    elif args.opt == 'adamw':
        learning_rate = 0.001 
        weight_decay = 0.01
        optimizer = optim.AdamW(leaf_params.values(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError('Optimizer not supported.')
    
    num_steps = 5001  # Define the number of gradient steps
    
    for step in range(num_steps):
        optimizer.zero_grad()
        predicted_sdf, colors = construct_sdf(csg_tree_outline, leaf_params, points)
        predicted_sdf_flat = predicted_sdf.view(-1)
        loss = torch.nn.functional.mse_loss(predicted_sdf_flat, ground_truth_sdf_flat)
        loss.backward()
        optimizer.step()
        
        if step % 500 == 0:
            print(f'Step {step}, Loss: {loss.item()}')
            predicted_sdf = predicted_sdf.reshape(grid_side, grid_side, grid_side)
            plot_sdf(predicted_sdf, colors, "Predicted SDF", viz=False, step=step, save_path=save_path)

