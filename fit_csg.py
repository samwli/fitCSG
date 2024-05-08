import torch
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Fit CSG parameters to a given SDF.")
    parser.add_argument('-i', '--input_dir', required=True, help='Input directory containing SDF and CSG outline files.')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory for visualizations.')
    parser.add_argument('-opt', required=True, help='optimizer')
    return parser.parse_args()

def load_and_initialize_tree(filename, full=False):
    """Load the CSG tree from a pickle file, strip the old parameters, and initialize new parameters."""
    with open(filename, 'rb') as f:
        tree = pickle.load(f)
    
    leaf_params = {}
    processed_tree = process_tree(tree, leaf_params, full)
    return processed_tree, leaf_params

def process_tree(tree, leaf_params, full):
    """Recursively process the tree to strip parameters and initialize new ones or keep existing ones."""
    if isinstance(tree, dict):
        if 'type' in tree:  # It's a leaf node with parameters
            leaf_name = tree['type']
            # Conditionally keep existing parameters or initialize new ones
            if full:
                # Use existing parameters
                leaf_params[leaf_name] = torch.tensor(tree['params'], dtype=torch.float32)
            else:
                # Initialize new random parameters
                leaf_params[leaf_name] = initialize_params(leaf_name)
            return leaf_name  # Return the leaf name as a node in the tree
        else:  # It's an internal node
            # Recursively handle the left and right children
            new_tree = {'operation': tree['operation']}
            if 'left' in tree:
                new_tree['left'] = process_tree(tree['left'], leaf_params, full)
            if 'right' in tree:
                new_tree['right'] = process_tree(tree['right'], leaf_params, full)
            return new_tree
    elif isinstance(tree, str):  # Direct leaf node as a string
        # Initialize parameters for this directly specified leaf, assuming full is False for simple types
        leaf_params[tree] = initialize_params(tree)
        return tree

def initialize_params(shape_type):
    """Generates new parameters for a given shape type."""
    position = np.random.uniform(-1, 1, 3)
    size = np.random.uniform(0.5, 1.5, 3)
    params = np.concatenate([position, size])
    return torch.nn.Parameter(torch.tensor(params, dtype=torch.float32, requires_grad=True))


# Define the SDF functions for the primitives
def sdf_ellipsoid(params, points):
    center, sizes = params[:3], params[3:6]
    distances = torch.linalg.norm((points - center) / sizes, dim=1) - 1
    return distances

def sdf_prism(params, points):
    center, sizes = params[:3], params[3:6]
    rel_pos = points - center
    outside = torch.abs(rel_pos) - sizes
    sdf_values = torch.maximum(outside, torch.zeros_like(outside)).sum(dim=1) + torch.minimum(torch.max(outside, dim=1).values, torch.zeros_like(points[:, 0]))
    return sdf_values

# Define a function that takes the tree structure and leaf parameters to construct the SDF
def construct_sdf(tree, leaf_params, points):
    if isinstance(tree, str):  # Leaf node case
        if 'Ellip' in tree:
            return sdf_ellipsoid(leaf_params[tree], points)
        elif 'Prism' in tree:
            return sdf_prism(leaf_params[tree], points)
    else:  # Recursive case for operations
        left_sdf = construct_sdf(tree['left'], leaf_params, points)
        right_sdf = construct_sdf(tree['right'], leaf_params, points)

        if tree['operation'].lower() == 'union':
            return torch.minimum(left_sdf, right_sdf)
        elif tree['operation'].lower() == 'intersection':
            return torch.maximum(left_sdf, right_sdf)
        elif tree['operation'].lower() == 'subtraction':
            return torch.maximum(left_sdf, -right_sdf)


def plot_and_save_sdf(sdf_values, grid_side, step, save_path='vis'):
    x = np.linspace(-2, 2, grid_side)
    y = np.linspace(-2, 2, grid_side)
    z = np.linspace(-2, 2, grid_side)
    
    # Create the meshgrid for plotting
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Extract the points where the SDF is near zero (surface boundary)
    surface_mask = sdf_values.detach().cpu().numpy() < 0
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot only the surface points
    ax.scatter(X[surface_mask], Y[surface_mask], Z[surface_mask], color='blue', s=1)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, 'predicted_sdf{}.png'.format(step))

    # Save the plot
    plt.savefig(filename)
    plt.close(fig)  # Close the figure to free memory

if __name__ == '__main__':
    args = parse_args()
    # Define the grid of points
    grid_side = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.linspace(-2, 2, grid_side, device=device)
    y = torch.linspace(-2, 2, grid_side, device=device)
    z = torch.linspace(-2, 2, grid_side, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    points = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], dim=-1).to(device)

    # Load the gt tree and parameters
    gt_tree, gt_params = load_and_initialize_tree(os.path.join(args.input_dir, 'csg_outline.pkl'), True)
    for param in gt_params.values():
        param.data = param.data.to(device)

    ground_truth_sdf_tensor = construct_sdf(gt_tree, gt_params, points).detach().to(device)
    plot_and_save_sdf(ground_truth_sdf_tensor.reshape(grid_side, grid_side, grid_side), grid_side, '_gt', os.path.join(args.input_dir, args.output_dir))
    ground_truth_sdf_flat = ground_truth_sdf_tensor.view(-1)
    
    # Load the CSG tree outline and initialize parameters
    csg_tree_outline, leaf_params = load_and_initialize_tree(os.path.join(args.input_dir, 'csg_outline.pkl'))
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
        predicted_sdf = construct_sdf(csg_tree_outline, leaf_params, points)
        predicted_sdf_flat = predicted_sdf.view(-1)
        loss = torch.nn.functional.mse_loss(predicted_sdf_flat, ground_truth_sdf_flat)
        loss.backward()
        optimizer.step()
        
        if step % 500 == 0:  # Print the loss every 10 steps
            print(f'Step {step}, Loss: {loss.item()}')
            predicted_sdf = predicted_sdf.reshape(grid_side, grid_side, grid_side)
            plot_and_save_sdf(predicted_sdf, grid_side, step, os.path.join(args.input_dir, args.output_dir))
        
    # After optimization, the leaf_params will contain the optimized parameters.
