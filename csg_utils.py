import random
import json

import torch
from scipy import ndimage
import numpy as np

from sdf_shapes import sdf_ellipsoid, sdf_prism, filtered_colors


def create_grid(num_points=50, device='cpu'):
    x = torch.linspace(-2, 2, num_points, device=device)
    y = torch.linspace(-2, 2, num_points, device=device)
    z = torch.linspace(-2, 2, num_points, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    points = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], dim=-1)
    return points


def initialize_params(shape_type):
    """Generates new parameters for a given shape type."""
    position = np.random.uniform(-1, 1, 3)
    size = np.random.uniform(0.5, 1.5, 3)
    params = np.concatenate([position, size])
    return torch.nn.Parameter(torch.tensor(params, dtype=torch.float32, requires_grad=True))


def get_tree(tree, full=False):
    """Get tree structure and params (true or random)"""
    if isinstance(tree, str):
        with open(tree, 'rb') as f:
            tree = json.load(open(tree))
    
    leaf_params = {}
    tree_outline = process_tree(tree, leaf_params, full)
    return tree_outline, leaf_params


def process_tree(tree, leaf_params, full):
    """Recursively process the tree to return the outline and initialize new or keep existing leaf params."""
    if 'type' in tree:  # It's a leaf node with parameters
        leaf_name = tree['type']
        # Conditionally keep existing parameters or initialize new ones
        if full:
            # Use existing parameters
            center = torch.tensor(tree['params']['center'], dtype=torch.float32)
            sizes = torch.tensor(tree['params']['sizes'], dtype=torch.float32)
            rotation = torch.tensor(tree['params']['rotation'], dtype=torch.float32)
            leaf_params[leaf_name] = torch.cat([center, sizes, rotation])
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

        if tree['operation'].lower() == 'union':
            final_sdf, color_mask = torch.minimum(left_sdf, right_sdf), left_sdf < right_sdf
        elif tree['operation'].lower() == 'intersection':
            final_sdf, color_mask = torch.maximum(left_sdf, right_sdf), left_sdf > right_sdf
        elif tree['operation'].lower() == 'subtraction':
            final_sdf, color_mask = torch.maximum(left_sdf, -right_sdf), left_sdf > -right_sdf
        
        if get_colors:
            left_color = left_color.to(final_sdf.device)
            right_color = right_color.to(final_sdf.device)
            colors = torch.where(color_mask.unsqueeze(1), left_color, right_color)
            return final_sdf, colors
        else:
            return final_sdf, None