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