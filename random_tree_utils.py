import random
import json

import torch
from scipy import ndimage

from sdf_shapes import sdf_ellipsoid, sdf_prism, sdf_zero, filtered_colors
from csgsdf import create_grid


def generate_parameters():
    """Generates parameters for 3D shapes: position, size, orientation (not used in SDF)."""
    params = torch.zeros(9)
    params[:3] = torch.empty(3).uniform_(-1, 1)  # Position: uniform between -1 and 1
    params[3:6] = torch.empty(3).uniform_(0.5, 1.5)  # Size: uniform between 0.5 and 1.5
    params[6:9] = torch.empty(3).uniform_(0, 360)  # Orientation: uniform between 0 and 360
    return params


class Leaf:
    """A leaf node in the CSG tree, representing a shape with an SDF."""
    def __init__(self, indexed_name, params=None):
        self.indexed_name = indexed_name  # Directly use the indexed name
        self.params = params if params is not None else generate_parameters()
        self.color = list(filtered_colors.values())[random.randint(0, len(filtered_colors)-1)]

    def sdf(self, points):
        if 'Prism' in self.indexed_name:
            return sdf_prism(self.params, self.color, points)
        elif 'Ellipsoid' in self.indexed_name:
            return sdf_ellipsoid(self.params, self.color, points)
        else:
            return sdf_zero(self.params, self.color, points)


class Node:
    def __init__(self, left, right, operation):
        self.left = left
        self.right = right
        self.operation = operation

    def sdf(self, points):
        sdf_left, color_left = self.left.sdf(points)
        sdf_right, color_right = self.right.sdf(points)
        
        # Apply boolean operations
        if self.operation == 'Union':
            final_sdf = torch.minimum(sdf_left, sdf_right)
        elif self.operation == 'Intersection':
            final_sdf = torch.maximum(sdf_left, sdf_right)
        elif self.operation == 'Subtraction':
            final_sdf = torch.maximum(sdf_left, -sdf_right)
        
        final_sdf_expanded = final_sdf.unsqueeze(1).expand(-1, 3)
        origins = torch.where(final_sdf_expanded == sdf_left.unsqueeze(1), color_left, color_right)
        return final_sdf, origins


# try different operations to create valid SDFs
def choose_valid_operations(left, right):
    points = create_grid()
    sdf_left, _ = left.sdf(points) 
    sdf_right, _ = right.sdf(points)

    valid_ops = ['Union']
    if torch.any(torch.maximum(sdf_left, sdf_right) < 0):
        valid_ops.append('Intersection')
    if torch.any(torch.maximum(sdf_left, -sdf_right) < 0):
        valid_ops.append('Subtraction')
    
    return valid_ops


# final SDF should be a single connected component
def choose_valid_operations_for_root(left, right):
    """
    Determine valid operations for the root node, ensuring the result is a single connected component.
    """
    points = create_grid()
    sdf_left, _ = left.sdf(points)
    sdf_right, _ = right.sdf(points)
    valid_ops = []
    sdf_union = torch.minimum(sdf_left, sdf_right)
    
    if torch.sum(sdf_union < 0) > 100 and check_sdf_connectivity(sdf_union, round(len(points)**(1/3))):
        valid_ops.append('Union')
    else:
        return valid_ops
    
    sdf_intersection = torch.maximum(sdf_left, sdf_right)
    if torch.sum(sdf_intersection < 0) > 100 and check_sdf_connectivity(sdf_intersection, round(len(points)**(1/3))):
        valid_ops.append('Intersection')
    else:
        return valid_ops
    
    sdf_subtraction = torch.maximum(sdf_left, -sdf_right)
    if torch.sum(sdf_subtraction < 0) > 100 and check_sdf_connectivity(sdf_subtraction, round(len(points)**(1/3))):
        valid_ops.append('Subtraction')
    
    return valid_ops


def check_sdf_connectivity(sdf_values_flat, grid_size):
    sdf_grid = sdf_values_flat.reshape((grid_size, grid_size, grid_size))
    binary_grid = sdf_grid < 0
    # Use a more inclusive structure for connectivity
    structure = ndimage.generate_binary_structure(3, 2)  # Full connectivity including diagonal
    labeled_grid, num_features = ndimage.label(binary_grid, structure=structure)
    return num_features == 1


def random_csg_tree(depth):
    index_counter = {}
    primitives = ['Prism', 'Ellipsoid']
    leaves = []

    # Generate leaves with unique indices
    for i in range(2 ** depth):
        shape = random.choice(primitives)
        index = index_counter.get(shape, 0)
        indexed_name = f"{shape}{index}"
        index_counter[shape] = index + 1
        leaves.append(Leaf(indexed_name))

    # Combine leaves to form the tree
    while len(leaves) > 1:
        new_leaves = []
        for i in range(0, len(leaves), 2):
            left = leaves[i]
            right = leaves[i + 1] if i + 1 < len(leaves) else Leaf('Zero0', torch.zeros(9))
            valid_ops = choose_valid_operations(left, right)
            operation = random.choice(valid_ops) if valid_ops else 'Union'
            new_node = Node(left, right, operation)
            new_leaves.append(new_node)
        leaves = new_leaves
    
    return leaves[0]  # Return the root of the tree


def csg_tree_to_dict(node):
    if isinstance(node, Leaf):
        params = node.params.tolist()
        center = params[:3]
        sizes = params[3:6]
        rotation = params[6:9]
        return {'type': node.indexed_name, 'params': {"center": center, "sizes": sizes, "rotation": rotation}}
    elif isinstance(node, Node):
        return {
            'operation': node.operation,
            'left': csg_tree_to_dict(node.left),
            'right': csg_tree_to_dict(node.right)
        }
        
def save_csg_tree(csg_tree, filename):
    """Save the CSG tree dictionary to a file using json."""
    tree_dict = csg_tree_to_dict(csg_tree)
    with open(filename, 'w') as f:
        json.dump(tree_dict, f, indent=4)
    
    return tree_dict
