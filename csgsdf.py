import numpy as np
import random
from graphviz import Digraph
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy import ndimage
import pickle

operation_abbr = {'Union': 'U', 'Intersection': 'I', 'Subtraction': 'S'}
vibrant_colors = [
    'blue', 'red', 'green', 'yellow', 'purple', 'deepskyblue', 'darkorange', 'mediumseagreen'
]

# Filter only those keys from CSS4_COLORS that match the vibrant list
filtered_colors = {key: mcolors.CSS4_COLORS[key] for key in vibrant_colors}

def sdf_ellipsoid(params, color, points):
    center, sizes = params[:3], params[3:6]
    distances = np.linalg.norm((points - center) / sizes, axis=1) - 1
    colors = np.full(points.shape[0], color)
    return distances, colors

def sdf_prism(params, color, points):
    center, sizes = params[:3], params[3:6]
    rel_pos = points - center
    outside = np.abs(rel_pos) - sizes
    sdf_values = np.maximum(outside, 0).sum(axis=1) + np.minimum(np.max(outside, axis=1), 0)
    colors = np.full(points.shape[0], color)
    
    return sdf_values, colors

def sdf_zero(params, color, points):
    distances = np.ones(points.shape[0])
    colors = np.full(points.shape[0], color)
    return distances, colors


def generate_parameters():
    """Generates parameters for 3D shapes: position, size, orientation (not used in SDF)."""
    params = np.zeros(9)
    params[:3] = np.random.uniform(-1, 1, 3)
    params[3:6] = np.random.uniform(0.5, 1.5, 3)
    params[6:9] = np.random.uniform(0, 360, 3)
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
        sdf_left, origin_left = self.left.sdf(points)
        sdf_right, origin_right = self.right.sdf(points)
        
        # Apply boolean operations
        if self.operation == 'Union':
            final_sdf = np.minimum(sdf_left, sdf_right)
        elif self.operation == 'Intersection':
            final_sdf = np.maximum(sdf_left, sdf_right)
        elif self.operation == 'Subtraction':
            final_sdf = np.maximum(sdf_left, -sdf_right)
        
        origins = np.where(final_sdf == sdf_left, origin_left, origin_right)
        return final_sdf, origins


def create_grid(num_points=50):
    x = np.linspace(-2, 2, num_points)
    y = np.linspace(-2, 2, num_points)
    z = np.linspace(-2, 2, num_points)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    return points

# try different operations to create valid SDFs
def choose_valid_operations(left, right):
    points = create_grid()
    sdf_left, _ = left.sdf(points) 
    sdf_right, _ = right.sdf(points)

    valid_ops = ['Union']
    if np.any(np.maximum(sdf_left, sdf_right) < 0):
        valid_ops.append('Intersection')
    if np.any(np.maximum(sdf_left, -sdf_right) < 0):
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
    sdf_union = np.minimum(sdf_left, sdf_right)
    
    if np.sum(sdf_union < 0) > 100 and check_sdf_connectivity(sdf_union, round(len(points)**(1/3))):
        valid_ops.append('Union')
    else:
        return valid_ops
    
    sdf_intersection = np.maximum(sdf_left, sdf_right)
    if np.sum(sdf_intersection < 0) > 100 and check_sdf_connectivity(sdf_intersection, round(len(points)**(1/3))):
        valid_ops.append('Intersection')
    else:
        return valid_ops
    
    sdf_subtraction = np.maximum(sdf_left, -sdf_right)
    if np.sum(sdf_subtraction < 0) > 100 and check_sdf_connectivity(sdf_subtraction, round(len(points)**(1/3))):
        valid_ops.append('Subtraction')
    
    return valid_ops

def check_sdf_connectivity(sdf_values_flat, grid_size):
    sdf_grid = sdf_values_flat.reshape((grid_size, grid_size, grid_size))
    binary_grid = sdf_grid < 0
    # Use a more inclusive structure for connectivity
    structure = ndimage.generate_binary_structure(3, 2)  # Full connectivity including diagonal
    labeled_grid, num_features = ndimage.label(binary_grid, structure=structure)
    return num_features == 1

def generate_csg_tree(depth):
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
            right = leaves[i + 1] if i + 1 < len(leaves) else Leaf('Zero0', np.zeros(9))
            valid_ops = choose_valid_operations(left, right)
            operation = random.choice(valid_ops) if valid_ops else 'Union'
            new_node = Node(left, right, operation)
            new_leaves.append(new_node)
        leaves = new_leaves
    
    return leaves[0]  # Return the root of the tree


def visualize_csg_tree(node, graph=None):
    """ Visualize the CSG tree using Graphviz. Displays node information succinctly. """
    if graph is None:
        graph = Digraph()
        graph.attr('node', shape='ellipse')
    
    if isinstance(node, Leaf):
        # Extract and format the parameters for display
        params_text_lines = []
        for index, value in enumerate(node.params):
            if index % 3 == 0 and index > 0:
                params_text_lines.append('\n')
            params_text_lines.append(f"{value:.2f} ")

        # Join all parts to form the final text for the node label
        params_text = ''.join(params_text_lines).strip()
        label = f"{node.indexed_name} [{params_text}]"
        graph.node(str(id(node)), label=label)
    else:
        # For internal nodes, display the operation abbreviation
        label = f"{operation_abbr[node.operation]}"
        graph.node(str(id(node)), label=label)
        graph.edge(str(id(node)), str(id(node.left)))
        graph.edge(str(id(node)), str(id(node.right)))
        visualize_csg_tree(node.left, graph)
        visualize_csg_tree(node.right, graph)

    return graph


def csg_tree_to_dict(node):
    if isinstance(node, Leaf):
        return {'type': node.indexed_name, 'params': node.params.tolist()}
    elif isinstance(node, Node):
        return {
            'operation': node.operation,
            'left': csg_tree_to_dict(node.left),
            'right': csg_tree_to_dict(node.right)
        }

        
def save_csg_tree(csg_tree, filename):
    """Save the CSG tree dictionary to a file using pickle."""
    tree_dict = csg_tree_to_dict(csg_tree)
    with open(filename, 'wb') as f:
        pickle.dump(tree_dict, f)

def plot_sdf(sdf_values, origins, title):
    grid_side = int(np.cbrt(sdf_values.size))
    x, y, z = np.meshgrid(np.linspace(-2, 2, grid_side), 
                          np.linspace(-2, 2, grid_side), 
                          np.linspace(-2, 2, grid_side), indexing='ij')

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    origins = origins.flatten()

    mask = sdf_values.flatten() < 0
    color_array = origins[mask]  # Directly use origins for colors

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[mask], y[mask], z[mask], color=color_array, s=5)

    # Set fixed and square axes
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_box_aspect([1,1,1])  # Ensuring equal aspect ratio for all axes to make the plot truly square
    ax.set_title(title)
    plt.show()


def main():
    depth = 1
    csg_tree = generate_csg_tree(depth)
    
    if csg_tree:
        graph = visualize_csg_tree(csg_tree)
        graph.render('csg_tree', view=True)

        grid_side = 50
        points = create_grid(grid_side)
        
        final_sdf, origins = csg_tree.sdf(points)
        save_csg_tree(csg_tree, 'csg_outline.pkl')
        final_sdf = np.array(final_sdf).reshape((grid_side, grid_side, grid_side))
        np.save('final_sdf.npy', final_sdf)
        origins = np.array(origins).reshape((grid_side, grid_side, grid_side))

        plot_sdf(final_sdf, origins, "Final CSG Shape SDF")

if __name__ == "__main__":
    main()
