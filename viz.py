import os

import numpy as np
import torch
from graphviz import Digraph
import matplotlib.pyplot as plt

from csg_utils import create_grid


def visualize_csg_tree(node, graph=None):
    """ Visualize the CSG tree using Graphviz. Displays node information succinctly. """
    if graph is None:
        graph = Digraph()

    if "type" in node:
        # Extract and format the parameters for display
        params_text_lines = []
        # Add the shape name as the first line
        shape_name = node["type"]
        params_text_lines.append(f"{shape_name}\n")

        # Add the center, sizes, and rotation on new lines
        center = f"Center: {', '.join(str(round(val, 2)) for val in node['params']['center'])}"
        sizes = f"Sizes: {', '.join(str(round(val, 2)) for val in node['params']['sizes'])}"
        rotation = f"Rot: {', '.join(str(round(val)) for val in node['params']['rotation'])}"

        params_text_lines.append(f"{center}\n")
        params_text_lines.append(f"{sizes}\n")
        params_text_lines.append(f"{rotation}")

        # Join all parts to form the final text for the node label
        params_text = ''.join(params_text_lines).strip()
        label = f"{params_text}"

        # Create the node in the graph with the new label format
        graph.node(str(id(node)), label=label)
    else:
        # For internal nodes, display the operation abbreviation
        label = node["operation"][0]
        graph.node(str(id(node)), label=label)
        graph.edge(str(id(node)), str(id(node["left"])))
        graph.edge(str(id(node)), str(id(node["right"])))
        visualize_csg_tree(node["left"], graph)
        visualize_csg_tree(node["right"], graph)

    return graph


def plot_sdf(sdf_values, colors, title, viz=True, step=None, save_path='viz'):
    grid_size = int(np.cbrt(sdf_values.numel()))
    points = create_grid(grid_size).numpy()
    mask = sdf_values.detach().cpu().numpy().flatten() < 0

    if colors is not None:
        colors = colors.cpu().numpy()[mask]  # Directly use origins for colors
    else:
        colors = 'blue'

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = points[mask].T
    ax.scatter(xs=x, ys=y, zs=z, c=colors, s=5)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_box_aspect([1,1,1])  # Ensuring equal aspect ratio for all axes to make the plot truly square
    ax.set_title(title)
    
    if viz:
        plt.show()
    else:     
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, 'predicted_sdf{}.png'.format(step))
        plt.savefig(filename)
    plt.close(fig)  # Close the figure to free memory
    