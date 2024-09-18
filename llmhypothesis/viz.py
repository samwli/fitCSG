import json
import sys
import numpy as np
import plotly.graph_objects as go

def create_cylinder(center, radius, height, rotation):
    """Creates a mesh for a cylinder given center, radius, height, and rotation."""
    theta = np.linspace(0, 2 * np.pi, 30)
    z = np.linspace(-height / 2, height / 2, 30)
    theta, z = np.meshgrid(theta, z)
    x = radius * np.cos(theta) + center[0]
    y = radius * np.sin(theta) + center[1]
    z = z + center[2]
    
    # No rotation applied in this simple example
    return go.Surface(x=x, y=y, z=z, showscale=False, opacity=0.5)

def create_cone(center, radius, height, rotation):
    """Creates a mesh for a cone given center, radius, height, and rotation."""
    theta = np.linspace(0, 2 * np.pi, 30)
    z = np.linspace(0, height, 30)
    theta, z = np.meshgrid(theta, z)
    r = (height - z) / height * radius
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    z = z + center[2]

    # No rotation applied
    return go.Surface(x=x, y=y, z=z, showscale=False, opacity=0.5)

def create_ellipsoid(center, sizes, rotation):
    """Creates a mesh for an ellipsoid given center, sizes, and rotation."""
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = sizes[0] * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = sizes[1] * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = sizes[2] * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

    return go.Surface(x=x, y=y, z=z, showscale=False, opacity=0.5)

def create_prism(center, sizes, rotation):
    """Creates a mesh for a prism (rectangular cuboid) given center, sizes, and rotation."""
    x = [center[0] - sizes[0] / 2.0, center[0] + sizes[0] / 2.0]
    y = [center[1] - sizes[1] / 2.0, center[1] + sizes[1] / 2.0]
    z = [center[2] - sizes[2] / 2.0, center[2] + sizes[2] / 2.0]

    # Define the vertices of the cuboid
    vertices_x = [x[0], x[1], x[1], x[0], x[0], x[1], x[1], x[0]]
    vertices_y = [y[0], y[0], y[1], y[1], y[0], y[0], y[1], y[1]]
    vertices_z = [z[0], z[0], z[0], z[0], z[1], z[1], z[1], z[1]]

    # Define the correct indices that form the triangles of each face of the cuboid
    faces_i = [0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 4, 5, 0, 1, 2, 3]
    faces_j = [1, 2, 3, 2, 6, 3, 7, 0, 5, 6, 7, 6, 4, 5, 6, 7]
    faces_k = [2, 3, 1, 6, 7, 7, 6, 5, 6, 7, 4, 5, 0, 4, 5, 3]

    return go.Mesh3d(
        x=vertices_x,
        y=vertices_y,
        z=vertices_z,
        i=faces_i,
        j=faces_j,
        k=faces_k,
        opacity=0.5,
        color='lightblue'
    )

def visualize_csg(node):
    """Recursively visualize a CSG node."""
    if 'type' in node:
        shape_type = node['type']
        params = node['params']
        if shape_type == 'Cylinder':
            return [create_cylinder(params['center'], params['radius'], params['height'], params['rotation'])]
        elif shape_type == 'Cone':
            return [create_cone(params['center'], params['radius'], params['height'], params['rotation'])]
        elif shape_type == 'Ellipsoid':
            return [create_ellipsoid(params['center'], params['sizes'], params['rotation'])]
        elif shape_type == 'Prism':
            return [create_prism(params['center'], params['sizes'], params['rotation'])]
    elif 'operation' in node:
        left_traces = visualize_csg(node['left'])
        right_traces = visualize_csg(node['right'])
        return left_traces + right_traces  # Combine traces
    return []

def flatten_traces(traces):
    """Flatten nested lists of traces into a single list."""
    flat_list = []
    for trace in traces:
        if isinstance(trace, list):
            flat_list.extend(flatten_traces(trace))
        else:
            flat_list.append(trace)
    return flat_list

def main(filename):
    # Load the CSG JSON file
    with open(filename, 'r') as file:
        csg_data = json.load(file)

    # Initialize the figure
    fig = go.Figure()

    # Visualize the CSG structure
    traces = visualize_csg(csg_data)

    # Flatten the traces to ensure each trace is correctly added
    traces = flatten_traces(traces)

    # Add each trace individually to the figure
    for trace in traces:
        fig.add_trace(trace)

    # Calculate the ranges for each axis
    x_vals = np.concatenate([trace.x for trace in traces if hasattr(trace, 'x')])
    y_vals = np.concatenate([trace.y for trace in traces if hasattr(trace, 'y')])
    z_vals = np.concatenate([trace.z for trace in traces if hasattr(trace, 'z')])

    margin = 0.1
    # Determine min and max values for each axis
    x_range = [np.min(x_vals) - margin, np.max(x_vals) + margin]
    y_range = [np.min(y_vals) - margin, np.max(y_vals) + margin]
    z_range = [np.min(z_vals) - margin, np.max(z_vals) + margin]

    # Set the layout for the figure with uniform scaling
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=x_range),
            yaxis=dict(nticks=4, range=y_range),
            zaxis=dict(nticks=4, range=z_range),
            aspectmode='data'  # Ensures the scale is uniform across axes
        ),
        title="CSG Visualization",
    )
    # Render the figure
    fig.show()

if __name__ == "__main__":
    # Check if the filename argument is provided
    if len(sys.argv) != 2:
        print("Usage: python viz.py <filename>")
        sys.exit(1)  # Exit the program with a non-zero status to indicate an error

    # Get the filename from the arguments
    filename = sys.argv[1]
    
    # You can now use the filename for further processing
    print(f"Visualizing provided: {filename}")
    main(filename)