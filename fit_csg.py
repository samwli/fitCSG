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
from multiprocessing import Process, Manager

# Function to perform the optimization
def optimize(input_dir, output_name, opt, grid_size, viz_outputs, num_steps, log_steps, tree_path, seed, return_dict=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}, Seed: {seed}')
    
    # Load the gt tree and parameters
    tree_outline, gt_params = get_tree(os.path.join(input_dir, tree_path), True)
    _, leaf_params = get_tree(os.path.join(input_dir, tree_path))

    for param_key in gt_params.keys():
        gt_params[param_key].data = gt_params[param_key].data.to(device)
        leaf_params[param_key].data = leaf_params[param_key].data.to(device)
    
    # Choose optimizer
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
    get_colors = False
    ground_truth_sdf_tensor, colors = construct_sdf(tree_outline, gt_params, points, get_colors)
    ground_truth_sdf_flat = ground_truth_sdf_tensor.view(-1).to(device)
    
    final_loss = None
    best_predicted_sdf = None
    
    for step in range(num_steps):
        optimizer.zero_grad()
        predicted_sdf, colors = construct_sdf(tree_outline, leaf_params, points, get_colors)
        loss = torch.nn.functional.mse_loss(predicted_sdf.view(-1), ground_truth_sdf_flat)
        loss.backward()
        optimizer.step()
        
        if step % log_steps == 0:
            print(f'Seed {seed}, Step {step}, Loss: {loss.item()}')
            # plot_sdf(predicted_sdf, colors, "Predicted SDF", viz=False, step=step, save_path=os.path.join(input_dir, f'{output_name}_seed_{seed}'))

        if step == num_steps - 1:
            final_loss = loss.item()
            best_predicted_sdf = predicted_sdf.detach()  # Detach the tensor before sharing it

    if return_dict is not None:
        return_dict[seed] = (final_loss, best_predicted_sdf.cpu(), colors)  # Move tensors to CPU and store them

    return final_loss, best_predicted_sdf, colors  # Return values if running in a single process mode


# Function to start multiple processes or run single-threaded based on num_processes
def main(input_dir, output_name, opt, grid_size, viz_outputs, num_steps, log_steps, tree_path, num_processes):
    if num_processes is None or num_processes == 1:
        # Run single-threaded if num_processes is None or 1
        print("Running single process...")
        seed = np.random.randint(0, 10000)  # Random seed for single run
        final_loss, best_predicted_sdf, colors = optimize(input_dir, output_name, opt, grid_size, viz_outputs, num_steps, log_steps, tree_path, seed)
        print(f'Final loss: {final_loss}')
        
        # Only plot the final SDF for the single process run
        if viz_outputs:
            save_path = os.path.join(input_dir, f'{output_name}_single_seed_{seed}')
            plot_sdf(best_predicted_sdf, colors, "Best Predicted SDF", viz=False, step='final', save_path=save_path)
    else:
        # Run multi-process if num_processes is greater than 1
        print(f"Running {num_processes} processes in parallel...")
        manager = Manager()
        return_dict = manager.dict()
        processes = []
        
        for i in range(num_processes):
            seed = np.random.randint(0, 10000)  # Random seed for each process
            process = Process(target=optimize, args=(input_dir, output_name, opt, grid_size, viz_outputs, num_steps, log_steps, tree_path, seed, return_dict))
            process.start()
            processes.append(process)
        
        for process in processes:
            process.join()
        
        # Find the process with the lowest final loss
        best_seed = min(return_dict, key=lambda seed: return_dict[seed][0])
        final_loss, best_predicted_sdf, colors = return_dict[best_seed]

        print(f'Best seed: {best_seed}, Final loss: {final_loss}')
        
        # Only plot the final SDF of the best process
        if viz_outputs:
            save_path = os.path.join(input_dir, f'{output_name}_best_seed_{best_seed}')
            plot_sdf(best_predicted_sdf, colors, "Best Predicted SDF", viz=False, step='final', save_path=save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit CSG parameters to a given SDF.")
    parser.add_argument('-i', '--input_dir', default='.', help='Input directory containing the tree json')
    parser.add_argument('-o', '--output_name', default='viz', help='Output directory for visualizations.')
    parser.add_argument('-opt', default='adam', help='optimizer')
    parser.add_argument("--grid_size", type=int, default=50, help="Size of the grid for visualization.")
    parser.add_argument("--viz_outputs", action="store_true", help="Render the CSG tree graph.")
    parser.add_argument("--num_steps", type=int, default=5001, help="Number of optimization steps.")
    parser.add_argument("--log_steps", type=int, default=500, help="Log every n steps.")
    parser.add_argument("--tree_path", type=str, default="csg_tree.json", help="Path read/write csg json trees.")
    parser.add_argument("--num_processes", type=int, default=None, help="Number of parallel processes to run. Set to None for a single process.")
    
    args = parser.parse_args()
    
    main(args.input_dir, args.output_name, args.opt, args.grid_size, args.viz_outputs, args.num_steps, args.log_steps, args.tree_path, args.num_processes)
