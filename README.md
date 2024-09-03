# fitCSG
This project currently has two components. The first is for data generation `csgsdf.py` By design, the script generates full, binary CSG trees. There are checks to keep each selected boolean operation "valid" from the leaf shape primitives. Set the desired depth in the main function. This script will output a visualization of the CSG tree, the SDF field (with each leaf a different color), and the CSG in dictionary format (`csg_tree.json`). You can visually inspect outputs to run experiments on for the next script. I recommend running this script, which does not require gpu acceleration, on a local machine if you want to visualize the plots. Note: the random tree generation aspect is less useful as we proceed. As such, you can hand-craft a `csg_tree.json` and use the script to inspect the tree graph and point cloud. This will now be the default behavior (`--random_tree=False`)

The second component is `fit_csg.py`. Args: input path (where the `csg_outline.json` from the data generation module should be found), output path, and optimizer (Adam, AdamW, SGD, RMSProp). The optimizer parameters and num steps can be tuned in the script for now. The visualized gt SDF and optimized SDFs are written to examine the convergence and correctness. 

## Installation
First install the conda env: `conda env create -f environment.yml`.
Next, install pytorch that fits your system.


## TODOs
1. Try SAM and Levenberg-Marquardt optimizers and tune parameters
2. Use multiprocessing to optimize from different random inits in parallel
3. Develop the shape primitive SDFs. We need more shapes, and to include rotations in the generatation, which should be then be included as optimizable parameters during the CSG fitting.
4. The target SDF is given at the moment, since we are generating it. For real world experiments, we need to convert point clouds to SDF fields.
