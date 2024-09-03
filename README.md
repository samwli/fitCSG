# fitCSG
This project currently has two components. The first is `csgsdf.py`, which visualizes a signed-distance field (SDF) and pretty(-ish) constructive solid geometry (CSG) graph from a json CSG graph. The script can process a user-provided json, or generate and save a random json tree up to a given depth (there are some limited checks for shape and operation selection to try to create "valid" trees). Note: the random tree generation is less useful as we proceed, so the default behavior will be (`--random_tree=False`) for hand-crafted json CSG trees. I recommend running this script, which does not require gpu acceleration, on a local machine for visualization.

The second component is `fit_csg.py`, which for now depends on a `csg_tree.json` that is processed into a CSG tree outline, a target SDF, and random initial leaf params. The tree outline and leaf paramters are used to construct a predicted SDF, and the leaf params are optimized to minimize the loss with the target SDF.

## Installation
First install the conda env: `conda env create -f environment.yml`.
Next, install pytorch that fits your system.

## TODOs
1. Try SAM and Levenberg-Marquardt optimizers and tune parameters.
2. Use multiprocessing to optimize from different random inits in parallel.
3. Optimization should occur in a normalize space, so solve ICP/Procrustes for scale, shift, and rotation.
4. Bridge the gaps from monocular RGB -> full object point cloud -> target SDF.
5. Create a dataset to develop the optimization and LLM modules. This requires defining further SDF shapes (with rotation) and hand-crafting the tree and parameter values.
6. Experiment with LLM prompting to give us the tree outline for real world objects. We should also get initial parameters from LLM since we are optimizing in a normalized space, LLM may be able to provide relatively meaningful params. 
