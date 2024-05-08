# fitCSG
This project currently has two components. The first is for data generation "csgsdf.py" By design, the script generates full, binary CSG trees. There are checks to keep each selected boolean operation "valid" from the leaf shape primitives. Set the desired depth in the main function. This script will output a visualization of the CSG tree, the SDF field (with each leaf a different color), and the CSG in dictionary format which is pickled and saved (`csg_outline.pkl`). You can visually inspect outputs to run experiments on for the next script.

The second component is "fit_csg.py". Specify args: input path (where the `csg_outline.pkl` from the data generation module should be found), output path, and optimizer (Adam, AdamW, SGD, RMSProp). The optimizer parameters and num steps can be tuned in the script for now. The visualized gt SDF and optimized SDFs are written to examine the convergence and correctness. 

TODOs:
1. Try SAM optimizer
2. Tune optimizer parameters
3. Develop the shape primitive SDFs: we need more shapes, and to include rotations in the generatation, which should be then be included as optimizable parameters during the CSG fitting.
4. Try optimization on point clouds of objects (ShapeNet, real world)
5. The target SDF is given at the moment, since we are generating it. In real world, we need to convert point clouds to SDF fields.
