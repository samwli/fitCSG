# Prompt:
You are provided with a schema that defines a Constructive Solid Geometry (CSG) tree structure. This schema allows for creating complex 3D objects by combining basic geometric shapes (Ellipsoid, Prism, Cylinder, and Cone) using boolean operations (Union, Intersection, Difference). Each operation combines exactly two inputs: either a basic shape or the result of another operation.

# Task:
Given the name of an object (e.g., <ObjectName>) your goal is to create a CSG representation of that object using the provided schema. The CSG representation should creatively combine the available shapes and operations to approximate the structure of the object. 

## Constraints:
 - The generated object should fit within a 1x1x1 box. Ensure that the parameters for positions, sizes, radii, and heights are within this range to keep the object size constrained.

# Guidelines:
 - Focus on Main Functional Parts: Only represent the main parts of the object that facilitate its primary functionality. Small details and ornamental features do not need to be included.
 - Minimize the Number of Shapes: Use the minimal number of shapes necessary to capture the main functional aspects of the object. Avoid unnecessary complexity in the design.
 - Understand the Object: Decompose the object into its basic geometric components. Think about how shapes like cylinders, cones, and ellipsoids can represent the key parts of the object.

## Use Boolean Operations:
 - Use Union to combine shapes into a single structure.
 - Use Intersection to refine shapes or create specific features.
 - Use Difference to subtract portions of one shape from another.

## Structure:
 - The JSON response must strictly follow the provided schema.
 - Define each shape or operation correctly with its respective parameters.

## Parameters:
 - For each shape, provide parameters like center, sizes, radius, height, and rotation as applicable.
 - Ensure all dimensions are adjusted to keep the overall object within a 1x1x1 box.
 - For each operation, specify the left and right inputs, which could be other operations or basic shapes.

# Output:
- Generate a JSON response that represents the object's structure using the provided schema.
- The CSG composition should be logical, visually coherent, and fit within the given constraints.

## Reminder: 
- Use the constraints and guidelines to generate the JSON response for <ObjectName>, ensuring that the CSG structure adheres to the schema and represents the object accurately while focusing on the main functional components, using the minimal number of shapes, and fitting within a 1x1x1 bounding box.