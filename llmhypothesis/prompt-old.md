# Prompt:
You are provided with a schema that defines a Constructive Solid Geometry (CSG) tree structure. This schema allows for creating 3D objects by combining basic geometric shapes (Ellipsoid, Prism, Cylinder, Cone, and Sphere) using boolean operations (Union, Intersection, Subtraction). Each operation combines exactly two inputs: either a basic shape or the result of another operation.

# Task:
Given the name of an object (e.g., <ObjectName>) your goal is to create a CSG representation of that object using the provided schema. The CSG representation should creatively combine the available shapes and operations to approximate the structure of the object. 

## Constraints:
 - The generated object should fit and be centered within a 1x1x1 box. Ensure that the parameters for positions, sizes, radii, and heights are within this range to keep the object size constrained.

# Guidelines:
 - Focus on main parts: Only represent the major components of the object that are suitable for interaction/grasping. Small details and ornamental features do not need to be included.
 - Minimize the Number of Shapes: Use the minimal number of shapes and operations necessary to capture the coarse reprsentation of the object.
 - Understand the Object: Decompose the object into its basic geometric and semantic components. Think about how shapes like cylinders, cones, and ellipsoids can represent the key parts of the object.

## Boolean Operations:
- **Union**: Combine two shapes into a single structure.
- **Intersection**: Keep only the overlapping region of two shapes.
- **Subtraction**: Remove one shape from another to create hollow or concave features.

## Structure:
- Use the provided schema to define shapes and operations.
- Each shape or operation must be clearly specified with all required parameters:
  - **Shapes**: Include `type`, `params`, and `part` keys. Parameters depend on the shape type (e.g., `center`, `sizes`, `radius`, `height`, `rotation`).
  - **Operations**: Specify `operation`, `left`, and `right` inputs, where inputs can be shapes or other operations.

# Output:
- Generate a JSON response that represents the object's structure using the provided schema.
- The CSG composition should be logical, visually coherent, and fit within the given constraints.

## Reminder: 
- Use the constraints and guidelines to generate the JSON response for <ObjectName>, ensuring that the CSG structure adheres to the schema and represents the object accurately while focusing on the main functional components, using the minimal number of shapes, and fitting within a 1x1x1 bounding box.