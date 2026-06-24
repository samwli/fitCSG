# Task: Generate a CSG Representation of an Object

## Context:
You are tasked with creating a 3D representation of an object (e.g., `<ObjectName>`) using the Constructive Solid Geometry (CSG) schema. This schema combines basic geometric shapes (Ellipsoid, Prism, Cylinder, Cone, Sphere) with boolean operations (Union, Intersection, Subtraction) to form complex objects.

## Constraints:
- **Bounding Box**: The entire object must fit within a 1x1x1 bounding box, centered at the origin. All dimensions (positions, sizes, radii, heights) must remain within this range.
- **Minimize Complexity**: Use the fewest possible shapes and operations to capture the object's main geometry and semantics. Simplify wherever possible while maintaining visual coherence.
- **Focus on Critical Components**: Include primary geometric and semantic parts of the object that contribute to its overall design or interaction. For example, the sunglasses can be minimally representated by its two lenses and temples (legs).
-**Avoid Unnecessary Granularity**: Small and ornamental details, such as edges and tips, should generally be ignored or merged into broader parts unless they are critical to the object’s characterization. The sunglasses bridge or nodepads, for example, should not be included. A good rule of thumb is to omit parts that are too small to grasp.

## Boolean Operations:
- **Union**: Combine two shapes into a single structure.
- **Intersection**: Keep only the overlapping region of two shapes.
- **Subtraction**: Remove one shape from another to create hollow or concave features.

## Structure:
- Use the provided schema to define shapes and operations.
- Each shape or operation must be clearly specified with all required parameters:
  - **Shapes**: Include `type`, `params`, and `part` keys. Parameters depend on the shape type:
    - **center**: The position of the shape.
    - **sizes/radius/height**: The dimensions of the shape.
    - **axis**: A normalized vector `[x, y, z]` defining the orientation of the shape.
  - **Operations**: Specify `operation`, `left`, and `right` inputs, where inputs can be shapes or other operations.

## Guidelines:
1. **High-level Completeness**: Ensure the object is reconstructed with the geometric and/or semantic components necessary to define and understand it.
2. **Decomposition**: Break the object into basic geometric components. Use cylinders for elongated features, cones for tapered shapes, and ellipsoids or spheres for rounded parts.
3. **Logical Assembly**: Combine shapes step-by-step using boolean operations to form a coherent structure.
4. **Validate Dimensions**: Confirm that all parts remain within the 1x1x1 bounding box and are appropriately scaled and positioned.

# Output:
- Provide a JSON response that adheres strictly to the schema and represents the object fully.
- Ensure the structure includes all geometrically and semantically significant components while keeping the tree simple and concise.
