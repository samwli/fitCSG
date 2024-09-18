// Main export type that represents the CSG tree structure.
// It can either be a boolean operation (CSGOperation) or a basic geometric shape (CSGShape).
export type CSGResponse = CSGOperation | CSGShape;

// Represents a boolean operation in the CSG tree: Union, Intersection, or Difference.
// Each operation has two inputs, which can be another operation or a shape.
export interface CSGOperation {
  operation: 'Union' | 'Intersection' | 'Difference'; // The type of operation.
  left: CSGResponse; // The left input of the operation, which can be another operation or a shape.
  right: CSGResponse; // The right input of the operation, which can be another operation or a shape.
}

// A union type that includes all possible basic geometric shapes in the CSG tree.
export type CSGShape = Ellipsoid | Prism | Cylinder | Cone;

/**
 * Interface for Ellipsoid shape parameters.
 * Defines the center, sizes (radii along x, y, z axes), and rotation.
 */
export interface Ellipsoid {
  type: 'Ellipsoid'; // Type of the shape, which is Ellipsoid in this case.
  params: EllipsoidParams; // Parameters specific to Ellipsoid.
}

export interface EllipsoidParams {
  center: [number, number, number]; // Center of the ellipsoid [x, y, z].
  sizes: [number, number, number]; // Radii of the ellipsoid along the x, y, and z axes.
  rotation: [number, number, number]; // Rotation of the ellipsoid in degrees around the x, y, and z axes.
}

/**
 * Interface for Prism shape parameters.
 * Defines the center, sizes along x, y, z, and rotation.
 */
export interface Prism {
  type: 'Prism'; // Type of the shape, which is Prism in this case.
  params: PrismParams; // Parameters specific to Prism.
}

export interface PrismParams {
  center: [number, number, number]; // Center of the prism [x, y, z].
  sizes: [number, number, number]; // Dimensions of the prism along the x, y, and z axes.
  rotation: [number, number, number]; // Rotation of the prism in degrees around the x, y, and z axes.
}

/**
 * Interface for Cylinder shape parameters.
 * Defines the center, radius, height, and rotation.
 */
export interface Cylinder {
  type: 'Cylinder'; // Type of the shape, which is Cylinder in this case.
  params: CylinderParams; // Parameters specific to Cylinder.
}

export interface CylinderParams {
  center: [number, number, number]; // Center of the cylinder [x, y, z].
  radius: number; // Radius of the cylinder base.
  height: number; // Height of the cylinder.
  rotation: [number, number, number]; // Rotation of the cylinder in degrees around the x, y, and z axes.
}

/**
 * Interface for Cone shape parameters.
 * Defines the center, base radius, height, and rotation.
 */
export interface Cone {
  type: 'Cone'; // Type of the shape, which is Cone in this case.
  params: ConeParams; // Parameters specific to Cone.
}

export interface ConeParams {
  center: [number, number, number]; // Center of the cone [x, y, z].
  radius: number; // Radius of the cone base.
  height: number; // Height of the cone.
  rotation: [number, number, number]; // Rotation of the cone in degrees around the x, y, and z axes.
}
