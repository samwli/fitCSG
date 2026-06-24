// Main export type that represents the CSG tree structure.
// It can either be a boolean operation (CSGOperation) or a basic geometric shape (CSGShape).
export type CSGResponse = CSGOperation | CSGShape;

// Represents a boolean operation in the CSG tree: Union, Intersection, or Difference.
// Each operation has two inputs, which can be another operation or a shape.
export interface CSGOperation {
  operation: 'Union' | 'Intersection' | 'Subtraction'; // The type of operation.
  left: CSGResponse; // The left input of the operation, which can be another operation or a shape.
  right: CSGResponse; // The right input of the operation, which can be another operation or a shape.
}

export type CSGShape = ShapeWithID<'Ellipsoid', EllipsoidParams> | 
                        ShapeWithID<'Prism', PrismParams> | 
                        ShapeWithID<'Cylinder', CylinderParams> | 
                        ShapeWithID<'Cone', ConeParams> | 
                        ShapeWithID<'Sphere', SphereParams>;

/**
 * Generic interface for a shape with an ID and a part label.
 * Each shape type is appended with an integer ID (e.g., "Cylinder0", "Ellipsoid1").
 */
export interface ShapeWithID<Type extends string, Params> {
  type: `${Type}${number}`; // Dynamic type with the shape name and an integer ID.
  params: Params; // Parameters specific to the shape.
  part: string; // A short label describing the part (e.g., "Base", "Handle", "Body").
}

// Parameter interfaces for each shape type.
export interface EllipsoidParams {
  center: [number, number, number];
  sizes: [number, number, number];
  axis: [number, number, number]; // Normalized vector defining orientation.
}

export interface PrismParams {
  center: [number, number, number];
  sizes: [number, number, number];
  axis: [number, number, number]; // Normalized vector defining orientation.
}

export interface CylinderParams {
  center: [number, number, number];
  radius: number;
  height: number;
  axis: [number, number, number]; // Normalized vector defining orientation.
}

export interface ConeParams {
  center: [number, number, number];
  radius: number;
  height: number;
  axis: [number, number, number]; // Normalized vector defining orientation.
}

export interface SphereParams {
  center: [number, number, number];
  radius: number;
}
