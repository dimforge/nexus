//! Geometric shape definitions.
//!
//! This module provides shape types and their geometric operations:
//! - Ball (sphere/circle)
//! - Capsule (swept sphere)
//! - Cuboid (box/rectangle)
//! - Segment (line segment)
//! - Triangle
//! - Tetrahedron (3D only)
//! - Cone (3D only)
//! - Cylinder (3D only)
//! - ConvexPolyhedron (convex polygon/polyhedron)
//! - TriMesh (triangle mesh)
//! - Polyline

mod capsule;
#[cfg(feature = "dim3")]
mod cone;
mod convex_polyhedron;
#[cfg(feature = "dim3")]
mod cylinder;
mod polyline;
mod segment;
mod shape;
#[cfg(feature = "dim3")]
mod tetrahedron;
mod triangle;
mod trimesh;

// Re-export struct types only to avoid ambiguous function re-exports.
pub use parry::shape::{Ball, Cuboid};

pub use capsule::Capsule;
#[cfg(feature = "dim3")]
pub use cone::Cone;
pub use convex_polyhedron::ConvexPolyhedron;
#[cfg(feature = "dim3")]
pub use cylinder::Cylinder;
pub use polyline::Polyline;
pub use segment::Segment;
pub use shape::*;
#[cfg(feature = "dim3")]
pub use tetrahedron::Tetrahedron;
pub use triangle::Triangle;
pub use trimesh::TriMesh;
