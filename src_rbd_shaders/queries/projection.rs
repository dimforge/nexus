//! Point Projection Result Module
//!
//! This module defines the common return type for all point projection operations.
//! `ProjectionResult` carries both the projected point location and information about
//! whether the original point was inside the shape.

use crate::Vector;
use glamx::Vec3;
use parry::query::PointProjection;

/// Epsilon for floating-point comparisons.
pub const EPSILON: Vector = Vector::splat(1.1920929e-7);

/// The result of a point projection operation.
///
/// This structure is returned by all point projection functions.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct ProjectionResult {
    /// The point's projection on the shape.
    /// This can be equal to the original point if the point was inside
    /// of the shape and the projection function doesn't always project
    /// on the boundary.
    pub point: Vector,
    /// Is the point inside of the shape?
    pub is_inside: bool,
}

impl From<PointProjection> for ProjectionResult {
    fn from(value: PointProjection) -> Self {
        ProjectionResult {
            point: value.point,
            is_inside: value.is_inside,
        }
    }
}

impl ProjectionResult {
    /// Creates a new projection result.
    #[inline]
    pub fn new(point: Vector, is_inside: bool) -> Self {
        Self { point, is_inside }
    }
}

/// Feature type constants.
pub const FEATURE_VERTEX: u32 = 0;
pub const FEATURE_EDGE: u32 = 1;
pub const FEATURE_FACE: u32 = 2;
pub const FEATURE_SOLID: u32 = 3;

/// Projection result with location information.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct ProjectionWithLocation {
    /// The projected point.
    pub point: Vector,
    /// Is the point inside the shape?
    pub inside: bool,
    /// Barycentric coordinates of the projection on the feature.
    pub bcoords: Vec3,
    /// The type of feature (vertex, edge, face, solid).
    pub feature_type: u32,
    /// The feature ID.
    pub id: u32,
    /// Explicit padding, required for spv -> WGSL conversion.
    pub padding: [u32; 3],
}

impl ProjectionWithLocation {
    /// Creates a vertex projection result.
    #[inline]
    pub fn vertex(pt: Vector, id: u32, inside: bool) -> Self {
        Self {
            point: pt,
            bcoords: Vec3::ZERO,
            feature_type: FEATURE_VERTEX,
            id,
            inside,
            padding: [0; _],
        }
    }

    /// Creates an edge projection result.
    #[inline]
    pub fn edge(pt: Vector, bcoords: glamx::Vec2, id: u32, inside: bool) -> Self {
        Self {
            point: pt,
            bcoords: Vec3::new(bcoords.x, bcoords.y, 0.0),
            feature_type: FEATURE_EDGE,
            id,
            inside,
            padding: [0; _],
        }
    }

    /// Creates a face projection result.
    #[inline]
    pub fn face(pt: Vector, bcoords: Vec3, id: u32, inside: bool) -> Self {
        Self {
            point: pt,
            bcoords,
            feature_type: FEATURE_FACE,
            id,
            inside,
            padding: [0; _],
        }
    }

    /// Creates a solid projection result.
    #[inline]
    pub fn solid(pt: Vector) -> Self {
        Self {
            point: pt,
            bcoords: Vec3::ZERO,
            feature_type: FEATURE_SOLID,
            id: 0,
            inside: true,
            padding: [0; _],
        }
    }

    /// Gets the barycentric coordinates for a 3D projection.
    #[cfg(feature = "dim3")]
    pub fn barycentric_coordinates(&self) -> Vec3 {
        let mut bcoords = Vec3::ZERO;

        match self.feature_type {
            FEATURE_VERTEX => match self.id {
                0 => bcoords.x = 1.0,
                1 => bcoords.y = 1.0,
                2 => bcoords.z = 1.0,
                _ => {}
            },
            FEATURE_EDGE => match self.id {
                0 => {
                    bcoords.x = self.bcoords.x;
                    bcoords.y = self.bcoords.y;
                }
                1 => {
                    bcoords.y = self.bcoords.x;
                    bcoords.z = self.bcoords.y;
                }
                2 => {
                    bcoords.x = self.bcoords.x;
                    bcoords.z = self.bcoords.y;
                }
                _ => {}
            },
            FEATURE_FACE => {
                bcoords = self.bcoords;
            }
            _ => {}
        }

        bcoords
    }
}
