//! Configuration-Space Obstacle Point Module
//!
//! This module provides the CSO point structure used in GJK algorithm.

use crate::Vector;

pub const FLT_EPS: f32 = 1.0e-7;
pub const EPS_TOL: f32 = 1.0e-6;

/// A point of a Configuration-Space Obstacle.
///
/// A Configuration-Space Obstacle (CSO) is the result of the
/// Minkowski Difference of two solids. In other words, each of its
/// points correspond to the difference of two point, each belonging
/// to a different solid.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct CsoPoint {
    /// The point on the CSO. This is equal to `self.orig1 - self.orig2`, unless this CsoPoint
    /// has been translated with self.translate.
    pub point: Vector,
    /// The original point on the first shape used to compute `self.point`.
    pub orig_a: Vector,
    /// The original point on the second shape used to compute `self.point`.
    pub orig_b: Vector,
}

impl CsoPoint {
    /// Creates a new CSO point with all information provided.
    #[inline]
    pub fn new(point: Vector, orig_a: Vector, orig_b: Vector) -> Self {
        Self {
            point,
            orig_a,
            orig_b,
        }
    }

    /// Initializes a CSO point with `orig1 - orig2`.
    pub fn from_points(orig1: Vector, orig2: Vector) -> CsoPoint {
        CsoPoint::new(orig1 - orig2, orig1, orig2)
    }

    /// CSO point where all components are set to zero.
    pub fn origin() -> CsoPoint {
        CsoPoint::new(Vector::ZERO, Vector::ZERO, Vector::ZERO)
    }
}
