//! Ray definition and operations.

use crate::Vector;

/// A ray defined by an origin point and direction (not necessarily normalized).
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Ray {
    /// The ray's starting point.
    pub origin: Vector,
    /// The ray's direction vector.
    pub dir: Vector,
}

impl Ray {
    /// Creates a new ray from origin and direction.
    #[inline]
    pub fn new(origin: Vector, dir: Vector) -> Self {
        Self { origin, dir }
    }

    /// Computes the point at `origin + t * dir`.
    #[inline]
    pub fn point_at(&self, t: f32) -> Vector {
        self.origin + self.dir * t
    }
}
