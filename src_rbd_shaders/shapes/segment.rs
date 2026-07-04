//! Segment (Line Segment) Module
//!
//! This module provides geometric operations for line segments.
//! A segment is defined by two endpoints and represents the straight line
//! connecting them.

use crate::Vector;
use crate::queries::ProjectionWithLocation;

/// A line segment defined by two endpoints.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Segment {
    /// First endpoint of the segment.
    pub a: Vector,
    #[cfg(feature = "dim3")]
    pub padding0: u32,
    /// Second endpoint of the segment.
    pub b: Vector,
    #[cfg(feature = "dim3")]
    pub padding1: u32,
}

impl Segment {
    /// Creates a new segment from two endpoints.
    #[inline]
    pub fn new(a: Vector, b: Vector) -> Self {
        Self {
            a,
            b,
            #[cfg(feature = "dim3")]
            padding0: 0,
            #[cfg(feature = "dim3")]
            padding1: 0,
        }
    }

    /// Projects a point onto a line segment in local coordinates.
    ///
    /// Returns the closest point on the segment to `pt`.
    /// TODO: implement the other projection functions
    #[inline]
    pub fn project_local_point(&self, pt: Vector) -> Vector {
        let ab = self.b - self.a;
        let ap = pt - self.a;
        let ab_ap = ab.dot(ap);
        let sqnab = ab.dot(ab);

        // PERF: would it be faster to do a bunch of `select` instead of `if`?
        if ab_ap <= 0.0 {
            // Voronoi region of vertex 'a'.
            self.a
        } else if ab_ap >= sqnab {
            // Voronoi region of vertex 'b'.
            self.b
        } else {
            // Voronoi region of the segment interior.
            let u = ab_ap / sqnab;
            self.a + ab * u
        }
    }

    /// Projects a point onto a segment and returns location information.
    #[inline]
    pub fn project_local_point_and_get_location(
        &self,
        pt: Vector,
        _solid: bool,
    ) -> ProjectionWithLocation {
        let ab = self.b - self.a;
        let ap = pt - self.a;
        let ab_ap = ab.dot(ap);
        let sqnab = ab.dot(ab);

        // TODO: is this acceptable?

        if ab_ap <= 0.0 {
            // Voronoi region of vertex 'a'.
            let inside = crate::queries::relative_eq(self.a, pt);
            ProjectionWithLocation::vertex(self.a, 0, inside)
        } else if ab_ap >= sqnab {
            // Voronoi region of vertex 'b'.
            let inside = crate::queries::relative_eq(self.b, pt);
            ProjectionWithLocation::vertex(self.b, 1, inside)
        } else {
            // Voronoi region of the segment interior.
            let u = ab_ap / sqnab;
            let bcoords = glamx::Vec2::new(1.0 - u, u);
            let proj = self.a + ab * u;
            let inside = crate::queries::relative_eq(proj, pt);
            ProjectionWithLocation::edge(proj, bcoords, 0, inside)
        }
    }
}
