//! Capsule shape (swept sphere along a segment).

use crate::queries::{PolygonalFeature, ProjectionResult};
use crate::shapes::segment::Segment;
use crate::{Pose, Vector};
use khal_std::index::MaybeIndexUnchecked;

/// A capsule shape, defined by a central segment and a radius.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Capsule {
    /// The capsule's principal axis (central line segment).
    /// The segment endpoints define the capsule's orientation and length.
    pub segment: Segment,
    /// The capsule's radius (distance from the central axis to the surface).
    /// Must be positive.
    pub radius: f32,
}

impl Capsule {
    /// Creates a new capsule from a segment and radius.
    #[inline]
    pub fn new(segment: Segment, radius: f32) -> Self {
        Self { segment, radius }
    }

    /// Creates a new capsule from two endpoints and a radius.
    #[inline]
    pub fn from_endpoints(a: Vector, b: Vector, radius: f32) -> Self {
        Self {
            segment: Segment::new(a, b),
            radius,
        }
    }

    /// Projects a point onto a capsule in its local coordinate frame.
    #[inline]
    pub fn project_local_point(&self, pt: Vector) -> Vector {
        let proj_on_axis = self.segment.project_local_point(pt);
        let dproj = pt - proj_on_axis;
        let dist_to_axis = dproj.length();

        // PERF: call `select` instead?
        if dist_to_axis > self.radius {
            proj_on_axis + dproj * (self.radius / dist_to_axis)
        } else {
            pt
        }
    }

    /// Projects a point onto a transformed capsule in world space.
    #[inline]
    pub fn project_point(&self, pose: Pose, pt: Vector) -> Vector {
        let local_pt = pose.inverse_transform_point(pt);
        pose * self.project_local_point(local_pt)
    }

    /// Projects a point onto the boundary (surface) of a capsule in its local frame.
    ///
    /// Always projects onto the surface, even if the point is inside.
    #[inline]
    pub fn project_local_point_on_boundary(&self, pt: Vector) -> ProjectionResult {
        let proj_on_axis = self.segment.project_local_point(pt);
        let dproj = pt - proj_on_axis;
        let dist_to_axis = dproj.length();

        if dist_to_axis > 0.0 {
            let is_inside = dist_to_axis <= self.radius;
            ProjectionResult::new(
                proj_on_axis + dproj * (self.radius / dist_to_axis),
                is_inside,
            )
        } else {
            // Very rare occurrence: the point lies on the capsule's axis exactly.
            // Pick an arbitrary projection direction along an axis orthogonal to the principal axis.
            let axis_seg = self.segment.b - self.segment.a;
            let axis_len = axis_seg.length();
            let proj_dir = any_orthogonal_vector(axis_seg / axis_len);
            ProjectionResult::new(proj_on_axis + proj_dir * self.radius, true)
        }
    }

    /// Projects a point onto the boundary of a transformed capsule in world space.
    #[inline]
    pub fn project_point_on_boundary(&self, pose: Pose, pt: Vector) -> ProjectionResult {
        let local_pt = pose.inverse_transform_point(pt);
        let mut result = self.project_local_point_on_boundary(local_pt);
        result.point = pose * result.point;
        result
    }

    /// Computes the local support point of a capsule.
    #[inline]
    pub fn local_support_point(&self, dir: Vector) -> Vector {
        let seg_dir = self.segment.b - self.segment.a;
        let endpoint = if seg_dir.dot(dir) >= 0.0 {
            self.segment.b
        } else {
            self.segment.a
        };

        if self.radius == 0.0 {
            return endpoint;
        }

        let dir_len = dir.length();
        let normal = if dir_len != 0.0 {
            dir / dir_len
        } else {
            Vector::Y
        };
        endpoint + normal * self.radius
    }

    /// Computes the support face of a capsule.
    #[inline]
    pub fn support_face(&self, dir: Vector) -> PolygonalFeature {
        let mut result = PolygonalFeature::default();

        if self.radius == 0.0 {
            result.vertices.write(0, self.segment.a);
            result.vertices.write(1, self.segment.b);
            result.num_vertices = 2;
        } else {
            let seg_dir = self.segment.b - self.segment.a;
            if seg_dir.dot(dir).abs() <= 1.0e-6 {
                result.vertices.write(0, self.segment.a);
                result.vertices.write(1, self.segment.b);
                result.num_vertices = 2;
            } else {
                let endpoint = if seg_dir.dot(dir) >= 0.0 {
                    self.segment.b
                } else {
                    self.segment.a
                };
                result
                    .vertices
                    .write(0, endpoint + dir * (self.radius / dir.length()));
                result.num_vertices = 1;
            }
        }

        result
    }
}

/// Computes an orthonormal basis from a single normalized 3D vector.
#[cfg(feature = "dim3")]
#[inline]
pub fn orthonormal_basis3(v: Vector) -> [Vector; 2] {
    // NOTE: not using `sign` because we don't want the 0.0 case to return 0.0.
    let sign = if v.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + v.z);
    let b = v.x * v.y * a;

    [
        Vector::new(1.0 + sign * v.x * v.x * a, sign * b, -sign * v.x),
        Vector::new(b, sign + v.y * v.y * a, -v.y),
    ]
}

/// Finds an arbitrary vector orthogonal to the given vector.
#[inline]
pub fn any_orthogonal_vector(v: Vector) -> Vector {
    #[cfg(feature = "dim2")]
    {
        Vector::new(v.y, -v.x)
    }
    #[cfg(feature = "dim3")]
    {
        orthonormal_basis3(v)[0]
    }
}
