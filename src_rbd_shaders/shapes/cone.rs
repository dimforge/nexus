//! Cone Shape Module (3D only)
//!
//! This module provides geometric operations for cones.
//! A cone is defined by its half-height (distance from base center to apex)
//! and base radius.
//!
//! The cone is oriented along the Y axis with:
//! - Apex at (0, half_height, 0)
//! - Base center at (0, -half_height, 0)
//! - Base is a circle in the XZ plane

use crate::queries::{PolygonalFeature, ProjectionResult};
use crate::shapes::Segment;
use crate::{Pose, Vector};
use glamx::{Vec2, Vec3};
use khal_std::index::MaybeIndexUnchecked;

/// A cone shape with circular base (3D only).
///
/// The cone is aligned with the Y axis:
/// - Base at y = -half_height
/// - Apex at y = +half_height
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Cone {
    /// Half of the cone's height (distance from base center to apex).
    pub half_height: f32,
    /// Radius of the circular base.
    pub radius: f32,
}

impl Cone {
    /// Creates a new cone with the given half-height and radius.
    #[inline]
    pub fn new(half_height: f32, radius: f32) -> Self {
        Self {
            half_height,
            radius,
        }
    }

    /// Projects a point on a cone.
    ///
    /// If the point is inside the cone, the point itself is returned.
    pub fn project_local_point(&self, pt: Vector) -> Vector {
        // Project on the basis.
        let planar_dist_from_basis_center = Vec2::new(pt.x, pt.z).length();
        let dir_from_basis_center = if planar_dist_from_basis_center > 0.0 {
            Vec2::new(pt.x, pt.z) / planar_dist_from_basis_center
        } else {
            Vec2::new(1.0, 0.0)
        };

        let projection_on_basis = Vec3::new(pt.x, -self.half_height, pt.z);

        if pt.y < -self.half_height && planar_dist_from_basis_center <= self.radius {
            // The projection is on the basis.
            return projection_on_basis;
        }

        // Project on the basis circle.
        let proj2d = dir_from_basis_center * self.radius;
        let projection_on_basis_circle = Vec3::new(proj2d.x, -self.half_height, proj2d.y);

        // Project on the conic side.
        let apex_point = Vec3::new(0.0, self.half_height, 0.0);
        let conic_side_segment = Segment::new(apex_point, projection_on_basis_circle);
        let conic_side_segment_dir = conic_side_segment.b - conic_side_segment.a;
        let proj = conic_side_segment.project_local_point(pt);

        let apex_to_basis_center = Vec3::new(0.0, -2.0 * self.half_height, 0.0);

        // Now determine if the point is inside the cone.
        if pt.y >= -self.half_height
            && pt.y <= self.half_height
            && conic_side_segment_dir
                .cross(pt - apex_point)
                .dot(conic_side_segment_dir.cross(apex_to_basis_center))
                >= 0.0
        {
            // We are inside the cone.
            pt
        } else {
            // We are outside the cone, return the computed segment projection.
            proj
        }
    }

    /// Projects a point on a transformed cone.
    ///
    /// If the point is inside the cone, the point itself is returned.
    pub fn project_point(&self, pose: Pose, pt: Vector) -> Vector {
        let local_pt = pose.inverse_transform_point(pt);
        pose * self.project_local_point(local_pt)
    }

    /// Projects a point on the boundary of a cone.
    pub fn project_local_point_on_boundary(&self, pt: Vector) -> ProjectionResult {
        // Project on the basis.
        let planar_dist_from_basis_center = Vec2::new(pt.x, pt.z).length();
        let dir_from_basis_center = if planar_dist_from_basis_center > 0.0 {
            Vec2::new(pt.x, pt.z) / planar_dist_from_basis_center
        } else {
            Vec2::new(1.0, 0.0)
        };

        let projection_on_basis = Vec3::new(pt.x, -self.half_height, pt.z);

        if pt.y < -self.half_height && planar_dist_from_basis_center <= self.radius {
            // The projection is on the basis.
            return ProjectionResult::new(projection_on_basis, false);
        }

        // Project on the basis circle.
        let proj2d = dir_from_basis_center * self.radius;
        let projection_on_basis_circle = Vec3::new(proj2d.x, -self.half_height, proj2d.y);

        // Project on the conic side.
        let apex_point = Vec3::new(0.0, self.half_height, 0.0);
        let conic_side_segment = Segment::new(apex_point, projection_on_basis_circle);
        let conic_side_segment_dir = conic_side_segment.b - conic_side_segment.a;
        let proj = conic_side_segment.project_local_point(pt);

        let apex_to_basis_center = Vec3::new(0.0, -2.0 * self.half_height, 0.0);

        // Now determine if the point is inside of the cone.
        if pt.y >= -self.half_height
            && pt.y <= self.half_height
            && conic_side_segment_dir
                .cross(pt - apex_point)
                .dot(conic_side_segment_dir.cross(apex_to_basis_center))
                >= 0.0
        {
            // We are inside the cone, so the correct projection is
            // either on the basis of the cone, or on the conic side.
            let pt_to_proj = proj - pt;
            let pt_to_basis_proj = projection_on_basis - pt;
            if pt_to_proj.dot(pt_to_proj) > pt_to_basis_proj.dot(pt_to_basis_proj) {
                ProjectionResult::new(projection_on_basis, true)
            } else {
                ProjectionResult::new(proj, true)
            }
        } else {
            // We are outside the cone, return the computed segment projection as-is.
            ProjectionResult::new(proj, false)
        }
    }

    /// Project a point of a transformed cone's boundary.
    pub fn project_point_on_boundary(&self, pose: Pose, pt: Vector) -> ProjectionResult {
        let local_pt = pose.inverse_transform_point(pt);
        let mut result = self.project_local_point_on_boundary(local_pt);
        result.point = pose * result.point;
        result
    }

    /// Computes the local support point of a cone.
    pub fn local_support_point(&self, dir: Vector) -> Vector {
        let mut vres = dir;
        vres.y = 0.0;

        let planar_dir_len = vres.length();

        if planar_dir_len == 0.0 {
            vres = Vec3::ZERO;
            vres.y = if dir.y >= 0.0 {
                self.half_height
            } else {
                -self.half_height
            };
        } else {
            vres *= self.radius / planar_dir_len;
            vres.y = -self.half_height;

            if dir.dot(vres) < dir.y * self.half_height {
                vres = Vec3::ZERO;
                vres.y = self.half_height;
            }
        }

        vres
    }

    /// Computes the support face of a cone.
    pub fn support_face(&self, dir: Vector) -> PolygonalFeature {
        let mut result = PolygonalFeature::default();

        let mut dir2 = Vec2::new(dir.x, dir.z);
        let dir2_len = dir2.length();
        if dir2_len < crate::F32_EPSILON {
            dir2 = Vec2::new(1.0, 0.0);
        } else {
            dir2 /= dir2_len;
        }

        if dir.y > 0.0 {
            // We return a segment lying on the cone's curved part.
            result.vertices.write(
                0,
                Vec3::new(
                    dir2.x * self.radius,
                    -self.half_height,
                    dir2.y * self.radius,
                ),
            );
            result
                .vertices
                .write(1, Vec3::new(0.0, self.half_height, 0.0));
            result.num_vertices = 2;
        } else {
            // We return a square approximation of the cone cap.
            let y = -self.half_height;
            result
                .vertices
                .write(0, Vec3::new(dir2.x * self.radius, y, dir2.y * self.radius));
            result
                .vertices
                .write(1, Vec3::new(-dir2.y * self.radius, y, dir2.x * self.radius));
            result.vertices.write(
                2,
                Vec3::new(-dir2.x * self.radius, y, -dir2.y * self.radius),
            );
            result
                .vertices
                .write(3, Vec3::new(dir2.y * self.radius, y, -dir2.x * self.radius));
            result.num_vertices = 4;
        }

        result
    }
}
