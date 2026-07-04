//! Cylinder Shape Module (3D only)
//!
//! This module provides geometric operations for cylinders.
//! A cylinder is defined by its half-height and radius.
//!
//! The cylinder is oriented along the Y axis with:
//! - Top at y = +half_height.
//! - Bottom at y = -half_height.
//! - Circular cross-section in the XZ plane.

use crate::queries::{PolygonalFeature, ProjectionResult};
use crate::{Pose, Vector};
use glamx::{Vec2, Vec3};
use khal_std::index::MaybeIndexUnchecked;

/// A cylinder shape with circular cross-section (3D only).
///
/// The cylinder is aligned with the Y axis, extending from
/// y = -half_height to y = +half_height.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Cylinder {
    /// Half of the cylinder's height.
    pub half_height: f32,
    /// Radius of the circular cross-section.
    pub radius: f32,
}

impl Cylinder {
    /// Creates a new cylinder with the given half-height and radius.
    #[inline]
    pub fn new(half_height: f32, radius: f32) -> Self {
        Self {
            half_height,
            radius,
        }
    }

    /// Projects a point on a cylinder.
    ///
    /// If the point is inside the cylinder, the point itself is returned.
    pub fn project_local_point(&self, pt: Vector) -> Vector {
        // Project on the basis.
        let planar_dist_from_basis_center = Vec2::new(pt.x, pt.z).length();
        let dir_from_basis_center = if planar_dist_from_basis_center > 0.0 {
            Vec2::new(pt.x, pt.z) / planar_dist_from_basis_center
        } else {
            Vec2::new(1.0, 0.0)
        };

        let proj2d = dir_from_basis_center * self.radius;

        // PERF: reduce branching
        if pt.y >= -self.half_height
            && pt.y <= self.half_height
            && planar_dist_from_basis_center <= self.radius
        {
            pt
        } else {
            // The point is outside of the cylinder.
            if pt.y > self.half_height {
                if planar_dist_from_basis_center <= self.radius {
                    Vec3::new(pt.x, self.half_height, pt.z)
                } else {
                    Vec3::new(proj2d.x, self.half_height, proj2d.y)
                }
            } else if pt.y < -self.half_height {
                // Project on the bottom plane or the bottom circle.
                if planar_dist_from_basis_center <= self.radius {
                    Vec3::new(pt.x, -self.half_height, pt.z)
                } else {
                    Vec3::new(proj2d.x, -self.half_height, proj2d.y)
                }
            } else {
                // Project on the side.
                Vec3::new(proj2d.x, pt.y, proj2d.y)
            }
        }
    }

    /// Projects a point on a transformed cylinder.
    ///
    /// If the point is inside the cylinder, the point itself is returned.
    pub fn project_point(&self, pose: Pose, pt: Vector) -> Vector {
        let local_pt = pose.inverse_transform_point(pt);
        pose * self.project_local_point(local_pt)
    }

    /// Projects a point on the boundary of a cylinder.
    pub fn project_local_point_on_boundary(&self, pt: Vector) -> ProjectionResult {
        // Project on the basis.
        let planar_dist_from_basis_center = Vec2::new(pt.x, pt.z).length();
        let dir_from_basis_center = if planar_dist_from_basis_center > 0.0 {
            Vec2::new(pt.x, pt.z) / planar_dist_from_basis_center
        } else {
            Vec2::new(1.0, 0.0)
        };

        let proj2d = dir_from_basis_center * self.radius;

        // PERF: reduce branching
        if pt.y >= -self.half_height
            && pt.y <= self.half_height
            && planar_dist_from_basis_center <= self.radius
        {
            // The point is inside of the cylinder.
            let dist_to_top = self.half_height - pt.y;
            let dist_to_bottom = pt.y - (-self.half_height);
            let dist_to_side = self.radius - planar_dist_from_basis_center;

            if dist_to_top < dist_to_bottom && dist_to_top < dist_to_side {
                let projection_on_top = Vec3::new(pt.x, self.half_height, pt.z);
                ProjectionResult::new(projection_on_top, true)
            } else if dist_to_bottom < dist_to_top && dist_to_bottom < dist_to_side {
                let projection_on_bottom = Vec3::new(pt.x, -self.half_height, pt.z);
                ProjectionResult::new(projection_on_bottom, true)
            } else {
                let projection_on_side = Vec3::new(proj2d.x, pt.y, proj2d.y);
                ProjectionResult::new(projection_on_side, true)
            }
        } else {
            // The point is outside of the cylinder.
            if pt.y > self.half_height {
                if planar_dist_from_basis_center <= self.radius {
                    let projection_on_top = Vec3::new(pt.x, self.half_height, pt.z);
                    ProjectionResult::new(projection_on_top, false)
                } else {
                    let projection_on_top_circle = Vec3::new(proj2d.x, self.half_height, proj2d.y);
                    ProjectionResult::new(projection_on_top_circle, false)
                }
            } else if pt.y < -self.half_height {
                // Project on the bottom plane or the bottom circle.
                if planar_dist_from_basis_center <= self.radius {
                    let projection_on_bottom = Vec3::new(pt.x, -self.half_height, pt.z);
                    ProjectionResult::new(projection_on_bottom, false)
                } else {
                    let projection_on_bottom_circle =
                        Vec3::new(proj2d.x, -self.half_height, proj2d.y);
                    ProjectionResult::new(projection_on_bottom_circle, false)
                }
            } else {
                // Project on the side.
                let projection_on_side = Vec3::new(proj2d.x, pt.y, proj2d.y);
                ProjectionResult::new(projection_on_side, false)
            }
        }
    }

    /// Project a point of a transformed cylinder's boundary.
    pub fn project_point_on_boundary(&self, pose: Pose, pt: Vector) -> ProjectionResult {
        let local_pt = pose.inverse_transform_point(pt);
        let mut result = self.project_local_point_on_boundary(local_pt);
        result.point = pose * result.point;
        result
    }

    /// Computes the local support point of a cylinder.
    pub fn local_support_point(&self, dir: Vector) -> Vector {
        let mut vres = dir;
        vres.y = 0.0;

        let planar_dir_len = vres.length();
        let factor = if planar_dir_len != 0.0 {
            self.radius / planar_dir_len
        } else {
            0.0
        };
        vres *= factor;
        vres.y = if dir.y >= 0.0 {
            self.half_height
        } else {
            -self.half_height
        };
        vres
    }

    /// Computes the support face of a cylinder.
    pub fn support_face(&self, dir: Vector) -> PolygonalFeature {
        let mut result = PolygonalFeature::default();

        let mut dir2 = Vec2::new(dir.x, dir.z);
        let dir2_len = dir2.length();
        if dir2_len < crate::F32_EPSILON {
            dir2 = Vec2::new(1.0, 0.0);
        } else {
            dir2 /= dir2_len;
        }

        if dir.y.abs() < 0.5 {
            // We return a segment lying on the cylinder's curved part.
            result.vertices.write(
                0,
                Vec3::new(
                    dir2.x * self.radius,
                    -self.half_height,
                    dir2.y * self.radius,
                ),
            );
            result.vertices.write(
                1,
                Vec3::new(dir2.x * self.radius, self.half_height, dir2.y * self.radius),
            );
            result.num_vertices = 2;
        } else {
            // We return a square approximation of the cylinder cap.
            let y = if dir.y >= 0.0 {
                self.half_height
            } else {
                -self.half_height
            };
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
