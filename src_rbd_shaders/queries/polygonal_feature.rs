//! Polygonal Feature Contact Generation
//!
//! This module implements contact point generation between polygonal features of
//! convex shapes: after SAT identifies a separating axis, it clips the support
//! features to generate a complete contact manifold.

#[cfg(feature = "dim3")]
use crate::queries::contact_manifold::MAX_MANIFOLD_POINTS;
use crate::queries::contact_manifold::{ContactManifold, ContactPoint};
use crate::{Pose, Vector};
use glamx::Vec2;
use khal_std::index::MaybeIndexUnchecked;

#[cfg(feature = "dim3")]
use crate::utils::orthonormal_basis3;
#[cfg(feature = "dim3")]
use glamx::Vec3;

// TODO: share the epsilon value across modules?
#[cfg(feature = "dim3")]
const EPSILON: f32 = 1.1920929e-7;
/// Cosine of pi/8 (approximately 22.5 degrees), used for parallelism tests.
#[cfg(feature = "dim3")]
const COS_FRAC_PI_8: f32 = 0.923_879_5;

/// Maximum vertices in a 2D polygonal feature (edge).
#[cfg(feature = "dim2")]
pub const MAX_VERTICES: usize = 2;
/// Maximum vertices in a 3D polygonal feature (quad face).
#[cfg(feature = "dim3")]
pub const MAX_VERTICES: usize = 4;

/// A polygonal feature representing the local polygonal approximation of
/// a vertex, face, or edge of a convex shape.
///
/// This can represent:
/// - A vertex (num_vertices = 1)
/// - An edge (num_vertices = 2)
/// - A face (num_vertices = 3 or 4 in 3D)
#[derive(Clone, Copy)]
#[repr(C)]
pub struct PolygonalFeature {
    /// Up to four vertices forming this polygonal feature.
    pub vertices: [Vector; MAX_VERTICES],
    /// The number of vertices in this feature.
    pub num_vertices: u32,
}

impl From<parry::shape::PolygonalFeature> for PolygonalFeature {
    fn from(value: parry::shape::PolygonalFeature) -> Self {
        Self {
            vertices: value.vertices,
            num_vertices: value.num_vertices as u32,
        }
    }
}

impl Default for PolygonalFeature {
    fn default() -> Self {
        Self {
            vertices: [Vector::default(); MAX_VERTICES],
            num_vertices: 0,
        }
    }
}

impl PolygonalFeature {
    /// Transform each vertex of this polygonal feature by the given pose.
    #[inline]
    pub fn transform_by(&self, pose: Pose) -> PolygonalFeature {
        #[cfg(feature = "dim2")]
        {
            PolygonalFeature {
                vertices: [pose * self.vertices.read(0), pose * self.vertices.read(1)],
                num_vertices: self.num_vertices,
            }
        }
        #[cfg(feature = "dim3")]
        {
            PolygonalFeature {
                vertices: [
                    pose * self.vertices.read(0),
                    pose * self.vertices.read(1),
                    pose * self.vertices.read(2),
                    pose * self.vertices.read(3),
                ],
                num_vertices: self.num_vertices,
            }
        }
    }
}

/// 2D "cross product" (perp dot product).
#[cfg(feature = "dim3")]
#[inline]
fn perp(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

/// Pseudo-inverse of a scalar (returns 0 if x is 0).
#[cfg(feature = "dim2")]
#[inline]
fn pseudo_inv(x: f32) -> f32 {
    if x == 0.0 { 0.0 } else { 1.0 / x }
}

/// Approximate equality check for scalars.
#[cfg(feature = "dim3")]
#[inline]
fn relative_eq_scalar(a: f32, b: f32) -> bool {
    let abs_diff = (a - b).abs();

    // For when the numbers are really close together
    if abs_diff <= EPSILON {
        return true;
    }

    let abs_a = a.abs();
    let abs_b = b.abs();

    // Use a relative difference comparison
    abs_diff <= abs_a.max(abs_b) * EPSILON
}

// ====================
// 2D Implementation
// ====================

#[cfg(feature = "dim2")]
mod dim2 {
    use super::*;

    #[derive(Clone, Copy, Default)]
    pub struct ClippingPoints {
        pub seg1_a: Vec2,
        pub seg2_a: Vec2,
        pub seg1_b: Vec2,
        pub seg2_b: Vec2,
        pub empty: bool,
    }

    pub fn clip_segment_segment_with_normal(
        mut seg1_a: Vec2,
        mut seg1_b: Vec2,
        mut seg2_a: Vec2,
        mut seg2_b: Vec2,
        normal: Vec2,
    ) -> ClippingPoints {
        let tangent = Vec2::new(-normal.y, normal.x);
        let mut result = ClippingPoints::default();
        let mut range1 = [seg1_a.dot(tangent), seg1_b.dot(tangent)];
        let mut range2 = [seg2_a.dot(tangent), seg2_b.dot(tangent)];

        if range1.read(1) < range1.read(0) {
            core::mem::swap(&mut seg1_a, &mut seg1_b);
            range1 = [range1.read(1), range1.read(0)];
        }

        if range2.read(1) < range2.read(0) {
            core::mem::swap(&mut seg2_a, &mut seg2_b);
            range2 = [range2.read(1), range2.read(0)];
        }

        if range2.read(0) > range1.read(1) || range1.read(0) > range2.read(1) {
            // No clip point.
            result.empty = true;
            return result;
        }

        if range2.read(0) > range1.read(0) {
            let bcoord =
                (range2.read(0) - range1.read(0)) * pseudo_inv(range1.read(1) - range1.read(0));
            result.seg1_a = seg1_a + (seg1_b - seg1_a) * bcoord;
            result.seg2_a = seg2_a;
        } else {
            let bcoord =
                (range1.read(0) - range2.read(0)) * pseudo_inv(range2.read(1) - range2.read(0));
            result.seg1_a = seg1_a;
            result.seg2_a = seg2_a + (seg2_b - seg2_a) * bcoord;
        }

        if range2.read(1) < range1.read(1) {
            let bcoord =
                (range2.read(1) - range1.read(0)) * pseudo_inv(range1.read(1) - range1.read(0));
            result.seg1_b = seg1_a + (seg1_b - seg1_a) * bcoord;
            result.seg2_b = seg2_b;
        } else {
            let bcoord =
                (range1.read(1) - range2.read(0)) * pseudo_inv(range2.read(1) - range2.read(0));
            result.seg1_b = seg1_b;
            result.seg2_b = seg2_a + (seg2_b - seg2_a) * bcoord;
        }

        result
    }

    /// Compute contacts points between a face and a vertex.
    ///
    /// This method assume we already know that at least one contact exists.
    pub fn face_vertex_contacts(
        pose12: glamx::Pose2,
        face1: &PolygonalFeature,
        sep_axis1: Vec2,
        vertex2: &PolygonalFeature,
        prediction: f32,
        flipped: bool,
    ) -> ContactManifold {
        let mut result = ContactManifold::default();
        let v2_1 = pose12.transform_point(vertex2.vertices.read(0));
        let tangent1 = face1.vertices.read(1) - face1.vertices.read(0);
        let normal1 = Vec2::new(-tangent1.y, tangent1.x);
        let denom = -normal1.dot(sep_axis1);
        let dist = (face1.vertices.read(0) - v2_1).dot(normal1) / denom;

        if dist < prediction {
            let local_p1 = v2_1 - dist * normal1;

            if !flipped {
                result.points_a.write(0, ContactPoint::new(local_p1, dist));
            } else {
                let local_p2 = pose12.inverse_transform_point(v2_1);
                result.points_a.write(0, ContactPoint::new(local_p2, dist));
            }
            result.len = 1;
        }

        result
    }

    /// Computes the contacts between two polygonal faces.
    pub fn face_face_contacts(
        pose12: glamx::Pose2,
        face1: &PolygonalFeature,
        normal1: Vec2,
        face2: &PolygonalFeature,
        prediction: f32,
        flipped: bool,
    ) -> ContactManifold {
        let mut result = ContactManifold::default();

        let clip = clip_segment_segment_with_normal(
            face1.vertices.read(0),
            face1.vertices.read(1),
            pose12.transform_point(face2.vertices.read(0)),
            pose12.transform_point(face2.vertices.read(1)),
            normal1,
        );

        if !clip.empty {
            let dist_a = (clip.seg2_a - clip.seg1_a).dot(normal1);

            if dist_a < prediction {
                if !flipped {
                    result
                        .points_a
                        .write(0, ContactPoint::new(clip.seg1_a, dist_a));
                } else {
                    let local_p2 = pose12.inverse_transform_point(clip.seg2_a);
                    result
                        .points_a
                        .write(0, ContactPoint::new(local_p2, dist_a));
                }
                result.len = 1;
            }

            let dist_b = (clip.seg2_b - clip.seg1_b).dot(normal1);
            if dist_b < prediction {
                let i = result.len as usize;
                if !flipped {
                    result
                        .points_a
                        .write(i, ContactPoint::new(clip.seg1_b, dist_b));
                } else {
                    let local_p2 = pose12.inverse_transform_point(clip.seg2_b);
                    result
                        .points_a
                        .write(i, ContactPoint::new(local_p2, dist_b));
                }
                result.len += 1;
            }
        }

        result
    }
}

// ====================
// 3D Implementation
// ====================

/// Re-export of the ≤8 → ≤4 keep-deepest-then-spread contact selector for the
/// optional contact-reduction pass (see `gpu_reduce_contacts`).
#[cfg(feature = "dim3")]
pub use dim3::manifold_reduction;

#[cfg(feature = "dim3")]
mod dim3 {
    use super::*;
    use crate::MAX_FLT;

    const MAX_CANDIDATE_POINTS: usize = 8;

    #[derive(Clone, Copy, Default)]
    pub struct ClippingPoints {
        pub seg1_a: Vec3,
        pub seg2_a: Vec3,
        pub seg1_b: Vec3,
        pub seg2_b: Vec3,
        pub empty: bool,
    }

    /// Returns the barycentric coordinates of the closest point on each segment.
    pub fn closest_points_segment_segment(
        seg1_a: Vec2,
        seg1_b: Vec2,
        seg2_a: Vec2,
        seg2_b: Vec2,
    ) -> Vec2 {
        // Inspired by real-time collision detection by Christer Ericson.
        let d1 = seg1_b - seg1_a;
        let d2 = seg2_b - seg2_a;
        let r = seg1_a - seg2_a;

        let a = d1.dot(d1);
        let e = d2.dot(d2);
        let f = d2.dot(r);

        let (s, t) = if a <= EPSILON && e <= EPSILON {
            (0.0, 0.0)
        } else if a <= EPSILON {
            (0.0, (f / e).clamp(0.0, 1.0))
        } else {
            let c = d1.dot(r);
            if e <= EPSILON {
                ((-c / a).clamp(0.0, 1.0), 0.0)
            } else {
                let b = d1.dot(d2);
                let ae = a * e;
                let bb = b * b;
                let denom = ae - bb;

                let mut s = if denom > EPSILON {
                    ((b * f - c * e) / denom).clamp(0.0, 1.0)
                } else {
                    0.0
                };

                let mut t = (b * s + f) / e;

                if t < 0.0 {
                    t = 0.0;
                    s = (-c / a).clamp(0.0, 1.0);
                } else if t > 1.0 {
                    t = 1.0;
                    s = ((b - c) / a).clamp(0.0, 1.0);
                }
                (s, t)
            }
        };

        Vec2::new(s, t)
    }

    /// Compute the barycentric coordinates of the intersection between the two given lines.
    /// Returns `vec2(MAX_FLT, MAX_FLT)` if the lines are parallel.
    pub fn closest_points_line2d(
        edge1_a: Vec2,
        edge1_b: Vec2,
        edge2_a: Vec2,
        edge2_b: Vec2,
    ) -> Vec2 {
        // Inspired by Real-time collision detection by Christer Ericson.
        let dir1 = edge1_b - edge1_a;
        let dir2 = edge2_b - edge2_a;
        let r = edge1_a - edge2_a;

        let a = dir1.dot(dir1);
        let e = dir2.dot(dir2);
        let f = dir2.dot(r);

        if a <= EPSILON && e <= EPSILON {
            Vec2::new(0.0, 0.0)
        } else if a <= EPSILON {
            Vec2::new(0.0, f / e)
        } else {
            let c = dir1.dot(r);
            if e <= EPSILON {
                Vec2::new(-c / a, 0.0)
            } else {
                let b = dir1.dot(dir2);
                let ae = a * e;
                let bb = b * b;
                let denom = ae - bb;

                // Use absolute and ulps error to test collinearity.
                let parallel = denom <= EPSILON;

                if !parallel {
                    let s = (b * f - c * e) / denom;
                    let t = (b * s + f) / e;
                    Vec2::new(s, t)
                } else {
                    Vec2::new(MAX_FLT, MAX_FLT)
                }
            }
        }
    }

    /// Projects two segments on one another and compute their intersection.
    pub fn clip_segment_segment(
        mut seg1_a: Vec3,
        mut seg1_b: Vec3,
        mut seg2_a: Vec3,
        mut seg2_b: Vec3,
    ) -> ClippingPoints {
        let mut result = ClippingPoints::default();
        let tangent1 = seg1_b - seg1_a;
        let sqnorm_tangent1 = tangent1.dot(tangent1);

        let mut range1 = [0.0, sqnorm_tangent1];
        let mut range2 = [
            (seg2_a - seg1_a).dot(tangent1),
            (seg2_b - seg1_a).dot(tangent1),
        ];

        if range1.read(1) < range1.read(0) {
            core::mem::swap(&mut seg1_a, &mut seg1_b);
            range1 = [range1.read(1), range1.read(0)];
        }

        if range2.read(1) < range2.read(0) {
            core::mem::swap(&mut seg2_a, &mut seg2_b);
            range2 = [range2.read(1), range2.read(0)];
        }

        if range2.read(0) > range1.read(1) || range1.read(0) > range2.read(1) {
            // No clip point.
            result.empty = true;
            return result;
        }

        let length1 = range1.read(1) - range1.read(0);
        let length2 = range2.read(1) - range2.read(0);

        if range2.read(0) > range1.read(0) {
            let bcoord = (range2.read(0) - range1.read(0)) / length1;
            result.seg1_a = seg1_a + tangent1 * bcoord;
            result.seg2_a = seg2_a;
        } else {
            let bcoord = (range1.read(0) - range2.read(0)) / length2;
            result.seg1_a = seg1_a;
            result.seg2_a = seg2_a + (seg2_b - seg2_a) * bcoord;
        }

        if range2.read(1) < range1.read(1) {
            let bcoord = (range2.read(1) - range1.read(0)) / length1;
            result.seg1_b = seg1_a + tangent1 * bcoord;
            result.seg2_b = seg2_b;
        } else {
            let bcoord = (range1.read(1) - range2.read(0)) / length2;
            result.seg1_b = seg1_b;
            result.seg2_b = seg2_a + (seg2_b - seg2_a) * bcoord;
        }

        result.empty = false;
        result
    }

    pub fn contacts_edge_edge(
        pose12: glamx::Pose3,
        face1: &PolygonalFeature,
        sep_axis1: Vec3,
        face2: &PolygonalFeature,
        prediction: f32,
        flipped: bool,
    ) -> ContactManifold {
        let mut result = ContactManifold::default();
        let basis = orthonormal_basis3(sep_axis1);

        let projected_edge1 = [
            Vec2::new(
                face1.vertices.read(0).dot(basis.read(0)),
                face1.vertices.read(0).dot(basis.read(1)),
            ),
            Vec2::new(
                face1.vertices.read(1).dot(basis.read(0)),
                face1.vertices.read(1).dot(basis.read(1)),
            ),
        ];

        let vertices2_1 = [
            pose12.transform_point(face2.vertices.read(0)),
            pose12.transform_point(face2.vertices.read(1)),
        ];
        let projected_edge2 = [
            Vec2::new(
                vertices2_1.read(0).dot(basis.read(0)),
                vertices2_1.read(0).dot(basis.read(1)),
            ),
            Vec2::new(
                vertices2_1.read(1).dot(basis.read(0)),
                vertices2_1.read(1).dot(basis.read(1)),
            ),
        ];

        let mut tangent1 = projected_edge1.read(1) - projected_edge1.read(0);
        let mut tangent2 = projected_edge2.read(1) - projected_edge2.read(0);
        let tangent_len1 = tangent1.length();
        let tangent_len2 = tangent2.length();

        if tangent_len1 > EPSILON && tangent_len2 > EPSILON {
            tangent1 /= tangent_len1;
            tangent2 /= tangent_len2;

            let parallel = tangent1.dot(tangent2) >= COS_FRAC_PI_8;

            if !parallel {
                let bcoords = closest_points_segment_segment(
                    projected_edge1.read(0),
                    projected_edge1.read(1),
                    projected_edge2.read(0),
                    projected_edge2.read(1),
                );

                // Found a contact between the two edges.
                let local_p1 =
                    face1.vertices.read(0) * (1.0 - bcoords.x) + face1.vertices.read(1) * bcoords.x;
                let local_p2_1 =
                    vertices2_1.read(0) * (1.0 - bcoords.y) + vertices2_1.read(1) * bcoords.y;
                let dist = (local_p2_1 - local_p1).dot(sep_axis1);

                if dist <= prediction {
                    if !flipped {
                        result.points_a.write(0, ContactPoint::new(local_p1, dist));
                    } else {
                        let local_p2 = pose12.inverse_transform_point(local_p2_1);
                        result.points_a.write(0, ContactPoint::new(local_p2, dist));
                    }
                    result.len = 1;
                }
                return result;
            }
        }

        // The lines are parallel so we are having a conformal contact.
        let clips = clip_segment_segment(
            face1.vertices.read(0),
            face1.vertices.read(1),
            vertices2_1.read(0),
            vertices2_1.read(1),
        );

        if !clips.empty {
            let dist0 = (clips.seg2_a - clips.seg1_a).dot(sep_axis1);
            let dist1 = (clips.seg2_b - clips.seg1_b).dot(sep_axis1);

            if dist0 <= prediction {
                if !flipped {
                    result
                        .points_a
                        .write(0, ContactPoint::new(clips.seg1_a, dist0));
                } else {
                    let local_p2 = pose12.inverse_transform_point(clips.seg2_a);
                    result.points_a.write(0, ContactPoint::new(local_p2, dist0));
                }
                result.len = 1;
            }

            let k = result.len as usize;

            if dist1 <= prediction {
                if !flipped {
                    result
                        .points_a
                        .write(k, ContactPoint::new(clips.seg1_b, dist1));
                } else {
                    let local_p2 = pose12.inverse_transform_point(clips.seg2_b);
                    result.points_a.write(k, ContactPoint::new(local_p2, dist1));
                }
                result.len += 1;
            }
        }

        result
    }

    pub fn manifold_reduction(
        candidates: &[ContactPoint; MAX_CANDIDATE_POINTS],
        num_candidates: u32,
        normal: Vector,
    ) -> ContactManifold {
        let mut result = ContactManifold::default();
        let num = num_candidates as usize;

        if num <= MAX_MANIFOLD_POINTS {
            result.points_a.write(0, candidates.read(0));
            result.points_a.write(1, candidates.read(1));
            result.points_a.write(2, candidates.read(2));
            result.points_a.write(3, candidates.read(3));
            result.len = num_candidates;
            return result;
        }

        // Run contact reduction so we only have up to four solver contacts.
        // 1. Find the deepest contact.
        let mut deepest_dist = candidates.at(0).dist;
        let mut selected = [
            0usize,
            MAX_CANDIDATE_POINTS,
            MAX_CANDIDATE_POINTS,
            MAX_CANDIDATE_POINTS,
        ];

        for i in 1..num {
            if candidates.at(i).dist < deepest_dist {
                deepest_dist = candidates.at(i).dist;
                selected.write(0, i);
            }
        }

        // 2. Find the point that is the furthest from the deepest one.
        let selected_a = candidates.at(selected.read(0)).pt;
        let mut furthest_dist = -MAX_FLT;

        for i in 0..num {
            let pt_sel = selected_a - candidates.at(i).pt;
            let dist = pt_sel.dot(pt_sel);
            if i != selected.read(0) && dist > furthest_dist {
                furthest_dist = dist;
                selected.write(1, i);
            }
        }

        // 3. Now find the two points furthest from the segment we built so far.
        let selected_b = candidates.at(selected.read(1)).pt;
        let selected_ab = selected_b - selected_a;
        let tangent = selected_ab.cross(normal);

        let mut min_dot = MAX_FLT;
        let mut max_dot = -MAX_FLT;

        for i in 0..num {
            if i == selected.read(0) || i == selected.read(1) {
                continue;
            }

            let d = (candidates.at(i).pt - selected_a).dot(tangent);
            if d < min_dot {
                min_dot = d;
                selected.write(2, i);
            }

            if d > max_dot {
                max_dot = d;
                selected.write(3, i);
            }
        }

        if selected.read(2) == MAX_CANDIDATE_POINTS {
            selected.write(2, selected.read(3));
            selected.write(3, MAX_CANDIDATE_POINTS);
        }

        result.points_a.write(0, candidates.read(selected.read(0)));
        result.points_a.write(1, candidates.read(selected.read(1)));
        result.len = 2;

        if selected.read(2) != MAX_CANDIDATE_POINTS {
            result.points_a.write(2, candidates.read(selected.read(2)));
            result.len = 3;

            if selected.read(3) != MAX_CANDIDATE_POINTS {
                result.points_a.write(3, candidates.read(selected.read(3)));
                result.len = 4;
            }
        }

        result
    }

    pub fn contacts_face_face(
        pose12: glamx::Pose3,
        face1: &PolygonalFeature,
        sep_axis1: Vec3,
        face2: &PolygonalFeature,
        prediction: f32,
        flipped: bool,
    ) -> ContactManifold {
        let mut candidates = [ContactPoint::default(); MAX_CANDIDATE_POINTS];
        let mut num_candidates = 0u32;

        let basis = orthonormal_basis3(sep_axis1);
        let projected_face1 = [
            Vec2::new(
                face1.vertices.read(0).dot(basis.read(0)),
                face1.vertices.read(0).dot(basis.read(1)),
            ),
            Vec2::new(
                face1.vertices.read(1).dot(basis.read(0)),
                face1.vertices.read(1).dot(basis.read(1)),
            ),
            Vec2::new(
                face1.vertices.read(2).dot(basis.read(0)),
                face1.vertices.read(2).dot(basis.read(1)),
            ),
            Vec2::new(
                face1.vertices.read(3).dot(basis.read(0)),
                face1.vertices.read(3).dot(basis.read(1)),
            ),
        ];

        let vertices2_1 = [
            pose12.transform_point(face2.vertices.read(0)),
            pose12.transform_point(face2.vertices.read(1)),
            pose12.transform_point(face2.vertices.read(2)),
            pose12.transform_point(face2.vertices.read(3)),
        ];
        let projected_face2 = [
            Vec2::new(
                vertices2_1.read(0).dot(basis.read(0)),
                vertices2_1.read(0).dot(basis.read(1)),
            ),
            Vec2::new(
                vertices2_1.read(1).dot(basis.read(0)),
                vertices2_1.read(1).dot(basis.read(1)),
            ),
            Vec2::new(
                vertices2_1.read(2).dot(basis.read(0)),
                vertices2_1.read(2).dot(basis.read(1)),
            ),
            Vec2::new(
                vertices2_1.read(3).dot(basis.read(0)),
                vertices2_1.read(3).dot(basis.read(1)),
            ),
        ];

        // Check vertices of face1 inside face2
        if face2.num_vertices > 2 {
            let normal2_1 = (vertices2_1.read(2) - vertices2_1.read(1))
                .cross(vertices2_1.read(0) - vertices2_1.read(1));
            let denom = normal2_1.dot(sep_axis1);

            if !relative_eq_scalar(denom, 0.0) {
                let last_index2 = face2.num_vertices as usize - 1;
                let mut any_point_is_outside = false;

                for i in 0..face1.num_vertices as usize {
                    let p1 = projected_face1.read(i);

                    let mut sign = perp(
                        projected_face2.read(0) - projected_face2.read(last_index2),
                        p1 - projected_face2.read(last_index2),
                    );

                    let mut point_is_outside = false;
                    for j in 0..last_index2 {
                        let new_sign = perp(
                            projected_face2.read(j + 1) - projected_face2.read(j),
                            p1 - projected_face2.read(j),
                        );

                        if sign == 0.0 {
                            sign = new_sign;
                        } else if sign * new_sign < 0.0 {
                            point_is_outside = true;
                            break;
                        }
                    }

                    any_point_is_outside = any_point_is_outside || point_is_outside;
                    let dist =
                        (vertices2_1.read(0) - face1.vertices.read(i)).dot(normal2_1) / denom;

                    if !point_is_outside && dist <= prediction {
                        let local_p1 = face1.vertices.read(i);
                        let local_p2_1 = face1.vertices.read(i) + dist * sep_axis1;

                        if !flipped {
                            candidates.at_mut(num_candidates as usize).pt = local_p1;
                            candidates.at_mut(num_candidates as usize).dist = dist;
                        } else {
                            let local_p2 = pose12.inverse_transform_point(local_p2_1);
                            candidates.at_mut(num_candidates as usize).pt = local_p2;
                            candidates.at_mut(num_candidates as usize).dist = dist;
                        }
                        num_candidates += 1;
                    }
                }

                if !any_point_is_outside {
                    return manifold_reduction(&candidates, num_candidates, sep_axis1);
                }
            }
        }

        // Check vertices of face2 inside face1
        if face1.num_vertices > 2 {
            let normal1 = (face1.vertices.read(2) - face1.vertices.read(1))
                .cross(face1.vertices.read(0) - face1.vertices.read(1));

            let denom = -normal1.dot(sep_axis1);
            if !relative_eq_scalar(denom, 0.0) {
                let last_index1 = face1.num_vertices as usize - 1;
                let mut any_point_is_outside = false;

                for i in 0..face2.num_vertices as usize {
                    let p2 = projected_face2.read(i);

                    let mut sign = perp(
                        projected_face1.read(0) - projected_face1.read(last_index1),
                        p2 - projected_face1.read(last_index1),
                    );

                    let mut point_is_outside = false;
                    for j in 0..last_index1 {
                        let new_sign = perp(
                            projected_face1.read(j + 1) - projected_face1.read(j),
                            p2 - projected_face1.read(j),
                        );

                        if sign == 0.0 {
                            sign = new_sign;
                        } else if sign * new_sign < 0.0 {
                            point_is_outside = true;
                            break;
                        }
                    }

                    any_point_is_outside = any_point_is_outside || point_is_outside;
                    let dist = (face1.vertices.read(0) - vertices2_1.read(i)).dot(normal1) / denom;

                    if !point_is_outside && dist <= prediction {
                        let local_p2_1 = vertices2_1.read(i);
                        let local_p1 = vertices2_1.read(i) - dist * sep_axis1;

                        if !flipped {
                            candidates.at_mut(num_candidates as usize).pt = local_p1;
                            candidates.at_mut(num_candidates as usize).dist = dist;
                        } else {
                            let local_p2 = pose12.inverse_transform_point(local_p2_1);
                            candidates.at_mut(num_candidates as usize).pt = local_p2;
                            candidates.at_mut(num_candidates as usize).dist = dist;
                        }
                        num_candidates += 1;
                    }
                }

                if !any_point_is_outside {
                    return manifold_reduction(&candidates, num_candidates, sep_axis1);
                }
            }
        }

        // Check edge-edge intersections
        for j in 0..face2.num_vertices as usize {
            for i in 0..face1.num_vertices as usize {
                let bcoords = closest_points_line2d(
                    projected_face1.read(i),
                    projected_face1.read((i + 1) % face1.num_vertices as usize),
                    projected_face2.read(j),
                    projected_face2.read((j + 1) % face2.num_vertices as usize),
                );
                if bcoords.x > 0.0 && bcoords.x < 1.0 && bcoords.y > 0.0 && bcoords.y < 1.0 {
                    let edge1_a = face1.vertices.read(i);
                    let edge1_b = face1.vertices.read((i + 1) % face1.num_vertices as usize);
                    let edge2_a = vertices2_1.read(j);
                    let edge2_b = vertices2_1.read((j + 1) % face2.num_vertices as usize);
                    let local_p1 = edge1_a * (1.0 - bcoords.x) + edge1_b * bcoords.x;
                    let local_p2_1 = edge2_a * (1.0 - bcoords.y) + edge2_b * bcoords.y;
                    let dist = (local_p2_1 - local_p1).dot(sep_axis1);

                    if dist <= prediction {
                        if !flipped {
                            candidates.at_mut(num_candidates as usize).pt = local_p1;
                            candidates.at_mut(num_candidates as usize).dist = dist;
                        } else {
                            let local_p2 = pose12.inverse_transform_point(local_p2_1);
                            candidates.at_mut(num_candidates as usize).pt = local_p2;
                            candidates.at_mut(num_candidates as usize).dist = dist;
                        }
                        num_candidates += 1;
                    }

                    if num_candidates as usize == MAX_CANDIDATE_POINTS {
                        return manifold_reduction(&candidates, num_candidates, sep_axis1);
                    }
                }
            }
        }

        manifold_reduction(&candidates, num_candidates, sep_axis1)
    }
}

/// Computes the contacts between two polygonal features (2D version).
#[cfg(feature = "dim2")]
pub fn contacts(
    pose12: Pose,
    pose21: Pose,
    sep_axis1: Vector,
    sep_axis2: Vector,
    feature1: &PolygonalFeature,
    feature2: &PolygonalFeature,
    prediction: f32,
    flipped: bool,
) -> ContactManifold {
    if feature1.num_vertices == 2 {
        if feature2.num_vertices == 2 {
            dim2::face_face_contacts(pose12, feature1, sep_axis1, feature2, prediction, flipped)
        } else {
            dim2::face_vertex_contacts(pose12, feature1, sep_axis1, feature2, prediction, flipped)
        }
    } else {
        dim2::face_vertex_contacts(pose21, feature2, sep_axis2, feature1, prediction, !flipped)
    }
}

/// Computes all the contacts between two polygonal features (3D version).
#[cfg(feature = "dim3")]
pub fn contacts(
    pose12: Pose,
    _pose21: Pose, // Unused argument, to match the 2D definition.
    sep_axis1: Vector,
    _sep_axis2: Vector,
    feature1: &PolygonalFeature,
    feature2: &PolygonalFeature,
    prediction: f32,
    flipped: bool,
) -> ContactManifold {
    if feature1.num_vertices == 2 && feature2.num_vertices == 2 {
        dim3::contacts_edge_edge(pose12, feature1, sep_axis1, feature2, prediction, flipped)
    } else {
        dim3::contacts_face_face(pose12, feature1, sep_axis1, feature2, prediction, flipped)
    }
}
