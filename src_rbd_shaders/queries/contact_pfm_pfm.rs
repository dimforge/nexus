//! Contact computation for polygonal feature-based shapes using GJK/EPA.
//!
//! This module provides contact manifold computation between convex shapes
//! that use the support function interface (support maps).

use crate::queries::contact_manifold::{ContactManifold, ContactPoint, MAX_MANIFOLD_POINTS};
use crate::queries::gjk::{
    self, CLOSEST_POINTS, Epa, FLT_EPS, GjkResult, INTERSECTION, VoronoiSimplex,
    cso_point_from_shapes,
};
use crate::queries::polygonal_feature;
use crate::shapes::Shape;
#[cfg(feature = "dim3")]
use crate::shapes::{SHAPE_TYPE_TRIANGLE, Triangle};
use crate::{DIM, PaddedVector, Pose, Vector};
use khal_std::index::MaybeIndexUnchecked;

/// Computes contact between two support map shapes using GJK.
pub fn contact_support_map_support_map(
    pose12: Pose,
    g1: &Shape,
    g2: &Shape,
    prediction: f32,
    vertices: &[PaddedVector],
) -> GjkResult {
    let mut dir = pose12.translation;

    let dir_len_sq = dir.dot(dir);
    if dir_len_sq > FLT_EPS * FLT_EPS {
        dir /= crate::sqrt(dir_len_sq);
    } else {
        dir = Vector::X;
    }

    let cso_point = cso_point_from_shapes(pose12, g1, g2, dir, vertices);
    let mut simplex = VoronoiSimplex::init(cso_point);

    let cpts = gjk::closest_points(pose12, g1, g2, prediction, true, &mut simplex, vertices);
    if cpts.status != INTERSECTION {
        return cpts;
    }

    // The point is inside the CSO: use the fallback algorithm
    let mut epa = Epa::default();
    let penetration = epa.closest_points(pose12, g1, g2, &simplex, vertices);
    if penetration.valid {
        return gjk::gjk_result_closest_points(
            penetration.pt_a,
            penetration.pt_b,
            penetration.normal,
        );
    }

    // Everything failed
    gjk::gjk_result_no_intersection(Vector::X)
}

/// Unit normal of `tri`'s plane oriented towards `point`, or zero for degenerate triangles or
/// when `point` (nearly) lies on the plane. Independent of the triangle's winding.
#[cfg(feature = "dim3")]
#[inline]
fn triangle_face_normal_towards(tri: &Triangle, point: Vector) -> Vector {
    let n = (tri.b - tri.a).cross(tri.c - tri.a);
    let len = n.length();
    if len <= FLT_EPS {
        return Vector::ZERO;
    }
    let n = n / len;
    let side = (point - tri.a).dot(n);
    if side > FLT_EPS {
        n
    } else if side < -FLT_EPS {
        -n
    } else {
        Vector::ZERO
    }
}

/// Merges near-coincident points of `manifold` (within a quarter of `prediction`, keeping the
/// deeper one). The clipped features and the appended GJK point often coincide, and duplicated
/// points skew the solver and the per-point warmstart.
#[inline]
fn dedup_manifold_points(manifold: &mut ContactManifold, prediction: f32) {
    let eps = (0.25 * prediction).max(1.0e-5);
    let eps_sq = eps * eps;
    let len = manifold.len;
    let mut out = 0u32;
    for i in 0..MAX_MANIFOLD_POINTS as u32 {
        if i < len {
            let p = *manifold.points_a.at(i as usize);
            let mut merged = false;
            for k in 0..MAX_MANIFOLD_POINTS as u32 {
                if k < out && !merged {
                    let q = *manifold.points_a.at(k as usize);
                    let d = q.pt - p.pt;
                    if d.dot(d) < eps_sq {
                        if p.dist < q.dist {
                            manifold.points_a.write(k as usize, p);
                        }
                        merged = true;
                    }
                }
            }
            if !merged {
                manifold.points_a.write(out as usize, p);
                out += 1;
            }
        }
    }
    manifold.len = out;
}

/// Computes the contact manifold between two polygonal feature-based shapes.
pub fn contact_manifold_pfm_pfm(
    pose12: Pose,
    pfm1: &Shape,
    border_radius1: f32,
    pfm2: &Shape,
    border_radius2: f32,
    prediction: f32,
    vertices: &[PaddedVector],
    #[cfg(feature = "dim3")] indices: &[u32],
) -> ContactManifold {
    let total_prediction = prediction + border_radius1 + border_radius2;
    let contact = contact_support_map_support_map(pose12, pfm1, pfm2, total_prediction, vertices);

    match contact.status {
        CLOSEST_POINTS => {
            #[allow(unused_mut)]
            let mut p1 = contact.a;
            #[allow(unused_mut)]
            let mut p2_1 = contact.b;
            #[allow(unused_mut)]
            let mut local_n1 = contact.dir;
            #[allow(unused_mut)]
            let mut fallback = false;

            // A triangle has no thickness, so EPA can pick its back face when the other shape
            // barely touches it. Re-derive the contact along the face normal on the side where
            // the other shape sits.
            #[cfg(feature = "dim3")]
            if pfm1.shape_type() == SHAPE_TYPE_TRIANGLE {
                let tri = pfm1.to_triangle();
                let face_n = triangle_face_normal_towards(&tri, pose12.translation);
                // `face_n` is zero when undecidable, which fails this test.
                if local_n1.dot(face_n) < 0.0 {
                    let support_dir2 = pose12.inverse_transform_vector(-face_n);
                    p2_1 = pose12.transform_point(pfm2.local_support_point(support_dir2, vertices));
                    let dist = (p2_1 - tri.a).dot(face_n);
                    if dist > total_prediction {
                        return ContactManifold::default();
                    }
                    p1 = p2_1 - face_n * dist;
                    local_n1 = face_n;
                    fallback = true;
                }
            }

            let local_n2 = pose12.inverse_transform_vector(-local_n1);

            #[cfg(feature = "dim2")]
            let feature1 = pfm1.support_face(local_n1, vertices);
            #[cfg(feature = "dim2")]
            let feature2 = pfm2.support_face(local_n2, vertices);
            #[cfg(feature = "dim3")]
            let feature1 = pfm1.support_face(local_n1, vertices, indices);
            #[cfg(feature = "dim3")]
            let feature2 = pfm2.support_face(local_n2, vertices, indices);
            let mut manifold = polygonal_feature::contacts(
                pose12,
                pose12.inverse(),
                local_n1,
                local_n2,
                &feature1,
                &feature2,
                total_prediction,
                false,
            );

            // The fallback's projected deepest point may lie outside the triangle, so it is
            // only used when feature clipping produced nothing.
            if manifold.len < MAX_MANIFOLD_POINTS as u32
                && (DIM == 3 || (DIM == 2 && manifold.len == 0))
                && !(fallback && manifold.len > 0)
            {
                let dist = (p2_1 - p1).dot(local_n1);
                manifold
                    .points_a
                    .write(manifold.len as usize, ContactPoint::new(p1, dist));
                manifold.len += 1;
            }

            // Adjust points to take the radius into account.
            if border_radius1 != 0.0 || border_radius2 != 0.0 {
                for i in 0..manifold.len as usize {
                    manifold.points_a.at_mut(i).pt += local_n1 * border_radius1;
                    manifold.points_a.at_mut(i).dist -= border_radius1 + border_radius2;
                }
            }

            dedup_manifold_points(&mut manifold, prediction);
            manifold.normal_a = local_n1;
            manifold
        }
        _ => {
            // No collisions.
            ContactManifold::default()
        }
    }
}
