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
            let p1 = contact.a;
            let p2_1 = contact.b;
            let local_n1 = contact.dir;
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

            if manifold.len < MAX_MANIFOLD_POINTS as u32
                && (DIM == 3 || (DIM == 2 && manifold.len == 0))
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

            manifold.normal_a = local_n1;
            manifold
        }
        _ => {
            // No collisions.
            ContactManifold::default()
        }
    }
}
