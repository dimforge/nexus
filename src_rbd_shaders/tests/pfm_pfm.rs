//! Tests for the PFM-PFM (Polygonal Feature Map) contact manifold generation.

use crate::queries::contact::pfm_pfm;
use crate::shapes::Shape;
use crate::{Pose, Vector};
use khal_std::index::MaybeIndexUnchecked;

#[test]
fn test_separated_cuboids() {
    // Two cuboids separated along the X axis
    let half_extents = Vector::ONE;

    let shape1 = Shape::cuboid(half_extents);
    let shape2 = Shape::cuboid(half_extents);

    // Place shape2 at x = 4.0, so there's a gap of 2.0 between them
    let translation = Vector::X * 4.0;

    let pose12 = Pose::from_translation(translation);
    let vertices = vec![];
    let prediction = 3.0; // Large prediction to detect the contact

    #[cfg(feature = "dim2")]
    let manifold = pfm_pfm(pose12, &shape1, 0.0, &shape2, 0.0, prediction, &vertices);
    #[cfg(feature = "dim3")]
    let manifold = pfm_pfm(
        pose12,
        &shape1,
        0.0,
        &shape2,
        0.0,
        prediction,
        &vertices,
        &[],
    );

    // With large prediction, we should detect the contact
    assert!(
        manifold.len > 0,
        "Should detect contact with large prediction"
    );

    // The distance should be ~2.0
    assert!(
        (manifold.points_a.at(0).dist - 2.0).abs() < 1.0e-3,
        "Expected distance ~2.0, got {}",
        manifold.points_a.at(0).dist
    );
}

#[test]
fn test_nearly_touching_cuboids() {
    // Two cuboids with a small gap (nearly touching)
    let half_extents = Vector::ONE;

    let shape1 = Shape::cuboid(half_extents);
    let shape2 = Shape::cuboid(half_extents);

    // Place shape2 at x = 2.05, so there's a small gap
    let translation = Vector::X * 2.05;

    let pose12 = Pose::from_translation(translation);
    let vertices = vec![];
    let prediction = 0.1;

    let manifold = pfm_pfm(
        pose12,
        &shape1,
        0.0,
        &shape2,
        0.0,
        prediction,
        &vertices,
        #[cfg(feature = "dim3")]
        &[],
    );

    assert!(manifold.len > 0, "Should detect nearly touching contact");
    // Distance should be small and positive (small gap)
    assert!(
        manifold.points_a.at(0).dist > 0.0 && (manifold.points_a.at(0).dist - 0.05).abs() < 1.0e-3,
        "Expected small positive distance, got {}",
        manifold.points_a.at(0).dist
    );
}

#[test]
fn test_overlapping_cuboids() {
    // Two overlapping cuboids
    let half_extents = Vector::ONE;

    let shape1 = Shape::cuboid(half_extents);
    let shape2 = Shape::cuboid(half_extents);

    // Place shape2 at x = 1.0, so they overlap by 1.0
    let translation = Vector::X;

    let pose12 = Pose::from_translation(translation);
    let vertices = vec![];
    let prediction = 0.1;

    let manifold = pfm_pfm(
        pose12,
        &shape1,
        0.0,
        &shape2,
        0.0,
        prediction,
        &vertices,
        #[cfg(feature = "dim3")]
        &[],
    );

    assert!(manifold.len > 0, "Should detect overlapping contact");
    // Distance should be negative (penetrating)
    assert!(
        manifold.points_a.at(0).dist < 0.0,
        "Expected negative distance (penetration), got {}",
        manifold.points_a.at(0).dist
    );
}

#[test]
fn test_capsule_capsule() {
    // Two capsules
    let (a, b) = (Vector::Y * -0.5, Vector::Y * 0.5);

    let shape1 = Shape::capsule(a, b, 0.5);
    let shape2 = Shape::capsule(a, b, 0.5);

    // Place shape2 at x = 2.0, so they're touching (radius 0.5 each)
    let translation = Vector::X * 2.0;

    let pose12 = Pose::from_translation(translation);
    let vertices = vec![];
    let prediction = 1.5; // Large prediction to detect the contact

    let manifold = pfm_pfm(
        pose12,
        &shape1,
        0.0,
        &shape2,
        0.0,
        prediction,
        &vertices,
        #[cfg(feature = "dim3")]
        &[],
    );

    assert!(manifold.len > 0, "Should detect capsule-capsule contact");
    // Distance should be ~1.0 (gap between capsule surfaces)
    assert!(
        (manifold.points_a.at(0).dist - 1.0).abs() < 1.0e-3,
        "Expected distance ~1.0, got {}",
        manifold.points_a.at(0).dist
    );
}

#[test]
fn test_capsule_cuboid() {
    // Capsule and cuboid
    let (a, b) = (Vector::Y * -0.5, Vector::Y * 0.5);
    let shape1 = Shape::capsule(a, b, 0.5);

    let half_extents = Vector::ONE;
    let shape2 = Shape::cuboid(half_extents);

    // Place cuboid at x = 2.0
    let translation = Vector::X * 2.0;
    let pose12 = Pose::from_translation(translation);
    let vertices = vec![];
    let prediction = 1.0;

    let manifold = pfm_pfm(
        pose12,
        &shape1,
        0.0,
        &shape2,
        0.0,
        prediction,
        &vertices,
        #[cfg(feature = "dim3")]
        &[],
    );

    assert!(manifold.len > 0, "Should detect capsule-cuboid contact");
    // Distance should be ~0.5 (capsule radius to cuboid face)
    assert!(
        (manifold.points_a.at(0).dist - 0.5).abs() < 1.0e-3,
        "Expected distance ~0.5, got {}",
        manifold.points_a.at(0).dist
    );
}

#[cfg(feature = "dim3")]
#[test]
fn degenerate_epa_faces_do_not_create_zero_normal_contacts() {
    use crate::{PaddedVector, shapes::Triangle};
    use glamx::{Quat, Vec3};
    let half = Vec3::new(0.5, 0.7, 0.3);
    let vertices = (0..8)
        .map(|i| {
            PaddedVector::new(Vec3::new(
                if i & 1 == 0 { -half.x } else { half.x },
                if i & 2 == 0 { -half.y } else { half.y },
                if i & 4 == 0 { -half.z } else { half.z },
            ))
        })
        .collect::<Vec<_>>();
    let indices = [
        0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5, 0, 1, 5, 0, 5, 4, 2, 6, 7, 2, 7, 3, 0, 2, 3, 0, 3, 1,
        4, 5, 7, 4, 7, 6,
    ];
    let cuboid = Shape::cuboid(half);
    let polyhedron = Shape::convex_poly(0, 8, 0, indices.len() as u32);
    let triangle = Shape::from_triangle(&Triangle::new(
        Vec3::new(-0.5, -0.5, 0.0),
        Vec3::new(0.5, -0.5, 0.0),
        Vec3::new(0.0, 0.5, 0.1),
    ));
    for (a, b, rotation_scale) in [
        (cuboid, polyhedron, 3.0),
        (polyhedron, polyhedron, 2.0),
        (triangle, triangle, 0.0),
    ] {
        let pose = Pose::from_rotation(Quat::from_scaled_axis(
            Vec3::new(0.17, -0.11, 0.23) * rotation_scale,
        ));
        let manifold = pfm_pfm(pose, &a, 0.0, &b, 0.0, 0.02, &vertices, &indices);
        // EPA is bounded and may fail, but a reported manifold needs a usable
        // contact direction. These cases previously returned 1, 4 and 1 rows
        // respectively, all with a zero normal.
        assert!(
            manifold.len == 0 || (manifold.normal_a.length_squared() - 1.0).abs() < 2.0e-6,
            "invalid normal {:?} for {} contacts",
            manifold.normal_a,
            manifold.len
        );
    }
}

#[test]
#[cfg(feature = "dim3")]
fn triangle_backface_fallback_stays_on_the_finite_triangle() {
    use crate::{PaddedVector, shapes::Triangle};
    use glamx::{Quat, Vec3};
    let vertices: Vec<_> = (0..8)
        .map(|i| {
            PaddedVector::new(Vec3::new(
                if i & 1 == 0 { -0.5 } else { 0.5 },
                if i & 2 == 0 { -0.7 } else { 0.7 },
                if i & 4 == 0 { -0.3 } else { 0.3 },
            ))
        })
        .collect();
    let indices = [
        0, 0, 0, 0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5, 0, 1, 5, 0, 5, 4, 2, 6, 7, 2, 7, 3, 0, 2, 3,
        0, 3, 1, 4, 5, 7, 4, 7, 6,
    ];
    let convex = Shape::convex_poly(0, 8, 0, indices.len() as u32);
    // Exact relative pose from the mixed mesh/polyhedron deferred regression.
    let f = f32::from_bits;
    let pose = Pose::from_parts(
        Vec3::new(f(1067869798), f(1045220559), f(3184315596)),
        Quat::from_xyzw(f(1048315297), f(3189969584), f(1051360326), f(1063611899)),
    );
    let a = Vec3::new(-0.6, -0.5, 0.0);
    let b = Vec3::new(0.6, -0.5, 0.0);
    let c = Vec3::new(0.6, 0.5, 0.0);
    let d = Vec3::new(-0.6, 0.5, 0.0);
    let bad = pfm_pfm(
        pose,
        &Shape::from_triangle(&Triangle::new(a, b, c)),
        0.0,
        &convex,
        0.0,
        0.02,
        &vertices,
        &indices,
    );
    // Previously this returned (1.1546427, 0.75503933, 0) at depth -0.60767436:
    // more than 0.6 units outside the triangle, despite only a nearby corner.
    assert_eq!(
        bad.len, 0,
        "Reject the off-triangle infinite-plane fallback"
    );
    let tri = Triangle::new(a, c, d);
    let neighbor = pfm_pfm(
        pose,
        &Shape::from_triangle(&tri),
        0.0,
        &convex,
        0.0,
        0.02,
        &vertices,
        &indices,
    );
    assert!(
        neighbor.len > 0,
        "Keep the neighboring triangle's valid speculative contact"
    );
    for point in neighbor.points_a.iter().take(neighbor.len as usize) {
        let projected = tri.project_local_point_and_get_location(point.pt, true);
        assert!((projected.point - point.pt).length_squared() < 1e-10);
        assert!(point.dist >= 0.0 && point.dist < 0.02);
    }
}
