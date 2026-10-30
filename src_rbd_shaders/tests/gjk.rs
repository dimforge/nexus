//! Tests for the GJK (Gilbert-Johnson-Keerthi) algorithm.

use crate::queries::gjk::{
    CLOSEST_POINTS, INTERSECTION, VoronoiSimplex, closest_points, cso_point_from_shapes,
};
use crate::shapes::Shape;
use crate::{Pose, Vector};

#[test]
fn test_separated_cuboids() {
    // Two cuboids separated along the X axis
    let half_extents = Vector::ONE;

    let shape1 = Shape::cuboid(half_extents);
    let shape2 = Shape::cuboid(half_extents);

    // Place shape2 at x = 4.0, so there's a gap of 2.0 between them
    #[cfg(feature = "dim2")]
    let translation = glamx::Vec2::new(4.0, 0.0);
    #[cfg(feature = "dim3")]
    let translation = glamx::Vec3::new(4.0, 0.0, 0.0);

    let pose12 = Pose::from_translation(translation);
    let vertices = vec![];

    // Initialize simplex with a point from the CSO
    let init_dir = Vector::X;

    let cso_point = cso_point_from_shapes(pose12, &shape1, &shape2, init_dir, &vertices);
    let mut simplex = VoronoiSimplex::init(cso_point);

    let result = closest_points(
        pose12,
        &shape1,
        &shape2,
        10.0,
        true,
        &mut simplex,
        &vertices,
    );

    assert_eq!(result.status, CLOSEST_POINTS);
    // Distance should be approximately 2.0 (gap between the two cuboids)
    let dist = (result.b - result.a).length();
    assert!(
        (dist - 2.0).abs() < 1.0e-3,
        "Expected distance ~2.0, got {}",
        dist
    );
}

#[test]
fn test_nearly_touching_cuboids() {
    // Two cuboids with a very small gap
    let half_extents = Vector::ONE;

    let shape1 = Shape::cuboid(half_extents);
    let shape2 = Shape::cuboid(half_extents);

    // Place shape2 at x = 2.01, so there's a tiny gap
    #[cfg(feature = "dim2")]
    let translation = glamx::Vec2::new(2.01, 0.0);
    #[cfg(feature = "dim3")]
    let translation = glamx::Vec3::new(2.01, 0.0, 0.0);

    let pose12 = Pose::from_translation(translation);
    let vertices = vec![];

    #[cfg(feature = "dim2")]
    let init_dir = glamx::Vec2::new(1.0, 0.0);
    #[cfg(feature = "dim3")]
    let init_dir = glamx::Vec3::new(1.0, 0.0, 0.0);

    let cso_point = cso_point_from_shapes(pose12, &shape1, &shape2, init_dir, &vertices);
    let mut simplex = VoronoiSimplex::init(cso_point);

    let result = closest_points(
        pose12,
        &shape1,
        &shape2,
        10.0,
        true,
        &mut simplex,
        &vertices,
    );

    // Should report closest points with small distance
    assert_eq!(result.status, CLOSEST_POINTS);
    let dist = (result.b - result.a).length();
    assert!(dist < 0.1, "Expected distance ~1.0e-3, got {}", dist);
}

#[test]
fn test_intersecting_cuboids() {
    // Two overlapping cuboids
    let half_extents = Vector::ONE;

    let shape1 = Shape::cuboid(half_extents);
    let shape2 = Shape::cuboid(half_extents);

    // Place shape2 at x = 1.0, so they overlap by 1.0
    let translation = Vector::X;

    let pose12 = Pose::from_translation(translation);
    let vertices = vec![];
    let init_dir = Vector::ONE;

    let cso_point = cso_point_from_shapes(pose12, &shape1, &shape2, init_dir, &vertices);
    let mut simplex = VoronoiSimplex::init(cso_point);

    let result = closest_points(
        pose12,
        &shape1,
        &shape2,
        10.0,
        true,
        &mut simplex,
        &vertices,
    );

    // Should report intersection
    assert_eq!(result.status, INTERSECTION);
}

#[test]
fn test_separated_capsules() {
    // Two capsules separated along the X axis
    let (a1, b1) = (Vector::Y * -0.5, Vector::Y * 0.5);

    let shape1 = Shape::capsule(a1, b1, 0.5);
    let shape2 = Shape::capsule(a1, b1, 0.5);

    // Place shape2 at x = 3.0, so there's a gap of 2.0 between them (radius 0.5 each)
    let translation = Vector::X * 3.0;

    let pose12 = Pose::from_translation(translation);
    let vertices = vec![];
    let init_dir = Vector::X;

    let cso_point = cso_point_from_shapes(pose12, &shape1, &shape2, init_dir, &vertices);
    let mut simplex = VoronoiSimplex::init(cso_point);

    let result = closest_points(
        pose12,
        &shape1,
        &shape2,
        10.0,
        true,
        &mut simplex,
        &vertices,
    );

    assert_eq!(result.status, CLOSEST_POINTS);
    // Distance should be approximately 2.0
    let dist = (result.b - result.a).length();
    assert!(
        (dist - 2.0).abs() < 1.0e-3,
        "Expected distance ~2.0, got {}",
        dist
    );
}

#[cfg(feature = "dim3")]
#[test]
fn epa_coincident_cuboids_preserve_depth_after_support_roundoff() {
    use crate::queries::gjk::Epa;
    use glamx::{Quat, Vec3};
    let half = Vec3::new(0.5, 0.7, 0.3);
    let shape = Shape::cuboid(half);
    let poly = Shape::convex_poly(0, 8, 0, 0);
    let vertices = (0..8)
        .map(|i| {
            crate::PaddedVector::new(Vec3::new(
                if i & 1 == 0 { -half.x } else { half.x },
                if i & 2 == 0 { -half.y } else { half.y },
                if i & 4 == 0 { -half.z } else { half.z },
            ))
        })
        .collect::<Vec<_>>();
    let pose = Pose::from_rotation(Quat::from_scaled_axis(Vec3::new(0.68, -0.44, 0.92)));
    let direction = Vec3::new(0.2, -0.3, -0.7);
    let mut simplex = VoronoiSimplex::init(cso_point_from_shapes(
        pose, &poly, &shape, direction, &vertices,
    ));
    assert_eq!(
        closest_points(pose, &poly, &shape, 10.0, true, &mut simplex, &vertices).status,
        INTERSECTION
    );
    assert_eq!(simplex.dim, 1);
    // Independent cuboid SAT gives the penetration depth. All centers coincide,
    // so opposite normal signs are equally valid.
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    let rotated = axes.map(|axis| pose.transform_vector(axis));
    let mut depth = f32::MAX;
    for axis in axes
        .into_iter()
        .chain(rotated)
        .chain(axes.into_iter().flat_map(|a| rotated.map(|b| a.cross(b))))
    {
        if axis.length_squared() > 1.0e-12 {
            let normal = axis.normalize();
            let radius =
                half.dot(normal.abs()) + half.dot(pose.inverse_transform_vector(normal).abs());
            depth = depth.min(radius);
        }
    }
    for offset in [-2i64, -1, 0, 1, 2] {
        let mut perturbed = simplex;
        let b = &mut perturbed.vertices[1].orig_b;
        b.x = f32::from_bits((i64::from(b.x.to_bits()) + offset) as u32);
        perturbed.vertices[1].point = perturbed.vertices[1].orig_a - perturbed.vertices[1].orig_b;
        let result = Epa::default().closest_points(pose, &poly, &shape, &perturbed, &vertices);
        assert!(result.valid, "EPA failed after {offset}-ulp perturbation");
        let actual = (result.pt_a - result.pt_b).dot(result.normal);
        assert!(
            (actual - depth).abs() < 2.0e-5,
            "EPA depth {actual}, SAT depth {depth}, perturbation {offset}"
        );
        assert!((result.normal.length() - 1.0).abs() < 1.0e-6);
    }
}
