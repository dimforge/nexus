//! Tests for the GJK (Gilbert-Johnson-Keerthi) algorithm.

use crate::queries::gjk::{
    CLOSEST_POINTS, INTERSECTION, VoronoiSimplex, closest_points, cso_point_from_shapes,
};
use crate::shapes::Shape;
use crate::{PaddedVector, Pose, Vector};

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
