//! Tests for the EPA (Expanding Polytope Algorithm).

use crate::queries::gjk::{
    Epa, INTERSECTION, VoronoiSimplex, closest_points, cso_point_from_shapes,
};
use crate::shapes::Shape;
use crate::{PaddedVector, Pose, Vector};

#[test]
fn test_penetration_depth() {
    // Two overlapping cuboids
    #[cfg(feature = "dim2")]
    let half_extents = glamx::Vec2::new(1.0, 1.0);
    #[cfg(feature = "dim3")]
    let half_extents = glamx::Vec3::new(1.0, 1.0, 1.0);

    let shape1 = Shape::cuboid(half_extents);
    let shape2 = Shape::cuboid(half_extents);

    // Place shape2 at x = 1.0, so they overlap by 1.0
    #[cfg(feature = "dim2")]
    let translation = glamx::Vec2::new(1.0, 0.0);
    #[cfg(feature = "dim3")]
    let translation = glamx::Vec3::new(1.0, 0.0, 0.0);

    let pose12 = Pose::from_translation(translation);
    let vertices = vec![];

    let init_dir = Vector::X;

    let cso_point = cso_point_from_shapes(pose12, &shape1, &shape2, init_dir, &vertices);
    let mut simplex = VoronoiSimplex::init(cso_point);

    // First run GJK to confirm intersection and build simplex
    let gjk_result = closest_points(
        pose12,
        &shape1,
        &shape2,
        10.0,
        true,
        &mut simplex,
        &vertices,
    );
    assert_eq!(gjk_result.status, INTERSECTION);

    // Now run EPA to get penetration depth
    let mut epa = Epa::default();
    let epa_result = epa.closest_points(pose12, &shape1, &shape2, &simplex, &vertices);

    assert!(epa_result.valid, "EPA should return valid result");

    // The penetration depth should be ~1.0 (the overlap amount)
    let penetration = (epa_result.pt_b - epa_result.pt_a).dot(epa_result.normal);
    assert!(
        penetration < 0.0,
        "Penetration should be negative (overlapping)"
    );
    assert!(
        (penetration.abs() - 1.0).abs() < 1.0e-3,
        "Expected penetration ~1.0, got {}",
        penetration.abs()
    );
}

#[test]
fn test_deep_penetration() {
    // Two cuboids with significant overlap
    let half_extents = Vector::splat(1.0);

    let shape1 = Shape::cuboid(half_extents);
    let shape2 = Shape::cuboid(half_extents);

    // Place shape2 at x = 0.5, so they overlap by 1.5
    let translation = Vector::X * 0.5;

    let pose12 = Pose::from_translation(translation);
    let vertices = vec![];

    let init_dir = Vector::X;

    let cso_point = cso_point_from_shapes(pose12, &shape1, &shape2, init_dir, &vertices);
    let mut simplex = VoronoiSimplex::init(cso_point);

    let gjk_result = closest_points(
        pose12,
        &shape1,
        &shape2,
        10.0,
        true,
        &mut simplex,
        &vertices,
    );
    assert_eq!(gjk_result.status, INTERSECTION);

    let mut epa = Epa::default();
    let epa_result = epa.closest_points(pose12, &shape1, &shape2, &simplex, &vertices);

    assert!(epa_result.valid, "EPA should return valid result");
}
