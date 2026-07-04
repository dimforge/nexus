//! Tests for the PFM-PFM (Polygonal Feature Map) contact manifold generation.

use crate::queries::contact::pfm_pfm;
use crate::shapes::Shape;
use crate::{PaddedVector, Pose, Vector};
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
