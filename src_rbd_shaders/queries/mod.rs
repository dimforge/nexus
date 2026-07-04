//! Collision query algorithms.
//!
//! This module provides various query algorithms for collision detection:
//! - Point projection onto shapes
//! - Ray casting
//! - Contact manifold generation
//! - GJK/EPA algorithms for convex shapes
//! - SAT (Separating Axis Theorem) for specific shape pairs

pub(crate) mod contact; // This is pub(crate) for tests.
mod contact_manifold;
mod contact_pfm_pfm;
pub(crate) mod gjk; // This is pub(crate) for tests.
mod polygonal_feature;
mod projection;
mod ray;
mod sat;

pub use contact::*;
pub use contact_manifold::*;
pub use contact_pfm_pfm::*;
pub use polygonal_feature::*;
// Re-export projection items explicitly; EPSILON also exists in sat
pub use projection::{
    FEATURE_EDGE, FEATURE_FACE, FEATURE_SOLID, FEATURE_VERTEX, ProjectionResult,
    ProjectionWithLocation,
};
pub use ray::*;
// Re-export sat items explicitly; EPSILON comes from projection
use crate::Vector;
use crate::queries::projection::EPSILON;
pub use sat::{SeparatingAxis, cuboid_cuboid_find_local_separating_normal_oneway};
#[cfg(feature = "dim3")]
pub use sat::{
    cuboid_cuboid_compute_separation_wrt_local_line,
    cuboid_cuboid_find_local_separating_edge_twoway,
};

// TODO: move this elsewhere
/// Approximate equality check for vectors.
#[inline]
pub fn relative_eq(a: Vector, b: Vector) -> bool {
    let abs_diff = (a - b).abs();

    // For when the numbers are really close together
    #[cfg(feature = "dim2")]
    let close = abs_diff.x <= EPSILON.x && abs_diff.y <= EPSILON.y;
    #[cfg(feature = "dim3")]
    let close = abs_diff.x <= EPSILON.x && abs_diff.y <= EPSILON.y && abs_diff.z <= EPSILON.z;

    if close {
        return true;
    }

    let abs_a = a.abs();
    let abs_b = b.abs();
    let max_ab = abs_a.max(abs_b);

    // Use a relative difference comparison
    #[cfg(feature = "dim2")]
    let result = abs_diff.x <= max_ab.x * EPSILON.x && abs_diff.y <= max_ab.y * EPSILON.y;
    #[cfg(feature = "dim3")]
    let result = abs_diff.x <= max_ab.x * EPSILON.x
        && abs_diff.y <= max_ab.y * EPSILON.y
        && abs_diff.z <= max_ab.z * EPSILON.z;

    result
}
