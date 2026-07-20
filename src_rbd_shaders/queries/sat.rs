//! Separating Axis Theorem (SAT) for cuboid-cuboid collision detection.
//!
//! Tests potential separating axes to determine whether two cuboids overlap, and
//! returns the axis of maximum separation (or, when penetrating, the contact normal).

use crate::shapes::Cuboid;
use crate::{MAX_FLT, Pose, Vector};
use parry::shape::SupportMap;

#[cfg(feature = "dim3")]
use glamx::Vec3;
#[cfg(feature = "dim3")]
use khal_std::index::MaybeIndexUnchecked;

/// Result of a separating axis test.
#[derive(Clone, Copy, Default)]
pub struct SeparatingAxis {
    /// The separating axis direction (normalized).
    pub axis: Vector,
    /// Separation distance along the axis (negative = penetrating).
    pub separation: f32,
    #[cfg(feature = "dim2")]
    pub padding: u32,
}

impl SeparatingAxis {
    pub fn new(separation: f32, axis: Vector) -> Self {
        Self {
            separation,
            axis,
            #[cfg(feature = "dim2")]
            padding: 0,
        }
    }
}

/// Machine epsilon for floating-point comparisons.
#[cfg(feature = "dim3")]
pub const EPSILON: f32 = 1.1920929E-7;

#[cfg(feature = "dim3")]
/// Computes the separation of two cuboids along `axis1`.
#[inline(always)]
pub fn cuboid_cuboid_compute_separation_wrt_local_line(
    cuboid1: &Cuboid,
    cuboid2: &Cuboid,
    pos12: Pose,
    line_axis1: Vector,
) -> SeparatingAxis {
    let signum = if pos12.translation.dot(line_axis1) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let axis1 = line_axis1 * signum;
    let axis2 = pos12.rotation.inverse() * (-axis1);
    let local_pt1 = cuboid1.local_support_point(axis1);
    let local_pt2 = cuboid2.local_support_point(axis2);
    let pt2 = pos12 * local_pt2;
    let separation = (pt2 - local_pt1).dot(axis1);
    SeparatingAxis::new(separation, axis1)
}

#[cfg(feature = "dim3")]
/// Finds the best separating edge between two cuboids.
///
/// All combinations of edges from both cuboids are taken into
/// account.
#[inline(always)]
pub fn cuboid_cuboid_find_local_separating_edge_twoway(
    cuboid1: &Cuboid,
    cuboid2: &Cuboid,
    pos12: Pose,
) -> SeparatingAxis {
    let mut best_sep = SeparatingAxis::new(-MAX_FLT, Vec3::ZERO);

    let x2 = pos12.rotation * Vec3::new(1.0, 0.0, 0.0);
    let y2 = pos12.rotation * Vec3::new(0.0, 1.0, 0.0);
    let z2 = pos12.rotation * Vec3::new(0.0, 0.0, 1.0);

    // We have 3 * 3 = 9 axes to test.
    let axes = [
        // Vector::{x, y ,z}().cross(x2)
        Vec3::new(0.0, -x2.z, x2.y),
        Vec3::new(x2.z, 0.0, -x2.x),
        Vec3::new(-x2.y, x2.x, 0.0),
        // Vector::{x, y ,z}().cross(y2)
        Vec3::new(0.0, -y2.z, y2.y),
        Vec3::new(y2.z, 0.0, -y2.x),
        Vec3::new(-y2.y, y2.x, 0.0),
        // Vector::{x, y ,z}().cross(z2)
        Vec3::new(0.0, -z2.z, z2.y),
        Vec3::new(z2.z, 0.0, -z2.x),
        Vec3::new(-z2.y, z2.x, 0.0),
    ];

    // TODO: unroll loop
    for i in 0..9 {
        let axis1 = axes.read(i);
        let norm1 = axis1.length();
        if norm1 > EPSILON {
            let sep = cuboid_cuboid_compute_separation_wrt_local_line(
                cuboid1,
                cuboid2,
                pos12,
                axis1 / norm1,
            );

            if sep.separation > best_sep.separation {
                best_sep = sep;
            }
        }
    }

    best_sep
}

/// Finds the best separating normal between two cuboids.
///
/// Only the normals from `cuboid1` are tested.
#[inline(always)]
pub fn cuboid_cuboid_find_local_separating_normal_oneway(
    cuboid1: &Cuboid,
    cuboid2: &Cuboid,
    pos12: Pose,
) -> SeparatingAxis {
    let mut best_separation = -MAX_FLT;
    let mut best_dir = Vector::ZERO;

    macro_rules! check_axis(
        ($x: ident) => {
            let sign = if pos12.translation.$x >= 0.0 { 1.0 } else { -1.0 };
            let mut axis1 = Vector::ZERO;
            axis1.$x = sign;
            let axis2 = pos12.rotation.inverse() * -axis1;
            let local_pt2 = cuboid2.local_support_point(axis2);
            let pt2 = pos12 * local_pt2;
            let separation = pt2.$x * sign - cuboid1.half_extents.$x;

            if separation > best_separation {
                best_separation = separation;
                best_dir = axis1;
            }
        }
    );
    check_axis!(x);
    check_axis!(y);
    #[cfg(feature = "dim3")]
    check_axis!(z);

    SeparatingAxis::new(best_separation, best_dir)
}
