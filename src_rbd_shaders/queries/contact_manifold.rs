//! Contact Manifold Module
//!
//! This module defines structures for storing contact information between colliders.
//! Contact manifolds are the output of narrow-phase collision detection and serve
//! as input to physics solvers.
//!
//! Dimension-specific limits:
//! - 2D: Up to 2 contact points per manifold
//! - 3D: Up to 4 contact points per manifold
//!
//! Contact points are stored with their location (in the first object's local frame)
//! and penetration distance. The manifold also includes a shared contact normal.

use crate::{Pose, Vector};
use khal_std::index::MaybeIndexUnchecked;

/// Maximum number of contact points in a 2D contact manifold.
#[cfg(feature = "dim2")]
pub const MAX_MANIFOLD_POINTS: usize = 2;
/// Maximum number of contact points in a 3D contact manifold.
#[cfg(feature = "dim3")]
pub const MAX_MANIFOLD_POINTS: usize = 4;

/// A single contact point within a manifold.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct ContactPoint {
    // NOTE: field order is important here to make this struct as compact as possible.
    /// Contact point location (in object A's local frame).
    pub pt: Vector,
    /// Signed penetration distance (negative = penetrating).
    pub dist: f32,
    #[cfg(feature = "dim2")]
    pub padding: u32,
}

impl ContactPoint {
    /// Creates a new contact point.
    #[inline]
    pub fn new(pt: Vector, dist: f32) -> Self {
        Self {
            pt,
            dist,
            #[cfg(feature = "dim2")]
            padding: 0,
        }
    }
}

/// A contact manifold containing multiple contact points.
///
/// Represents the contact region between two colliders. Multiple points
/// provide stability for physics simulation.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct ContactManifold {
    // NOTE: fields order is important here to make this struct as compact as possible.
    pub points_a: [ContactPoint; MAX_MANIFOLD_POINTS],
    pub normal_a: Vector,
    pub len: u32,
    #[cfg(feature = "dim2")]
    pub padding: u32,
}

impl Default for ContactManifold {
    fn default() -> Self {
        Self {
            points_a: [ContactPoint::default(); MAX_MANIFOLD_POINTS],
            normal_a: Vector::default(),
            len: 0,
            #[cfg(feature = "dim2")]
            padding: 0,
        }
    }
}

impl ContactManifold {
    /// Creates an empty contact manifold.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a manifold with a single contact point.
    #[inline]
    pub fn single_point(pt: Vector, dist: f32, normal: Vector) -> Self {
        let mut result = ContactManifold::default();
        result.points_a.write(0, ContactPoint::new(pt, dist));
        result.normal_a = normal;
        result.len = 1;
        result
    }

    /// Clears all contact points from the manifold.
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Flips a contact manifold to be from the perspective of the other collider.
    #[inline]
    pub fn flip(&self, transform: Pose) -> ContactManifold {
        let mut result = *self;
        let normal = transform.rotation * -result.normal_a;

        result.points_a.at_mut(0).pt =
            transform * result.points_a.at(0).pt - normal * result.points_a.at(0).dist;
        result.points_a.at_mut(1).pt =
            transform * result.points_a.at(1).pt - normal * result.points_a.at(1).dist;

        #[cfg(feature = "dim3")]
        {
            result.points_a.at_mut(2).pt =
                transform * result.points_a.at(2).pt - normal * result.points_a.at(2).dist;
            result.points_a.at_mut(3).pt =
                transform * result.points_a.at(3).pt - normal * result.points_a.at(3).dist;
        }

        result.normal_a = normal;
        result
    }
}
