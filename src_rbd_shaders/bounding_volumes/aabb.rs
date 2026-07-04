//! Axis-Aligned Bounding Box (AABB)
//!
//! This module provides the AABB structure and related operations.
//! An AABB is defined by its minimum and maximum corners.
//! It represents the tightest axis-aligned box that contains a shape.

use crate::{MAX_FLT, Pose, Vector};

/// An axis-aligned bounding box (AABB).
///
/// The AABB is defined by its minimum and maximum corners.
/// All points P inside the AABB satisfy: mins <= P <= maxs (component-wise).
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct Aabb {
    /// The corner with the smallest coordinates (lower-left in 2D).
    pub mins: Vector,
    #[cfg(feature = "dim3")]
    pub padding0: u32,
    /// The corner with the largest coordinates (upper-right in 2D).
    pub maxs: Vector,
    #[cfg(feature = "dim3")]
    pub padding1: u32,
}

impl Aabb {
    /// Creates a new AABB from min and max corners.
    #[inline]
    pub fn new(mins: Vector, maxs: Vector) -> Self {
        Self {
            mins,
            maxs,
            #[cfg(feature = "dim3")]
            padding0: 0,
            #[cfg(feature = "dim3")]
            padding1: 0,
        }
    }

    /// Creates an AABB with infinite bounds (covers everything).
    #[inline]
    #[cfg(feature = "dim2")]
    pub fn infinite() -> Self {
        Self {
            mins: Vector::splat(-MAX_FLT),
            maxs: Vector::splat(MAX_FLT),
        }
    }

    /// Creates an AABB with infinite bounds (covers everything).
    #[inline]
    #[cfg(feature = "dim3")]
    pub fn infinite() -> Self {
        Self {
            mins: Vector::splat(-MAX_FLT),
            maxs: Vector::splat(MAX_FLT),
            #[cfg(feature = "dim3")]
            padding0: 0,
            #[cfg(feature = "dim3")]
            padding1: 0,
        }
    }

    /// Computes the center point of the AABB.
    #[inline]
    pub fn center(&self) -> Vector {
        (self.mins + self.maxs) * 0.5
    }

    /// Computes the half-extents (half-widths) of the AABB.
    #[inline]
    pub fn half_extents(&self) -> Vector {
        (self.maxs - self.mins) * 0.5
    }

    /// Computes the full extents (widths) of the AABB.
    #[inline]
    pub fn extents(&self) -> Vector {
        self.maxs - self.mins
    }

    /// Returns the AABB transformed by a pose.
    ///
    /// This computes a new AABB that tightly bounds the original AABB
    /// after applying the given transformation.
    #[cfg(feature = "dim2")]
    #[inline]
    pub fn transform_by(&self, pose: Pose) -> Aabb {
        let center = self.center();
        let half_extents = self.half_extents();

        let new_center = pose.transform_point(center);

        // Compute new half-extents by considering how the rotation affects the extents
        let rot_mat = pose.rotation.to_mat();
        let abs_rot_mat = glamx::Mat2::from_cols(rot_mat.x_axis.abs(), rot_mat.y_axis.abs());
        let new_half_extents = abs_rot_mat * half_extents;

        Aabb {
            mins: new_center - new_half_extents,
            maxs: new_center + new_half_extents,
            #[cfg(feature = "dim3")]
            padding0: 0,
            #[cfg(feature = "dim3")]
            padding1: 0,
        }
    }

    /// Returns the AABB transformed by a pose.
    #[cfg(feature = "dim3")]
    #[inline]
    pub fn transform_by(&self, pose: Pose) -> Aabb {
        let center = self.center();
        let half_extents = self.half_extents();

        let new_center = pose.transform_point(center);

        // Compute new half-extents by considering how the rotation affects the extents
        let rot_mat = glamx::Mat3::from_quat(pose.rotation);
        let abs_rot_mat = glamx::Mat3::from_cols(
            rot_mat.x_axis.abs(),
            rot_mat.y_axis.abs(),
            rot_mat.z_axis.abs(),
        );
        let new_half_extents = abs_rot_mat * half_extents;

        Aabb {
            mins: new_center - new_half_extents,
            maxs: new_center + new_half_extents,
            #[cfg(feature = "dim3")]
            padding0: 0,
            #[cfg(feature = "dim3")]
            padding1: 0,
        }
    }

    /// Merges this AABB with another, returning the smallest AABB containing both.
    #[inline]
    pub fn merged(&self, other: &Aabb) -> Aabb {
        Aabb {
            mins: self.mins.min(other.mins),
            maxs: self.maxs.max(other.maxs),
            #[cfg(feature = "dim3")]
            padding0: 0,
            #[cfg(feature = "dim3")]
            padding1: 0,
        }
    }

    /// Grows this AABB to include a point.
    #[inline]
    pub fn grow(&mut self, pt: Vector) {
        self.mins = self.mins.min(pt);
        self.maxs = self.maxs.max(pt);
    }

    /// Tests if this AABB intersects another AABB.
    #[cfg(feature = "dim2")]
    #[inline]
    pub fn intersects(&self, other: &Aabb) -> bool {
        // TODO PERF: is could we use some sort of `self.mins <= other.maxs`
        //            directly instead of detailing each component.
        self.mins.x <= other.maxs.x
            && self.maxs.x >= other.mins.x
            && self.mins.y <= other.maxs.y
            && self.maxs.y >= other.mins.y
    }

    /// Tests if this AABB intersects another AABB.
    #[cfg(feature = "dim3")]
    #[inline]
    pub fn intersects(&self, other: &Aabb) -> bool {
        // TODO PERF: is could we use some sort of `self.mins <= other.maxs`
        //            directly instead of detailing each component.
        self.mins.x <= other.maxs.x
            && self.maxs.x >= other.mins.x
            && self.mins.y <= other.maxs.y
            && self.maxs.y >= other.mins.y
            && self.mins.z <= other.maxs.z
            && self.maxs.z >= other.mins.z
    }

    /// Tests if this AABB contains a point.
    #[cfg(feature = "dim2")]
    #[inline]
    pub fn contains_point(&self, pt: Vector) -> bool {
        pt.x >= self.mins.x && pt.x <= self.maxs.x && pt.y >= self.mins.y && pt.y <= self.maxs.y
    }

    /// Tests if this AABB contains a point.
    #[cfg(feature = "dim3")]
    #[inline]
    pub fn contains_point(&self, pt: Vector) -> bool {
        pt.x >= self.mins.x
            && pt.x <= self.maxs.x
            && pt.y >= self.mins.y
            && pt.y <= self.maxs.y
            && pt.z >= self.mins.z
            && pt.z <= self.maxs.z
    }

    /// Loosens the AABB by a given margin on all sides.
    #[cfg(feature = "dim2")]
    #[inline]
    pub fn loosened(&self, margin: f32) -> Aabb {
        let margin_vec = Vector::splat(margin);
        Aabb {
            mins: self.mins - margin_vec,
            maxs: self.maxs + margin_vec,
            #[cfg(feature = "dim3")]
            padding0: 0,
            #[cfg(feature = "dim3")]
            padding1: 0,
        }
    }

    /// Loosens the AABB by a given margin on all sides.
    #[cfg(feature = "dim3")]
    #[inline]
    pub fn loosened(&self, margin: f32) -> Aabb {
        let margin_vec = Vector::splat(margin);
        Aabb {
            mins: self.mins - margin_vec,
            maxs: self.maxs + margin_vec,
            #[cfg(feature = "dim3")]
            padding0: 0,
            #[cfg(feature = "dim3")]
            padding1: 0,
        }
    }
}
