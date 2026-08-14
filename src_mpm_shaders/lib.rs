//! Nexus MPM (Material Point Method) GPU shaders.
//!
//! This crate contains Rust GPU shaders for the nexus_mpm solver,
//! providing GPU-accelerated MPM simulation.

#![cfg_attr(target_arch_is_gpu, no_std)]
#![cfg_attr(target_arch = "spirv", feature(asm_experimental_arch))]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::module_inception)]
#![allow(unused_imports)]

// Re-export the nexus_rbd_shaders crate for shared types.
#[cfg(feature = "dim2")]
pub extern crate nexus_rbd_shaders2d as nexus_rbd_shaders;
#[cfg(feature = "dim3")]
pub extern crate nexus_rbd_shaders3d as nexus_rbd_shaders;

// Re-export parry for collision shapes.
#[cfg(feature = "dim2")]
pub extern crate parry2d as parry;
#[cfg(feature = "dim3")]
pub extern crate parry3d as parry;

// Re-export glamx for convenience.
pub use glamx;
use glamx::*;

// Re-export key types and utilities from nexus_rbd_shaders.
pub use nexus_rbd_shaders::{AngVector, DIM, Pad, PaddedVector, Pose, RotMatrix, Rotation, Vector};
pub use nexus_rbd_shaders::{abs, acos, asin, atan2, cos, safe_div, sin, sqrt};
pub use nexus_rbd_shaders::{gcross, gcross_av, gdot, maybe_inv, rotation_to_matrix};
pub use nexus_rbd_shaders::{udiv, umod};

// NOTE: we disable two-ways coupling for now since it’s quite expensive and not very
//       stable anyway. We’d need some deeper integration with Nexus’ solver for it
//       to look really good.
pub const TWO_WAYS_COUPLING_ENABLED: bool = false;

//
// MPM-specific type aliases
//

/// Signed integer vector type (IVec2 in 2D, IVec3 in 3D).
#[cfg(feature = "dim2")]
pub type IVector = IVec2;
/// Signed integer vector type (IVec2 in 2D, IVec3 in 3D).
#[cfg(feature = "dim3")]
pub type IVector = IVec3;

/// Unsigned integer vector type (UVec2 in 2D, UVec3 in 3D).
#[cfg(feature = "dim2")]
pub type UVector = UVec2;
/// Unsigned integer vector type (UVec2 in 2D, UVec3 in 3D).
#[cfg(feature = "dim3")]
pub type UVector = UVec3;

/// The square matrix type for the current dimension (Mat2 in 2D, Mat3 in 3D).
#[cfg(feature = "dim2")]
pub type Matrix = Mat2;
/// The square matrix type for the current dimension (Mat2 in 2D, Mat3 in 3D).
#[cfg(feature = "dim3")]
pub type Matrix = Mat3;

/// The square matrix type for the current dimension (Mat2 in 2D, Mat3 in 3D).
#[cfg(feature = "dim2")]
pub type PaddedMatrix = Mat2;
/// The square matrix type for the current dimension (Mat2 in 2D, Mat3 in 3D), with explicit padding.
#[cfg(feature = "dim3")]
pub type PaddedMatrix = Mat4;

/// The dimension constant as usize (for array indexing).
#[cfg(feature = "dim2")]
pub const DIM_USIZE: usize = 2;
/// The dimension constant as usize (for array indexing).
#[cfg(feature = "dim3")]
pub const DIM_USIZE: usize = 3;

//
// Helper function: construct a diagonal matrix.
//
#[cfg(feature = "dim2")]
#[inline]
pub fn diag(v: Vector) -> Matrix {
    Mat2::from_cols(Vec2::new(v.x, 0.0), Vec2::new(0.0, v.y))
}

#[cfg(feature = "dim3")]
#[inline]
pub fn diag(v: Vector) -> Matrix {
    Mat3::from_cols(
        Vec3::new(v.x, 0.0, 0.0),
        Vec3::new(0.0, v.y, 0.0),
        Vec3::new(0.0, 0.0, v.z),
    )
}

/// Helper to compute the trace of a matrix.
#[cfg(feature = "dim2")]
#[inline]
pub fn trace(m: Matrix) -> f32 {
    m.x_axis.x + m.y_axis.y
}

#[cfg(feature = "dim3")]
#[inline]
pub fn trace(m: Matrix) -> f32 {
    m.x_axis.x + m.y_axis.y + m.z_axis.z
}

/// The length of a vector as AngVector. In 2D this is abs, in 3D it's the standard length.
#[cfg(feature = "dim2")]
#[inline]
pub fn ang_length(v: AngVector) -> f32 {
    abs(v)
}

#[cfg(feature = "dim3")]
#[inline]
pub fn ang_length(v: AngVector) -> f32 {
    v.length()
}

pub trait PaddingExt {
    type WithoutPadding;
    fn remove_padding(self) -> Self::WithoutPadding;
    fn add_padding(without_padding: Self::WithoutPadding) -> Self;
}

impl PaddingExt for Mat2 {
    type WithoutPadding = Mat2;
    #[inline]
    fn remove_padding(self) -> Mat2 {
        self
    }
    #[inline]
    fn add_padding(without_padding: Mat2) -> Mat2 {
        without_padding
    }
}

impl PaddingExt for Mat4 {
    type WithoutPadding = Mat3;
    #[inline]
    fn remove_padding(self) -> Mat3 {
        Mat3::from_cols(self.x_axis.xyz(), self.y_axis.xyz(), self.z_axis.xyz())
    }
    #[inline]
    fn add_padding(without_padding: Mat3) -> Mat4 {
        Mat4::from_mat3(without_padding)
    }
}

//
// Modules
//
pub mod collision;
pub mod grid;
pub mod models;
pub mod solver;
