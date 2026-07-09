//! Nexus GPU physics engine shaders.
//!
//! This crate contains Rust GPU shaders for the nexus physics engine,
//! providing GPU-accelerated collision detection and physics simulation.

// Only no_std when targeting GPU (spirv or nvptx64). On CPU, we need std for generated ShaderArgs.
#![cfg_attr(target_arch_is_gpu, no_std)]
#![cfg_attr(target_arch = "spirv", feature(asm_experimental_arch))]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::module_inception)]

use core::ops::{Deref, DerefMut};
use glamx::*;

// Re-export glamx types for convenience
pub use glamx;

//
// Dimension-specific type aliases
//

#[cfg(feature = "dim2")]
pub extern crate parry2d as parry;
#[cfg(feature = "dim3")]
pub extern crate parry3d as parry;
#[cfg(feature = "dim2")]
pub extern crate rapier2d as rapier;
#[cfg(feature = "dim3")]
pub extern crate rapier3d as rapier;

/// The vector type for the current dimension (Vec2 in 2D, Vec3 in 3D).
#[cfg(feature = "dim2")]
pub type Vector = Vec2;
/// The vector type for the current dimension (Vec2 in 2D, Vec3 in 3D).
#[cfg(feature = "dim3")]
pub type Vector = Vec3;

/// The angular vector type (f32 in 2D, Vec3 in 3D).
/// In 2D, angular velocity is a scalar (rotation around Z axis).
/// In 3D, angular velocity is a vector (rotation around arbitrary axis).
#[cfg(feature = "dim2")]
pub type AngVector = f32;
/// The angular vector type (f32 in 2D, Vec3 in 3D).
#[cfg(feature = "dim3")]
pub type AngVector = Vec3;

/// The rotation type (Rot2 in 2D, Quat in 3D).
#[cfg(feature = "dim2")]
pub type Rotation = Rot2;
/// The rotation type (Rot2 in 2D, Quat in 3D).
#[cfg(feature = "dim3")]
pub type Rotation = Quat;

/// The pose type (Pose2 in 2D, Pose3 in 3D).
/// Represents a rigid body transformation (rotation + translation).
#[cfg(feature = "dim2")]
pub type Pose = Pose2;
/// The pose type (Pose2 in 2D, Pose3 in 3D).
#[cfg(feature = "dim3")]
pub type Pose = Pose3;

/// The matrix type for rotations (Mat2 in 2D, Mat3 in 3D).
#[cfg(feature = "dim2")]
pub type RotMatrix = Mat2;
/// The matrix type for rotations (Mat2 in 2D, Mat3 in 3D).
#[cfg(feature = "dim3")]
pub type RotMatrix = Mat3;

/// The dimension constant.
#[cfg(feature = "dim2")]
pub const DIM: u32 = 2;
/// The dimension constant.
#[cfg(feature = "dim3")]
pub const DIM: u32 = 3;

/// Number of rotational degrees of freedom of a rigid-body (1 in 2D, 3 in 3D).
#[cfg(feature = "dim2")]
pub const ANG_DIM: u32 = 1;
/// Number of rotational degrees of freedom of a rigid-body (1 in 2D, 3 in 3D).
#[cfg(feature = "dim3")]
pub const ANG_DIM: u32 = 3;

//
// Rotation helper functions
//

/// Converts a rotation to a rotation matrix.
#[cfg(feature = "dim2")]
#[inline]
pub fn rotation_to_matrix(rot: Rotation) -> RotMatrix {
    rot.to_mat()
}

/// Converts a rotation to a rotation matrix.
#[cfg(feature = "dim3")]
#[inline]
pub fn rotation_to_matrix(rot: Rotation) -> RotMatrix {
    Mat3::from_quat(rot)
}

/// Creates a rotation from an angle (2D only).
#[cfg(feature = "dim2")]
#[inline]
pub fn rotation_from_angle(angle: f32) -> Rotation {
    Rot2::new(angle)
}

/// Creates a rotation from a scaled axis (axis * angle) (3D only).
#[cfg(feature = "dim3")]
#[inline]
pub fn rotation_from_scaled_axis(scaled_axis: Vec3) -> Rotation {
    Quat::from_scaled_axis(scaled_axis)
}

/// Fast renormalization for quaternions (3D only).
#[cfg(feature = "dim3")]
#[inline]
pub fn rotation_renormalize_fast(q: Rotation) -> Rotation {
    q.normalize()
}

/// Gets the angle of a 2D rotation.
#[cfg(feature = "dim2")]
#[inline]
pub fn rotation_angle(rot: Rotation) -> f32 {
    rot.angle()
}

/// Gets the imaginary part of a quaternion (3D only).
#[cfg(feature = "dim3")]
#[inline]
pub fn rotation_imag(rot: Rotation) -> Vec3 {
    Vec3::new(rot.x, rot.y, rot.z)
}

//
// Generic cross product (dimension-agnostic)
//

/// Computes the 2D "cross product" (perp dot product): a.x * b.y - a.y * b.x
#[cfg(feature = "dim2")]
#[inline]
pub fn gcross(a: Vector, b: Vector) -> AngVector {
    a.x * b.y - a.y * b.x
}

/// Computes the 3D cross product.
#[cfg(feature = "dim3")]
#[inline]
pub fn gcross(a: Vector, b: Vector) -> AngVector {
    a.cross(b)
}

/// Computes angular_velocity x vector (2D version).
#[cfg(feature = "dim2")]
#[inline]
pub fn gcross_av(angular: AngVector, v: Vector) -> Vector {
    Vec2::new(-angular * v.y, angular * v.x)
}

/// Computes angular_velocity x vector (3D version).
#[cfg(feature = "dim3")]
#[inline]
pub fn gcross_av(angular: AngVector, v: Vector) -> Vector {
    angular.cross(v)
}

/// Dot product for angular vectors.
#[cfg(feature = "dim2")]
#[inline]
pub fn gdot(a: AngVector, b: AngVector) -> f32 {
    a * b
}

/// Dot product for angular vectors.
#[cfg(feature = "dim3")]
#[inline]
pub fn gdot(a: AngVector, b: AngVector) -> f32 {
    a.dot(b)
}

//
// Utility functions
//

/// Unsigned integer division without Rust's division-by-zero check.
///
/// Bypasses the compiler-inserted zero-check branch that breaks uniform control
/// flow (required for workgroup barriers on WebGPU).
///
/// # Safety
/// The caller must ensure `b != 0`. Behavior is implementation-defined if `b == 0`.
#[cfg(target_arch = "spirv")]
#[inline(always)]
pub fn udiv(a: u32, b: u32) -> u32 {
    let mut result: u32 = 0;
    unsafe {
        core::arch::asm! {
            "%a = OpLoad _ {a}",
            "%b = OpLoad _ {b}",
            "%result = OpUDiv _ %a %b",
            "OpStore {result} %result",
            a = in(reg) &a,
            b = in(reg) &b,
            result = in(reg) &mut result,
        }
    }
    result
}

/// Unsigned integer modulo without Rust's division-by-zero check.
///
/// # Safety
/// The caller must ensure `b != 0`. Behavior is implementation-defined if `b == 0`.
#[cfg(target_arch = "spirv")]
#[inline(always)]
pub fn umod(a: u32, b: u32) -> u32 {
    let mut result: u32 = 0;
    unsafe {
        core::arch::asm! {
            "%a = OpLoad _ {a}",
            "%b = OpLoad _ {b}",
            "%result = OpUMod _ %a %b",
            "OpStore {result} %result",
            a = in(reg) &a,
            b = in(reg) &b,
            result = in(reg) &mut result,
        }
    }
    result
}

/// Fallback for non-SPIR-V targets (CPU-side).
#[cfg(not(target_arch = "spirv"))]
#[inline(always)]
pub fn udiv(a: u32, b: u32) -> u32 {
    a / b
}

/// Fallback for non-SPIR-V targets (CPU-side).
#[cfg(not(target_arch = "spirv"))]
#[inline(always)]
pub fn umod(a: u32, b: u32) -> u32 {
    a % b
}

/// Safe division that returns 0 if the denominator is 0.
#[inline]
pub fn safe_div(num: f32, denom: f32) -> f32 {
    if denom == 0.0 { 0.0 } else { num / denom }
}

/// Returns 1/x if x != 0, otherwise 0.
#[inline]
pub fn maybe_inv(x: f32) -> f32 {
    const INV_EPSILON: f32 = 1.0e-20;
    if x < -INV_EPSILON || x > INV_EPSILON {
        1.0 / x
    } else {
        0.0
    }
}

/// Clamps the magnitude of a Vec2 to at most `limit`.
#[inline]
pub fn cap_magnitude_vec2(v: Vec2, limit: f32) -> Vec2 {
    let n = v.length();
    if n > limit { v * (limit / n) } else { v }
}

/// Machine epsilon for f32.
pub const F32_EPSILON: f32 = 1.1920929e-7;

/// Tolerance for floating point comparisons.
pub const EPS_TOL: f32 = 1.0e-6;

/// Maximum floating-point value (used for unbounded parameters).
///
/// Note that this does **not** match the usual max float value (3.40282347E+38). On some browsers
/// like safari, very large constants don’t work, likely due to safari compiling the shader with --fast-math.
/// The value here has been empirically tested to work. Larger values (like 1.0e19) don’t seem to work either.
pub const MAX_FLT: f32 = 1.0e18;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Default)]
#[repr(C)]
pub struct Pad<T, P>(pub T, pub P);

// Pod/Zeroable impls for specific Pad instantiations that have no internal padding.
// We can't use a generic impl because Pad<T, P> may have padding between T and P
// if align_of::<P>() > size_of::<T>() % align_of::<P>() (e.g., Pad<u8, u32>).
#[cfg(not(target_arch_is_gpu))]
mod pad_bytemuck_impls {
    use super::Pad;
    use glamx::{Vec2, Vec3};

    // Pad<Vec3, u32>: 12 + 4 = 16 bytes, 4-byte aligned, no padding
    unsafe impl bytemuck::Zeroable for Pad<Vec3, u32> {}
    unsafe impl bytemuck::Pod for Pad<Vec3, u32> {}

    // Pad<Vec2, ()>: 8 + 0 = 8 bytes, no padding
    unsafe impl bytemuck::Zeroable for Pad<Vec2, ()> {}
    unsafe impl bytemuck::Pod for Pad<Vec2, ()> {}

    // Pad<f32, u32>: 4 + 4 = 8 bytes, 4-byte aligned, no padding
    unsafe impl bytemuck::Zeroable for Pad<f32, u32> {}
    unsafe impl bytemuck::Pod for Pad<f32, u32> {}
}

impl<T, P> Pad<T, P> {
    #[inline(always)]
    pub fn new(x: T) -> Self
    where
        P: Default,
    {
        Pad(x, P::default())
    }
}

#[cfg(feature = "dim2")]
pub type PaddedVector = Pad<Vector, ()>;
#[cfg(feature = "dim3")]
pub type PaddedVector = Pad<Vector, u32>;

impl<T, P: Default> From<T> for Pad<T, P> {
    #[inline(always)]
    fn from(x: T) -> Pad<T, P> {
        Pad(x, P::default())
    }
}

impl<T, P> Deref for Pad<T, P> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, P> DerefMut for Pad<T, P> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

//
// Math helper functions for SPIR-V compatibility
//
// In SPIR-V/no_std, f32 doesn't have methods like sqrt(), sin(), etc.
// We use the num_traits::Float trait to provide these operations.
//

use khal_std::num_traits::Float;

/// Computes the square root of a value.
#[inline]
pub fn sqrt(x: f32) -> f32 {
    Float::sqrt(x)
}

/// Computes the sine of a value (in radians).
#[inline]
pub fn sin(x: f32) -> f32 {
    Float::sin(x)
}

/// Computes the cosine of a value (in radians).
#[inline]
pub fn cos(x: f32) -> f32 {
    Float::cos(x)
}

/// Computes the arcsine of a value.
#[inline]
pub fn asin(x: f32) -> f32 {
    Float::asin(x)
}

/// Computes the arccosine of a value.
#[inline]
pub fn acos(x: f32) -> f32 {
    Float::acos(x)
}

/// Computes the arctangent of y/x.
#[inline]
pub fn atan2(y: f32, x: f32) -> f32 {
    Float::atan2(y, x)
}

/// Computes the absolute value.
#[inline]
pub fn abs(x: f32) -> f32 {
    Float::abs(x)
}

//
// Modules
//
pub mod bounding_volumes;
pub mod broad_phase;
pub mod dynamics;
pub mod queries;
pub mod shapes;
pub mod utils;

#[cfg(test)]
mod tests;

/// Branchless-ish, panic-free matrix column access for device code.
///
/// `Mat::col` panics with a formatted message on an out-of-range index, which
/// cuda-oxide device kernels cannot carry (panic message formatting is
/// unsupported on the GPU).
/// `index` **must** be in range (`0..2` for `Mat2`, `0..3` for `Mat3`); every
/// caller in this crate derives it from an axis mask, so this always holds.
/// An out-of-range index is a bug: debug (host) builds catch it with a
/// `debug_assert!`, release/device builds fold it to the last column rather
/// than carrying panic machinery into the kernel.
pub trait ColBranchless {
    type Column;
    fn col_b(&self, index: usize) -> Self::Column;
}

impl ColBranchless for glamx::Mat3 {
    type Column = glamx::Vec3;
    #[inline(always)]
    fn col_b(&self, index: usize) -> glamx::Vec3 {
        debug_assert!(index < 3, "Mat3 column index out of range");
        match index {
            0 => self.x_axis,
            1 => self.y_axis,
            _ => self.z_axis,
        }
    }
}

impl ColBranchless for glamx::Mat2 {
    type Column = glamx::Vec2;
    #[inline(always)]
    fn col_b(&self, index: usize) -> glamx::Vec2 {
        debug_assert!(index < 2, "Mat2 column index out of range");
        match index {
            0 => self.x_axis,
            _ => self.y_axis,
        }
    }
}
