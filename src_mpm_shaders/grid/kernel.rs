//! Quadratic B-spline kernel for MPM grid transfers.
//!
//! Provides the quadratic B-spline weights and the neighbour stencil. The kernel
//! spans 3 nodes per dimension, giving 9 neighbors in 2D and 27 in 3D.

use crate::{DIM_USIZE, Vector, abs};
use glamx::*;

/*
 * Neighborhood stencil constants.
 *
 * These define the iteration order and shared-memory indices for the
 * 3x3 (2D) or 3x3x3 (3D) neighborhood around a particle.
 */

/// Number of neighbor nodes in the kernel stencil.
#[cfg(feature = "dim2")]
pub const NBH_LEN: usize = 9;
/// Number of neighbor nodes in the kernel stencil.
#[cfg(feature = "dim3")]
pub const NBH_LEN: usize = 27;

/// Returns the stencil offset for neighbor `i` as a UVector.
///
/// In 2D, these are UVec2 offsets into a 3x3 grid.
/// In 3D, these are UVec3 offsets into a 3x3x3 grid.
#[cfg(feature = "dim2")]
pub const NBH_SHIFTS: [UVec2; 9] = [
    UVec2::new(2, 2),
    UVec2::new(2, 0),
    UVec2::new(2, 1),
    UVec2::new(0, 2),
    UVec2::new(0, 0),
    UVec2::new(0, 1),
    UVec2::new(1, 2),
    UVec2::new(1, 0),
    UVec2::new(1, 1),
];

/// Returns the stencil offset for neighbor `i` as a UVector.
#[cfg(feature = "dim3")]
pub const NBH_SHIFTS: [UVec3; 27] = [
    UVec3::new(2, 2, 2),
    UVec3::new(2, 0, 2),
    UVec3::new(2, 1, 2),
    UVec3::new(0, 2, 2),
    UVec3::new(0, 0, 2),
    UVec3::new(0, 1, 2),
    UVec3::new(1, 2, 2),
    UVec3::new(1, 0, 2),
    UVec3::new(1, 1, 2),
    UVec3::new(2, 2, 0),
    UVec3::new(2, 0, 0),
    UVec3::new(2, 1, 0),
    UVec3::new(0, 2, 0),
    UVec3::new(0, 0, 0),
    UVec3::new(0, 1, 0),
    UVec3::new(1, 2, 0),
    UVec3::new(1, 0, 0),
    UVec3::new(1, 1, 0),
    UVec3::new(2, 2, 1),
    UVec3::new(2, 0, 1),
    UVec3::new(2, 1, 1),
    UVec3::new(0, 2, 1),
    UVec3::new(0, 0, 1),
    UVec3::new(0, 1, 1),
    UVec3::new(1, 2, 1),
    UVec3::new(1, 0, 1),
    UVec3::new(1, 1, 1),
];

/// Flattens the 2D/3D stencil offset of neighbor `i` into a workgroup
/// shared-memory index.
#[cfg(feature = "dim2")]
pub const NBH_SHIFT_SHARED: [u32; 9] = [22, 2, 12, 20, 0, 10, 21, 1, 11];
#[cfg(feature = "dim3")]
pub const NBH_SHIFT_SHARED: [u32; 27] = [
    86, 74, 80, 84, 72, 78, 85, 73, 79, 14, 2, 8, 12, 0, 6, 13, 1, 7, 50, 38, 44, 48, 36, 42, 49,
    37, 43,
];

/// Extracts a component from a Vec3 by dynamic index.
///
/// This avoids `Vec3::Index<usize>` which generates SPIR-V pointer phi nodes
/// (requiring the VariablePointers capability). Instead, this function
/// produces value phi nodes which are always valid in SPIR-V.
#[inline]
pub fn vec3_extract(v: Vec3, index: u32) -> f32 {
    if index == 0 {
        v.x
    } else if index == 1 {
        v.y
    } else {
        v.z
    }
}

/// Quadratic B-spline kernel.
///
/// This kernel provides quadratic (degree 2) B-spline basis functions for the
/// MPM particle-grid transfers. Each basis function has support over 3 cells,
/// giving a smooth C1-continuous interpolation.
pub struct QuadraticKernel;

impl QuadraticKernel {
    /// Computes the inverse of the D matrix diagonal for APIC transfers.
    ///
    /// For the quadratic B-spline, `inv_d = 4 / h^2` where `h` is the cell width.
    #[inline]
    pub fn inv_d(cell_width: f32) -> f32 {
        4.0 / (cell_width * cell_width)
    }

    /// Evaluates all three quadratic B-spline basis functions at position `x`,
    /// returning `Vec3(w_left, w_center, w_right)`. `x` is the distance from the
    /// associated (leftmost) grid node, in cell units.
    #[inline]
    pub fn eval_all(x: f32) -> Vec3 {
        Vec3::new(
            0.5 * (1.5 - x) * (1.5 - x),
            0.75 - (x - 1.0) * (x - 1.0),
            0.5 * (x - 0.5) * (x - 0.5),
        )
    }

    /// Evaluates a single quadratic B-spline basis function at position `x`.
    ///
    /// Uses absolute value of `x` and selects the appropriate piece of the
    /// piecewise-quadratic function based on distance from center.
    #[inline]
    pub fn eval(x: f32) -> f32 {
        // Branchless: rust-gpu lowers `if/else if/else` to real branches, which are
        // expensive in the per-particle inner loop of the transfer kernels. Computing both
        // pieces and masking compiles to selects/arithmetic instead (matching what the
        // slang version produces).
        let x_abs = abs(x);
        let part1 = 0.75 - x_abs * x_abs;
        let part2 = 0.5 * (1.5 - x_abs) * (1.5 - x_abs);
        let lt05 = (x_abs < 0.5) as u32 as f32;
        let lt15 = (x_abs < 1.5) as u32 as f32;
        lt05 * part1 + (1.0 - lt05) * lt15 * part2
    }

    /// Evaluates the derivative of a single quadratic B-spline basis function at position `x`.
    #[inline]
    pub fn eval_derivative(x: f32) -> f32 {
        // Branchless, see `eval`.
        let x_abs = abs(x);
        let sign = (x >= 0.0) as u32 as f32 * 2.0 - 1.0;
        let part1 = -2.0 * sign * x_abs;
        let part2 = -sign * (1.5 - x_abs);
        let lt05 = (x_abs < 0.5) as u32 as f32;
        let lt15 = (x_abs < 1.5) as u32 as f32;
        lt05 * part1 + (1.0 - lt05) * lt15 * part2
    }

    /// Precomputes all kernel weights for a particle at position `ref_pos` relative
    /// to the associated grid node, with cell width `h`.
    ///
    /// Returns an array of `DIM_USIZE` Vec3 values, one per spatial dimension.
    /// Each Vec3 contains the three basis function weights for that dimension.
    #[inline]
    pub fn precompute_weights(ref_elt_pos_minus_particle_pos: Vector, h: f32) -> [Vec3; DIM_USIZE] {
        #[cfg(feature = "dim2")]
        {
            [
                Self::eval_all(-ref_elt_pos_minus_particle_pos.x / h),
                Self::eval_all(-ref_elt_pos_minus_particle_pos.y / h),
            ]
        }
        #[cfg(feature = "dim3")]
        {
            [
                Self::eval_all(-ref_elt_pos_minus_particle_pos.x / h),
                Self::eval_all(-ref_elt_pos_minus_particle_pos.y / h),
                Self::eval_all(-ref_elt_pos_minus_particle_pos.z / h),
            ]
        }
    }
}
