//! Joint-constraint columns (`M⁻¹ Jᵀ`) for small multibodies, from the LU
//! factors of their mass matrix.
//!
//! Lane `c` back-solves the unit vector `e_c` with the same operations, in the
//! same order, as `lu_solve_in_place`, holding the solution and its LU row in
//! registers; each constraint column is then gathered from the resulting
//! `M⁻¹` instead of being back-solved on its own in device memory.
//!
//! Generated substitution code: indices must be literals so the solution stays
//! in registers.

use super::types::{MultibodyInfo, MultibodyJointConstraint};
use crate::utils::BatchIndices;
use crate::utils::linalg::MatSlice;
use glamx::Vec4;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

/// A 32-entry column held in registers.
#[derive(Copy, Clone)]
struct Column {
    c0: Vec4,
    c1: Vec4,
    c2: Vec4,
    c3: Vec4,
    c4: Vec4,
    c5: Vec4,
    c6: Vec4,
    c7: Vec4,
}

impl Column {
    #[inline(always)]
    fn unit(index: u32) -> Self {
        let e = |k: u32| if index == k { 1.0 } else { 0.0 };
        Self {
            c0: Vec4::new(e(0), e(1), e(2), e(3)),
            c1: Vec4::new(e(4), e(5), e(6), e(7)),
            c2: Vec4::new(e(8), e(9), e(10), e(11)),
            c3: Vec4::new(e(12), e(13), e(14), e(15)),
            c4: Vec4::new(e(16), e(17), e(18), e(19)),
            c5: Vec4::new(e(20), e(21), e(22), e(23)),
            c6: Vec4::new(e(24), e(25), e(26), e(27)),
            c7: Vec4::new(e(28), e(29), e(30), e(31)),
        }
    }
}

#[inline(always)]
fn shuffle_u32(value: u32, lane: u32) -> u32 {
    #[cfg(target_arch = "spirv")]
    {
        spirv_std::arch::subgroup_shuffle(value, lane)
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        let _ = lane;
        value
    }
}

#[inline(always)]
fn shuffle(value: f32, lane: u32) -> f32 {
    #[cfg(target_arch = "spirv")]
    {
        spirv_std::arch::subgroup_shuffle(value, lane)
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        let _ = lane;
        value
    }
}

/// `x ← U⁻¹ L⁻¹ x` (the substitution half of `lu_solve_in_place`, same
/// operations in the same order). Lane `i` holds row `i` of the factors in
/// `lu`; the entries are broadcast by shuffles. Terms past `n` multiply zero
/// rows and columns, so only the rows themselves need a bound check.
#[inline(always)]
fn substitute(n: u32, lu: &Column, x: &mut Column) {
    // Row 0 of `L` is the identity row: nothing to subtract.
    if 1 < n {
        let mut s = x.c0.y;
        s -= shuffle(lu.c0.x, 1) * x.c0.x;
        x.c0.y = s;
    }
    if 2 < n {
        let mut s = x.c0.z;
        s -= shuffle(lu.c0.x, 2) * x.c0.x;
        s -= shuffle(lu.c0.y, 2) * x.c0.y;
        x.c0.z = s;
    }
    if 3 < n {
        let mut s = x.c0.w;
        s -= shuffle(lu.c0.x, 3) * x.c0.x;
        s -= shuffle(lu.c0.y, 3) * x.c0.y;
        s -= shuffle(lu.c0.z, 3) * x.c0.z;
        x.c0.w = s;
    }
    if 4 < n {
        let mut s = x.c1.x;
        s -= shuffle(lu.c0.x, 4) * x.c0.x;
        s -= shuffle(lu.c0.y, 4) * x.c0.y;
        s -= shuffle(lu.c0.z, 4) * x.c0.z;
        s -= shuffle(lu.c0.w, 4) * x.c0.w;
        x.c1.x = s;
    }
    if 5 < n {
        let mut s = x.c1.y;
        s -= shuffle(lu.c0.x, 5) * x.c0.x;
        s -= shuffle(lu.c0.y, 5) * x.c0.y;
        s -= shuffle(lu.c0.z, 5) * x.c0.z;
        s -= shuffle(lu.c0.w, 5) * x.c0.w;
        s -= shuffle(lu.c1.x, 5) * x.c1.x;
        x.c1.y = s;
    }
    if 6 < n {
        let mut s = x.c1.z;
        s -= shuffle(lu.c0.x, 6) * x.c0.x;
        s -= shuffle(lu.c0.y, 6) * x.c0.y;
        s -= shuffle(lu.c0.z, 6) * x.c0.z;
        s -= shuffle(lu.c0.w, 6) * x.c0.w;
        s -= shuffle(lu.c1.x, 6) * x.c1.x;
        s -= shuffle(lu.c1.y, 6) * x.c1.y;
        x.c1.z = s;
    }
    if 7 < n {
        let mut s = x.c1.w;
        s -= shuffle(lu.c0.x, 7) * x.c0.x;
        s -= shuffle(lu.c0.y, 7) * x.c0.y;
        s -= shuffle(lu.c0.z, 7) * x.c0.z;
        s -= shuffle(lu.c0.w, 7) * x.c0.w;
        s -= shuffle(lu.c1.x, 7) * x.c1.x;
        s -= shuffle(lu.c1.y, 7) * x.c1.y;
        s -= shuffle(lu.c1.z, 7) * x.c1.z;
        x.c1.w = s;
    }
    if 8 < n {
        let mut s = x.c2.x;
        s -= shuffle(lu.c0.x, 8) * x.c0.x;
        s -= shuffle(lu.c0.y, 8) * x.c0.y;
        s -= shuffle(lu.c0.z, 8) * x.c0.z;
        s -= shuffle(lu.c0.w, 8) * x.c0.w;
        s -= shuffle(lu.c1.x, 8) * x.c1.x;
        s -= shuffle(lu.c1.y, 8) * x.c1.y;
        s -= shuffle(lu.c1.z, 8) * x.c1.z;
        s -= shuffle(lu.c1.w, 8) * x.c1.w;
        x.c2.x = s;
    }
    if 9 < n {
        let mut s = x.c2.y;
        s -= shuffle(lu.c0.x, 9) * x.c0.x;
        s -= shuffle(lu.c0.y, 9) * x.c0.y;
        s -= shuffle(lu.c0.z, 9) * x.c0.z;
        s -= shuffle(lu.c0.w, 9) * x.c0.w;
        s -= shuffle(lu.c1.x, 9) * x.c1.x;
        s -= shuffle(lu.c1.y, 9) * x.c1.y;
        s -= shuffle(lu.c1.z, 9) * x.c1.z;
        s -= shuffle(lu.c1.w, 9) * x.c1.w;
        s -= shuffle(lu.c2.x, 9) * x.c2.x;
        x.c2.y = s;
    }
    if 10 < n {
        let mut s = x.c2.z;
        s -= shuffle(lu.c0.x, 10) * x.c0.x;
        s -= shuffle(lu.c0.y, 10) * x.c0.y;
        s -= shuffle(lu.c0.z, 10) * x.c0.z;
        s -= shuffle(lu.c0.w, 10) * x.c0.w;
        s -= shuffle(lu.c1.x, 10) * x.c1.x;
        s -= shuffle(lu.c1.y, 10) * x.c1.y;
        s -= shuffle(lu.c1.z, 10) * x.c1.z;
        s -= shuffle(lu.c1.w, 10) * x.c1.w;
        s -= shuffle(lu.c2.x, 10) * x.c2.x;
        s -= shuffle(lu.c2.y, 10) * x.c2.y;
        x.c2.z = s;
    }
    if 11 < n {
        let mut s = x.c2.w;
        s -= shuffle(lu.c0.x, 11) * x.c0.x;
        s -= shuffle(lu.c0.y, 11) * x.c0.y;
        s -= shuffle(lu.c0.z, 11) * x.c0.z;
        s -= shuffle(lu.c0.w, 11) * x.c0.w;
        s -= shuffle(lu.c1.x, 11) * x.c1.x;
        s -= shuffle(lu.c1.y, 11) * x.c1.y;
        s -= shuffle(lu.c1.z, 11) * x.c1.z;
        s -= shuffle(lu.c1.w, 11) * x.c1.w;
        s -= shuffle(lu.c2.x, 11) * x.c2.x;
        s -= shuffle(lu.c2.y, 11) * x.c2.y;
        s -= shuffle(lu.c2.z, 11) * x.c2.z;
        x.c2.w = s;
    }
    if 12 < n {
        let mut s = x.c3.x;
        s -= shuffle(lu.c0.x, 12) * x.c0.x;
        s -= shuffle(lu.c0.y, 12) * x.c0.y;
        s -= shuffle(lu.c0.z, 12) * x.c0.z;
        s -= shuffle(lu.c0.w, 12) * x.c0.w;
        s -= shuffle(lu.c1.x, 12) * x.c1.x;
        s -= shuffle(lu.c1.y, 12) * x.c1.y;
        s -= shuffle(lu.c1.z, 12) * x.c1.z;
        s -= shuffle(lu.c1.w, 12) * x.c1.w;
        s -= shuffle(lu.c2.x, 12) * x.c2.x;
        s -= shuffle(lu.c2.y, 12) * x.c2.y;
        s -= shuffle(lu.c2.z, 12) * x.c2.z;
        s -= shuffle(lu.c2.w, 12) * x.c2.w;
        x.c3.x = s;
    }
    if 13 < n {
        let mut s = x.c3.y;
        s -= shuffle(lu.c0.x, 13) * x.c0.x;
        s -= shuffle(lu.c0.y, 13) * x.c0.y;
        s -= shuffle(lu.c0.z, 13) * x.c0.z;
        s -= shuffle(lu.c0.w, 13) * x.c0.w;
        s -= shuffle(lu.c1.x, 13) * x.c1.x;
        s -= shuffle(lu.c1.y, 13) * x.c1.y;
        s -= shuffle(lu.c1.z, 13) * x.c1.z;
        s -= shuffle(lu.c1.w, 13) * x.c1.w;
        s -= shuffle(lu.c2.x, 13) * x.c2.x;
        s -= shuffle(lu.c2.y, 13) * x.c2.y;
        s -= shuffle(lu.c2.z, 13) * x.c2.z;
        s -= shuffle(lu.c2.w, 13) * x.c2.w;
        s -= shuffle(lu.c3.x, 13) * x.c3.x;
        x.c3.y = s;
    }
    if 14 < n {
        let mut s = x.c3.z;
        s -= shuffle(lu.c0.x, 14) * x.c0.x;
        s -= shuffle(lu.c0.y, 14) * x.c0.y;
        s -= shuffle(lu.c0.z, 14) * x.c0.z;
        s -= shuffle(lu.c0.w, 14) * x.c0.w;
        s -= shuffle(lu.c1.x, 14) * x.c1.x;
        s -= shuffle(lu.c1.y, 14) * x.c1.y;
        s -= shuffle(lu.c1.z, 14) * x.c1.z;
        s -= shuffle(lu.c1.w, 14) * x.c1.w;
        s -= shuffle(lu.c2.x, 14) * x.c2.x;
        s -= shuffle(lu.c2.y, 14) * x.c2.y;
        s -= shuffle(lu.c2.z, 14) * x.c2.z;
        s -= shuffle(lu.c2.w, 14) * x.c2.w;
        s -= shuffle(lu.c3.x, 14) * x.c3.x;
        s -= shuffle(lu.c3.y, 14) * x.c3.y;
        x.c3.z = s;
    }
    if 15 < n {
        let mut s = x.c3.w;
        s -= shuffle(lu.c0.x, 15) * x.c0.x;
        s -= shuffle(lu.c0.y, 15) * x.c0.y;
        s -= shuffle(lu.c0.z, 15) * x.c0.z;
        s -= shuffle(lu.c0.w, 15) * x.c0.w;
        s -= shuffle(lu.c1.x, 15) * x.c1.x;
        s -= shuffle(lu.c1.y, 15) * x.c1.y;
        s -= shuffle(lu.c1.z, 15) * x.c1.z;
        s -= shuffle(lu.c1.w, 15) * x.c1.w;
        s -= shuffle(lu.c2.x, 15) * x.c2.x;
        s -= shuffle(lu.c2.y, 15) * x.c2.y;
        s -= shuffle(lu.c2.z, 15) * x.c2.z;
        s -= shuffle(lu.c2.w, 15) * x.c2.w;
        s -= shuffle(lu.c3.x, 15) * x.c3.x;
        s -= shuffle(lu.c3.y, 15) * x.c3.y;
        s -= shuffle(lu.c3.z, 15) * x.c3.z;
        x.c3.w = s;
    }
    if 16 < n {
        let mut s = x.c4.x;
        s -= shuffle(lu.c0.x, 16) * x.c0.x;
        s -= shuffle(lu.c0.y, 16) * x.c0.y;
        s -= shuffle(lu.c0.z, 16) * x.c0.z;
        s -= shuffle(lu.c0.w, 16) * x.c0.w;
        s -= shuffle(lu.c1.x, 16) * x.c1.x;
        s -= shuffle(lu.c1.y, 16) * x.c1.y;
        s -= shuffle(lu.c1.z, 16) * x.c1.z;
        s -= shuffle(lu.c1.w, 16) * x.c1.w;
        s -= shuffle(lu.c2.x, 16) * x.c2.x;
        s -= shuffle(lu.c2.y, 16) * x.c2.y;
        s -= shuffle(lu.c2.z, 16) * x.c2.z;
        s -= shuffle(lu.c2.w, 16) * x.c2.w;
        s -= shuffle(lu.c3.x, 16) * x.c3.x;
        s -= shuffle(lu.c3.y, 16) * x.c3.y;
        s -= shuffle(lu.c3.z, 16) * x.c3.z;
        s -= shuffle(lu.c3.w, 16) * x.c3.w;
        x.c4.x = s;
    }
    if 17 < n {
        let mut s = x.c4.y;
        s -= shuffle(lu.c0.x, 17) * x.c0.x;
        s -= shuffle(lu.c0.y, 17) * x.c0.y;
        s -= shuffle(lu.c0.z, 17) * x.c0.z;
        s -= shuffle(lu.c0.w, 17) * x.c0.w;
        s -= shuffle(lu.c1.x, 17) * x.c1.x;
        s -= shuffle(lu.c1.y, 17) * x.c1.y;
        s -= shuffle(lu.c1.z, 17) * x.c1.z;
        s -= shuffle(lu.c1.w, 17) * x.c1.w;
        s -= shuffle(lu.c2.x, 17) * x.c2.x;
        s -= shuffle(lu.c2.y, 17) * x.c2.y;
        s -= shuffle(lu.c2.z, 17) * x.c2.z;
        s -= shuffle(lu.c2.w, 17) * x.c2.w;
        s -= shuffle(lu.c3.x, 17) * x.c3.x;
        s -= shuffle(lu.c3.y, 17) * x.c3.y;
        s -= shuffle(lu.c3.z, 17) * x.c3.z;
        s -= shuffle(lu.c3.w, 17) * x.c3.w;
        s -= shuffle(lu.c4.x, 17) * x.c4.x;
        x.c4.y = s;
    }
    if 18 < n {
        let mut s = x.c4.z;
        s -= shuffle(lu.c0.x, 18) * x.c0.x;
        s -= shuffle(lu.c0.y, 18) * x.c0.y;
        s -= shuffle(lu.c0.z, 18) * x.c0.z;
        s -= shuffle(lu.c0.w, 18) * x.c0.w;
        s -= shuffle(lu.c1.x, 18) * x.c1.x;
        s -= shuffle(lu.c1.y, 18) * x.c1.y;
        s -= shuffle(lu.c1.z, 18) * x.c1.z;
        s -= shuffle(lu.c1.w, 18) * x.c1.w;
        s -= shuffle(lu.c2.x, 18) * x.c2.x;
        s -= shuffle(lu.c2.y, 18) * x.c2.y;
        s -= shuffle(lu.c2.z, 18) * x.c2.z;
        s -= shuffle(lu.c2.w, 18) * x.c2.w;
        s -= shuffle(lu.c3.x, 18) * x.c3.x;
        s -= shuffle(lu.c3.y, 18) * x.c3.y;
        s -= shuffle(lu.c3.z, 18) * x.c3.z;
        s -= shuffle(lu.c3.w, 18) * x.c3.w;
        s -= shuffle(lu.c4.x, 18) * x.c4.x;
        s -= shuffle(lu.c4.y, 18) * x.c4.y;
        x.c4.z = s;
    }
    if 19 < n {
        let mut s = x.c4.w;
        s -= shuffle(lu.c0.x, 19) * x.c0.x;
        s -= shuffle(lu.c0.y, 19) * x.c0.y;
        s -= shuffle(lu.c0.z, 19) * x.c0.z;
        s -= shuffle(lu.c0.w, 19) * x.c0.w;
        s -= shuffle(lu.c1.x, 19) * x.c1.x;
        s -= shuffle(lu.c1.y, 19) * x.c1.y;
        s -= shuffle(lu.c1.z, 19) * x.c1.z;
        s -= shuffle(lu.c1.w, 19) * x.c1.w;
        s -= shuffle(lu.c2.x, 19) * x.c2.x;
        s -= shuffle(lu.c2.y, 19) * x.c2.y;
        s -= shuffle(lu.c2.z, 19) * x.c2.z;
        s -= shuffle(lu.c2.w, 19) * x.c2.w;
        s -= shuffle(lu.c3.x, 19) * x.c3.x;
        s -= shuffle(lu.c3.y, 19) * x.c3.y;
        s -= shuffle(lu.c3.z, 19) * x.c3.z;
        s -= shuffle(lu.c3.w, 19) * x.c3.w;
        s -= shuffle(lu.c4.x, 19) * x.c4.x;
        s -= shuffle(lu.c4.y, 19) * x.c4.y;
        s -= shuffle(lu.c4.z, 19) * x.c4.z;
        x.c4.w = s;
    }
    if 20 < n {
        let mut s = x.c5.x;
        s -= shuffle(lu.c0.x, 20) * x.c0.x;
        s -= shuffle(lu.c0.y, 20) * x.c0.y;
        s -= shuffle(lu.c0.z, 20) * x.c0.z;
        s -= shuffle(lu.c0.w, 20) * x.c0.w;
        s -= shuffle(lu.c1.x, 20) * x.c1.x;
        s -= shuffle(lu.c1.y, 20) * x.c1.y;
        s -= shuffle(lu.c1.z, 20) * x.c1.z;
        s -= shuffle(lu.c1.w, 20) * x.c1.w;
        s -= shuffle(lu.c2.x, 20) * x.c2.x;
        s -= shuffle(lu.c2.y, 20) * x.c2.y;
        s -= shuffle(lu.c2.z, 20) * x.c2.z;
        s -= shuffle(lu.c2.w, 20) * x.c2.w;
        s -= shuffle(lu.c3.x, 20) * x.c3.x;
        s -= shuffle(lu.c3.y, 20) * x.c3.y;
        s -= shuffle(lu.c3.z, 20) * x.c3.z;
        s -= shuffle(lu.c3.w, 20) * x.c3.w;
        s -= shuffle(lu.c4.x, 20) * x.c4.x;
        s -= shuffle(lu.c4.y, 20) * x.c4.y;
        s -= shuffle(lu.c4.z, 20) * x.c4.z;
        s -= shuffle(lu.c4.w, 20) * x.c4.w;
        x.c5.x = s;
    }
    if 21 < n {
        let mut s = x.c5.y;
        s -= shuffle(lu.c0.x, 21) * x.c0.x;
        s -= shuffle(lu.c0.y, 21) * x.c0.y;
        s -= shuffle(lu.c0.z, 21) * x.c0.z;
        s -= shuffle(lu.c0.w, 21) * x.c0.w;
        s -= shuffle(lu.c1.x, 21) * x.c1.x;
        s -= shuffle(lu.c1.y, 21) * x.c1.y;
        s -= shuffle(lu.c1.z, 21) * x.c1.z;
        s -= shuffle(lu.c1.w, 21) * x.c1.w;
        s -= shuffle(lu.c2.x, 21) * x.c2.x;
        s -= shuffle(lu.c2.y, 21) * x.c2.y;
        s -= shuffle(lu.c2.z, 21) * x.c2.z;
        s -= shuffle(lu.c2.w, 21) * x.c2.w;
        s -= shuffle(lu.c3.x, 21) * x.c3.x;
        s -= shuffle(lu.c3.y, 21) * x.c3.y;
        s -= shuffle(lu.c3.z, 21) * x.c3.z;
        s -= shuffle(lu.c3.w, 21) * x.c3.w;
        s -= shuffle(lu.c4.x, 21) * x.c4.x;
        s -= shuffle(lu.c4.y, 21) * x.c4.y;
        s -= shuffle(lu.c4.z, 21) * x.c4.z;
        s -= shuffle(lu.c4.w, 21) * x.c4.w;
        s -= shuffle(lu.c5.x, 21) * x.c5.x;
        x.c5.y = s;
    }
    if 22 < n {
        let mut s = x.c5.z;
        s -= shuffle(lu.c0.x, 22) * x.c0.x;
        s -= shuffle(lu.c0.y, 22) * x.c0.y;
        s -= shuffle(lu.c0.z, 22) * x.c0.z;
        s -= shuffle(lu.c0.w, 22) * x.c0.w;
        s -= shuffle(lu.c1.x, 22) * x.c1.x;
        s -= shuffle(lu.c1.y, 22) * x.c1.y;
        s -= shuffle(lu.c1.z, 22) * x.c1.z;
        s -= shuffle(lu.c1.w, 22) * x.c1.w;
        s -= shuffle(lu.c2.x, 22) * x.c2.x;
        s -= shuffle(lu.c2.y, 22) * x.c2.y;
        s -= shuffle(lu.c2.z, 22) * x.c2.z;
        s -= shuffle(lu.c2.w, 22) * x.c2.w;
        s -= shuffle(lu.c3.x, 22) * x.c3.x;
        s -= shuffle(lu.c3.y, 22) * x.c3.y;
        s -= shuffle(lu.c3.z, 22) * x.c3.z;
        s -= shuffle(lu.c3.w, 22) * x.c3.w;
        s -= shuffle(lu.c4.x, 22) * x.c4.x;
        s -= shuffle(lu.c4.y, 22) * x.c4.y;
        s -= shuffle(lu.c4.z, 22) * x.c4.z;
        s -= shuffle(lu.c4.w, 22) * x.c4.w;
        s -= shuffle(lu.c5.x, 22) * x.c5.x;
        s -= shuffle(lu.c5.y, 22) * x.c5.y;
        x.c5.z = s;
    }
    if 23 < n {
        let mut s = x.c5.w;
        s -= shuffle(lu.c0.x, 23) * x.c0.x;
        s -= shuffle(lu.c0.y, 23) * x.c0.y;
        s -= shuffle(lu.c0.z, 23) * x.c0.z;
        s -= shuffle(lu.c0.w, 23) * x.c0.w;
        s -= shuffle(lu.c1.x, 23) * x.c1.x;
        s -= shuffle(lu.c1.y, 23) * x.c1.y;
        s -= shuffle(lu.c1.z, 23) * x.c1.z;
        s -= shuffle(lu.c1.w, 23) * x.c1.w;
        s -= shuffle(lu.c2.x, 23) * x.c2.x;
        s -= shuffle(lu.c2.y, 23) * x.c2.y;
        s -= shuffle(lu.c2.z, 23) * x.c2.z;
        s -= shuffle(lu.c2.w, 23) * x.c2.w;
        s -= shuffle(lu.c3.x, 23) * x.c3.x;
        s -= shuffle(lu.c3.y, 23) * x.c3.y;
        s -= shuffle(lu.c3.z, 23) * x.c3.z;
        s -= shuffle(lu.c3.w, 23) * x.c3.w;
        s -= shuffle(lu.c4.x, 23) * x.c4.x;
        s -= shuffle(lu.c4.y, 23) * x.c4.y;
        s -= shuffle(lu.c4.z, 23) * x.c4.z;
        s -= shuffle(lu.c4.w, 23) * x.c4.w;
        s -= shuffle(lu.c5.x, 23) * x.c5.x;
        s -= shuffle(lu.c5.y, 23) * x.c5.y;
        s -= shuffle(lu.c5.z, 23) * x.c5.z;
        x.c5.w = s;
    }
    if 24 < n {
        let mut s = x.c6.x;
        s -= shuffle(lu.c0.x, 24) * x.c0.x;
        s -= shuffle(lu.c0.y, 24) * x.c0.y;
        s -= shuffle(lu.c0.z, 24) * x.c0.z;
        s -= shuffle(lu.c0.w, 24) * x.c0.w;
        s -= shuffle(lu.c1.x, 24) * x.c1.x;
        s -= shuffle(lu.c1.y, 24) * x.c1.y;
        s -= shuffle(lu.c1.z, 24) * x.c1.z;
        s -= shuffle(lu.c1.w, 24) * x.c1.w;
        s -= shuffle(lu.c2.x, 24) * x.c2.x;
        s -= shuffle(lu.c2.y, 24) * x.c2.y;
        s -= shuffle(lu.c2.z, 24) * x.c2.z;
        s -= shuffle(lu.c2.w, 24) * x.c2.w;
        s -= shuffle(lu.c3.x, 24) * x.c3.x;
        s -= shuffle(lu.c3.y, 24) * x.c3.y;
        s -= shuffle(lu.c3.z, 24) * x.c3.z;
        s -= shuffle(lu.c3.w, 24) * x.c3.w;
        s -= shuffle(lu.c4.x, 24) * x.c4.x;
        s -= shuffle(lu.c4.y, 24) * x.c4.y;
        s -= shuffle(lu.c4.z, 24) * x.c4.z;
        s -= shuffle(lu.c4.w, 24) * x.c4.w;
        s -= shuffle(lu.c5.x, 24) * x.c5.x;
        s -= shuffle(lu.c5.y, 24) * x.c5.y;
        s -= shuffle(lu.c5.z, 24) * x.c5.z;
        s -= shuffle(lu.c5.w, 24) * x.c5.w;
        x.c6.x = s;
    }
    if 25 < n {
        let mut s = x.c6.y;
        s -= shuffle(lu.c0.x, 25) * x.c0.x;
        s -= shuffle(lu.c0.y, 25) * x.c0.y;
        s -= shuffle(lu.c0.z, 25) * x.c0.z;
        s -= shuffle(lu.c0.w, 25) * x.c0.w;
        s -= shuffle(lu.c1.x, 25) * x.c1.x;
        s -= shuffle(lu.c1.y, 25) * x.c1.y;
        s -= shuffle(lu.c1.z, 25) * x.c1.z;
        s -= shuffle(lu.c1.w, 25) * x.c1.w;
        s -= shuffle(lu.c2.x, 25) * x.c2.x;
        s -= shuffle(lu.c2.y, 25) * x.c2.y;
        s -= shuffle(lu.c2.z, 25) * x.c2.z;
        s -= shuffle(lu.c2.w, 25) * x.c2.w;
        s -= shuffle(lu.c3.x, 25) * x.c3.x;
        s -= shuffle(lu.c3.y, 25) * x.c3.y;
        s -= shuffle(lu.c3.z, 25) * x.c3.z;
        s -= shuffle(lu.c3.w, 25) * x.c3.w;
        s -= shuffle(lu.c4.x, 25) * x.c4.x;
        s -= shuffle(lu.c4.y, 25) * x.c4.y;
        s -= shuffle(lu.c4.z, 25) * x.c4.z;
        s -= shuffle(lu.c4.w, 25) * x.c4.w;
        s -= shuffle(lu.c5.x, 25) * x.c5.x;
        s -= shuffle(lu.c5.y, 25) * x.c5.y;
        s -= shuffle(lu.c5.z, 25) * x.c5.z;
        s -= shuffle(lu.c5.w, 25) * x.c5.w;
        s -= shuffle(lu.c6.x, 25) * x.c6.x;
        x.c6.y = s;
    }
    if 26 < n {
        let mut s = x.c6.z;
        s -= shuffle(lu.c0.x, 26) * x.c0.x;
        s -= shuffle(lu.c0.y, 26) * x.c0.y;
        s -= shuffle(lu.c0.z, 26) * x.c0.z;
        s -= shuffle(lu.c0.w, 26) * x.c0.w;
        s -= shuffle(lu.c1.x, 26) * x.c1.x;
        s -= shuffle(lu.c1.y, 26) * x.c1.y;
        s -= shuffle(lu.c1.z, 26) * x.c1.z;
        s -= shuffle(lu.c1.w, 26) * x.c1.w;
        s -= shuffle(lu.c2.x, 26) * x.c2.x;
        s -= shuffle(lu.c2.y, 26) * x.c2.y;
        s -= shuffle(lu.c2.z, 26) * x.c2.z;
        s -= shuffle(lu.c2.w, 26) * x.c2.w;
        s -= shuffle(lu.c3.x, 26) * x.c3.x;
        s -= shuffle(lu.c3.y, 26) * x.c3.y;
        s -= shuffle(lu.c3.z, 26) * x.c3.z;
        s -= shuffle(lu.c3.w, 26) * x.c3.w;
        s -= shuffle(lu.c4.x, 26) * x.c4.x;
        s -= shuffle(lu.c4.y, 26) * x.c4.y;
        s -= shuffle(lu.c4.z, 26) * x.c4.z;
        s -= shuffle(lu.c4.w, 26) * x.c4.w;
        s -= shuffle(lu.c5.x, 26) * x.c5.x;
        s -= shuffle(lu.c5.y, 26) * x.c5.y;
        s -= shuffle(lu.c5.z, 26) * x.c5.z;
        s -= shuffle(lu.c5.w, 26) * x.c5.w;
        s -= shuffle(lu.c6.x, 26) * x.c6.x;
        s -= shuffle(lu.c6.y, 26) * x.c6.y;
        x.c6.z = s;
    }
    if 27 < n {
        let mut s = x.c6.w;
        s -= shuffle(lu.c0.x, 27) * x.c0.x;
        s -= shuffle(lu.c0.y, 27) * x.c0.y;
        s -= shuffle(lu.c0.z, 27) * x.c0.z;
        s -= shuffle(lu.c0.w, 27) * x.c0.w;
        s -= shuffle(lu.c1.x, 27) * x.c1.x;
        s -= shuffle(lu.c1.y, 27) * x.c1.y;
        s -= shuffle(lu.c1.z, 27) * x.c1.z;
        s -= shuffle(lu.c1.w, 27) * x.c1.w;
        s -= shuffle(lu.c2.x, 27) * x.c2.x;
        s -= shuffle(lu.c2.y, 27) * x.c2.y;
        s -= shuffle(lu.c2.z, 27) * x.c2.z;
        s -= shuffle(lu.c2.w, 27) * x.c2.w;
        s -= shuffle(lu.c3.x, 27) * x.c3.x;
        s -= shuffle(lu.c3.y, 27) * x.c3.y;
        s -= shuffle(lu.c3.z, 27) * x.c3.z;
        s -= shuffle(lu.c3.w, 27) * x.c3.w;
        s -= shuffle(lu.c4.x, 27) * x.c4.x;
        s -= shuffle(lu.c4.y, 27) * x.c4.y;
        s -= shuffle(lu.c4.z, 27) * x.c4.z;
        s -= shuffle(lu.c4.w, 27) * x.c4.w;
        s -= shuffle(lu.c5.x, 27) * x.c5.x;
        s -= shuffle(lu.c5.y, 27) * x.c5.y;
        s -= shuffle(lu.c5.z, 27) * x.c5.z;
        s -= shuffle(lu.c5.w, 27) * x.c5.w;
        s -= shuffle(lu.c6.x, 27) * x.c6.x;
        s -= shuffle(lu.c6.y, 27) * x.c6.y;
        s -= shuffle(lu.c6.z, 27) * x.c6.z;
        x.c6.w = s;
    }
    if 28 < n {
        let mut s = x.c7.x;
        s -= shuffle(lu.c0.x, 28) * x.c0.x;
        s -= shuffle(lu.c0.y, 28) * x.c0.y;
        s -= shuffle(lu.c0.z, 28) * x.c0.z;
        s -= shuffle(lu.c0.w, 28) * x.c0.w;
        s -= shuffle(lu.c1.x, 28) * x.c1.x;
        s -= shuffle(lu.c1.y, 28) * x.c1.y;
        s -= shuffle(lu.c1.z, 28) * x.c1.z;
        s -= shuffle(lu.c1.w, 28) * x.c1.w;
        s -= shuffle(lu.c2.x, 28) * x.c2.x;
        s -= shuffle(lu.c2.y, 28) * x.c2.y;
        s -= shuffle(lu.c2.z, 28) * x.c2.z;
        s -= shuffle(lu.c2.w, 28) * x.c2.w;
        s -= shuffle(lu.c3.x, 28) * x.c3.x;
        s -= shuffle(lu.c3.y, 28) * x.c3.y;
        s -= shuffle(lu.c3.z, 28) * x.c3.z;
        s -= shuffle(lu.c3.w, 28) * x.c3.w;
        s -= shuffle(lu.c4.x, 28) * x.c4.x;
        s -= shuffle(lu.c4.y, 28) * x.c4.y;
        s -= shuffle(lu.c4.z, 28) * x.c4.z;
        s -= shuffle(lu.c4.w, 28) * x.c4.w;
        s -= shuffle(lu.c5.x, 28) * x.c5.x;
        s -= shuffle(lu.c5.y, 28) * x.c5.y;
        s -= shuffle(lu.c5.z, 28) * x.c5.z;
        s -= shuffle(lu.c5.w, 28) * x.c5.w;
        s -= shuffle(lu.c6.x, 28) * x.c6.x;
        s -= shuffle(lu.c6.y, 28) * x.c6.y;
        s -= shuffle(lu.c6.z, 28) * x.c6.z;
        s -= shuffle(lu.c6.w, 28) * x.c6.w;
        x.c7.x = s;
    }
    if 29 < n {
        let mut s = x.c7.y;
        s -= shuffle(lu.c0.x, 29) * x.c0.x;
        s -= shuffle(lu.c0.y, 29) * x.c0.y;
        s -= shuffle(lu.c0.z, 29) * x.c0.z;
        s -= shuffle(lu.c0.w, 29) * x.c0.w;
        s -= shuffle(lu.c1.x, 29) * x.c1.x;
        s -= shuffle(lu.c1.y, 29) * x.c1.y;
        s -= shuffle(lu.c1.z, 29) * x.c1.z;
        s -= shuffle(lu.c1.w, 29) * x.c1.w;
        s -= shuffle(lu.c2.x, 29) * x.c2.x;
        s -= shuffle(lu.c2.y, 29) * x.c2.y;
        s -= shuffle(lu.c2.z, 29) * x.c2.z;
        s -= shuffle(lu.c2.w, 29) * x.c2.w;
        s -= shuffle(lu.c3.x, 29) * x.c3.x;
        s -= shuffle(lu.c3.y, 29) * x.c3.y;
        s -= shuffle(lu.c3.z, 29) * x.c3.z;
        s -= shuffle(lu.c3.w, 29) * x.c3.w;
        s -= shuffle(lu.c4.x, 29) * x.c4.x;
        s -= shuffle(lu.c4.y, 29) * x.c4.y;
        s -= shuffle(lu.c4.z, 29) * x.c4.z;
        s -= shuffle(lu.c4.w, 29) * x.c4.w;
        s -= shuffle(lu.c5.x, 29) * x.c5.x;
        s -= shuffle(lu.c5.y, 29) * x.c5.y;
        s -= shuffle(lu.c5.z, 29) * x.c5.z;
        s -= shuffle(lu.c5.w, 29) * x.c5.w;
        s -= shuffle(lu.c6.x, 29) * x.c6.x;
        s -= shuffle(lu.c6.y, 29) * x.c6.y;
        s -= shuffle(lu.c6.z, 29) * x.c6.z;
        s -= shuffle(lu.c6.w, 29) * x.c6.w;
        s -= shuffle(lu.c7.x, 29) * x.c7.x;
        x.c7.y = s;
    }
    if 30 < n {
        let mut s = x.c7.z;
        s -= shuffle(lu.c0.x, 30) * x.c0.x;
        s -= shuffle(lu.c0.y, 30) * x.c0.y;
        s -= shuffle(lu.c0.z, 30) * x.c0.z;
        s -= shuffle(lu.c0.w, 30) * x.c0.w;
        s -= shuffle(lu.c1.x, 30) * x.c1.x;
        s -= shuffle(lu.c1.y, 30) * x.c1.y;
        s -= shuffle(lu.c1.z, 30) * x.c1.z;
        s -= shuffle(lu.c1.w, 30) * x.c1.w;
        s -= shuffle(lu.c2.x, 30) * x.c2.x;
        s -= shuffle(lu.c2.y, 30) * x.c2.y;
        s -= shuffle(lu.c2.z, 30) * x.c2.z;
        s -= shuffle(lu.c2.w, 30) * x.c2.w;
        s -= shuffle(lu.c3.x, 30) * x.c3.x;
        s -= shuffle(lu.c3.y, 30) * x.c3.y;
        s -= shuffle(lu.c3.z, 30) * x.c3.z;
        s -= shuffle(lu.c3.w, 30) * x.c3.w;
        s -= shuffle(lu.c4.x, 30) * x.c4.x;
        s -= shuffle(lu.c4.y, 30) * x.c4.y;
        s -= shuffle(lu.c4.z, 30) * x.c4.z;
        s -= shuffle(lu.c4.w, 30) * x.c4.w;
        s -= shuffle(lu.c5.x, 30) * x.c5.x;
        s -= shuffle(lu.c5.y, 30) * x.c5.y;
        s -= shuffle(lu.c5.z, 30) * x.c5.z;
        s -= shuffle(lu.c5.w, 30) * x.c5.w;
        s -= shuffle(lu.c6.x, 30) * x.c6.x;
        s -= shuffle(lu.c6.y, 30) * x.c6.y;
        s -= shuffle(lu.c6.z, 30) * x.c6.z;
        s -= shuffle(lu.c6.w, 30) * x.c6.w;
        s -= shuffle(lu.c7.x, 30) * x.c7.x;
        s -= shuffle(lu.c7.y, 30) * x.c7.y;
        x.c7.z = s;
    }
    if 31 < n {
        let mut s = x.c7.w;
        s -= shuffle(lu.c0.x, 31) * x.c0.x;
        s -= shuffle(lu.c0.y, 31) * x.c0.y;
        s -= shuffle(lu.c0.z, 31) * x.c0.z;
        s -= shuffle(lu.c0.w, 31) * x.c0.w;
        s -= shuffle(lu.c1.x, 31) * x.c1.x;
        s -= shuffle(lu.c1.y, 31) * x.c1.y;
        s -= shuffle(lu.c1.z, 31) * x.c1.z;
        s -= shuffle(lu.c1.w, 31) * x.c1.w;
        s -= shuffle(lu.c2.x, 31) * x.c2.x;
        s -= shuffle(lu.c2.y, 31) * x.c2.y;
        s -= shuffle(lu.c2.z, 31) * x.c2.z;
        s -= shuffle(lu.c2.w, 31) * x.c2.w;
        s -= shuffle(lu.c3.x, 31) * x.c3.x;
        s -= shuffle(lu.c3.y, 31) * x.c3.y;
        s -= shuffle(lu.c3.z, 31) * x.c3.z;
        s -= shuffle(lu.c3.w, 31) * x.c3.w;
        s -= shuffle(lu.c4.x, 31) * x.c4.x;
        s -= shuffle(lu.c4.y, 31) * x.c4.y;
        s -= shuffle(lu.c4.z, 31) * x.c4.z;
        s -= shuffle(lu.c4.w, 31) * x.c4.w;
        s -= shuffle(lu.c5.x, 31) * x.c5.x;
        s -= shuffle(lu.c5.y, 31) * x.c5.y;
        s -= shuffle(lu.c5.z, 31) * x.c5.z;
        s -= shuffle(lu.c5.w, 31) * x.c5.w;
        s -= shuffle(lu.c6.x, 31) * x.c6.x;
        s -= shuffle(lu.c6.y, 31) * x.c6.y;
        s -= shuffle(lu.c6.z, 31) * x.c6.z;
        s -= shuffle(lu.c6.w, 31) * x.c6.w;
        s -= shuffle(lu.c7.x, 31) * x.c7.x;
        s -= shuffle(lu.c7.y, 31) * x.c7.y;
        s -= shuffle(lu.c7.z, 31) * x.c7.z;
        x.c7.w = s;
    }
    if 31 < n {
        let s = x.c7.w;
        let u = shuffle(lu.c7.w, 31);
        x.c7.w = if u != 0.0 { s / u } else { 0.0 };
    }
    if 30 < n {
        let mut s = x.c7.z;
        s -= shuffle(lu.c7.w, 30) * x.c7.w;
        let u = shuffle(lu.c7.z, 30);
        x.c7.z = if u != 0.0 { s / u } else { 0.0 };
    }
    if 29 < n {
        let mut s = x.c7.y;
        s -= shuffle(lu.c7.z, 29) * x.c7.z;
        s -= shuffle(lu.c7.w, 29) * x.c7.w;
        let u = shuffle(lu.c7.y, 29);
        x.c7.y = if u != 0.0 { s / u } else { 0.0 };
    }
    if 28 < n {
        let mut s = x.c7.x;
        s -= shuffle(lu.c7.y, 28) * x.c7.y;
        s -= shuffle(lu.c7.z, 28) * x.c7.z;
        s -= shuffle(lu.c7.w, 28) * x.c7.w;
        let u = shuffle(lu.c7.x, 28);
        x.c7.x = if u != 0.0 { s / u } else { 0.0 };
    }
    if 27 < n {
        let mut s = x.c6.w;
        s -= shuffle(lu.c7.x, 27) * x.c7.x;
        s -= shuffle(lu.c7.y, 27) * x.c7.y;
        s -= shuffle(lu.c7.z, 27) * x.c7.z;
        s -= shuffle(lu.c7.w, 27) * x.c7.w;
        let u = shuffle(lu.c6.w, 27);
        x.c6.w = if u != 0.0 { s / u } else { 0.0 };
    }
    if 26 < n {
        let mut s = x.c6.z;
        s -= shuffle(lu.c6.w, 26) * x.c6.w;
        s -= shuffle(lu.c7.x, 26) * x.c7.x;
        s -= shuffle(lu.c7.y, 26) * x.c7.y;
        s -= shuffle(lu.c7.z, 26) * x.c7.z;
        s -= shuffle(lu.c7.w, 26) * x.c7.w;
        let u = shuffle(lu.c6.z, 26);
        x.c6.z = if u != 0.0 { s / u } else { 0.0 };
    }
    if 25 < n {
        let mut s = x.c6.y;
        s -= shuffle(lu.c6.z, 25) * x.c6.z;
        s -= shuffle(lu.c6.w, 25) * x.c6.w;
        s -= shuffle(lu.c7.x, 25) * x.c7.x;
        s -= shuffle(lu.c7.y, 25) * x.c7.y;
        s -= shuffle(lu.c7.z, 25) * x.c7.z;
        s -= shuffle(lu.c7.w, 25) * x.c7.w;
        let u = shuffle(lu.c6.y, 25);
        x.c6.y = if u != 0.0 { s / u } else { 0.0 };
    }
    if 24 < n {
        let mut s = x.c6.x;
        s -= shuffle(lu.c6.y, 24) * x.c6.y;
        s -= shuffle(lu.c6.z, 24) * x.c6.z;
        s -= shuffle(lu.c6.w, 24) * x.c6.w;
        s -= shuffle(lu.c7.x, 24) * x.c7.x;
        s -= shuffle(lu.c7.y, 24) * x.c7.y;
        s -= shuffle(lu.c7.z, 24) * x.c7.z;
        s -= shuffle(lu.c7.w, 24) * x.c7.w;
        let u = shuffle(lu.c6.x, 24);
        x.c6.x = if u != 0.0 { s / u } else { 0.0 };
    }
    if 23 < n {
        let mut s = x.c5.w;
        s -= shuffle(lu.c6.x, 23) * x.c6.x;
        s -= shuffle(lu.c6.y, 23) * x.c6.y;
        s -= shuffle(lu.c6.z, 23) * x.c6.z;
        s -= shuffle(lu.c6.w, 23) * x.c6.w;
        s -= shuffle(lu.c7.x, 23) * x.c7.x;
        s -= shuffle(lu.c7.y, 23) * x.c7.y;
        s -= shuffle(lu.c7.z, 23) * x.c7.z;
        s -= shuffle(lu.c7.w, 23) * x.c7.w;
        let u = shuffle(lu.c5.w, 23);
        x.c5.w = if u != 0.0 { s / u } else { 0.0 };
    }
    if 22 < n {
        let mut s = x.c5.z;
        s -= shuffle(lu.c5.w, 22) * x.c5.w;
        s -= shuffle(lu.c6.x, 22) * x.c6.x;
        s -= shuffle(lu.c6.y, 22) * x.c6.y;
        s -= shuffle(lu.c6.z, 22) * x.c6.z;
        s -= shuffle(lu.c6.w, 22) * x.c6.w;
        s -= shuffle(lu.c7.x, 22) * x.c7.x;
        s -= shuffle(lu.c7.y, 22) * x.c7.y;
        s -= shuffle(lu.c7.z, 22) * x.c7.z;
        s -= shuffle(lu.c7.w, 22) * x.c7.w;
        let u = shuffle(lu.c5.z, 22);
        x.c5.z = if u != 0.0 { s / u } else { 0.0 };
    }
    if 21 < n {
        let mut s = x.c5.y;
        s -= shuffle(lu.c5.z, 21) * x.c5.z;
        s -= shuffle(lu.c5.w, 21) * x.c5.w;
        s -= shuffle(lu.c6.x, 21) * x.c6.x;
        s -= shuffle(lu.c6.y, 21) * x.c6.y;
        s -= shuffle(lu.c6.z, 21) * x.c6.z;
        s -= shuffle(lu.c6.w, 21) * x.c6.w;
        s -= shuffle(lu.c7.x, 21) * x.c7.x;
        s -= shuffle(lu.c7.y, 21) * x.c7.y;
        s -= shuffle(lu.c7.z, 21) * x.c7.z;
        s -= shuffle(lu.c7.w, 21) * x.c7.w;
        let u = shuffle(lu.c5.y, 21);
        x.c5.y = if u != 0.0 { s / u } else { 0.0 };
    }
    if 20 < n {
        let mut s = x.c5.x;
        s -= shuffle(lu.c5.y, 20) * x.c5.y;
        s -= shuffle(lu.c5.z, 20) * x.c5.z;
        s -= shuffle(lu.c5.w, 20) * x.c5.w;
        s -= shuffle(lu.c6.x, 20) * x.c6.x;
        s -= shuffle(lu.c6.y, 20) * x.c6.y;
        s -= shuffle(lu.c6.z, 20) * x.c6.z;
        s -= shuffle(lu.c6.w, 20) * x.c6.w;
        s -= shuffle(lu.c7.x, 20) * x.c7.x;
        s -= shuffle(lu.c7.y, 20) * x.c7.y;
        s -= shuffle(lu.c7.z, 20) * x.c7.z;
        s -= shuffle(lu.c7.w, 20) * x.c7.w;
        let u = shuffle(lu.c5.x, 20);
        x.c5.x = if u != 0.0 { s / u } else { 0.0 };
    }
    if 19 < n {
        let mut s = x.c4.w;
        s -= shuffle(lu.c5.x, 19) * x.c5.x;
        s -= shuffle(lu.c5.y, 19) * x.c5.y;
        s -= shuffle(lu.c5.z, 19) * x.c5.z;
        s -= shuffle(lu.c5.w, 19) * x.c5.w;
        s -= shuffle(lu.c6.x, 19) * x.c6.x;
        s -= shuffle(lu.c6.y, 19) * x.c6.y;
        s -= shuffle(lu.c6.z, 19) * x.c6.z;
        s -= shuffle(lu.c6.w, 19) * x.c6.w;
        s -= shuffle(lu.c7.x, 19) * x.c7.x;
        s -= shuffle(lu.c7.y, 19) * x.c7.y;
        s -= shuffle(lu.c7.z, 19) * x.c7.z;
        s -= shuffle(lu.c7.w, 19) * x.c7.w;
        let u = shuffle(lu.c4.w, 19);
        x.c4.w = if u != 0.0 { s / u } else { 0.0 };
    }
    if 18 < n {
        let mut s = x.c4.z;
        s -= shuffle(lu.c4.w, 18) * x.c4.w;
        s -= shuffle(lu.c5.x, 18) * x.c5.x;
        s -= shuffle(lu.c5.y, 18) * x.c5.y;
        s -= shuffle(lu.c5.z, 18) * x.c5.z;
        s -= shuffle(lu.c5.w, 18) * x.c5.w;
        s -= shuffle(lu.c6.x, 18) * x.c6.x;
        s -= shuffle(lu.c6.y, 18) * x.c6.y;
        s -= shuffle(lu.c6.z, 18) * x.c6.z;
        s -= shuffle(lu.c6.w, 18) * x.c6.w;
        s -= shuffle(lu.c7.x, 18) * x.c7.x;
        s -= shuffle(lu.c7.y, 18) * x.c7.y;
        s -= shuffle(lu.c7.z, 18) * x.c7.z;
        s -= shuffle(lu.c7.w, 18) * x.c7.w;
        let u = shuffle(lu.c4.z, 18);
        x.c4.z = if u != 0.0 { s / u } else { 0.0 };
    }
    if 17 < n {
        let mut s = x.c4.y;
        s -= shuffle(lu.c4.z, 17) * x.c4.z;
        s -= shuffle(lu.c4.w, 17) * x.c4.w;
        s -= shuffle(lu.c5.x, 17) * x.c5.x;
        s -= shuffle(lu.c5.y, 17) * x.c5.y;
        s -= shuffle(lu.c5.z, 17) * x.c5.z;
        s -= shuffle(lu.c5.w, 17) * x.c5.w;
        s -= shuffle(lu.c6.x, 17) * x.c6.x;
        s -= shuffle(lu.c6.y, 17) * x.c6.y;
        s -= shuffle(lu.c6.z, 17) * x.c6.z;
        s -= shuffle(lu.c6.w, 17) * x.c6.w;
        s -= shuffle(lu.c7.x, 17) * x.c7.x;
        s -= shuffle(lu.c7.y, 17) * x.c7.y;
        s -= shuffle(lu.c7.z, 17) * x.c7.z;
        s -= shuffle(lu.c7.w, 17) * x.c7.w;
        let u = shuffle(lu.c4.y, 17);
        x.c4.y = if u != 0.0 { s / u } else { 0.0 };
    }
    if 16 < n {
        let mut s = x.c4.x;
        s -= shuffle(lu.c4.y, 16) * x.c4.y;
        s -= shuffle(lu.c4.z, 16) * x.c4.z;
        s -= shuffle(lu.c4.w, 16) * x.c4.w;
        s -= shuffle(lu.c5.x, 16) * x.c5.x;
        s -= shuffle(lu.c5.y, 16) * x.c5.y;
        s -= shuffle(lu.c5.z, 16) * x.c5.z;
        s -= shuffle(lu.c5.w, 16) * x.c5.w;
        s -= shuffle(lu.c6.x, 16) * x.c6.x;
        s -= shuffle(lu.c6.y, 16) * x.c6.y;
        s -= shuffle(lu.c6.z, 16) * x.c6.z;
        s -= shuffle(lu.c6.w, 16) * x.c6.w;
        s -= shuffle(lu.c7.x, 16) * x.c7.x;
        s -= shuffle(lu.c7.y, 16) * x.c7.y;
        s -= shuffle(lu.c7.z, 16) * x.c7.z;
        s -= shuffle(lu.c7.w, 16) * x.c7.w;
        let u = shuffle(lu.c4.x, 16);
        x.c4.x = if u != 0.0 { s / u } else { 0.0 };
    }
    if 15 < n {
        let mut s = x.c3.w;
        s -= shuffle(lu.c4.x, 15) * x.c4.x;
        s -= shuffle(lu.c4.y, 15) * x.c4.y;
        s -= shuffle(lu.c4.z, 15) * x.c4.z;
        s -= shuffle(lu.c4.w, 15) * x.c4.w;
        s -= shuffle(lu.c5.x, 15) * x.c5.x;
        s -= shuffle(lu.c5.y, 15) * x.c5.y;
        s -= shuffle(lu.c5.z, 15) * x.c5.z;
        s -= shuffle(lu.c5.w, 15) * x.c5.w;
        s -= shuffle(lu.c6.x, 15) * x.c6.x;
        s -= shuffle(lu.c6.y, 15) * x.c6.y;
        s -= shuffle(lu.c6.z, 15) * x.c6.z;
        s -= shuffle(lu.c6.w, 15) * x.c6.w;
        s -= shuffle(lu.c7.x, 15) * x.c7.x;
        s -= shuffle(lu.c7.y, 15) * x.c7.y;
        s -= shuffle(lu.c7.z, 15) * x.c7.z;
        s -= shuffle(lu.c7.w, 15) * x.c7.w;
        let u = shuffle(lu.c3.w, 15);
        x.c3.w = if u != 0.0 { s / u } else { 0.0 };
    }
    if 14 < n {
        let mut s = x.c3.z;
        s -= shuffle(lu.c3.w, 14) * x.c3.w;
        s -= shuffle(lu.c4.x, 14) * x.c4.x;
        s -= shuffle(lu.c4.y, 14) * x.c4.y;
        s -= shuffle(lu.c4.z, 14) * x.c4.z;
        s -= shuffle(lu.c4.w, 14) * x.c4.w;
        s -= shuffle(lu.c5.x, 14) * x.c5.x;
        s -= shuffle(lu.c5.y, 14) * x.c5.y;
        s -= shuffle(lu.c5.z, 14) * x.c5.z;
        s -= shuffle(lu.c5.w, 14) * x.c5.w;
        s -= shuffle(lu.c6.x, 14) * x.c6.x;
        s -= shuffle(lu.c6.y, 14) * x.c6.y;
        s -= shuffle(lu.c6.z, 14) * x.c6.z;
        s -= shuffle(lu.c6.w, 14) * x.c6.w;
        s -= shuffle(lu.c7.x, 14) * x.c7.x;
        s -= shuffle(lu.c7.y, 14) * x.c7.y;
        s -= shuffle(lu.c7.z, 14) * x.c7.z;
        s -= shuffle(lu.c7.w, 14) * x.c7.w;
        let u = shuffle(lu.c3.z, 14);
        x.c3.z = if u != 0.0 { s / u } else { 0.0 };
    }
    if 13 < n {
        let mut s = x.c3.y;
        s -= shuffle(lu.c3.z, 13) * x.c3.z;
        s -= shuffle(lu.c3.w, 13) * x.c3.w;
        s -= shuffle(lu.c4.x, 13) * x.c4.x;
        s -= shuffle(lu.c4.y, 13) * x.c4.y;
        s -= shuffle(lu.c4.z, 13) * x.c4.z;
        s -= shuffle(lu.c4.w, 13) * x.c4.w;
        s -= shuffle(lu.c5.x, 13) * x.c5.x;
        s -= shuffle(lu.c5.y, 13) * x.c5.y;
        s -= shuffle(lu.c5.z, 13) * x.c5.z;
        s -= shuffle(lu.c5.w, 13) * x.c5.w;
        s -= shuffle(lu.c6.x, 13) * x.c6.x;
        s -= shuffle(lu.c6.y, 13) * x.c6.y;
        s -= shuffle(lu.c6.z, 13) * x.c6.z;
        s -= shuffle(lu.c6.w, 13) * x.c6.w;
        s -= shuffle(lu.c7.x, 13) * x.c7.x;
        s -= shuffle(lu.c7.y, 13) * x.c7.y;
        s -= shuffle(lu.c7.z, 13) * x.c7.z;
        s -= shuffle(lu.c7.w, 13) * x.c7.w;
        let u = shuffle(lu.c3.y, 13);
        x.c3.y = if u != 0.0 { s / u } else { 0.0 };
    }
    if 12 < n {
        let mut s = x.c3.x;
        s -= shuffle(lu.c3.y, 12) * x.c3.y;
        s -= shuffle(lu.c3.z, 12) * x.c3.z;
        s -= shuffle(lu.c3.w, 12) * x.c3.w;
        s -= shuffle(lu.c4.x, 12) * x.c4.x;
        s -= shuffle(lu.c4.y, 12) * x.c4.y;
        s -= shuffle(lu.c4.z, 12) * x.c4.z;
        s -= shuffle(lu.c4.w, 12) * x.c4.w;
        s -= shuffle(lu.c5.x, 12) * x.c5.x;
        s -= shuffle(lu.c5.y, 12) * x.c5.y;
        s -= shuffle(lu.c5.z, 12) * x.c5.z;
        s -= shuffle(lu.c5.w, 12) * x.c5.w;
        s -= shuffle(lu.c6.x, 12) * x.c6.x;
        s -= shuffle(lu.c6.y, 12) * x.c6.y;
        s -= shuffle(lu.c6.z, 12) * x.c6.z;
        s -= shuffle(lu.c6.w, 12) * x.c6.w;
        s -= shuffle(lu.c7.x, 12) * x.c7.x;
        s -= shuffle(lu.c7.y, 12) * x.c7.y;
        s -= shuffle(lu.c7.z, 12) * x.c7.z;
        s -= shuffle(lu.c7.w, 12) * x.c7.w;
        let u = shuffle(lu.c3.x, 12);
        x.c3.x = if u != 0.0 { s / u } else { 0.0 };
    }
    if 11 < n {
        let mut s = x.c2.w;
        s -= shuffle(lu.c3.x, 11) * x.c3.x;
        s -= shuffle(lu.c3.y, 11) * x.c3.y;
        s -= shuffle(lu.c3.z, 11) * x.c3.z;
        s -= shuffle(lu.c3.w, 11) * x.c3.w;
        s -= shuffle(lu.c4.x, 11) * x.c4.x;
        s -= shuffle(lu.c4.y, 11) * x.c4.y;
        s -= shuffle(lu.c4.z, 11) * x.c4.z;
        s -= shuffle(lu.c4.w, 11) * x.c4.w;
        s -= shuffle(lu.c5.x, 11) * x.c5.x;
        s -= shuffle(lu.c5.y, 11) * x.c5.y;
        s -= shuffle(lu.c5.z, 11) * x.c5.z;
        s -= shuffle(lu.c5.w, 11) * x.c5.w;
        s -= shuffle(lu.c6.x, 11) * x.c6.x;
        s -= shuffle(lu.c6.y, 11) * x.c6.y;
        s -= shuffle(lu.c6.z, 11) * x.c6.z;
        s -= shuffle(lu.c6.w, 11) * x.c6.w;
        s -= shuffle(lu.c7.x, 11) * x.c7.x;
        s -= shuffle(lu.c7.y, 11) * x.c7.y;
        s -= shuffle(lu.c7.z, 11) * x.c7.z;
        s -= shuffle(lu.c7.w, 11) * x.c7.w;
        let u = shuffle(lu.c2.w, 11);
        x.c2.w = if u != 0.0 { s / u } else { 0.0 };
    }
    if 10 < n {
        let mut s = x.c2.z;
        s -= shuffle(lu.c2.w, 10) * x.c2.w;
        s -= shuffle(lu.c3.x, 10) * x.c3.x;
        s -= shuffle(lu.c3.y, 10) * x.c3.y;
        s -= shuffle(lu.c3.z, 10) * x.c3.z;
        s -= shuffle(lu.c3.w, 10) * x.c3.w;
        s -= shuffle(lu.c4.x, 10) * x.c4.x;
        s -= shuffle(lu.c4.y, 10) * x.c4.y;
        s -= shuffle(lu.c4.z, 10) * x.c4.z;
        s -= shuffle(lu.c4.w, 10) * x.c4.w;
        s -= shuffle(lu.c5.x, 10) * x.c5.x;
        s -= shuffle(lu.c5.y, 10) * x.c5.y;
        s -= shuffle(lu.c5.z, 10) * x.c5.z;
        s -= shuffle(lu.c5.w, 10) * x.c5.w;
        s -= shuffle(lu.c6.x, 10) * x.c6.x;
        s -= shuffle(lu.c6.y, 10) * x.c6.y;
        s -= shuffle(lu.c6.z, 10) * x.c6.z;
        s -= shuffle(lu.c6.w, 10) * x.c6.w;
        s -= shuffle(lu.c7.x, 10) * x.c7.x;
        s -= shuffle(lu.c7.y, 10) * x.c7.y;
        s -= shuffle(lu.c7.z, 10) * x.c7.z;
        s -= shuffle(lu.c7.w, 10) * x.c7.w;
        let u = shuffle(lu.c2.z, 10);
        x.c2.z = if u != 0.0 { s / u } else { 0.0 };
    }
    if 9 < n {
        let mut s = x.c2.y;
        s -= shuffle(lu.c2.z, 9) * x.c2.z;
        s -= shuffle(lu.c2.w, 9) * x.c2.w;
        s -= shuffle(lu.c3.x, 9) * x.c3.x;
        s -= shuffle(lu.c3.y, 9) * x.c3.y;
        s -= shuffle(lu.c3.z, 9) * x.c3.z;
        s -= shuffle(lu.c3.w, 9) * x.c3.w;
        s -= shuffle(lu.c4.x, 9) * x.c4.x;
        s -= shuffle(lu.c4.y, 9) * x.c4.y;
        s -= shuffle(lu.c4.z, 9) * x.c4.z;
        s -= shuffle(lu.c4.w, 9) * x.c4.w;
        s -= shuffle(lu.c5.x, 9) * x.c5.x;
        s -= shuffle(lu.c5.y, 9) * x.c5.y;
        s -= shuffle(lu.c5.z, 9) * x.c5.z;
        s -= shuffle(lu.c5.w, 9) * x.c5.w;
        s -= shuffle(lu.c6.x, 9) * x.c6.x;
        s -= shuffle(lu.c6.y, 9) * x.c6.y;
        s -= shuffle(lu.c6.z, 9) * x.c6.z;
        s -= shuffle(lu.c6.w, 9) * x.c6.w;
        s -= shuffle(lu.c7.x, 9) * x.c7.x;
        s -= shuffle(lu.c7.y, 9) * x.c7.y;
        s -= shuffle(lu.c7.z, 9) * x.c7.z;
        s -= shuffle(lu.c7.w, 9) * x.c7.w;
        let u = shuffle(lu.c2.y, 9);
        x.c2.y = if u != 0.0 { s / u } else { 0.0 };
    }
    if 8 < n {
        let mut s = x.c2.x;
        s -= shuffle(lu.c2.y, 8) * x.c2.y;
        s -= shuffle(lu.c2.z, 8) * x.c2.z;
        s -= shuffle(lu.c2.w, 8) * x.c2.w;
        s -= shuffle(lu.c3.x, 8) * x.c3.x;
        s -= shuffle(lu.c3.y, 8) * x.c3.y;
        s -= shuffle(lu.c3.z, 8) * x.c3.z;
        s -= shuffle(lu.c3.w, 8) * x.c3.w;
        s -= shuffle(lu.c4.x, 8) * x.c4.x;
        s -= shuffle(lu.c4.y, 8) * x.c4.y;
        s -= shuffle(lu.c4.z, 8) * x.c4.z;
        s -= shuffle(lu.c4.w, 8) * x.c4.w;
        s -= shuffle(lu.c5.x, 8) * x.c5.x;
        s -= shuffle(lu.c5.y, 8) * x.c5.y;
        s -= shuffle(lu.c5.z, 8) * x.c5.z;
        s -= shuffle(lu.c5.w, 8) * x.c5.w;
        s -= shuffle(lu.c6.x, 8) * x.c6.x;
        s -= shuffle(lu.c6.y, 8) * x.c6.y;
        s -= shuffle(lu.c6.z, 8) * x.c6.z;
        s -= shuffle(lu.c6.w, 8) * x.c6.w;
        s -= shuffle(lu.c7.x, 8) * x.c7.x;
        s -= shuffle(lu.c7.y, 8) * x.c7.y;
        s -= shuffle(lu.c7.z, 8) * x.c7.z;
        s -= shuffle(lu.c7.w, 8) * x.c7.w;
        let u = shuffle(lu.c2.x, 8);
        x.c2.x = if u != 0.0 { s / u } else { 0.0 };
    }
    if 7 < n {
        let mut s = x.c1.w;
        s -= shuffle(lu.c2.x, 7) * x.c2.x;
        s -= shuffle(lu.c2.y, 7) * x.c2.y;
        s -= shuffle(lu.c2.z, 7) * x.c2.z;
        s -= shuffle(lu.c2.w, 7) * x.c2.w;
        s -= shuffle(lu.c3.x, 7) * x.c3.x;
        s -= shuffle(lu.c3.y, 7) * x.c3.y;
        s -= shuffle(lu.c3.z, 7) * x.c3.z;
        s -= shuffle(lu.c3.w, 7) * x.c3.w;
        s -= shuffle(lu.c4.x, 7) * x.c4.x;
        s -= shuffle(lu.c4.y, 7) * x.c4.y;
        s -= shuffle(lu.c4.z, 7) * x.c4.z;
        s -= shuffle(lu.c4.w, 7) * x.c4.w;
        s -= shuffle(lu.c5.x, 7) * x.c5.x;
        s -= shuffle(lu.c5.y, 7) * x.c5.y;
        s -= shuffle(lu.c5.z, 7) * x.c5.z;
        s -= shuffle(lu.c5.w, 7) * x.c5.w;
        s -= shuffle(lu.c6.x, 7) * x.c6.x;
        s -= shuffle(lu.c6.y, 7) * x.c6.y;
        s -= shuffle(lu.c6.z, 7) * x.c6.z;
        s -= shuffle(lu.c6.w, 7) * x.c6.w;
        s -= shuffle(lu.c7.x, 7) * x.c7.x;
        s -= shuffle(lu.c7.y, 7) * x.c7.y;
        s -= shuffle(lu.c7.z, 7) * x.c7.z;
        s -= shuffle(lu.c7.w, 7) * x.c7.w;
        let u = shuffle(lu.c1.w, 7);
        x.c1.w = if u != 0.0 { s / u } else { 0.0 };
    }
    if 6 < n {
        let mut s = x.c1.z;
        s -= shuffle(lu.c1.w, 6) * x.c1.w;
        s -= shuffle(lu.c2.x, 6) * x.c2.x;
        s -= shuffle(lu.c2.y, 6) * x.c2.y;
        s -= shuffle(lu.c2.z, 6) * x.c2.z;
        s -= shuffle(lu.c2.w, 6) * x.c2.w;
        s -= shuffle(lu.c3.x, 6) * x.c3.x;
        s -= shuffle(lu.c3.y, 6) * x.c3.y;
        s -= shuffle(lu.c3.z, 6) * x.c3.z;
        s -= shuffle(lu.c3.w, 6) * x.c3.w;
        s -= shuffle(lu.c4.x, 6) * x.c4.x;
        s -= shuffle(lu.c4.y, 6) * x.c4.y;
        s -= shuffle(lu.c4.z, 6) * x.c4.z;
        s -= shuffle(lu.c4.w, 6) * x.c4.w;
        s -= shuffle(lu.c5.x, 6) * x.c5.x;
        s -= shuffle(lu.c5.y, 6) * x.c5.y;
        s -= shuffle(lu.c5.z, 6) * x.c5.z;
        s -= shuffle(lu.c5.w, 6) * x.c5.w;
        s -= shuffle(lu.c6.x, 6) * x.c6.x;
        s -= shuffle(lu.c6.y, 6) * x.c6.y;
        s -= shuffle(lu.c6.z, 6) * x.c6.z;
        s -= shuffle(lu.c6.w, 6) * x.c6.w;
        s -= shuffle(lu.c7.x, 6) * x.c7.x;
        s -= shuffle(lu.c7.y, 6) * x.c7.y;
        s -= shuffle(lu.c7.z, 6) * x.c7.z;
        s -= shuffle(lu.c7.w, 6) * x.c7.w;
        let u = shuffle(lu.c1.z, 6);
        x.c1.z = if u != 0.0 { s / u } else { 0.0 };
    }
    if 5 < n {
        let mut s = x.c1.y;
        s -= shuffle(lu.c1.z, 5) * x.c1.z;
        s -= shuffle(lu.c1.w, 5) * x.c1.w;
        s -= shuffle(lu.c2.x, 5) * x.c2.x;
        s -= shuffle(lu.c2.y, 5) * x.c2.y;
        s -= shuffle(lu.c2.z, 5) * x.c2.z;
        s -= shuffle(lu.c2.w, 5) * x.c2.w;
        s -= shuffle(lu.c3.x, 5) * x.c3.x;
        s -= shuffle(lu.c3.y, 5) * x.c3.y;
        s -= shuffle(lu.c3.z, 5) * x.c3.z;
        s -= shuffle(lu.c3.w, 5) * x.c3.w;
        s -= shuffle(lu.c4.x, 5) * x.c4.x;
        s -= shuffle(lu.c4.y, 5) * x.c4.y;
        s -= shuffle(lu.c4.z, 5) * x.c4.z;
        s -= shuffle(lu.c4.w, 5) * x.c4.w;
        s -= shuffle(lu.c5.x, 5) * x.c5.x;
        s -= shuffle(lu.c5.y, 5) * x.c5.y;
        s -= shuffle(lu.c5.z, 5) * x.c5.z;
        s -= shuffle(lu.c5.w, 5) * x.c5.w;
        s -= shuffle(lu.c6.x, 5) * x.c6.x;
        s -= shuffle(lu.c6.y, 5) * x.c6.y;
        s -= shuffle(lu.c6.z, 5) * x.c6.z;
        s -= shuffle(lu.c6.w, 5) * x.c6.w;
        s -= shuffle(lu.c7.x, 5) * x.c7.x;
        s -= shuffle(lu.c7.y, 5) * x.c7.y;
        s -= shuffle(lu.c7.z, 5) * x.c7.z;
        s -= shuffle(lu.c7.w, 5) * x.c7.w;
        let u = shuffle(lu.c1.y, 5);
        x.c1.y = if u != 0.0 { s / u } else { 0.0 };
    }
    if 4 < n {
        let mut s = x.c1.x;
        s -= shuffle(lu.c1.y, 4) * x.c1.y;
        s -= shuffle(lu.c1.z, 4) * x.c1.z;
        s -= shuffle(lu.c1.w, 4) * x.c1.w;
        s -= shuffle(lu.c2.x, 4) * x.c2.x;
        s -= shuffle(lu.c2.y, 4) * x.c2.y;
        s -= shuffle(lu.c2.z, 4) * x.c2.z;
        s -= shuffle(lu.c2.w, 4) * x.c2.w;
        s -= shuffle(lu.c3.x, 4) * x.c3.x;
        s -= shuffle(lu.c3.y, 4) * x.c3.y;
        s -= shuffle(lu.c3.z, 4) * x.c3.z;
        s -= shuffle(lu.c3.w, 4) * x.c3.w;
        s -= shuffle(lu.c4.x, 4) * x.c4.x;
        s -= shuffle(lu.c4.y, 4) * x.c4.y;
        s -= shuffle(lu.c4.z, 4) * x.c4.z;
        s -= shuffle(lu.c4.w, 4) * x.c4.w;
        s -= shuffle(lu.c5.x, 4) * x.c5.x;
        s -= shuffle(lu.c5.y, 4) * x.c5.y;
        s -= shuffle(lu.c5.z, 4) * x.c5.z;
        s -= shuffle(lu.c5.w, 4) * x.c5.w;
        s -= shuffle(lu.c6.x, 4) * x.c6.x;
        s -= shuffle(lu.c6.y, 4) * x.c6.y;
        s -= shuffle(lu.c6.z, 4) * x.c6.z;
        s -= shuffle(lu.c6.w, 4) * x.c6.w;
        s -= shuffle(lu.c7.x, 4) * x.c7.x;
        s -= shuffle(lu.c7.y, 4) * x.c7.y;
        s -= shuffle(lu.c7.z, 4) * x.c7.z;
        s -= shuffle(lu.c7.w, 4) * x.c7.w;
        let u = shuffle(lu.c1.x, 4);
        x.c1.x = if u != 0.0 { s / u } else { 0.0 };
    }
    if 3 < n {
        let mut s = x.c0.w;
        s -= shuffle(lu.c1.x, 3) * x.c1.x;
        s -= shuffle(lu.c1.y, 3) * x.c1.y;
        s -= shuffle(lu.c1.z, 3) * x.c1.z;
        s -= shuffle(lu.c1.w, 3) * x.c1.w;
        s -= shuffle(lu.c2.x, 3) * x.c2.x;
        s -= shuffle(lu.c2.y, 3) * x.c2.y;
        s -= shuffle(lu.c2.z, 3) * x.c2.z;
        s -= shuffle(lu.c2.w, 3) * x.c2.w;
        s -= shuffle(lu.c3.x, 3) * x.c3.x;
        s -= shuffle(lu.c3.y, 3) * x.c3.y;
        s -= shuffle(lu.c3.z, 3) * x.c3.z;
        s -= shuffle(lu.c3.w, 3) * x.c3.w;
        s -= shuffle(lu.c4.x, 3) * x.c4.x;
        s -= shuffle(lu.c4.y, 3) * x.c4.y;
        s -= shuffle(lu.c4.z, 3) * x.c4.z;
        s -= shuffle(lu.c4.w, 3) * x.c4.w;
        s -= shuffle(lu.c5.x, 3) * x.c5.x;
        s -= shuffle(lu.c5.y, 3) * x.c5.y;
        s -= shuffle(lu.c5.z, 3) * x.c5.z;
        s -= shuffle(lu.c5.w, 3) * x.c5.w;
        s -= shuffle(lu.c6.x, 3) * x.c6.x;
        s -= shuffle(lu.c6.y, 3) * x.c6.y;
        s -= shuffle(lu.c6.z, 3) * x.c6.z;
        s -= shuffle(lu.c6.w, 3) * x.c6.w;
        s -= shuffle(lu.c7.x, 3) * x.c7.x;
        s -= shuffle(lu.c7.y, 3) * x.c7.y;
        s -= shuffle(lu.c7.z, 3) * x.c7.z;
        s -= shuffle(lu.c7.w, 3) * x.c7.w;
        let u = shuffle(lu.c0.w, 3);
        x.c0.w = if u != 0.0 { s / u } else { 0.0 };
    }
    if 2 < n {
        let mut s = x.c0.z;
        s -= shuffle(lu.c0.w, 2) * x.c0.w;
        s -= shuffle(lu.c1.x, 2) * x.c1.x;
        s -= shuffle(lu.c1.y, 2) * x.c1.y;
        s -= shuffle(lu.c1.z, 2) * x.c1.z;
        s -= shuffle(lu.c1.w, 2) * x.c1.w;
        s -= shuffle(lu.c2.x, 2) * x.c2.x;
        s -= shuffle(lu.c2.y, 2) * x.c2.y;
        s -= shuffle(lu.c2.z, 2) * x.c2.z;
        s -= shuffle(lu.c2.w, 2) * x.c2.w;
        s -= shuffle(lu.c3.x, 2) * x.c3.x;
        s -= shuffle(lu.c3.y, 2) * x.c3.y;
        s -= shuffle(lu.c3.z, 2) * x.c3.z;
        s -= shuffle(lu.c3.w, 2) * x.c3.w;
        s -= shuffle(lu.c4.x, 2) * x.c4.x;
        s -= shuffle(lu.c4.y, 2) * x.c4.y;
        s -= shuffle(lu.c4.z, 2) * x.c4.z;
        s -= shuffle(lu.c4.w, 2) * x.c4.w;
        s -= shuffle(lu.c5.x, 2) * x.c5.x;
        s -= shuffle(lu.c5.y, 2) * x.c5.y;
        s -= shuffle(lu.c5.z, 2) * x.c5.z;
        s -= shuffle(lu.c5.w, 2) * x.c5.w;
        s -= shuffle(lu.c6.x, 2) * x.c6.x;
        s -= shuffle(lu.c6.y, 2) * x.c6.y;
        s -= shuffle(lu.c6.z, 2) * x.c6.z;
        s -= shuffle(lu.c6.w, 2) * x.c6.w;
        s -= shuffle(lu.c7.x, 2) * x.c7.x;
        s -= shuffle(lu.c7.y, 2) * x.c7.y;
        s -= shuffle(lu.c7.z, 2) * x.c7.z;
        s -= shuffle(lu.c7.w, 2) * x.c7.w;
        let u = shuffle(lu.c0.z, 2);
        x.c0.z = if u != 0.0 { s / u } else { 0.0 };
    }
    if 1 < n {
        let mut s = x.c0.y;
        s -= shuffle(lu.c0.z, 1) * x.c0.z;
        s -= shuffle(lu.c0.w, 1) * x.c0.w;
        s -= shuffle(lu.c1.x, 1) * x.c1.x;
        s -= shuffle(lu.c1.y, 1) * x.c1.y;
        s -= shuffle(lu.c1.z, 1) * x.c1.z;
        s -= shuffle(lu.c1.w, 1) * x.c1.w;
        s -= shuffle(lu.c2.x, 1) * x.c2.x;
        s -= shuffle(lu.c2.y, 1) * x.c2.y;
        s -= shuffle(lu.c2.z, 1) * x.c2.z;
        s -= shuffle(lu.c2.w, 1) * x.c2.w;
        s -= shuffle(lu.c3.x, 1) * x.c3.x;
        s -= shuffle(lu.c3.y, 1) * x.c3.y;
        s -= shuffle(lu.c3.z, 1) * x.c3.z;
        s -= shuffle(lu.c3.w, 1) * x.c3.w;
        s -= shuffle(lu.c4.x, 1) * x.c4.x;
        s -= shuffle(lu.c4.y, 1) * x.c4.y;
        s -= shuffle(lu.c4.z, 1) * x.c4.z;
        s -= shuffle(lu.c4.w, 1) * x.c4.w;
        s -= shuffle(lu.c5.x, 1) * x.c5.x;
        s -= shuffle(lu.c5.y, 1) * x.c5.y;
        s -= shuffle(lu.c5.z, 1) * x.c5.z;
        s -= shuffle(lu.c5.w, 1) * x.c5.w;
        s -= shuffle(lu.c6.x, 1) * x.c6.x;
        s -= shuffle(lu.c6.y, 1) * x.c6.y;
        s -= shuffle(lu.c6.z, 1) * x.c6.z;
        s -= shuffle(lu.c6.w, 1) * x.c6.w;
        s -= shuffle(lu.c7.x, 1) * x.c7.x;
        s -= shuffle(lu.c7.y, 1) * x.c7.y;
        s -= shuffle(lu.c7.z, 1) * x.c7.z;
        s -= shuffle(lu.c7.w, 1) * x.c7.w;
        let u = shuffle(lu.c0.y, 1);
        x.c0.y = if u != 0.0 { s / u } else { 0.0 };
    }
    if 0 < n {
        let mut s = x.c0.x;
        s -= shuffle(lu.c0.y, 0) * x.c0.y;
        s -= shuffle(lu.c0.z, 0) * x.c0.z;
        s -= shuffle(lu.c0.w, 0) * x.c0.w;
        s -= shuffle(lu.c1.x, 0) * x.c1.x;
        s -= shuffle(lu.c1.y, 0) * x.c1.y;
        s -= shuffle(lu.c1.z, 0) * x.c1.z;
        s -= shuffle(lu.c1.w, 0) * x.c1.w;
        s -= shuffle(lu.c2.x, 0) * x.c2.x;
        s -= shuffle(lu.c2.y, 0) * x.c2.y;
        s -= shuffle(lu.c2.z, 0) * x.c2.z;
        s -= shuffle(lu.c2.w, 0) * x.c2.w;
        s -= shuffle(lu.c3.x, 0) * x.c3.x;
        s -= shuffle(lu.c3.y, 0) * x.c3.y;
        s -= shuffle(lu.c3.z, 0) * x.c3.z;
        s -= shuffle(lu.c3.w, 0) * x.c3.w;
        s -= shuffle(lu.c4.x, 0) * x.c4.x;
        s -= shuffle(lu.c4.y, 0) * x.c4.y;
        s -= shuffle(lu.c4.z, 0) * x.c4.z;
        s -= shuffle(lu.c4.w, 0) * x.c4.w;
        s -= shuffle(lu.c5.x, 0) * x.c5.x;
        s -= shuffle(lu.c5.y, 0) * x.c5.y;
        s -= shuffle(lu.c5.z, 0) * x.c5.z;
        s -= shuffle(lu.c5.w, 0) * x.c5.w;
        s -= shuffle(lu.c6.x, 0) * x.c6.x;
        s -= shuffle(lu.c6.y, 0) * x.c6.y;
        s -= shuffle(lu.c6.z, 0) * x.c6.z;
        s -= shuffle(lu.c6.w, 0) * x.c6.w;
        s -= shuffle(lu.c7.x, 0) * x.c7.x;
        s -= shuffle(lu.c7.y, 0) * x.c7.y;
        s -= shuffle(lu.c7.z, 0) * x.c7.z;
        s -= shuffle(lu.c7.w, 0) * x.c7.w;
        let u = shuffle(lu.c0.x, 0);
        x.c0.x = if u != 0.0 { s / u } else { 0.0 };
    }
}

/// `1 / x`, or 0 when `x == 0` (rapier's `crate::utils::inv`).
#[inline(always)]
fn inv(x: f32) -> f32 {
    if x != 0.0 { 1.0 / x } else { 0.0 }
}

/// Same results as `gpu_mb_finalize_joint_constraints` (bit-identical for
/// single-DOF rows; coupling rows combine two `M⁻¹` columns), for up to 32 DOFs.
/// One 32-lane workgroup per multibody.
#[spirv_bindgen]
#[spirv(compute(threads(32)))]
pub fn gpu_mb_finalize_joint_simd(
    #[spirv(workgroup_id)] wid: khal_std::glamx::UVec3,
    #[spirv(local_invocation_id)] lid: khal_std::glamx::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    joint_constraint_columns: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] matrix: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] pivots: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
    #[spirv(workgroup)] inverse: &mut [f32; 1024],
    #[spirv(workgroup)] slot_meta: &mut [Vec4; 64],
) {
    let batch_id = wid.y;
    let mb_idx = wid.x;
    let lane = lid.x;
    if batch_id >= batch_ids.num_batches || mb_idx >= batch_ids.multibodies_len {
        return;
    }
    let mb = multibody_info.read(batch_ids.mbi(batch_id, mb_idx as usize));
    let n = mb.ndofs;
    if n == 0 || n > 32 || mb.max_constraints == 0 {
        return;
    }
    let m = MatSlice::dense(
        batch_ids.mb_region(batch_id, mb.mass_matrix_offset, n * n),
        n,
        n,
    );
    let piv = batch_ids.mb_region(batch_id, mb.first_dof, n);

    // `P e_c` is a unit vector: follow its non-zero through the row swaps.
    // Lane `k` loads pivot `k`; the walk reads them by shuffle, keeping the
    // loads off its dependency chain.
    let my_pivot = if lane < n {
        pivots.read(piv + lane as usize)
    } else {
        0
    };
    let mut pos = lane;
    for k in 0..n {
        let p = shuffle_u32(my_pivot, k);
        if pos == k {
            pos = p;
        } else if pos == p {
            pos = k;
        }
    }
    // Row `lane` of the LU factors (zero past `n`).
    let mut lu = Column::unit(32);
    if 0 < n && lane < n {
        lu.c0.x = matrix.read(m.idx(lane, 0));
    }
    if 1 < n && lane < n {
        lu.c0.y = matrix.read(m.idx(lane, 1));
    }
    if 2 < n && lane < n {
        lu.c0.z = matrix.read(m.idx(lane, 2));
    }
    if 3 < n && lane < n {
        lu.c0.w = matrix.read(m.idx(lane, 3));
    }
    if 4 < n && lane < n {
        lu.c1.x = matrix.read(m.idx(lane, 4));
    }
    if 5 < n && lane < n {
        lu.c1.y = matrix.read(m.idx(lane, 5));
    }
    if 6 < n && lane < n {
        lu.c1.z = matrix.read(m.idx(lane, 6));
    }
    if 7 < n && lane < n {
        lu.c1.w = matrix.read(m.idx(lane, 7));
    }
    if 8 < n && lane < n {
        lu.c2.x = matrix.read(m.idx(lane, 8));
    }
    if 9 < n && lane < n {
        lu.c2.y = matrix.read(m.idx(lane, 9));
    }
    if 10 < n && lane < n {
        lu.c2.z = matrix.read(m.idx(lane, 10));
    }
    if 11 < n && lane < n {
        lu.c2.w = matrix.read(m.idx(lane, 11));
    }
    if 12 < n && lane < n {
        lu.c3.x = matrix.read(m.idx(lane, 12));
    }
    if 13 < n && lane < n {
        lu.c3.y = matrix.read(m.idx(lane, 13));
    }
    if 14 < n && lane < n {
        lu.c3.z = matrix.read(m.idx(lane, 14));
    }
    if 15 < n && lane < n {
        lu.c3.w = matrix.read(m.idx(lane, 15));
    }
    if 16 < n && lane < n {
        lu.c4.x = matrix.read(m.idx(lane, 16));
    }
    if 17 < n && lane < n {
        lu.c4.y = matrix.read(m.idx(lane, 17));
    }
    if 18 < n && lane < n {
        lu.c4.z = matrix.read(m.idx(lane, 18));
    }
    if 19 < n && lane < n {
        lu.c4.w = matrix.read(m.idx(lane, 19));
    }
    if 20 < n && lane < n {
        lu.c5.x = matrix.read(m.idx(lane, 20));
    }
    if 21 < n && lane < n {
        lu.c5.y = matrix.read(m.idx(lane, 21));
    }
    if 22 < n && lane < n {
        lu.c5.z = matrix.read(m.idx(lane, 22));
    }
    if 23 < n && lane < n {
        lu.c5.w = matrix.read(m.idx(lane, 23));
    }
    if 24 < n && lane < n {
        lu.c6.x = matrix.read(m.idx(lane, 24));
    }
    if 25 < n && lane < n {
        lu.c6.y = matrix.read(m.idx(lane, 25));
    }
    if 26 < n && lane < n {
        lu.c6.z = matrix.read(m.idx(lane, 26));
    }
    if 27 < n && lane < n {
        lu.c6.w = matrix.read(m.idx(lane, 27));
    }
    if 28 < n && lane < n {
        lu.c7.x = matrix.read(m.idx(lane, 28));
    }
    if 29 < n && lane < n {
        lu.c7.y = matrix.read(m.idx(lane, 29));
    }
    if 30 < n && lane < n {
        lu.c7.z = matrix.read(m.idx(lane, 30));
    }
    if 31 < n && lane < n {
        lu.c7.w = matrix.read(m.idx(lane, 31));
    }
    let mut x = Column::unit(pos);
    substitute(n, &lu, &mut x);
    // Column `lane` of `M⁻¹`, stored column-major.
    if 0 < n {
        inverse[lane as usize * 32] = x.c0.x;
    }
    if 1 < n {
        inverse[lane as usize * 32 + 1] = x.c0.y;
    }
    if 2 < n {
        inverse[lane as usize * 32 + 2] = x.c0.z;
    }
    if 3 < n {
        inverse[lane as usize * 32 + 3] = x.c0.w;
    }
    if 4 < n {
        inverse[lane as usize * 32 + 4] = x.c1.x;
    }
    if 5 < n {
        inverse[lane as usize * 32 + 5] = x.c1.y;
    }
    if 6 < n {
        inverse[lane as usize * 32 + 6] = x.c1.z;
    }
    if 7 < n {
        inverse[lane as usize * 32 + 7] = x.c1.w;
    }
    if 8 < n {
        inverse[lane as usize * 32 + 8] = x.c2.x;
    }
    if 9 < n {
        inverse[lane as usize * 32 + 9] = x.c2.y;
    }
    if 10 < n {
        inverse[lane as usize * 32 + 10] = x.c2.z;
    }
    if 11 < n {
        inverse[lane as usize * 32 + 11] = x.c2.w;
    }
    if 12 < n {
        inverse[lane as usize * 32 + 12] = x.c3.x;
    }
    if 13 < n {
        inverse[lane as usize * 32 + 13] = x.c3.y;
    }
    if 14 < n {
        inverse[lane as usize * 32 + 14] = x.c3.z;
    }
    if 15 < n {
        inverse[lane as usize * 32 + 15] = x.c3.w;
    }
    if 16 < n {
        inverse[lane as usize * 32 + 16] = x.c4.x;
    }
    if 17 < n {
        inverse[lane as usize * 32 + 17] = x.c4.y;
    }
    if 18 < n {
        inverse[lane as usize * 32 + 18] = x.c4.z;
    }
    if 19 < n {
        inverse[lane as usize * 32 + 19] = x.c4.w;
    }
    if 20 < n {
        inverse[lane as usize * 32 + 20] = x.c5.x;
    }
    if 21 < n {
        inverse[lane as usize * 32 + 21] = x.c5.y;
    }
    if 22 < n {
        inverse[lane as usize * 32 + 22] = x.c5.z;
    }
    if 23 < n {
        inverse[lane as usize * 32 + 23] = x.c5.w;
    }
    if 24 < n {
        inverse[lane as usize * 32 + 24] = x.c6.x;
    }
    if 25 < n {
        inverse[lane as usize * 32 + 25] = x.c6.y;
    }
    if 26 < n {
        inverse[lane as usize * 32 + 26] = x.c6.z;
    }
    if 27 < n {
        inverse[lane as usize * 32 + 27] = x.c6.w;
    }
    if 28 < n {
        inverse[lane as usize * 32 + 28] = x.c7.x;
    }
    if 29 < n {
        inverse[lane as usize * 32 + 29] = x.c7.y;
    }
    if 30 < n {
        inverse[lane as usize * 32 + 30] = x.c7.z;
    }
    if 31 < n {
        inverse[lane as usize * 32 + 31] = x.c7.w;
    }
    workgroup_memory_barrier_with_group_sync();

    let cons_base = batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let col_base = batch_ids.mb_joint_constraint_columns_start(batch_id)
        + (mb.first_constraint as usize) * dofs_stride;
    let num_slots = mb.max_constraints.min(64);

    // Lane `l` finalizes slots `l` and `l + 32` (all row loads in flight at
    // once) and publishes what the column gather needs.
    for band in 0..2u32 {
        let s = lane + band * 32;
        if s < num_slots {
            let cons = joint_constraints.read(cons_base + s as usize);
            let coeff = cons.coupling_coeff;
            let d1 = cons.dof_id.min(31) as usize * 32;
            let d2 = cons.dof2_id.min(31) as usize * 32;
            if cons.kind != 0 {
                let mut at_d1 = inverse[d1 + cons.dof_id.min(31) as usize];
                let mut at_d2 = inverse[d1 + cons.dof2_id.min(31) as usize];
                if coeff != 0.0 {
                    at_d1 -= coeff * inverse[d2 + cons.dof_id.min(31) as usize];
                    at_d2 -= coeff * inverse[d2 + cons.dof2_id.min(31) as usize];
                }
                let lhs = at_d1 - coeff * at_d2;
                let cfm_gain = lhs * cons.cfm_coeff + cons.cfm_gain;
                let out = joint_constraints.at_mut(cons_base + s as usize);
                out.cfm_gain = cfm_gain;
                out.inv_lhs = inv(lhs + cfm_gain);
            }
            slot_meta[s as usize] = Vec4::new(
                f32::from_bits(cons.kind),
                f32::from_bits(cons.dof_id.min(31)),
                f32::from_bits(cons.dof2_id.min(31)),
                coeff,
            );
        }
    }
    workgroup_memory_barrier_with_group_sync();

    #[allow(clippy::needless_range_loop)]
    for s in 0..num_slots as usize {
        let meta = slot_meta[s];
        if meta.x.to_bits() == 0 || lane >= n {
            continue;
        }
        let coeff = meta.w;
        let mut col = inverse[meta.y.to_bits() as usize * 32 + lane as usize];
        if coeff != 0.0 {
            col -= coeff * inverse[meta.z.to_bits() as usize * 32 + lane as usize];
        }
        joint_constraint_columns.write(col_base + s * dofs_stride + lane as usize, col);
    }
}
