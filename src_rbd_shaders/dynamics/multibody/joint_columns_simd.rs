//! Joint-constraint columns (`M⁻¹ Jᵀ`) for small multibodies, from the LU
//! factors of their mass matrix.
//!
//! Lane `c` back-solves the unit vector `e_c` with the same operations, in the
//! same order, as `lu_solve_in_place`, holding the solution and its LU row in
//! registers; each constraint column is then gathered from the resulting
//! `M⁻¹` instead of being back-solved on its own in device memory.

use super::types::{MultibodyInfo, MultibodyJointConstraint};
use super::unroll::{Regs32, each_reg};
use crate::utils::BatchIndices;
use crate::utils::linalg::MatSlice;
use glamx::Vec4;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::iter::StepRng;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

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
#[allow(unused_comparisons, clippy::absurd_extreme_comparisons)]
fn substitute(n: u32, lu: &Regs32, x: &mut Regs32) {
    // `x_i -= l_ij x_j` for `j < i` (L has a unit diagonal).
    macro_rules! lower_term {
        ($i:literal, $s:ident, $j:literal, $q:ident, $l:ident) => {
            if $j < $i {
                $s -= shuffle(lu.$q.$l, $i) * x.$q.$l;
            }
        };
    }
    macro_rules! forward {
        ($i:literal, $q:ident, $l:ident) => {
            if 0 < $i && $i < n {
                let mut s = x.$q.$l;
                each_reg!(lower_term, $i, s);
                x.$q.$l = s;
            }
        };
    }
    each_reg!(forward);

    // `x_i = (x_i - Σ u_ij x_j) / u_ii` for `j > i`.
    macro_rules! upper_term {
        ($i:literal, $s:ident, $j:literal, $q:ident, $l:ident) => {
            if $j > $i {
                $s -= shuffle(lu.$q.$l, $i) * x.$q.$l;
            }
        };
    }
    macro_rules! backward {
        ($i:literal, $q:ident, $l:ident) => {
            if $i < n {
                let mut s = x.$q.$l;
                each_reg!(upper_term, $i, s);
                let u = shuffle(lu.$q.$l, $i);
                x.$q.$l = if u != 0.0 { s / u } else { 0.0 };
            }
        };
    }
    each_reg!(rev backward);
}

/// `1 / x`, or 0 when `x == 0` (rapier's `crate::utils::inv`).
#[inline(always)]
fn inv(x: f32) -> f32 {
    if x != 0.0 { 1.0 / x } else { 0.0 }
}

/// Same results as `gpu_mb_finalize_joint_constraints` (bit-identical for
/// single-DOF rows; coupling rows combine two `M⁻¹` columns), for up to 32 DOFs.
/// One 32-lane workgroup per multibody.
#[cfg(any(feature = "entry-points", not(target_arch = "spirv")))]
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
    // Nonzero: write the columns padded to whole quads into `padded_columns`
    // (read by `gpu_mb_substeps_packed`) instead of `joint_constraint_columns`.
    #[spirv(uniform, descriptor_set = 0, binding = 6)] padded: &u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] padded_columns: &mut [Vec4],
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
    let mut lu = Regs32::zero();
    macro_rules! load {
        ($c:literal, $q:ident, $l:ident) => {
            if $c < n && lane < n {
                lu.$q.$l = matrix.read(m.idx(lane, $c));
            }
        };
    }
    each_reg!(load);
    let mut x = Regs32::zero();
    macro_rules! unit {
        ($c:literal, $q:ident, $l:ident) => {
            x.$q.$l = if $c == pos { 1.0 } else { 0.0 };
        };
    }
    each_reg!(unit);
    substitute(n, &lu, &mut x);
    // Column `lane` of `M⁻¹`, stored column-major.
    macro_rules! store {
        ($c:literal, $q:ident, $l:ident) => {
            if $c < n {
                inverse[lane as usize * 32 + $c] = x.$q.$l;
            }
        };
    }
    each_reg!(store);
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

    if *padded != 0 {
        // `quads` quads per slot (`mb_max_ndofs` rounded up to 4); every lane
        // assembles whole quads so the stores coalesce.
        let quads = batch_ids.mb_max_ndofs.div_ceil(4);
        for idx in StepRng::new(lane..num_slots * quads, 32) {
            let s = (idx / quads) as usize;
            let q = idx % quads;
            let meta = slot_meta[s];
            if meta.x.to_bits() == 0 {
                continue;
            }
            let coeff = meta.w;
            let c1 = meta.y.to_bits() as usize * 32;
            let c2 = meta.z.to_bits() as usize * 32;
            let mut col = Vec4::ZERO;
            for k in 0..4u32 {
                let row = q * 4 + k;
                if row < n {
                    let mut x = inverse[c1 + row as usize];
                    if coeff != 0.0 {
                        x -= coeff * inverse[c2 + row as usize];
                    }
                    match k {
                        0 => col.x = x,
                        1 => col.y = x,
                        2 => col.z = x,
                        _ => col.w = x,
                    }
                }
            }
            padded_columns.write((cons_base + s) * quads as usize + q as usize, col);
        }
        return;
    }
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
