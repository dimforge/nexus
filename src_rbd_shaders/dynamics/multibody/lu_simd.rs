//! Register-resident LU for a small dense matrix, one row per SIMD lane.
use super::unroll::{Regs32, each_reg};
use crate::utils::linalg::{MatSlice, VSlice};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::subgroup_f_max;

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

/// Factors the `n × n` matrix `m` in place with partial pivoting (L with a unit diagonal below
/// U, as `lu_decompose`) and solves for `rhs` in place. Lane `i` holds row `i` in registers; the
/// pivot row's entries reach the other lanes by shuffles.
#[inline(always)]
#[allow(unused_comparisons, clippy::absurd_extreme_comparisons)]
pub(super) fn factor_solve(
    lane: u32,
    n: u32,
    matrix: &mut [f32],
    m: MatSlice,
    pivots: &mut [u32],
    piv: VSlice,
    forces: &mut [f32],
    rhs: VSlice,
) {
    let active = lane < n;
    let mut row = Regs32::zero();
    let mut x = if active {
        forces.read(rhs.at(lane))
    } else {
        0.0
    };

    macro_rules! load {
        ($c:literal, $q:ident, $l:ident) => {
            if active && $c < n {
                row.$q.$l = matrix.read(m.idx(lane, $c));
            }
        };
    }
    each_reg!(load);

    // One elimination step per column `k`.
    macro_rules! swap {
        ($src:ident, $c:literal, $q:ident, $l:ident) => {
            if $c < n {
                row.$q.$l = shuffle(row.$q.$l, $src);
            }
        };
    }
    macro_rules! eliminate {
        ($k:literal, $lower:ident, $c:literal, $q:ident, $l:ident) => {
            if $c > $k && $c < n {
                let u = shuffle(row.$q.$l, $k);
                if lane > $k && active {
                    row.$q.$l -= $lower * u;
                }
            }
        };
    }
    macro_rules! step {
        ($k:literal, $q:ident, $l:ident) => {
            if $k < n {
                // Pivot: the lowest lane `p >= k` holding the largest `|a_pk|`.
                let a = row.$q.$l;
                let abs = if a >= 0.0 { a } else { -a };
                let maximum = subgroup_f_max(if lane >= $k && active { abs } else { -1.0 });
                let p = (-subgroup_f_max(if lane >= $k && active && abs == maximum {
                    -(lane as f32)
                } else {
                    -32.0
                })) as u32;
                if lane == 0 {
                    pivots.write(piv.at($k), p);
                }
                if p != $k {
                    let src = if lane == $k {
                        p
                    } else if lane == p {
                        $k
                    } else {
                        lane
                    };
                    each_reg!(swap, src);
                    x = shuffle(x, src);
                }

                // Eliminate column `k` below the pivot, storing the multipliers in place.
                let akk = shuffle(row.$q.$l, $k);
                let inv = if akk != 0.0 { 1.0 / akk } else { 0.0 };
                let lower = row.$q.$l * inv;
                each_reg!(eliminate, $k, lower);
                if lane > $k && active {
                    row.$q.$l = lower;
                }
            }
        };
    }
    each_reg!(step);

    // Forward substitution with L, then back substitution with U.
    macro_rules! forward {
        ($k:literal, $q:ident, $l:ident) => {
            if $k < n {
                let y = shuffle(x, $k);
                if lane > $k && active {
                    x -= row.$q.$l * y;
                }
            }
        };
    }
    each_reg!(forward);
    macro_rules! backward {
        ($k:literal, $q:ident, $l:ident) => {
            if $k < n {
                if lane == $k {
                    let u = row.$q.$l;
                    x = if u != 0.0 { x / u } else { 0.0 };
                }
                let y = shuffle(x, $k);
                if lane < $k {
                    x -= row.$q.$l * y;
                }
            }
        };
    }
    each_reg!(rev backward);

    if active {
        forces.write(rhs.at(lane), x);
    }
    macro_rules! store {
        ($c:literal, $q:ident, $l:ident) => {
            if active && $c < n {
                matrix.write(m.idx(lane, $c), row.$q.$l);
            }
        };
    }
    each_reg!(store);
}

#[spirv_bindgen]
#[spirv(compute(threads(32)))]
pub fn gpu_mb_factor_solve_simd(
    #[spirv(workgroup_id)] wid: khal_std::glamx::UVec3,
    #[spirv(local_invocation_id)] lid: khal_std::glamx::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    info: &[super::types::MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] matrix: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] pivots: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] forces: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &crate::utils::BatchIndices,
) {
    let b = wid.y;
    let mb = wid.x;
    if b >= batch_ids.num_batches || mb >= batch_ids.multibodies_len {
        return;
    }
    let info = info.read(batch_ids.mbi(b, mb as usize));
    let n = info.ndofs;
    if n == 0 {
        return;
    }
    let m = MatSlice::dense(batch_ids.mb_region(b, info.mass_matrix_offset, n * n), n, n);
    let rhs = VSlice::dense(batch_ids.mb_region(b, info.first_dof, n));
    if n <= 32 && cfg!(target_arch = "spirv") && khal_std::sync::subgroup_f_add(1.0) == 32.0 {
        factor_solve(lid.x, n, matrix, m, pivots, rhs, forces, rhs);
    } else if lid.x == 0 {
        crate::utils::linalg::lu_decompose(matrix, m, pivots, rhs);
        crate::utils::linalg::lu_solve_in_place(matrix, m, pivots, rhs, forces, rhs);
    }
}
