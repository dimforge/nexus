//! Tree-sparse LᵀDL factorization + solve for the multibody mass matrix.
//!
//! The mass matrix of a kinematic tree has branch-induced sparsity:
//! `M[i, j] != 0` only when DOFs `i` and `j` lie on the same root-to-leaf
//! path. Eliminating leaves-to-root (descending DOF order, since parents are
//! assembled before children) factors `M = Lᵀ·D·L` with ZERO fill-in —
//! `L[k, i]` is nonzero only for `i` on `k`'s ancestor chain (Featherstone,
//! RBDA §6; MuJoCo's `mj_factorM`). Factor cost drops from O(n³) to
//! O(n·depth²) and every solve from O(n²) to O(n·depth).
//!
//! The per-DOF parent index (`u32::MAX` at roots) is stored in the buffer
//! that used to hold the LU pivots, so every downstream solve reads the tree
//! from a binding it already had. Factors live in the dense `mass_matrices`
//! storage: strict-lower ancestor-chain entries hold `L` (unit diagonal
//! implied), the diagonal holds `D`, and the upper triangle keeps stale
//! symmetric copies of M that no solve reads.
//!
//! The factor/solve loops are data-dependent but contain NO barriers, so a
//! single lane runs them serially — legal under WGSL uniformity rules (the
//! old dense kernels needed fixed 64-iteration loops only because they
//! barriered inside). At n ≤ 64, depth ≈ 7, the serial sparse path does a
//! few hundred FMAs versus the dense path's ~192 workgroup barriers.

use khal_std::index::MaybeIndexUnchecked;
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

use crate::utils::linalg::MAX_MB_DOFS;

/// Workgroup width for the multibody solver kernels. Must match the
/// `threads(N, 1, 1)` attribute and `MB_LU_LANES` on the host side.
pub(super) const LANES: u32 = 64;

/// Parent sentinel for root DOFs in the per-DOF parent array.
pub(super) const NO_PARENT: u32 = u32::MAX;

/// Side length of the workgroup-shared matrix tile. Must equal both the lane
/// count and the maximum supported `ndofs`.
const MAX_DOFS_U32: u32 = MAX_MB_DOFS as u32;

/// Index helper for the shared `mat[col * MAX_MB_DOFS + row]` tile.
#[inline]
pub(super) fn sm_idx(r: u32, c: u32) -> usize {
    (c * MAX_DOFS_U32 + r) as usize
}

/// Tree-sparse LᵀDL factorization of the shared `mat` tile in place, run
/// serially by lane 0 (no barriers inside — the loops are data-dependent).
/// `parents` holds the per-DOF parent indices (`NO_PARENT` at roots), read
/// from the pivots buffer at `parents_offset`.
///
/// Ends with a workgroup barrier so every lane sees the factors.
#[inline]
pub(super) fn ltdl_factor_in_shared(
    n: u32,
    lane: u32,
    mat: &mut impl MaybeIndexUnchecked<f32>,
    parents: &[u32],
    parents_offset: usize,
) {
    if lane == 0 {
        // Eliminate leaves-to-root: descending k, updating only the
        // ancestor-chain entries of rows above.
        for step in 0..n {
            let k = n - 1 - step;
            let d = mat.read(sm_idx(k, k));
            let inv_d = if d != 0.0 { 1.0 / d } else { 0.0 };
            let mut i = parents.read(parents_offset + k as usize);
            // Bounded loops (parents are strictly decreasing) so a corrupt
            // parent array can't hang the GPU.
            for _ in 0..MAX_DOFS_U32 {
                if i == NO_PARENT {
                    break;
                }
                let a = mat.read(sm_idx(k, i)) * inv_d;
                let mut j = i;
                for _ in 0..MAX_DOFS_U32 {
                    if j == NO_PARENT {
                        break;
                    }
                    let v = mat.read(sm_idx(i, j)) - a * mat.read(sm_idx(k, j));
                    mat.write(sm_idx(i, j), v);
                    j = parents.read(parents_offset + j as usize);
                }
                mat.write(sm_idx(k, i), a);
                i = parents.read(parents_offset + i as usize);
            }
        }
    }
    workgroup_memory_barrier_with_group_sync();
}

/// Solve `M·x = b` in place on the shared tile holding LᵀDL factors, run
/// serially by lane 0. Ends with a workgroup barrier.
#[inline]
pub(super) fn ltdl_solve_in_shared(
    n: u32,
    lane: u32,
    mat: &impl MaybeIndexUnchecked<f32>,
    parents: &[u32],
    parents_offset: usize,
    x: &mut impl MaybeIndexUnchecked<f32>,
) {
    if lane == 0 {
        // Solve Lᵀ·z = b: scatter descending (descendants before ancestors).
        for step in 0..n {
            let i = n - 1 - step;
            let xi = x.read(i as usize);
            let mut j = parents.read(parents_offset + i as usize);
            for _ in 0..MAX_DOFS_U32 {
                if j == NO_PARENT {
                    break;
                }
                let v = x.read(j as usize) - mat.read(sm_idx(i, j)) * xi;
                x.write(j as usize, v);
                j = parents.read(parents_offset + j as usize);
            }
        }
        // z = D⁻¹·z.
        for i in 0..n {
            let d = mat.read(sm_idx(i, i));
            let v = x.read(i as usize);
            x.write(i as usize, if d != 0.0 { v / d } else { 0.0 });
        }
        // Solve L·x = z: gather ascending (ancestors before descendants).
        for i in 0..n {
            let mut s = x.read(i as usize);
            let mut j = parents.read(parents_offset + i as usize);
            for _ in 0..MAX_DOFS_U32 {
                if j == NO_PARENT {
                    break;
                }
                s -= mat.read(sm_idx(i, j)) * x.read(j as usize);
                j = parents.read(parents_offset + j as usize);
            }
            x.write(i as usize, s);
        }
    }
    workgroup_memory_barrier_with_group_sync();
}
