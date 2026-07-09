//! Jacobian packing helpers shared by the multibody impulse-joint kernels:
//! per-side spatial-velocity access and `J` / `W·J` row construction.

use khal_std::index::MaybeIndexUnchecked;
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

use crate::dynamics::body::{Velocity, WorldMassProperties};
use crate::dynamics::joint::SPATIAL_DIM;
use crate::utils::linalg::MatSlice;
use crate::{ANG_DIM, AngVector, DIM, Vector};

use super::super::lu::LANES;
use super::super::types::MultibodyInfo;
use super::types::*;

// Returns the multibody side jacobian's offset / size in the per-batch
// jacobians buffer. `wj_id` is the start of the corresponding `M⁻¹·J`
// block (= `j_id + ndofs`).
#[inline]
pub(super) fn wj_id(j_id: u32, ndofs: u32) -> usize {
    (j_id + ndofs) as usize
}

/// `k`-th component of a free body's spatial velocity, in the same order
/// the jacobian rows are packed: `[lin (DIM), ang (ANG_DIM)]`. Returns a
/// value (not a reference) to avoid SPIR-V pointer-phi nodes.
#[inline]
pub(super) fn spatial_component(v: Velocity, k: u32) -> f32 {
    #[cfg(feature = "dim3")]
    {
        if k == 0 {
            v.linear.x
        } else if k == 1 {
            v.linear.y
        } else if k == 2 {
            v.linear.z
        } else if k == 3 {
            v.angular.x
        } else if k == 4 {
            v.angular.y
        } else {
            v.angular.z
        }
    }
    #[cfg(feature = "dim2")]
    {
        if k == 0 {
            v.linear.x
        } else if k == 1 {
            v.linear.y
        } else {
            v.angular
        }
    }
}

/// Workgroup-cooperative `J · v` for a generic side; the scalar result is
/// broadcast to all lanes.
///
/// The barrier sequence is **identical for every side kind and for
/// inactive constraints** (`active == false` just zeroes the term), so the
/// enclosing per-axis loop stays in workgroup-uniform control flow — no
/// lane ever skips a barrier another lane executes.
#[inline]
pub(super) fn side_dot_vel_par(
    active: bool,
    kind: u32,
    j_id: u32,
    ndofs: u32,
    body_id: u32,
    jacobians: &[f32],
    dof_vels: &[f32],
    dof_base_for_mb: usize,
    solver_vels: &[Velocity],
    colliders_start: usize,
    lane: u32,
    partial: &mut impl MaybeIndexUnchecked<f32>,
) -> f32 {
    // Lane → term mapping:
    //   * SIDE_KIND_MB:   lane l (< ndofs)      → J[l] · v_dof[l]
    //   * SIDE_KIND_BODY: lane k (< SPATIAL_DIM) → J[k] · v_spatial[k]
    //   * FIXED / inactive / out-of-range        → 0
    let term = if !active || kind == SIDE_KIND_FIXED {
        0.0f32
    } else if kind == SIDE_KIND_BODY {
        if lane < SPATIAL_DIM as u32 {
            let v = solver_vels.read(colliders_start + body_id as usize);
            jacobians.read(j_id as usize + lane as usize) * spatial_component(v, lane)
        } else {
            0.0f32
        }
    } else {
        // SIDE_KIND_MB
        if lane < ndofs {
            jacobians.read(j_id as usize + lane as usize)
                * dof_vels.read(dof_base_for_mb + lane as usize)
        } else {
            0.0f32
        }
    };

    partial.write(lane as usize, term);
    workgroup_memory_barrier_with_group_sync();
    // Tree reduction over the 64 lanes (2^6 == LANES).
    for step in 0..6u32 {
        let stride = 1u32 << (5 - step);
        if lane < stride {
            let v = partial.read(lane as usize) + partial.read((lane + stride) as usize);
            partial.write(lane as usize, v);
        }
        workgroup_memory_barrier_with_group_sync();
    }
    let result = partial.read(0);
    // Trailing barrier: guarantees every lane has read `partial[0]` before
    // the next reduction (or caller) overwrites `partial`.
    workgroup_memory_barrier_with_group_sync();
    result
}

/// Workgroup-cooperative `±delta · W·J` apply. Contains no barriers — the
/// caller must issue one unconditional barrier per axis after both apply
/// calls so the velocity writes are visible to the next axis's dot products.
#[inline]
pub(super) fn side_apply_impulse_par(
    active: bool,
    kind: u32,
    j_id: u32,
    ndofs: u32,
    body_id: u32,
    sign: f32,
    delta: f32,
    jacobians: &[f32],
    dof_vels: &mut [f32],
    dof_base_for_mb: usize,
    solver_vels: &mut [Velocity],
    colliders_start: usize,
    lane: u32,
) {
    // All operands are workgroup-uniform, so this early-out is uniform.
    if !active || kind == SIDE_KIND_FIXED || delta == 0.0 {
        return;
    }
    let wj0 = wj_id(j_id, ndofs);
    let scaled = sign * delta;
    if kind == SIDE_KIND_BODY {
        if lane == 0 {
            let coll_idx = colliders_start + body_id as usize;
            let mut v = solver_vels.read(coll_idx);
            #[cfg(feature = "dim3")]
            {
                v.linear.x += scaled * jacobians.read(wj0);
                v.linear.y += scaled * jacobians.read(wj0 + 1);
                v.linear.z += scaled * jacobians.read(wj0 + 2);
                v.angular.x += scaled * jacobians.read(wj0 + 3);
                v.angular.y += scaled * jacobians.read(wj0 + 4);
                v.angular.z += scaled * jacobians.read(wj0 + 5);
            }
            #[cfg(feature = "dim2")]
            {
                v.linear.x += scaled * jacobians.read(wj0);
                v.linear.y += scaled * jacobians.read(wj0 + 1);
                v.angular += scaled * jacobians.read(wj0 + 2);
            }
            solver_vels.write(coll_idx, v);
        }
        return;
    }
    // SIDE_KIND_MB — lane l owns DOF l (disjoint → race-free).
    if lane < ndofs {
        let v_idx = dof_base_for_mb + lane as usize;
        let cur = dof_vels.read(v_idx);
        let w = jacobians.read(wj0 + lane as usize);
        dof_vels.write(v_idx, cur + scaled * w);
    }
}

/// Stride (in floats) reserved per axis-constraint in the jacobians buffer:
/// `J_a (ndofs_a) + W·J_a (ndofs_a) + J_b (ndofs_b) + W·J_b (ndofs_b)`.
#[inline]
pub(super) fn axis_stride(ndofs_a: u32, ndofs_b: u32) -> u32 {
    2 * (ndofs_a + ndofs_b)
}

/// Write a free body's `(unit_force, unit_torque)` Jᵀ row + its `W·Jᵀ`
/// (= `(im⊙f, ii·t)`) into the jacobians buffer at `j_id`. Mirrors rapier's
/// `JointSolverBody::fill_jacobians`.
pub(super) fn fill_body_jacobians(
    jacobians: &mut [f32],
    j_id: u32,
    body_id: u32,
    unit_force: Vector,
    unit_torque: AngVector,
    mprops: &[WorldMassProperties],
    colliders_start: usize,
) {
    let mp = mprops.read(colliders_start + body_id as usize);
    let im = mp.inv_mass;

    let base = j_id as usize;
    #[cfg(feature = "dim3")]
    {
        jacobians.write(base, unit_force.x);
        jacobians.write(base + 1, unit_force.y);
        jacobians.write(base + 2, unit_force.z);
        jacobians.write(base + 3, unit_torque.x);
        jacobians.write(base + 4, unit_torque.y);
        jacobians.write(base + 5, unit_torque.z);
    }
    #[cfg(feature = "dim2")]
    {
        jacobians.write(base, unit_force.x);
        jacobians.write(base + 1, unit_force.y);
        jacobians.write(base + 2, unit_torque);
    }

    // W·J: linear part = im ⊙ unit_force; angular part = ii · unit_torque.
    let wbase = base + SPATIAL_DIM;
    #[cfg(feature = "dim3")]
    {
        jacobians.write(wbase, im.x * unit_force.x);
        jacobians.write(wbase + 1, im.y * unit_force.y);
        jacobians.write(wbase + 2, im.z * unit_force.z);
        let inv_i = mp.inv_inertia;
        let it = (inv_i * unit_torque.extend(0.0)).truncate();
        jacobians.write(wbase + 3, it.x);
        jacobians.write(wbase + 4, it.y);
        jacobians.write(wbase + 5, it.z);
    }
    #[cfg(feature = "dim2")]
    {
        jacobians.write(wbase, im.x * unit_force.x);
        jacobians.write(wbase + 1, im.y * unit_force.y);
        jacobians.write(wbase + 2, mp.inv_inertia * unit_torque);
    }
}

/// Write a multibody link's projected `Jᵀ` row into the jacobians buffer at
/// `j_id` (the `M⁻¹·Jᵀ` back-solve is deferred to the finalize kernel).
/// Mirrors rapier's `Multibody::fill_jacobians`.
pub(super) fn fill_mb_jacobians(
    jacobians: &mut [f32],
    j_id: u32,
    mb: &MultibodyInfo,
    link_id: u32,
    unit_force: Vector,
    unit_torque: AngVector,
    body_jacobians: &[f32],
    jac_start: usize,
) {
    let ndofs = mb.ndofs;
    let mb_jac_base = jac_start + mb.jacobian_offset as usize;
    let link_jac_base = mb_jac_base + (link_id as usize) * SPATIAL_DIM * (ndofs as usize);
    let link_j = MatSlice::dense(link_jac_base, SPATIAL_DIM as u32, ndofs);
    let (link_j_v, link_j_w) = link_j.rows_range_pair(0, DIM, DIM, ANG_DIM);

    // 1) j = link_J^T · (unit_force, unit_torque). Same kernel used by
    //    `fill_contact_jac_row` in `contact_constraints`.
    for k in 0..ndofs {
        let dot;
        #[cfg(feature = "dim3")]
        {
            let jv0 = body_jacobians.read(link_j_v.idx(0, k));
            let jv1 = body_jacobians.read(link_j_v.idx(1, k));
            let jv2 = body_jacobians.read(link_j_v.idx(2, k));
            let jw0 = body_jacobians.read(link_j_w.idx(0, k));
            let jw1 = body_jacobians.read(link_j_w.idx(1, k));
            let jw2 = body_jacobians.read(link_j_w.idx(2, k));
            dot = unit_force.x * jv0
                + unit_force.y * jv1
                + unit_force.z * jv2
                + unit_torque.x * jw0
                + unit_torque.y * jw1
                + unit_torque.z * jw2;
        }
        #[cfg(feature = "dim2")]
        {
            let jv0 = body_jacobians.read(link_j_v.idx(0, k));
            let jv1 = body_jacobians.read(link_j_v.idx(1, k));
            let jw0 = body_jacobians.read(link_j_w.idx(0, k));
            dot = unit_force.x * jv0 + unit_force.y * jv1 + unit_torque * jw0;
        }
        jacobians.write(j_id as usize + k as usize, dot);
    }
    // The `M⁻¹·Jᵀ` (W·J) back-solve is deferred to
    // `gpu_mb_finalize_impulse_joint_constraints` (so this build pass stays
    // within the 8-storage-buffer WebGPU limit — it no longer binds
    // `mass_matrices` / `lu_pivots`).
}

/// Same-multibody (loop-closure) variant of [`fill_mb_jacobians`]: when BOTH
/// attachment links of an impulse joint belong to the same multibody, combines
/// the two body jacobians into ONE relative jacobian `J_rel = J_bᵀ·f_b -
/// J_aᵀ·f_a` (keeping a block per side would lose the coupling and blow up on
/// tight loops).
///
/// Writes `J_rel` at `j_id` (the `M⁻¹·J_rel` back-solve is deferred to the
/// finalize kernel). The caller stores this block on the constraint's "B" side
/// and sets `ndofs_a = 0`. Mirrors `Multibody::fill_relative_jacobians` (rapier
/// 9265a19b), including the cancellation guard. 3D only (2D falls back to the
/// separate-block path).
#[cfg(feature = "dim3")]
#[allow(clippy::too_many_arguments)]
pub(super) fn fill_relative_mb_jacobians(
    jacobians: &mut [f32],
    j_id: u32,
    mb: &MultibodyInfo,
    link_a: u32,
    lin_a: Vector,
    ang_a: AngVector,
    link_b: u32,
    lin_b: Vector,
    ang_b: AngVector,
    body_jacobians: &[f32],
    jac_start: usize,
) {
    let ndofs = mb.ndofs;
    let mb_jac_base = jac_start + mb.jacobian_offset as usize;
    let la_base = mb_jac_base + (link_a as usize) * SPATIAL_DIM * (ndofs as usize);
    let lb_base = mb_jac_base + (link_b as usize) * SPATIAL_DIM * (ndofs as usize);
    let la = MatSlice::dense(la_base, SPATIAL_DIM as u32, ndofs);
    let lb = MatSlice::dense(lb_base, SPATIAL_DIM as u32, ndofs);
    let (la_v, la_w) = la.rows_range_pair(0, DIM, DIM, ANG_DIM);
    let (lb_v, lb_w) = lb.rows_range_pair(0, DIM, DIM, ANG_DIM);

    // 1) J_rel[k] = (link_b_Jᵀ·f_b)[k] - (link_a_Jᵀ·f_a)[k]; accumulate the
    //    Frobenius norms of the two body jacobians for the cancellation guard.
    let mut jba_nsq = 0.0f32;
    let mut jbb_nsq = 0.0f32;
    for k in 0..ndofs {
        let av0 = body_jacobians.read(la_v.idx(0, k));
        let av1 = body_jacobians.read(la_v.idx(1, k));
        let av2 = body_jacobians.read(la_v.idx(2, k));
        let aw0 = body_jacobians.read(la_w.idx(0, k));
        let aw1 = body_jacobians.read(la_w.idx(1, k));
        let aw2 = body_jacobians.read(la_w.idx(2, k));
        let bv0 = body_jacobians.read(lb_v.idx(0, k));
        let bv1 = body_jacobians.read(lb_v.idx(1, k));
        let bv2 = body_jacobians.read(lb_v.idx(2, k));
        let bw0 = body_jacobians.read(lb_w.idx(0, k));
        let bw1 = body_jacobians.read(lb_w.idx(1, k));
        let bw2 = body_jacobians.read(lb_w.idx(2, k));

        let ja = lin_a.x * av0
            + lin_a.y * av1
            + lin_a.z * av2
            + ang_a.x * aw0
            + ang_a.y * aw1
            + ang_a.z * aw2;
        let jb = lin_b.x * bv0
            + lin_b.y * bv1
            + lin_b.z * bv2
            + ang_b.x * bw0
            + ang_b.y * bw1
            + ang_b.z * bw2;
        jacobians.write(j_id as usize + k as usize, jb - ja);

        jba_nsq += av0 * av0 + av1 * av1 + av2 * av2 + aw0 * aw0 + aw1 * aw1 + aw2 * aw2;
        jbb_nsq += bv0 * bv0 + bv1 * bv1 + bv2 * bv2 + bw0 * bw0 + bw1 * bw1 + bw2 * bw2;
    }

    // 2) Cancellation guard (rapier). The reference scale is the magnitude of
    //    the dot-product *operands* (body jacobian × force), not of the result
    //    `J_rel` (which may itself be pure cancellation noise when the
    //    constrained direction isn't expressible by the multibody's DOFs). A
    //    `J_rel` this small is noise, not a real constraint direction → zero it
    //    so it contributes no (huge, mass-near-zero) impulse.
    let fa_sq = lin_a.dot(lin_a) + ang_a.dot(ang_a);
    let fb_sq = lin_b.dot(lin_b) + ang_b.dot(ang_b);
    let scale_sq = jba_nsq * fa_sq + jbb_nsq * fb_sq;
    let mut jrel_nsq = 0.0f32;
    for k in 0..ndofs {
        let v = jacobians.read(j_id as usize + k as usize);
        jrel_nsq += v * v;
    }
    let eps = f32::EPSILON * 1.0e3;
    if jrel_nsq <= eps * eps * scale_sq {
        for k in 0..ndofs {
            jacobians.write(j_id as usize + k as usize, 0.0);
        }
    }
    // 3) `W·J_rel = M⁻¹·J_rel` back-solve is deferred to
    //    `gpu_mb_finalize_impulse_joint_constraints` (the relative block is
    //    stored on side B as a normal MB jacobian, so the finalize pass solves
    //    it like any other MB side).
}
