//! Fused multibody PGS sweep: joint limit/motor constraints followed by
//! contact constraints, in ONE dispatch per substep phase.
//!
//! Replaces the former `gpu_mb_solve_joint_constraints` /
//! `gpu_mb_remove_solve_joint_no_bias` / `gpu_mb_solve_contact_constraints` /
//! `gpu_mb_remove_contact_constraint_bias` chain (2-3 dispatches per phase,
//! each a fully serial one-thread-per-multibody loop):
//!
//! - One 64-lane workgroup per (multibody, batch); the multibody's
//!   generalized velocities live in WORKGROUP memory for the whole sweep, so
//!   the per-constraint `Δv = delta · column` updates and the `J·v` products
//!   run one-DOF-per-lane against shared memory instead of serial storage
//!   round-trips.
//! - The bias removal that used to be a separate read-modify-write dispatch
//!   is a `use_bias` uniform: the stabilization sweep simply reads
//!   `rhs_wo_bias` (the next substep re-initializes every constraint, so the
//!   persistent rewrite was never needed).
//!
//! The arithmetic per constraint is IDENTICAL to the serial kernels (same
//! product/sum order — lane 0 accumulates the lane products in DOF order), so
//! results are bit-exact with the former chain.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::iter::StepRng;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

use crate::dynamics::body::Velocity;
use crate::gdot;
use crate::utils::BatchIndices;
use crate::utils::linalg::MAX_MB_DOFS;

use super::types::{
    MAX_MB_CONTACT_CONSTRAINTS_PER_MB, MB_CONTACT_KIND_TANGENT, MultibodyContactConstraint,
    MultibodyInfo, MultibodyJointConstraint,
};

const LANES: u32 = 64;

/// One PGS sweep over a multibody's joint (limit/motor) constraints followed
/// by its contact constraints. `use_bias = 0` runs the stabilization form
/// (`rhs_wo_bias`); non-zero runs the biased form (`rhs`).
///
/// Dispatch: one 64-lane workgroup per (multibody, batch) — thread grid
/// `[multibodies_per_batch · 64, num_batches, 1]`.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_solve_constraints(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] joint_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] contact_constraint_jacs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] contact_constraint_columns: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 6)] use_bias: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] batch_ids: &BatchIndices,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] solver_vels: &mut [Velocity],
    #[spirv(workgroup)] dof_v: &mut [f32; MAX_MB_DOFS as usize],
    #[spirv(workgroup)] scratch: &mut [f32; LANES as usize],
    #[spirv(workgroup)] imp_shared: &mut [f32; MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize],
) {
    let batch_id = workgroup_id.y;
    let mb_idx = workgroup_id.x;
    let lane = local_id.x;
    let num_mb = batch_ids.multibodies_len;
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_ids.mb_start(batch_id);
    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let ndofs = mb.ndofs;
    // Uniform per workgroup: every lane of this group returns together.
    if ndofs == 0 {
        return;
    }
    let use_bias = *use_bias != 0;

    let v_base = batch_ids.dof_start(batch_id) + mb.first_dof as usize;
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let colliders_start = batch_ids.coll_start(batch_id);

    let jcons_base =
        batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;
    let jcol_base = batch_ids.mb_joint_constraint_columns_start(batch_id)
        + (mb.first_constraint as usize) * dofs_stride;

    let ccons_base = batch_ids.mb_contact_constraints_start(batch_id)
        + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize);
    let ccol_base = batch_ids.mb_contact_constraint_columns_start(batch_id)
        + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize) * dofs_stride;

    let contact_count = mb.contact_constraint_count;
    // Nothing to solve (common for freely-swinging multibodies in tiny
    // batched environments): skip the load/store round-trip entirely.
    // Workgroup-uniform.
    if mb.max_constraints == 0 && contact_count == 0 {
        return;
    }

    // Load the generalized velocities and accumulated contact impulses into
    // workgroup memory. Impulses stay shared for the whole sweep so the
    // tangent clamp can read its normal's impulse without a storage fence.
    if lane < ndofs {
        dof_v[lane as usize] = dof_state.read(v_base + lane as usize);
    }
    for s in StepRng::new(lane..contact_count, LANES) {
        imp_shared[s as usize] = contact_constraints.read(ccons_base + s as usize).impulse;
    }
    workgroup_memory_barrier_with_group_sync();

    /*
     * Joint (limit/motor) sweep — mirrors the former
     * `solve_joint_constraints_body`, one constraint at a time.
     */
    for s in 0..mb.max_constraints {
        // Every lane reads the same constraint: all per-constraint scalars
        // below are workgroup-uniform, so no broadcast is needed.
        let cons = joint_constraints.read(jcons_base + s as usize);
        if cons.kind == 0 {
            // Uniform skip: all lanes take it together (barrier-safe).
            continue;
        }

        let rhs = if use_bias { cons.rhs } else { cons.rhs_wo_bias };
        let v_d = dof_v[cons.dof_id as usize];
        let rhs_total = v_d + rhs;
        let raw_imp = cons.impulse + cons.inv_lhs * (rhs_total - cons.cfm_gain * cons.impulse);
        let mut new_imp = raw_imp;
        if new_imp < cons.impulse_lo {
            new_imp = cons.impulse_lo;
        }
        if new_imp > cons.impulse_hi {
            new_imp = cons.impulse_hi;
        }
        let delta = new_imp - cons.impulse;

        if lane == 0 {
            let mut cons = cons;
            cons.impulse = new_imp;
            joint_constraints.write(jcons_base + s as usize, cons);
        }

        // All lanes read `dof_v[dof_id]` above; sync before overwriting it.
        workgroup_memory_barrier_with_group_sync();
        if lane < ndofs {
            let col = joint_constraint_columns
                .read(jcol_base + (s as usize) * dofs_stride + lane as usize);
            dof_v[lane as usize] -= delta * col;
        }
        workgroup_memory_barrier_with_group_sync();
    }

    /*
     * Contact sweep — mirrors the former `gpu_mb_solve_contact_constraints`,
     * one constraint at a time; the `J·v` products run one-DOF-per-lane.
     */
    for s in 0..contact_count {
        let cons = contact_constraints.read(ccons_base + s as usize);
        let col_offset = ccol_base + (s as usize) * dofs_stride;
        let is_self = cons.free_body_id == u32::MAX;

        // Multibody side of J · u, one product per lane; lane 0 sums them in
        // DOF order (bit-identical to the old serial accumulation).
        scratch[lane as usize] = if lane < ndofs {
            contact_constraint_jacs.read(col_offset + lane as usize) * dof_v[lane as usize]
        } else {
            0.0
        };
        workgroup_memory_barrier_with_group_sync();

        if lane == 0 {
            let mut j_dot_v = 0.0f32;
            for i in 0..ndofs {
                j_dot_v += scratch[i as usize];
            }
            // Free-body side stays lane-0-local (reads its own prior writes in
            // program order, so no storage fence is needed within the sweep).
            let free = if is_self {
                Velocity::default()
            } else {
                solver_vels.read(colliders_start + cons.free_body_id as usize)
            };
            if !is_self {
                j_dot_v += cons.lin_jac.dot(free.linear) + gdot(cons.ang_jac, free.angular);
            }

            let rhs = if use_bias { cons.rhs } else { cons.rhs_wo_bias };
            let impulse = imp_shared[s as usize];
            let rhs_total = j_dot_v + rhs;
            // CFM-factor form (rapier's `*ContactConstraintNormalPart::generic_solve`).
            let raw_imp = cons.cfm_factor * (impulse - cons.inv_lhs * rhs_total);

            // Normal: clamp to ≥ 0. Friction tangent: clamp to
            // `±μ · normal_impulse` (box friction), reading the paired normal
            // slot's CURRENT impulse from shared memory.
            let new_imp = if cons.kind == MB_CONTACT_KIND_TANGENT {
                let limit =
                    cons.friction_coeff * imp_shared[cons.normal_constraint_slot as usize];
                if raw_imp > limit {
                    limit
                } else if raw_imp < -limit {
                    -limit
                } else {
                    raw_imp
                }
            } else if raw_imp < 0.0 {
                0.0
            } else {
                raw_imp
            };
            let delta = new_imp - impulse;
            imp_shared[s as usize] = new_imp;
            scratch[0] = delta;

            if delta != 0.0 && !is_self {
                let mut new_free = free;
                new_free.linear += cons.lin_jac * (cons.free_body_im * delta);
                new_free.angular += cons.ii_ang_jac * delta;
                solver_vels.write(colliders_start + cons.free_body_id as usize, new_free);
            }
        }
        workgroup_memory_barrier_with_group_sync();

        let delta = scratch[0];
        if delta != 0.0 && lane < ndofs {
            let col = contact_constraint_columns.read(col_offset + lane as usize);
            dof_v[lane as usize] += delta * col;
        }
        workgroup_memory_barrier_with_group_sync();
    }

    /*
     * Writeback: generalized velocities and accumulated contact impulses.
     */
    if lane < ndofs {
        dof_state.write(v_base + lane as usize, dof_v[lane as usize]);
    }
    for s in StepRng::new(lane..contact_count, LANES) {
        let mut cons = contact_constraints.read(ccons_base + s as usize);
        cons.impulse = imp_shared[s as usize];
        contact_constraints.write(ccons_base + s as usize, cons);
    }
}
