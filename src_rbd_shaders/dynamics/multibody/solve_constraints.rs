//! Fused multibody PGS iteration: joint limit/motor constraints followed by
//! contact constraints, in one dispatch per substep phase.

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
    MAX_MB_CONTACT_CONSTRAINTS_PER_MB, MB_CONTACT_KIND_TANGENT, MB_JOINT_KIND_COUPLING,
    MB_JOINT_KIND_LIMIT, MB_JOINT_KIND_MOTOR, MultibodyContactConstraint, MultibodyInfo,
    MultibodyJointConstraint,
};

const LANES: u32 = 64;

/// Caps a friction impulse to the circular cone of radius `limit`. In 3D both
/// tangent rows of a contact point are capped jointly; in 2D there is a single
/// row and this degenerates to a scalar clamp.
#[inline]
fn cap_friction(t0: f32, t1: f32, limit: f32) -> (f32, f32) {
    let norm_sq = t0 * t0 + t1 * t1;
    if norm_sq > limit * limit && norm_sq > 0.0 {
        let scale = limit / crate::sqrt(norm_sq);
        (t0 * scale, t1 * scale)
    } else {
        (t0, t1)
    }
}

/// One PGS iteration over a multibody's joint (limit/motor) constraints followed
/// by its contact constraints.
///
/// Dispatch: one 64-lane workgroup per (multibody, batch).
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
    #[spirv(workgroup)] delta_shared: &mut f32,
    #[spirv(workgroup)] delta2_shared: &mut f32,
) {
    let batch_id = workgroup_id.y;
    let mb_idx = workgroup_id.x;
    let lane = local_id.x;
    let num_mb = batch_ids.multibodies_len;
    if mb_idx >= num_mb {
        return;
    }

    let mb = multibody_info.read(batch_ids.mbi(batch_id, mb_idx as usize));
    let ndofs = mb.ndofs;
    // Uniform per workgroup: every lane of this group returns together.
    if ndofs == 0 {
        return;
    }
    let use_bias = *use_bias != 0;

    let v_base = mb.first_dof as usize;
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
    if mb.max_constraints == 0 && contact_count == 0 {
        // Nothing to solve.
        return;
    }

    // Load the generalized velocities and accumulated contact impulses into
    // workgroup memory.
    if lane < ndofs {
        dof_v[lane as usize] = dof_state.read(batch_ids.mbi(batch_id, v_base + lane as usize));
    }
    for s in StepRng::new(lane..contact_count, LANES) {
        imp_shared[s as usize] = contact_constraints.read(ccons_base + s as usize).impulse;
    }
    workgroup_memory_barrier_with_group_sync();


    // Joint limits/motors
    for s in 0..mb.max_constraints {
        let cons = joint_constraints.read(jcons_base + s as usize);
        if cons.kind != MB_JOINT_KIND_LIMIT
            && cons.kind != MB_JOINT_KIND_MOTOR
            && cons.kind != MB_JOINT_KIND_COUPLING
        {
            // Unused slot or inactive limit.
            continue;
        }

        let rhs = if use_bias { cons.rhs } else { cons.rhs_wo_bias };
        // Generalized `J·v` for `J = e_{dof_id} - coupling_coeff*e_{dof2_id}`
        // (coupling rows); collapses to `v[dof_id]` for limit / motor rows
        // (their `coupling_coeff` is 0).
        let v_d = dof_v[cons.dof_id as usize]
            - cons.coupling_coeff * dof_v[cons.dof2_id as usize];
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


    // Contacts. In 3D the two friction rows of a contact point are solved
    // together so their impulse can be capped to the friction cone; the second
    // row is handled by its sibling and skipped here.
    for s in 0..contact_count {
        let cons = contact_constraints.read(ccons_base + s as usize);
        let is_tangent = cons.kind == MB_CONTACT_KIND_TANGENT;
        // Friction is only solved during the relaxation phase.
        if use_bias && is_tangent {
            continue;
        }
        #[cfg(feature = "dim3")]
        if is_tangent && s != cons.normal_constraint_slot + 1 {
            continue;
        }
        #[cfg(feature = "dim3")]
        let has_pair = is_tangent;
        #[cfg(feature = "dim2")]
        let has_pair = false;

        let col_offset = ccol_base + (s as usize) * dofs_stride;
        let col_offset2 = col_offset + dofs_stride;
        let is_self = cons.free_body_id == u32::MAX;

        // Multibody side of J · u, one product per lane; lane 0 sums them in
        // DOF order.
        scratch[lane as usize] = if lane < ndofs {
            contact_constraint_jacs.read(col_offset + lane as usize) * dof_v[lane as usize]
        } else {
            0.0
        };
        workgroup_memory_barrier_with_group_sync();

        let mut j_dot_v0 = 0.0f32;
        if lane == 0 {
            for i in 0..ndofs {
                j_dot_v0 += scratch[i as usize];
            }
        }
        workgroup_memory_barrier_with_group_sync();

        if has_pair {
            scratch[lane as usize] = if lane < ndofs {
                contact_constraint_jacs.read(col_offset2 + lane as usize) * dof_v[lane as usize]
            } else {
                0.0
            };
        }
        workgroup_memory_barrier_with_group_sync();

        if lane == 0 {
            let cons2 = contact_constraints.read(ccons_base + (s + 1) as usize);
            let mut j_dot_v1 = 0.0f32;
            if has_pair {
                for i in 0..ndofs {
                    j_dot_v1 += scratch[i as usize];
                }
            }
            // Free-body side stays lane-0-local.
            let free = if is_self {
                Velocity::default()
            } else {
                solver_vels.read(colliders_start + cons.free_body_id as usize)
            };
            if !is_self {
                j_dot_v0 += cons.lin_jac.dot(free.linear) + gdot(cons.ang_jac, free.angular);
                if has_pair {
                    j_dot_v1 += cons2.lin_jac.dot(free.linear) + gdot(cons2.ang_jac, free.angular);
                }
            }

            let cfm_factor = if use_bias { cons.cfm_factor } else { 1.0 };
            let impulse0 = imp_shared[s as usize];
            let rhs0 = if use_bias { cons.rhs } else { cons.rhs_wo_bias };
            let raw0 = cfm_factor * (impulse0 - cons.inv_lhs * (j_dot_v0 + rhs0));

            let impulse1 = if has_pair {
                imp_shared[(s + 1) as usize]
            } else {
                0.0
            };
            let raw1 = if has_pair {
                let rhs1 = if use_bias { cons2.rhs } else { cons2.rhs_wo_bias };
                cfm_factor * (impulse1 - cons2.inv_lhs * (j_dot_v1 + rhs1))
            } else {
                0.0
            };

            // Normal: clamp to ≥ 0. Friction: cap the tangent pair to the
            // circular cone `μ · normal_impulse`.
            let (new0, new1) = if is_tangent {
                let limit = cons.friction_coeff * imp_shared[cons.normal_constraint_slot as usize];
                cap_friction(raw0, raw1, limit)
            } else if raw0 < 0.0 {
                (0.0, 0.0)
            } else {
                (raw0, 0.0)
            };

            let delta0 = new0 - impulse0;
            let delta1 = if has_pair { new1 - impulse1 } else { 0.0 };
            imp_shared[s as usize] = new0;
            if has_pair {
                imp_shared[(s + 1) as usize] = new1;
            }
            *delta_shared = delta0;
            *delta2_shared = delta1;

            if !is_self && (delta0 != 0.0 || delta1 != 0.0) {
                let mut new_free = free;
                new_free.linear += cons.lin_jac * (cons.free_body_im * delta0);
                new_free.angular += cons.ii_ang_jac * delta0;
                if has_pair {
                    new_free.linear += cons2.lin_jac * (cons2.free_body_im * delta1);
                    new_free.angular += cons2.ii_ang_jac * delta1;
                }
                solver_vels.write(colliders_start + cons.free_body_id as usize, new_free);
            }
        }
        workgroup_memory_barrier_with_group_sync();

        // Per-lane `dof_v[lane]` update.
        let delta0 = *delta_shared;
        let delta1 = *delta2_shared;
        if lane < ndofs {
            if delta0 != 0.0 {
                let col = contact_constraint_columns.read(col_offset + lane as usize);
                dof_v[lane as usize] += delta0 * col;
            }
            if has_pair && delta1 != 0.0 {
                let col = contact_constraint_columns.read(col_offset2 + lane as usize);
                dof_v[lane as usize] += delta1 * col;
            }
        }
        workgroup_memory_barrier_with_group_sync();
    }

    // Writeback
    if lane < ndofs {
        dof_state.write(
            batch_ids.mbi(batch_id, v_base + lane as usize),
            dof_v[lane as usize],
        );
    }
    for s in StepRng::new(lane..contact_count, LANES) {
        let mut cons = contact_constraints.read(ccons_base + s as usize);
        cons.impulse = imp_shared[s as usize];
        contact_constraints.write(ccons_base + s as usize, cons);
    }
}

/// Joint-only PGS iteration (the joint half of [`gpu_mb_solve_constraints`]),
/// used by the Delassus path where the contact half runs in constraint space
/// as a separate dispatch (to avoid exceeding the 8-storage-buffer budget).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_solve_joints(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] joint_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dof_state: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] use_bias: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
    #[spirv(workgroup)] dof_v: &mut [f32; MAX_MB_DOFS as usize],
) {
    let batch_id = workgroup_id.y;
    let mb_idx = workgroup_id.x;
    let lane = local_id.x;
    let num_mb = batch_ids.multibodies_len;
    if mb_idx >= num_mb {
        return;
    }

    let mb = multibody_info.read(batch_ids.mbi(batch_id, mb_idx as usize));
    let ndofs = mb.ndofs;
    // Uniform per workgroup: every lane of this group returns together.
    if ndofs == 0 || mb.max_constraints == 0 {
        return;
    }
    let use_bias = *use_bias != 0;

    let v_base = mb.first_dof as usize;
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let jcons_base =
        batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;
    let jcol_base = batch_ids.mb_joint_constraint_columns_start(batch_id)
        + (mb.first_constraint as usize) * dofs_stride;

    if lane < ndofs {
        dof_v[lane as usize] = dof_state.read(batch_ids.mbi(batch_id, v_base + lane as usize));
    }
    workgroup_memory_barrier_with_group_sync();

    for s in 0..mb.max_constraints {
        let cons = joint_constraints.read(jcons_base + s as usize);
        if cons.kind != MB_JOINT_KIND_LIMIT
            && cons.kind != MB_JOINT_KIND_MOTOR
            && cons.kind != MB_JOINT_KIND_COUPLING
        {
            // Unused slot or inactive limit.
            continue;
        }

        let rhs = if use_bias { cons.rhs } else { cons.rhs_wo_bias };
        // Generalized `J·v` for `J = e_{dof_id} - coupling_coeff*e_{dof2_id}`
        // (coupling rows); collapses to `v[dof_id]` for limit / motor rows
        // (their `coupling_coeff` is 0).
        let v_d = dof_v[cons.dof_id as usize]
            - cons.coupling_coeff * dof_v[cons.dof2_id as usize];
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

        workgroup_memory_barrier_with_group_sync();
        if lane < ndofs {
            let col = joint_constraint_columns
                .read(jcol_base + (s as usize) * dofs_stride + lane as usize);
            dof_v[lane as usize] -= delta * col;
        }
        workgroup_memory_barrier_with_group_sync();
    }

    if lane < ndofs {
        dof_state.write(
            batch_ids.mbi(batch_id, v_base + lane as usize),
            dof_v[lane as usize],
        );
    }
}

/// Fills the per-multibody Delassus block `D[s][j] = ∂a[j]/∂impulse[s]` (row
/// `s` = the effect of constraint `s`, laid out row-contiguously so the solve
/// kernel's per-iteration row update reads coalesced). Runs right after
/// `gpu_mb_finalize_contact_constraints` (it consumes the M⁻¹Jᵀ columns).
///
/// One 64-lane workgroup per (multibody, batch).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_build_contact_delassus(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    contact_constraints: &[MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contact_constraint_jacs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] contact_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] delassus: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
) {
    const MAXC: u32 = MAX_MB_CONTACT_CONSTRAINTS_PER_MB;
    let batch_id = workgroup_id.y;
    let mb_idx = workgroup_id.x;
    let lane = local_id.x;
    let num_mb = batch_ids.multibodies_len;
    if mb_idx >= num_mb {
        return;
    }

    let mb = multibody_info.read(batch_ids.mbi(batch_id, mb_idx as usize));
    let ndofs = mb.ndofs;
    let count = mb.contact_constraint_count;
    if ndofs == 0 || count == 0 {
        return;
    }

    let cons_base = batch_ids.mb_contact_constraints_start(batch_id)
        + (mb_idx as usize) * (MAXC as usize);
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let col_base = batch_ids.mb_contact_constraint_columns_start(batch_id)
        + (mb_idx as usize) * (MAXC as usize) * dofs_stride;
    let d_base = ((batch_id * batch_ids.multibodies_batch_capacity + mb_idx) as usize)
        * (MAXC as usize)
        * (MAXC as usize);

    // Pair `p = s · count + j`: consecutive lanes share the source row `s`
    // and vary the target `j`, so the column reads of `s` broadcast and the
    // `D` writes coalesce.
    let num_pairs = count * count;
    for p in StepRng::new(lane..num_pairs, LANES) {
        let s = p / count;
        let j = p % count;

        // Multibody coupling: jac_j · (M⁻¹ jac_sᵀ).
        let jac_j_off = col_base + (j as usize) * dofs_stride;
        let col_s_off = col_base + (s as usize) * dofs_stride;
        let mut v = 0.0f32;
        for i in 0..ndofs {
            let jj = contact_constraint_jacs.read(jac_j_off + i as usize);
            let cs = contact_constraint_columns.read(col_s_off + i as usize);
            v += jj * cs;
        }

        // Free-body coupling (impulse at `s` moves the shared free body,
        // which feeds `a[j]`'s free-side term). Zero for self-contacts and
        // static free bodies.
        let cons_s = contact_constraints.read(cons_base + s as usize);
        let cons_j = contact_constraints.read(cons_base + j as usize);
        if cons_s.free_body_id != u32::MAX && cons_s.free_body_id == cons_j.free_body_id {
            v += cons_s.free_body_im * cons_j.lin_jac.dot(cons_s.lin_jac)
                + gdot(cons_j.ang_jac, cons_s.ii_ang_jac);
        }

        delassus.write(d_base + (s * MAXC + j) as usize, v);
    }
}

/// Constraint-space contact sweep: tracks `a[s] = J_s · u` incrementally in
/// workgroup memory using the precomputed Delassus rows, so each PGS
/// iteration is a couple of shared-memory scalars plus one lane-parallel row
/// update.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_solve_contacts_delassus(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contact_constraint_jacs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] contact_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] delassus: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] use_bias: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] batch_ids: &BatchIndices,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] solver_vels: &mut [Velocity],
    #[spirv(workgroup)] dof_v: &mut [f32; MAX_MB_DOFS as usize],
    #[spirv(workgroup)] a_shared: &mut [f32; MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize],
    #[spirv(workgroup)] imp_shared: &mut [f32; MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize],
    #[spirv(workgroup)] rhs_shared: &mut [f32; MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize],
    #[spirv(workgroup)] inv_lhs_shared: &mut [f32; MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize],
    #[spirv(workgroup)] cfm_shared: &mut [f32; MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize],
    #[spirv(workgroup)] friction_shared: &mut [f32; MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize],
    #[spirv(workgroup)] meta_shared: &mut [u32; MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize],
) {
    const MAXC: u32 = MAX_MB_CONTACT_CONSTRAINTS_PER_MB;
    let batch_id = workgroup_id.y;
    let mb_idx = workgroup_id.x;
    let lane = local_id.x;
    let num_mb = batch_ids.multibodies_len;
    if mb_idx >= num_mb {
        return;
    }

    let mb = multibody_info.read(batch_ids.mbi(batch_id, mb_idx as usize));
    let ndofs = mb.ndofs;
    let count = mb.contact_constraint_count;
    // Uniform per workgroup: every lane of this group returns together.
    if ndofs == 0 || count == 0 {
        return;
    }
    let use_bias = *use_bias != 0;

    let v_base = mb.first_dof as usize;
    let colliders_start = batch_ids.coll_start(batch_id);
    let cons_base = batch_ids.mb_contact_constraints_start(batch_id)
        + (mb_idx as usize) * (MAXC as usize);
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let col_base = batch_ids.mb_contact_constraint_columns_start(batch_id)
        + (mb_idx as usize) * (MAXC as usize) * dofs_stride;
    let d_base = ((batch_id * batch_ids.multibodies_batch_capacity + mb_idx) as usize)
        * (MAXC as usize)
        * (MAXC as usize);

    if lane < ndofs {
        dof_v[lane as usize] = dof_state.read(batch_ids.mbi(batch_id, v_base + lane as usize));
    }

    // Preload the per-constraint solve scalars into shared SoA arrays so the
    // serial recurrence below never touches storage on its critical path.
    // `meta` packs the kind, the paired normal slot, and whether the
    // free-body side needs the fire-and-forget storage velocity update.
    for s in StepRng::new(lane..count, LANES) {
        let cons = contact_constraints.read(cons_base + s as usize);
        imp_shared[s as usize] = cons.impulse;
        rhs_shared[s as usize] = if use_bias { cons.rhs } else { cons.rhs_wo_bias };
        inv_lhs_shared[s as usize] = cons.inv_lhs;
        cfm_shared[s as usize] = if use_bias { cons.cfm_factor } else { 1.0 };
        friction_shared[s as usize] = cons.friction_coeff;
        let is_self = cons.free_body_id == u32::MAX;
        let free_active = !is_self
            && (cons.free_body_im != 0.0 || gdot(cons.ii_ang_jac, cons.ii_ang_jac) != 0.0);
        meta_shared[s as usize] = (cons.kind & 0xff)
            | ((cons.normal_constraint_slot & 0xffff) << 8)
            | (if free_active { 1 << 24 } else { 0 });
    }
    workgroup_memory_barrier_with_group_sync();

    // Fresh `a[s] = J_s · u` under the current (post-joint-sweep, post-
    // warmstart) velocities.
    for s in StepRng::new(lane..count, LANES) {
        let jac_off = col_base + (s as usize) * dofs_stride;
        let mut dot = 0.0f32;
        for i in 0..ndofs {
            dot += contact_constraint_jacs.read(jac_off + i as usize) * dof_v[i as usize];
        }
        let cons = contact_constraints.read(cons_base + s as usize);
        if cons.free_body_id != u32::MAX {
            let free = solver_vels.read(colliders_start + cons.free_body_id as usize);
            dot += cons.lin_jac.dot(free.linear) + gdot(cons.ang_jac, free.angular);
        }
        a_shared[s as usize] = dot;
    }
    workgroup_memory_barrier_with_group_sync();

    // In 3D the two friction rows of a contact point are solved together so
    // their impulse can be capped to the friction cone; the second row is
    // handled by its sibling and skipped here.
    for s in 0..count {
        let meta = meta_shared[s as usize];
        let kind = meta & 0xff;
        let normal_slot = (meta >> 8) & 0xffff;
        let free_active = (meta >> 24) != 0;
        let is_tangent = kind == MB_CONTACT_KIND_TANGENT;

        if use_bias && is_tangent {
            // Friction is only solved during the stabilization sweep.
            continue;
        }
        #[cfg(feature = "dim3")]
        if is_tangent && s != normal_slot + 1 {
            continue;
        }
        #[cfg(feature = "dim3")]
        let has_pair = is_tangent;
        #[cfg(feature = "dim2")]
        let has_pair = false;

        let impulse0 = imp_shared[s as usize];
        let raw0 = cfm_shared[s as usize]
            * (impulse0 - inv_lhs_shared[s as usize] * (a_shared[s as usize] + rhs_shared[s as usize]));
        let impulse1 = if has_pair {
            imp_shared[(s + 1) as usize]
        } else {
            0.0
        };
        let raw1 = if has_pair {
            cfm_shared[(s + 1) as usize]
                * (impulse1
                    - inv_lhs_shared[(s + 1) as usize]
                        * (a_shared[(s + 1) as usize] + rhs_shared[(s + 1) as usize]))
        } else {
            0.0
        };

        let (new0, new1) = if is_tangent {
            let limit = friction_shared[s as usize] * imp_shared[normal_slot as usize];
            cap_friction(raw0, raw1, limit)
        } else if raw0 < 0.0 {
            (0.0, 0.0)
        } else {
            (raw0, 0.0)
        };
        let delta0 = new0 - impulse0;
        let delta1 = if has_pair { new1 - impulse1 } else { 0.0 };

        if delta0 != 0.0 || delta1 != 0.0 {
            if lane == 0 {
                imp_shared[s as usize] = new0;
                if has_pair {
                    imp_shared[(s + 1) as usize] = new1;
                }

                if free_active {
                    let cons = contact_constraints.read(cons_base + s as usize);
                    let mut free =
                        solver_vels.read(colliders_start + cons.free_body_id as usize);
                    free.linear += cons.lin_jac * (cons.free_body_im * delta0);
                    free.angular += cons.ii_ang_jac * delta0;
                    if has_pair {
                        let cons2 = contact_constraints.read(cons_base + (s + 1) as usize);
                        free.linear += cons2.lin_jac * (cons2.free_body_im * delta1);
                        free.angular += cons2.ii_ang_jac * delta1;
                    }
                    solver_vels.write(colliders_start + cons.free_body_id as usize, free);
                }
            }
            // Lane-parallel Delassus row update (row `s` is contiguous), plus
            // the off-path dof update (each lane owns its DOF).
            let d_row = d_base + (s * MAXC) as usize;
            let d_row2 = d_base + ((s + 1) * MAXC) as usize;
            for j in StepRng::new(lane..count, LANES) {
                let mut acc = delta0 * delassus.read(d_row + j as usize);
                if has_pair {
                    acc += delta1 * delassus.read(d_row2 + j as usize);
                }
                a_shared[j as usize] += acc;
            }
            if lane < ndofs {
                let col = contact_constraint_columns
                    .read(col_base + (s as usize) * dofs_stride + lane as usize);
                dof_v[lane as usize] += delta0 * col;
                if has_pair {
                    let col2 = contact_constraint_columns
                        .read(col_base + ((s + 1) as usize) * dofs_stride + lane as usize);
                    dof_v[lane as usize] += delta1 * col2;
                }
            }
            workgroup_memory_barrier_with_group_sync();
        }
    }

    // Writeback.
    if lane < ndofs {
        dof_state.write(
            batch_ids.mbi(batch_id, v_base + lane as usize),
            dof_v[lane as usize],
        );
    }
    for s in StepRng::new(lane..count, LANES) {
        let mut cons = contact_constraints.read(cons_base + s as usize);
        cons.impulse = imp_shared[s as usize];
        contact_constraints.write(cons_base + s as usize, cons);
    }
}
