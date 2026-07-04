//! The four compute entry points of the multibody impulse-joint pipeline
//! (update / finalize / solve / remove-bias).

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

use crate::Pose;
use crate::dynamics::ConstraintSoftness;
use crate::dynamics::body::{Velocity, WorldMassProperties};
use crate::utils::BatchIndices;

use super::super::lu::LANES;
use super::super::types::{MultibodyInfo, MultibodyLinkWorkspace};

use super::jacobians::*;
use super::types::*;
use super::update::*;

/// (Re)build all axis constraints of every multibody-touching impulse joint.
///
/// Run once per substep. Mirrors rapier's
/// `JointGenericExternalConstraintBuilder::update`: rebuilds all `J / W·J`
/// rows + biases from current poses / mass matrix.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_update_impulse_joint_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] builders: &[MbImpulseJointBuilder],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    constraints: &mut [MbImpulseJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] jacobians: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] softness: &ConstraintSoftness,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)]
    links_workspace: &[MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] body_jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 3)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 4)] mprops: &[WorldMassProperties],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * 64;
    let batch_id = invocation_id.y;
    let cap = batch_ids.mb_imp_joints_batch_capacity;
    if invocation_id.x >= cap {
        return;
    }
    let dt = softness.dt;
    // Joint lock/limit softness — configurable via `joint_natural_frequency` /
    // `joint_damping_ratio` (rapier's `joint.softness`), replacing the old
    // hardcoded `0.8/dt` + `cfm = 0`.
    let lock_erp_inv_dt = softness.joint_erp_inv_dt;
    let lock_cfm_coeff = softness.joint_cfm_coeff;

    let joints_start = batch_ids.mb_imp_joints_start(batch_id);
    let cons_start = batch_ids.mb_imp_joint_constraints_start(batch_id);
    let jac_buf_start = batch_ids.mb_imp_joint_jacobians_start(batch_id);
    let mb_start = batch_ids.mb_start(batch_id);
    let links_start = batch_ids.links_start(batch_id);
    let body_jac_start = batch_ids.jac_start(batch_id);
    let colliders_start = batch_ids.coll_start(batch_id);

    // Loop chunked across `num_threads` so a single workgroup row processes
    // all joints in a batch (matches `gpu_init_joint_constraints` style).
    // Iterating to `cap` instead of `len` lets us drop the per-batch
    // `num_joints` storage binding — the host pads unused builder slots
    // with `side_a_kind == SIDE_KIND_FIXED && side_b_kind == SIDE_KIND_FIXED`
    // which we use as the inactive-slot sentinel below.
    let mut i = invocation_id.x;
    while i < cap {
        let builder = builders.read(joints_start + i as usize);
        let is_dummy =
            builder.side_a_kind == SIDE_KIND_FIXED && builder.side_b_kind == SIDE_KIND_FIXED;
        if !is_dummy {
            builder.update_one_joint(
                constraints,
                cons_start,
                jacobians,
                jac_buf_start,
                multibody_info,
                mb_start,
                links_workspace,
                links_start,
                body_jacobians,
                body_jac_start,
                poses,
                colliders_start,
                mprops,
                dt,
                lock_erp_inv_dt,
                lock_cfm_coeff,
            );
        }
        i += num_threads;
    }
}

/// Companion finalize pass to `gpu_mb_update_impulse_joint_constraints`: for
/// each active constraint, back-solve the multibody side(s)' `M⁻¹·Jᵀ` columns
/// and compute `inv_lhs`. Must run after `update` and before
/// `gpu_mb_solve_impulse_joint_constraints` each substep.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_finalize_impulse_joint_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] builders: &[MbImpulseJointBuilder],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    constraints: &mut [MbImpulseJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] jacobians: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] mass_matrices: &[f32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] lu_pivots: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * 64;
    let batch_id = invocation_id.y;
    let cap = batch_ids.mb_imp_joints_batch_capacity;
    if invocation_id.x >= cap {
        return;
    }

    let joints_start = batch_ids.mb_imp_joints_start(batch_id);
    let cons_start = batch_ids.mb_imp_joint_constraints_start(batch_id);
    let mb_start = batch_ids.mb_start(batch_id);
    let mm_start = batch_ids.mm_start(batch_id);
    let dof_start = batch_ids.dof_start(batch_id);

    let mut i = invocation_id.x;
    while i < cap {
        let builder = builders.read(joints_start + i as usize);
        let is_dummy =
            builder.side_a_kind == SIDE_KIND_FIXED && builder.side_b_kind == SIDE_KIND_FIXED;
        if !is_dummy {
            let cons_base = cons_start + builder.constraint_id as usize;
            for s in 0..MAX_AXIS_CONSTRAINTS {
                let mut c = constraints.read(cons_base + s as usize);
                if c.kind != 0 {
                    // Multibody side(s): LU back-solve `M⁻¹·Jᵀ`. Free-body sides
                    // already have their `W·J` (= M⁻¹·Jᵀ) filled by the build
                    // pass (it holds `mprops`). The loop-closure relative block
                    // is stored on side B as a normal MB jacobian, so it's
                    // handled here too.
                    if c.side_a_kind == SIDE_KIND_MB && c.ndofs_a > 0 {
                        let mb = multibody_info.read(mb_start + c.side_a_id as usize);
                        solve_mb_wj(
                            jacobians,
                            c.j_id_a,
                            c.ndofs_a,
                            &mb,
                            mass_matrices,
                            mm_start,
                            lu_pivots,
                            dof_start,
                        );
                    }
                    if c.side_b_kind == SIDE_KIND_MB && c.ndofs_b > 0 {
                        let mb = multibody_info.read(mb_start + c.side_b_id as usize);
                        solve_mb_wj(
                            jacobians,
                            c.j_id_b,
                            c.ndofs_b,
                            &mb,
                            mass_matrices,
                            mm_start,
                            lu_pivots,
                            dof_start,
                        );
                    }
                    c.finalize_generic_constraint(jacobians);
                    constraints.write(cons_base + s as usize, c);
                }
            }
        }
        i += num_threads;
    }
}

/// One PGS sweep over the multibody-touching impulse-joint axis constraints
/// of a single color — **one workgroup per joint**, the lanes cooperating on
/// that joint's per-axis `J·v` reductions and `W·J` applies.
///
/// Joints are graph-colored at init time (see `set_impulse_joints`): within
/// one color no two joints share a multibody or a free body, so the color's
/// joints run race-free in parallel. The host dispatches one color per
/// iteration, giving an exact sequential Gauss–Seidel sweep in color-sorted
/// order.
#[spirv_bindgen]
#[spirv(compute(threads(64, 1, 1)))]
pub fn gpu_mb_solve_impulse_joint_constraints(
    #[spirv(workgroup_id)] wg_id: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] builders: &[MbImpulseJointBuilder],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    constraints: &mut [MbImpulseJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] all_color_groups: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] curr_color: &u32,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] solver_vels: &mut [Velocity],
    // Per-lane scratch for the J·v tree reductions.
    #[spirv(workgroup)] partial: &mut [f32; LANES as usize],
) {
    let batch_id = wg_id.y;
    let lane = lid.x;

    let joints_start = batch_ids.mb_imp_joints_start(batch_id);
    let cons_start = batch_ids.mb_imp_joint_constraints_start(batch_id);
    let mb_start = batch_ids.mb_start(batch_id);
    let dof_start = batch_ids.dof_start(batch_id);
    let colliders_start = batch_ids.coll_start(batch_id);

    // `color_groups` is a per-batch prefix-sum over the color-sorted
    // builders: color `c` owns the sorted-builder range
    // `[color_groups[c-1], color_groups[c])` (start `0` for color `0`).
    let color = *curr_color as usize;
    let color_groups = batch_ids.mb_imp_joint_color_groups_batch(batch_id, all_color_groups);
    let start = if color > 0 {
        color_groups[color - 1]
    } else {
        0
    };
    let end = color_groups[color];

    let mut j = start + wg_id.x;
    let workgroup_is_active = j < end;
    if !workgroup_is_active {
        // Technically, if we enter here, we should return. However, on the web, a return would
        // break uniform control flow. This could be avoided with a `workgroupUniformLoad` but
        // that’s not supported by rust-gpu.
        j = start; // Any valid index will do.
    }

    let builder = builders.at(joints_start + j as usize);
    let cons_base = cons_start + builder.constraint_id as usize;

    // Per-multibody dof base: same for every axis constraint of this joint.
    let dof_base_a = if builder.side_a_kind == SIDE_KIND_MB {
        let mb = multibody_info.at(mb_start + builder.side_a_id as usize);
        dof_start + mb.first_dof as usize
    } else {
        0
    };
    let dof_base_b = if builder.side_b_kind == SIDE_KIND_MB {
        let mb = multibody_info.at(mb_start + builder.side_b_id as usize);
        dof_start + mb.first_dof as usize
    } else {
        0
    };

    // TODO(PERF): load jacobians into shared memory and keep the velocity deltat on shared
    //             memory and only writeback after all the axis constraints are solved.
    for s in 0..MAX_AXIS_CONSTRAINTS {
        let c = constraints.at_mut(cons_base + s as usize);
        let active = workgroup_is_active && c.kind != 0;

        // dvel = J_b · v_b - J_a · v_a   (rapier's `vel2 - vel1`).
        let v1 = side_dot_vel_par(
            active,
            c.side_a_kind,
            c.j_id_a,
            c.ndofs_a,
            c.side_a_id,
            jacobians,
            dof_state,
            dof_base_a,
            solver_vels,
            colliders_start,
            lane,
            partial,
        );
        let v2 = side_dot_vel_par(
            active,
            c.side_b_kind,
            c.j_id_b,
            c.ndofs_b,
            c.side_b_id,
            jacobians,
            dof_state,
            dof_base_b,
            solver_vels,
            colliders_start,
            lane,
            partial,
        );

        let delta = if active {
            let dvel = c.rhs + (v2 - v1);
            let total = (c.impulse + c.inv_lhs * (dvel - c.cfm_gain * c.impulse))
                // NOTE: should be `clamp`, but `clamp` breaks uniform control flow for some reasons.
                .max(c.impulse_lo)
                .min(c.impulse_hi);
            let d = total - c.impulse;
            if lane == 0 {
                c.impulse = total;
            }
            d
        } else {
            0.0f32
        };

        // Apply ±delta · W·J: sign +1 for side A, -1 for side B (matches
        // rapier `solver_vel1.axpy(delta_impulse, &wj1, 1.0)`,
        // `solver_vel2.axpy(-delta_impulse, &wj2, 1.0)`).
        side_apply_impulse_par(
            active,
            c.side_a_kind,
            c.j_id_a,
            c.ndofs_a,
            c.side_a_id,
            1.0,
            delta,
            jacobians,
            dof_state,
            dof_base_a,
            solver_vels,
            colliders_start,
            lane,
        );
        side_apply_impulse_par(
            active,
            c.side_b_kind,
            c.j_id_b,
            c.ndofs_b,
            c.side_b_id,
            -1.0,
            delta,
            jacobians,
            dof_state,
            dof_base_b,
            solver_vels,
            colliders_start,
            lane,
        );

        workgroup_memory_barrier_with_group_sync();
    }
}

/// Strip the positional bias from each active constraint's `rhs` for the
/// stabilization sweep — mirrors `gpu_mb_remove_joint_constraint_bias`.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_remove_impulse_joint_constraint_bias(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] builders: &[MbImpulseJointBuilder],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    constraints: &mut [MbImpulseJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] num_joints: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * 64;
    let batch_id = invocation_id.y;
    let len = num_joints.read(batch_id as usize);
    let joints_start = batch_ids.mb_imp_joints_start(batch_id);
    let cons_start = batch_ids.mb_imp_joint_constraints_start(batch_id);
    let mut i = invocation_id.x;
    while i < len {
        let builder = builders.at(joints_start + i as usize);
        let cons_base = cons_start + builder.constraint_id as usize;
        for s in 0..MAX_AXIS_CONSTRAINTS {
            let c = constraints.at_mut(cons_base + s as usize);
            if c.kind != 0 {
                c.rhs = c.rhs_wo_bias;
            }
        }
        i += num_threads;
    }
}
