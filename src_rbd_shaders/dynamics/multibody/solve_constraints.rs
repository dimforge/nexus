//! Fused multibody PGS iteration: joint limit/motor constraints followed by
//! contact constraints, in one dispatch per substep phase.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::iter::StepRng;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

use crate::dynamics::body::Velocity;
use crate::dynamics::decode_bias_mode;
use crate::gdot;
use crate::utils::BatchIndices;
use crate::utils::linalg::MAX_MB_DOFS;

use super::types::{
    MB_CONTACT_KIND_TANGENT, MB_JOINT_KIND_COUPLING, MB_JOINT_KIND_FRICTION, MB_JOINT_KIND_LIMIT,
    MB_JOINT_KIND_MOTOR, MultibodyContactConstraint, MultibodyInfo, MultibodyJointConstraint,
};

const LANES: u32 = 64;

/// Caps a friction impulse to the circular cone of radius `limit`. In 3D both
/// tangent rows of a contact point are capped jointly; in 2D there is a single
/// row and this degenerates to a scalar clamp.
#[inline]
pub(super) fn cap_friction(t0: f32, t1: f32, limit: f32) -> (f32, f32) {
    let norm_sq = t0 * t0 + t1 * t1;
    if norm_sq > limit * limit && norm_sq > 0.0 {
        let scale = limit / crate::sqrt(norm_sq);
        (t0 * scale, t1 * scale)
    } else {
        (t0, t1)
    }
}

/// Calculate the maximum `contact_constraint_count` over every (multibody, batch).
///
/// The output value is written into a uniform that will be passed to the other kernels
/// and used when `web-compat` is enabled. (Since it’s a uniform it can be used in conditions
/// without breaking uniform control flow.)
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_compute_solve_bounds(
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] max_contact_constraints: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] batch_ids: &BatchIndices,
    #[spirv(workgroup)] scratch: &mut [u32; LANES as usize],
) {
    let lane = local_id.x;
    let total = batch_ids.multibodies_len * batch_ids.num_batches;

    let mut lane_max = 0u32;
    for i in StepRng::new(lane..total, LANES) {
        let count = multibody_info.read(i as usize).contact_constraint_count;
        if count > lane_max {
            lane_max = count;
        }
    }
    scratch.write(lane as usize, lane_max);
    workgroup_memory_barrier_with_group_sync();

    if lane == 0 {
        let mut max_count = 0u32;
        for i in 0..LANES {
            let v = scratch.read(i as usize);
            if v > max_count {
                max_count = v;
            }
        }
        max_contact_constraints.write(0, max_count);
    }
}

macro_rules! solve_constraints_entry {
    ($name:ident, $lanes:literal, $simd:literal) => {
        /// One PGS iteration over a multibody's joint (limit/motor) constraints followed
        /// by its contact constraints.
        ///
        /// Dispatch: one workgroup per (multibody, batch). The host selects the
        /// smallest supported width that covers every DOF.
        #[spirv_bindgen]
        #[spirv(compute(threads($lanes)))]
        pub fn $name(
            #[spirv(workgroup_id)] workgroup_id: UVec3,
            #[spirv(local_invocation_id)] local_id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
            multibody_info: &[MultibodyInfo],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
            joint_constraints: &mut [MultibodyJointConstraint],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
            joint_constraint_columns: &[f32],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
            contact_constraints: &mut [MultibodyContactConstraint],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] contact_jac_cols: &[f32],
            #[spirv(uniform, descriptor_set = 0, binding = 5)] use_bias: &u32,
            #[spirv(uniform, descriptor_set = 0, binding = 6)] batch_ids: &BatchIndices,
            #[spirv(uniform, descriptor_set = 0, binding = 7)] max_contact_constraints: &u32,
            #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] dof_state: &mut [f32],
            #[spirv(storage_buffer, descriptor_set = 1, binding = 1)]
            solver_vels: &mut [Velocity],
            #[spirv(uniform, descriptor_set = 1, binding = 2)] num_iterations: &u32,
            #[spirv(workgroup)] dof_v: &mut [f32; MAX_MB_DOFS],
            #[spirv(workgroup)] scratch: &mut [f32; $lanes as usize],
            #[spirv(workgroup)] delta_shared: &mut f32,
            #[spirv(workgroup)] delta2_shared: &mut f32,
        ) {
            if cfg!(target_arch = "spirv") && $simd && khal_std::sync::subgroup_f_add(1.0) == 32.0 {
                super::solve_simd::solve_simd_32(
                    workgroup_id.y,
                    workgroup_id.x,
                    local_id.x,
                    multibody_info,
                    joint_constraints,
                    joint_constraint_columns,
                    contact_constraints,
                    contact_jac_cols,
                    *use_bias,
                    batch_ids,
                    dof_state,
                    solver_vels,
                    *num_iterations,
                );
                return;
            }
            for _iteration in 0..*num_iterations {
                let batch_id = workgroup_id.y;
                let mb_idx = workgroup_id.x;
                let lane = local_id.x;
                let num_mb = batch_ids.multibodies_len;
                let in_range = mb_idx < num_mb;
                #[cfg(not(feature = "web-compat"))]
                if !in_range {
                    return;
                }
                let slot = if in_range { mb_idx } else { 0 };

                let mb = multibody_info.read(batch_ids.mbi(batch_id, slot as usize));
                let ndofs = mb.ndofs;
                // Uniform per workgroup: every lane of this group returns together.
                #[cfg(not(feature = "web-compat"))]
                if ndofs == 0 {
                    return;
                }
                let (use_bias, solve_friction) = decode_bias_mode(*use_bias);

                let v_base = mb.first_dof as usize;
                let dofs_stride = batch_ids.dof_batch_capacity as usize;

                let jcons_base =
                    batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;
                let jcol_base = batch_ids.mb_joint_constraint_columns_start(batch_id)
                    + (mb.first_constraint as usize) * dofs_stride;

                // This multibody's dynamic segment in the flat constraint buffer, and the
                // paired jac/column arena: slot `g` owns `2 * dofs_stride` dense floats,
                // the `J` row first, then its `M^-1*J^T` column.
                let ccons_base = mb.contact_constraint_start as usize;
                let cjc_base = ccons_base * 2 * dofs_stride;

                let contact_count = mb.contact_constraint_count;
                #[cfg(not(feature = "web-compat"))]
                if mb.max_constraints == 0 && contact_count == 0 {
                    // Nothing to solve.
                    return;
                }
                let active =
                    in_range && ndofs != 0 && (mb.max_constraints != 0 || contact_count != 0);

                // Load the generalized velocities into workgroup memory. The accumulated
                // contact impulses stay in storage: every impulse access below is lane-0
                // only, so same-invocation ordering makes storage reads-after-writes safe
                // and no compile-time per-multibody bound is needed.
                if active && lane < ndofs {
                    dof_v.write(
                        lane as usize,
                        dof_state.read(batch_ids.mbi(batch_id, v_base + lane as usize)),
                    );
                }
                workgroup_memory_barrier_with_group_sync();

                #[cfg(feature = "web-compat")]
                let joint_iteration_len = batch_ids.mb_max_joint_constraints;
                #[cfg(not(feature = "web-compat"))]
                let joint_iteration_len = mb.max_constraints;

                // Joint limits/motors
                for s in 0..joint_iteration_len {
                    let slot_active = active && s < mb.max_constraints;
                    let cons_idx = if slot_active {
                        jcons_base + s as usize
                    } else {
                        0
                    };
                    let cons = joint_constraints.read(cons_idx);
                    let solve = slot_active
                        && (cons.kind == MB_JOINT_KIND_LIMIT
                            || cons.kind == MB_JOINT_KIND_MOTOR
                            || cons.kind == MB_JOINT_KIND_COUPLING
                            || cons.kind == MB_JOINT_KIND_FRICTION);
                    #[cfg(not(feature = "web-compat"))]
                    if !solve {
                        // Unused slot or inactive limit.
                        continue;
                    }

                    let mut delta = 0.0f32;
                    if solve {
                        let rhs = if use_bias { cons.rhs } else { cons.rhs_wo_bias };
                        // Generalized `J·v` for `J = e_{dof_id} - coupling_coeff*e_{dof2_id}`
                        // (coupling rows); collapses to `v[dof_id]` for limit / motor rows
                        // (their `coupling_coeff` is 0).
                        let v_d = dof_v.read(cons.dof_id as usize)
                            - cons.coupling_coeff * dof_v.read(cons.dof2_id as usize);
                        let rhs_total = v_d + rhs;
                        let raw_imp = cons.impulse
                            + cons.inv_lhs * (rhs_total - cons.cfm_gain * cons.impulse);
                        let mut new_imp = raw_imp;
                        if new_imp < cons.impulse_lo {
                            new_imp = cons.impulse_lo;
                        }
                        if new_imp > cons.impulse_hi {
                            new_imp = cons.impulse_hi;
                        }
                        delta = new_imp - cons.impulse;

                        if lane == 0 {
                            let mut cons = cons;
                            cons.impulse = new_imp;
                            joint_constraints.write(jcons_base + s as usize, cons);
                        }
                    }

                    // All lanes read `dof_v.read(dof_id)` above; sync before overwriting it.
                    workgroup_memory_barrier_with_group_sync();
                    if solve && lane < ndofs {
                        let col = joint_constraint_columns
                            .read(jcol_base + (s as usize) * dofs_stride + lane as usize);
                        dof_v.write(lane as usize, dof_v.read(lane as usize) - delta * col);
                    }
                    workgroup_memory_barrier_with_group_sync();
                }

                // Contacts. In 3D the two friction rows of a contact point are solved
                // together so their impulse can be capped to the friction cone; the second
                // row is handled by its sibling and skipped here.
                #[cfg(feature = "web-compat")]
                let contact_iteration_len = *max_contact_constraints;
                #[cfg(not(feature = "web-compat"))]
                let contact_iteration_len = contact_count;
                #[cfg(not(feature = "web-compat"))]
                let _ = max_contact_constraints;

                for s in 0..contact_iteration_len {
                    let slot_active = active && s < contact_count;
                    let cons_idx = if slot_active {
                        ccons_base + s as usize
                    } else {
                        0
                    };
                    let cons = contact_constraints.read(cons_idx);
                    let is_tangent = cons.kind == MB_CONTACT_KIND_TANGENT;
                    // Friction is solved during the relaxation phase (and during the
                    // biased pass when `friction_in_bias_pass` is set); in 3D a tangent
                    // pair is solved by its first row only.
                    #[cfg(feature = "dim3")]
                    let solve = slot_active
                        && !(is_tangent && !solve_friction)
                        && !(is_tangent && s != cons.normal_constraint_slot + 1);
                    #[cfg(feature = "dim2")]
                    let solve = slot_active && !(is_tangent && !solve_friction);
                    #[cfg(not(feature = "web-compat"))]
                    if !solve {
                        continue;
                    }

                    #[cfg(feature = "dim3")]
                    let has_pair = is_tangent;
                    #[cfg(feature = "dim2")]
                    let has_pair = false;

                    let jac_offset = cjc_base + (s as usize) * 2 * dofs_stride;
                    let jac_offset2 = jac_offset + 2 * dofs_stride;
                    let is_self = cons.free_body_id == u32::MAX;

                    // Multibody side of J · u, one product per lane; lane 0 sums them in
                    // DOF order.
                    if solve {
                        scratch.write(
                            lane as usize,
                            if lane < ndofs {
                                contact_jac_cols.read(jac_offset + lane as usize)
                                    * dof_v.read(lane as usize)
                            } else {
                                0.0
                            },
                        );
                    }
                    workgroup_memory_barrier_with_group_sync();

                    let mut j_dot_v0 = 0.0f32;
                    if solve && lane == 0 {
                        for i in 0..ndofs {
                            j_dot_v0 += scratch.read(i as usize);
                        }
                    }
                    workgroup_memory_barrier_with_group_sync();

                    if solve && has_pair {
                        scratch.write(
                            lane as usize,
                            if lane < ndofs {
                                contact_jac_cols.read(jac_offset2 + lane as usize)
                                    * dof_v.read(lane as usize)
                            } else {
                                0.0
                            },
                        );
                    }
                    workgroup_memory_barrier_with_group_sync();

                    if solve && lane == 0 {
                        let cons2 = contact_constraints.read(ccons_base + (s + 1) as usize);
                        let mut j_dot_v1 = 0.0f32;
                        if has_pair {
                            for i in 0..ndofs {
                                j_dot_v1 += scratch.read(i as usize);
                            }
                        }
                        // Free-body side stays lane-0-local (`free_body_id` is a global
                        // body id).
                        let free = if is_self {
                            Velocity::default()
                        } else {
                            solver_vels.read(cons.free_body_id as usize)
                        };
                        if !is_self {
                            j_dot_v0 +=
                                cons.lin_jac.dot(free.linear) + gdot(cons.ang_jac, free.angular);
                            if has_pair {
                                j_dot_v1 += cons2.lin_jac.dot(free.linear)
                                    + gdot(cons2.ang_jac, free.angular);
                            }
                        }

                        // Compliance only softens the normal rows; friction stays rigid.
                        let cfm_factor = if use_bias && !is_tangent {
                            cons.cfm_factor
                        } else {
                            1.0
                        };
                        let impulse0 = cons.impulse;
                        let rhs0 = if use_bias { cons.rhs } else { cons.rhs_wo_bias };
                        let raw0 = cfm_factor * (impulse0 - cons.inv_lhs * (j_dot_v0 + rhs0));

                        let impulse1 = if has_pair { cons2.impulse } else { 0.0 };
                        let raw1 = if has_pair {
                            let rhs1 = if use_bias {
                                cons2.rhs
                            } else {
                                cons2.rhs_wo_bias
                            };
                            cfm_factor * (impulse1 - cons2.inv_lhs * (j_dot_v1 + rhs1))
                        } else {
                            0.0
                        };

                        // Normal: clamp to ≥ 0. Friction: cap the tangent pair to the
                        // circular cone `μ · normal_impulse`.
                        let (new0, new1) = if is_tangent {
                            // The paired normal was updated earlier in this iteration by this
                            // same lane, so the storage read observes the fresh value.
                            let limit = cons.friction_coeff
                                * contact_constraints
                                    .at(ccons_base + cons.normal_constraint_slot as usize)
                                    .impulse;
                            cap_friction(raw0, raw1, limit)
                        } else if raw0 < 0.0 {
                            (0.0, 0.0)
                        } else {
                            (raw0, 0.0)
                        };

                        let delta0 = new0 - impulse0;
                        let delta1 = if has_pair { new1 - impulse1 } else { 0.0 };
                        contact_constraints.at_mut(cons_idx).impulse = new0;
                        if has_pair {
                            contact_constraints.at_mut(cons_idx + 1).impulse = new1;
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
                            solver_vels.write(cons.free_body_id as usize, new_free);
                        }
                    }
                    workgroup_memory_barrier_with_group_sync();

                    // Per-lane `dof_v.read(lane)` update.
                    let delta0 = *delta_shared;
                    let delta1 = *delta2_shared;
                    if solve && lane < ndofs {
                        if delta0 != 0.0 {
                            let col =
                                contact_jac_cols.read(jac_offset + dofs_stride + lane as usize);
                            dof_v.write(lane as usize, dof_v.read(lane as usize) + delta0 * col);
                        }
                        if has_pair && delta1 != 0.0 {
                            let col =
                                contact_jac_cols.read(jac_offset2 + dofs_stride + lane as usize);
                            dof_v.write(lane as usize, dof_v.read(lane as usize) + delta1 * col);
                        }
                    }
                    workgroup_memory_barrier_with_group_sync();
                }

                // Writeback (the contact impulses were updated in storage as they were
                // solved).
                if active && lane < ndofs {
                    dof_state.write(
                        batch_ids.mbi(batch_id, v_base + lane as usize),
                        dof_v.read(lane as usize),
                    );
                }
                // Storage writes from lane zero must be visible to all lanes in
                // the following iteration of the portable fallback.
                if _iteration + 1 < *num_iterations {
                    khal_std::sync::control_barrier::<2, 2, { 0x8 | 0x40 | 0x100 }>();
                }
            }
        }
    };
}

solve_constraints_entry!(gpu_mb_solve_constraints, 64, false);
solve_constraints_entry!(gpu_mb_solve_constraints_32, 32, false);
solve_constraints_entry!(gpu_mb_solve_constraints_simd, 32, true);
