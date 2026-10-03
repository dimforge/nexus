//! Native Metal PGS for small multibodies, preserving the cooperative solver's
//! constraint order, clamping, friction cone and iteration counts.
//!
//! The full-width path owns one DOF per lane. Large batches use four 8-lane
//! partitions per workgroup, with four DOFs per lane. Explicit register bands
//! avoid private-array spills. A serial sweep handles unsupported subgroup widths.
//! Keep the shader attributes unqualified: spirv_bindgen reads their workgroup
//! sizes to generate host dispatch metadata.
use super::solve_constraints::cap_friction;
use super::types::*;
use crate::dynamics::{Velocity, decode_bias_mode};
use crate::gdot;
use crate::utils::BatchIndices;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::subgroup_f_add;

#[inline(always)]
fn shuffle<const LANES: u32>(value: f32, lane: u32) -> f32 {
    if LANES == 1 {
        return value;
    }
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
fn sum<const LANES: u32>(value: f32, _lane: u32, base: u32) -> f32 {
    if LANES == 1 {
        return value;
    }
    if LANES == 32 {
        return subgroup_f_add(value);
    }
    // Masked native reductions avoid a shuffle loop that failed equivalence
    // checks after rust-gpu/SPIR-V-to-MSL translation. Each partition is kept
    // separate even when the other robots have different contact counts.
    let a = subgroup_f_add(if base == 0 { value } else { 0.0 });
    let b = subgroup_f_add(if base == 8 { value } else { 0.0 });
    let c = subgroup_f_add(if base == 16 { value } else { 0.0 });
    let d = subgroup_f_add(if base == 24 { value } else { 0.0 });
    if base == 0 {
        a
    } else if base == 8 {
        b
    } else if base == 16 {
        c
    } else {
        d
    }
}

/// Scalar fallback. LANES=1 does not call subgroup intrinsics.
#[inline]
fn solve_simd<const LANES: u32, const N: usize>(
    batch_id: u32,
    mb_idx: u32,
    lane: u32,
    base: u32,
    multibody_info: &[MultibodyInfo],
    joint_constraints: &mut [MultibodyJointConstraint],
    joint_constraint_columns: &[f32],
    contact_constraints: &mut [MultibodyContactConstraint],
    contact_jac_cols: &[f32],
    use_bias: u32,
    batch_ids: &BatchIndices,
    dof_state: &mut [f32],
    solver_vels: &mut [Velocity],
    num_iterations: u32,
) {
    if mb_idx >= batch_ids.multibodies_len {
        return;
    }
    let mb = multibody_info.read(batch_ids.mbi(batch_id, mb_idx as usize));
    let ndofs = mb.ndofs;
    if ndofs == 0 || (mb.max_constraints == 0 && mb.contact_constraint_count == 0) {
        return;
    }
    let (use_bias, solve_friction) = decode_bias_mode(use_bias);
    let v_base = mb.first_dof as usize;
    let stride = batch_ids.dof_batch_capacity as usize;
    let jcons_base = batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;
    let jcol_base = batch_ids.mb_joint_constraint_columns_start(batch_id)
        + mb.first_constraint as usize * stride;
    let ccons_base = mb.contact_constraint_start as usize;
    let cjc_base = ccons_base * 2 * stride;
    let mut velocity = [0.0f32; N];
    for band in 0..N {
        let dof = lane + band as u32 * LANES;
        if dof < ndofs {
            velocity.write(
                band,
                dof_state.read(batch_ids.mbi(batch_id, v_base + dof as usize)),
            );
        }
    }

    for _iteration in 0..num_iterations {
        for s in 0..mb.max_constraints as usize {
            let mut cons = joint_constraints.read(jcons_base + s);
            if cons.kind != MB_JOINT_KIND_LIMIT
                && cons.kind != MB_JOINT_KIND_MOTOR
                && cons.kind != MB_JOINT_KIND_COUPLING
                && cons.kind != MB_JOINT_KIND_FRICTION
            {
                continue;
            }
            cons.impulse = shuffle::<LANES>(cons.impulse, base);
            let rhs = if use_bias { cons.rhs } else { cons.rhs_wo_bias };
            let v_d = shuffle::<LANES>(
                velocity.read((cons.dof_id / LANES) as usize),
                base + cons.dof_id % LANES,
            ) - cons.coupling_coeff
                * shuffle::<LANES>(
                    velocity.read((cons.dof2_id / LANES) as usize),
                    base + cons.dof2_id % LANES,
                );
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
                joint_constraints.at_mut(jcons_base + s).impulse = new_imp;
            }
            for band in 0..N {
                let dof = lane + band as u32 * LANES;
                if dof < ndofs {
                    let col = joint_constraint_columns.read(jcol_base + s * stride + dof as usize);
                    velocity.write(band, velocity.read(band) - delta * col);
                }
            }
        }

        for s in 0..mb.contact_constraint_count {
            let cons_idx = ccons_base + s as usize;
            let cons = contact_constraints.read(cons_idx);
            let is_tangent = cons.kind == MB_CONTACT_KIND_TANGENT;
            if is_tangent && !solve_friction {
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
            let jac_offset = cjc_base + s as usize * 2 * stride;
            let jac_offset2 = jac_offset + 2 * stride;
            let mut product0 = 0.0;
            let mut product1 = 0.0;
            for band in 0..N {
                let dof = lane + band as u32 * LANES;
                if dof < ndofs {
                    product0 +=
                        contact_jac_cols.read(jac_offset + dof as usize) * velocity.read(band);
                    if has_pair {
                        product1 +=
                            contact_jac_cols.read(jac_offset2 + dof as usize) * velocity.read(band);
                    }
                }
            }
            let mut j_dot_v0 = sum::<LANES>(product0, lane, base);
            let mut j_dot_v1 = sum::<LANES>(product1, lane, base);
            let mut delta0 = 0.0;
            let mut delta1 = 0.0;
            if lane == 0 {
                let cons2 = if has_pair {
                    contact_constraints.read(cons_idx + 1)
                } else {
                    cons
                };
                let is_self = cons.free_body_id == u32::MAX;
                let free = if is_self {
                    Velocity::default()
                } else {
                    solver_vels.read(cons.free_body_id as usize)
                };
                if !is_self {
                    j_dot_v0 += cons.lin_jac.dot(free.linear) + gdot(cons.ang_jac, free.angular);
                    if has_pair {
                        j_dot_v1 +=
                            cons2.lin_jac.dot(free.linear) + gdot(cons2.ang_jac, free.angular);
                    }
                }
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
                let (new0, new1) = if is_tangent {
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
                delta0 = new0 - impulse0;
                delta1 = if has_pair { new1 - impulse1 } else { 0.0 };
                contact_constraints.at_mut(cons_idx).impulse = new0;
                if has_pair {
                    contact_constraints.at_mut(cons_idx + 1).impulse = new1;
                }
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
            let delta0 = shuffle::<LANES>(delta0, base);
            let delta1 = shuffle::<LANES>(delta1, base);
            for band in 0..N {
                let dof = lane + band as u32 * LANES;
                if dof < ndofs {
                    let mut v = velocity.read(band);
                    if delta0 != 0.0 {
                        v += delta0 * contact_jac_cols.read(jac_offset + stride + dof as usize);
                    }
                    if has_pair && delta1 != 0.0 {
                        v += delta1 * contact_jac_cols.read(jac_offset2 + stride + dof as usize);
                    }
                    velocity.write(band, v);
                }
            }
        }
    }
    for band in 0..N {
        let dof = lane + band as u32 * LANES;
        if dof < ndofs {
            dof_state.write(
                batch_ids.mbi(batch_id, v_base + dof as usize),
                velocity.read(band),
            );
        }
    }
}

macro_rules! each_band {
    ($band:ident, $body:block) => {{
        {
            let $band = 0usize;
            $body
        }
        {
            let $band = 1usize;
            $body
        }
        {
            let $band = 2usize;
            $body
        }
        {
            let $band = 3usize;
            $body
        }
    }};
}

#[inline(always)]
fn read_band(v: glamx::Vec4, i: usize) -> f32 {
    match i {
        0 => v.x,
        1 => v.y,
        2 => v.z,
        _ => v.w,
    }
}
#[inline(always)]
fn write_band(v: &mut glamx::Vec4, i: usize, x: f32) {
    match i {
        0 => v.x = x,
        1 => v.y = x,
        2 => v.z = x,
        _ => v.w = x,
    }
}
#[inline(always)]
fn solve_packed8<const MODE: u32>(
    batch_id: u32,
    mb_idx: u32,
    lane: u32,
    base: u32,
    multibody_info: &[MultibodyInfo],
    joint_constraints: &mut [MultibodyJointConstraint],
    joint_constraint_columns: &[f32],
    contact_constraints: &mut [MultibodyContactConstraint],
    contact_jac_cols: &[f32],
    use_bias: u32,
    batch_ids: &BatchIndices,
    dof_state: &mut [f32],
    solver_vels: &mut [Velocity],
    num_iterations: u32,
) {
    if mb_idx >= batch_ids.multibodies_len {
        return;
    }
    let mb = multibody_info.read(batch_ids.mbi(batch_id, mb_idx as usize));
    let ndofs = mb.ndofs;
    if ndofs == 0 || (mb.max_constraints == 0 && mb.contact_constraint_count == 0) {
        return;
    }
    let (use_bias, solve_friction) = decode_bias_mode(if MODE < 3 { MODE } else { use_bias });
    let v_base = mb.first_dof as usize;
    let stride = batch_ids.dof_batch_capacity as usize;
    let jcons_base = batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;
    let jcol_base = batch_ids.mb_joint_constraint_columns_start(batch_id)
        + mb.first_constraint as usize * stride;
    let ccons_base = mb.contact_constraint_start as usize;
    let cjc_base = ccons_base * 2 * stride;
    let mut velocity = glamx::Vec4::ZERO;
    each_band!(band, {
        let dof = lane + band as u32 * 8;
        if dof < ndofs {
            write_band(
                &mut velocity,
                band,
                dof_state.read(batch_ids.mbi(batch_id, v_base + dof as usize)),
            );
        }
    });

    for _iteration in 0..num_iterations {
        for s in 0..mb.max_constraints as usize {
            let mut cons = joint_constraints.read(jcons_base + s);
            if cons.kind != MB_JOINT_KIND_LIMIT
                && cons.kind != MB_JOINT_KIND_MOTOR
                && cons.kind != MB_JOINT_KIND_COUPLING
                && cons.kind != MB_JOINT_KIND_FRICTION
            {
                continue;
            }
            cons.impulse = shuffle::<8>(cons.impulse, base);
            let rhs = if use_bias { cons.rhs } else { cons.rhs_wo_bias };
            let v_d = shuffle::<8>(
                read_band(velocity, (cons.dof_id / 8) as usize),
                base + cons.dof_id % 8,
            ) - cons.coupling_coeff
                * shuffle::<8>(
                    read_band(velocity, (cons.dof2_id / 8) as usize),
                    base + cons.dof2_id % 8,
                );
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
                joint_constraints.at_mut(jcons_base + s).impulse = new_imp;
            }
            each_band!(band, {
                let dof = lane + band as u32 * 8;
                if dof < ndofs {
                    let col = joint_constraint_columns.read(jcol_base + s * stride + dof as usize);
                    let old = read_band(velocity, band);
                    write_band(&mut velocity, band, old - delta * col);
                }
            });
        }

        for s in 0..mb.contact_constraint_count {
            let cons_idx = ccons_base + s as usize;
            let cons = contact_constraints.read(cons_idx);
            let is_tangent = cons.kind == MB_CONTACT_KIND_TANGENT;
            if is_tangent && !solve_friction {
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
            let jac_offset = cjc_base + s as usize * 2 * stride;
            let jac_offset2 = jac_offset + 2 * stride;
            let mut product0 = 0.0;
            let mut product1 = 0.0;
            each_band!(band, {
                let dof = lane + band as u32 * 8;
                if dof < ndofs {
                    product0 += contact_jac_cols.read(jac_offset + dof as usize)
                        * read_band(velocity, band);
                    if has_pair {
                        product1 += contact_jac_cols.read(jac_offset2 + dof as usize)
                            * read_band(velocity, band);
                    }
                }
            });
            let mut j_dot_v0 = sum::<8>(product0, lane, base);
            let mut j_dot_v1 = sum::<8>(product1, lane, base);
            let mut delta0 = 0.0;
            let mut delta1 = 0.0;
            if lane == 0 {
                let cons2 = if has_pair {
                    contact_constraints.read(cons_idx + 1)
                } else {
                    cons
                };
                let is_self = cons.free_body_id == u32::MAX;
                let free = if is_self {
                    Velocity::default()
                } else {
                    solver_vels.read(cons.free_body_id as usize)
                };
                if !is_self {
                    j_dot_v0 += cons.lin_jac.dot(free.linear) + gdot(cons.ang_jac, free.angular);
                    if has_pair {
                        j_dot_v1 +=
                            cons2.lin_jac.dot(free.linear) + gdot(cons2.ang_jac, free.angular);
                    }
                }
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
                let (new0, new1) = if is_tangent {
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
                delta0 = new0 - impulse0;
                delta1 = if has_pair { new1 - impulse1 } else { 0.0 };
                contact_constraints.at_mut(cons_idx).impulse = new0;
                if has_pair {
                    contact_constraints.at_mut(cons_idx + 1).impulse = new1;
                }
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
            let delta0 = shuffle::<8>(delta0, base);
            let delta1 = shuffle::<8>(delta1, base);
            each_band!(band, {
                let dof = lane + band as u32 * 8;
                if dof < ndofs {
                    let mut v = read_band(velocity, band);
                    if delta0 != 0.0 {
                        v += delta0 * contact_jac_cols.read(jac_offset + stride + dof as usize);
                    }
                    if has_pair && delta1 != 0.0 {
                        v += delta1 * contact_jac_cols.read(jac_offset2 + stride + dof as usize);
                    }
                    write_band(&mut velocity, band, v);
                }
            });
        }
    }
    each_band!(band, {
        let dof = lane + band as u32 * 8;
        if dof < ndofs {
            dof_state.write(
                batch_ids.mbi(batch_id, v_base + dof as usize),
                read_band(velocity, band),
            );
        }
    });
}

#[spirv_bindgen]
#[spirv(compute(threads(32)))]
pub fn gpu_mb_solve_constraints_packed(
    #[spirv(global_invocation_id)] id: khal_std::glamx::UVec3,
    #[spirv(subgroup_local_invocation_id)] subgroup_lane: u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] joint_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] contact_jac_cols: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] use_bias: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] batch_ids: &BatchIndices,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] max_contact_constraints: &u32,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] solver_vels: &mut [Velocity],
    #[spirv(uniform, descriptor_set = 1, binding = 2)] num_iterations: &u32,
) {
    let width = subgroup_f_add(1.0) as u32;
    let _ = max_contact_constraints;
    let batch_id = id.x / 8;
    let lane = id.x % 8;
    if batch_id >= batch_ids.num_batches {
        return;
    }
    if cfg!(target_arch = "spirv") && width == 32 {
        solve_packed8::<3>(
            batch_id,
            id.y,
            lane,
            subgroup_lane - lane,
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
    } else if lane == 0 {
        // A fully serial fallback has no subgroup-width requirements.
        solve_simd::<1, 32>(
            batch_id,
            id.y,
            0,
            0,
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
    }
}

#[spirv_bindgen]
#[spirv(compute(threads(32)))]
pub fn gpu_mb_solve_constraints_packed_bias(
    #[spirv(global_invocation_id)] id: khal_std::glamx::UVec3,
    #[spirv(subgroup_local_invocation_id)] subgroup_lane: u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] joint_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] contact_jac_cols: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] use_bias: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] batch_ids: &BatchIndices,
    #[spirv(uniform, descriptor_set = 0, binding = 7)] max_contact_constraints: &u32,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] solver_vels: &mut [Velocity],
    #[spirv(uniform, descriptor_set = 1, binding = 2)] num_iterations: &u32,
) {
    let width = subgroup_f_add(1.0) as u32;
    let _ = max_contact_constraints;
    let batch_id = id.x / 8;
    let lane = id.x % 8;
    if batch_id >= batch_ids.num_batches {
        return;
    }
    if cfg!(target_arch = "spirv") && width == 32 {
        solve_packed8::<1>(
            batch_id,
            id.y,
            lane,
            subgroup_lane - lane,
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
    } else if lane == 0 {
        // A fully serial fallback has no subgroup-width requirements.
        solve_simd::<1, 32>(
            batch_id,
            id.y,
            0,
            0,
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
    }
}

/// All 32 invocations must be active, belong to one subgroup, and agree on the
/// multibody. Each lane owns its generalized velocity; only lane zero updates
/// impulses and the free-body velocity. The host retains the cooperative path
/// for larger models, and the entry point checks the active subgroup width.
#[inline]
pub(super) fn solve_simd_32(
    batch_id: u32,
    mb_idx: u32,
    lane: u32,
    multibody_info: &[MultibodyInfo],
    joint_constraints: &mut [MultibodyJointConstraint],
    joint_constraint_columns: &[f32],
    contact_constraints: &mut [MultibodyContactConstraint],
    contact_jac_cols: &[f32],
    use_bias: u32,
    batch_ids: &BatchIndices,
    dof_state: &mut [f32],
    solver_vels: &mut [Velocity],
    num_iterations: u32,
) {
    if mb_idx >= batch_ids.multibodies_len {
        return;
    }
    let mb = multibody_info.read(batch_ids.mbi(batch_id, mb_idx as usize));
    let ndofs = mb.ndofs;
    if ndofs == 0 || (mb.max_constraints == 0 && mb.contact_constraint_count == 0) {
        return;
    }
    let (use_bias, solve_friction) = decode_bias_mode(use_bias);
    let v_base = mb.first_dof as usize;
    let stride = batch_ids.dof_batch_capacity as usize;
    let jcons_base = batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;
    let jcol_base = batch_ids.mb_joint_constraint_columns_start(batch_id)
        + mb.first_constraint as usize * stride;
    let ccons_base = mb.contact_constraint_start as usize;
    let cjc_base = ccons_base * 2 * stride;
    let mut velocity = if lane < ndofs {
        dof_state.read(batch_ids.mbi(batch_id, v_base + lane as usize))
    } else {
        0.0
    };

    for _iteration in 0..num_iterations {
        for s in 0..mb.max_constraints as usize {
            let mut cons = joint_constraints.read(jcons_base + s);
            if cons.kind != MB_JOINT_KIND_LIMIT
                && cons.kind != MB_JOINT_KIND_MOTOR
                && cons.kind != MB_JOINT_KIND_COUPLING
                && cons.kind != MB_JOINT_KIND_FRICTION
            {
                continue;
            }
            cons.impulse = shuffle::<32>(cons.impulse, 0);
            let rhs = if use_bias { cons.rhs } else { cons.rhs_wo_bias };
            let v_d = shuffle::<32>(velocity, cons.dof_id)
                - cons.coupling_coeff * shuffle::<32>(velocity, cons.dof2_id);
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
                joint_constraints.at_mut(jcons_base + s).impulse = new_imp;
            }
            if lane < ndofs {
                let col = joint_constraint_columns.read(jcol_base + s * stride + lane as usize);
                velocity -= delta * col;
            }
        }

        for s in 0..mb.contact_constraint_count {
            let cons_idx = ccons_base + s as usize;
            let cons = contact_constraints.read(cons_idx);
            let is_tangent = cons.kind == MB_CONTACT_KIND_TANGENT;
            if is_tangent && !solve_friction {
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
            let jac_offset = cjc_base + s as usize * 2 * stride;
            let jac_offset2 = jac_offset + 2 * stride;
            let product0 = if lane < ndofs {
                contact_jac_cols.read(jac_offset + lane as usize) * velocity
            } else {
                0.0
            };
            let mut j_dot_v0 = subgroup_f_add(product0);
            let product1 = if has_pair && lane < ndofs {
                contact_jac_cols.read(jac_offset2 + lane as usize) * velocity
            } else {
                0.0
            };
            let mut j_dot_v1 = subgroup_f_add(product1);
            let mut delta0 = 0.0;
            let mut delta1 = 0.0;
            if lane == 0 {
                let cons2 = if has_pair {
                    contact_constraints.read(cons_idx + 1)
                } else {
                    cons
                };
                let is_self = cons.free_body_id == u32::MAX;
                let free = if is_self {
                    Velocity::default()
                } else {
                    solver_vels.read(cons.free_body_id as usize)
                };
                if !is_self {
                    j_dot_v0 += cons.lin_jac.dot(free.linear) + gdot(cons.ang_jac, free.angular);
                    if has_pair {
                        j_dot_v1 +=
                            cons2.lin_jac.dot(free.linear) + gdot(cons2.ang_jac, free.angular);
                    }
                }
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
                let (new0, new1) = if is_tangent {
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
                delta0 = new0 - impulse0;
                delta1 = if has_pair { new1 - impulse1 } else { 0.0 };
                contact_constraints.at_mut(cons_idx).impulse = new0;
                if has_pair {
                    contact_constraints.at_mut(cons_idx + 1).impulse = new1;
                }
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
            let delta0 = shuffle::<32>(delta0, 0);
            let delta1 = shuffle::<32>(delta1, 0);
            if lane < ndofs {
                if delta0 != 0.0 {
                    velocity += delta0 * contact_jac_cols.read(jac_offset + stride + lane as usize);
                }
                if has_pair && delta1 != 0.0 {
                    velocity +=
                        delta1 * contact_jac_cols.read(jac_offset2 + stride + lane as usize);
                }
            }
        }
    }
    if lane < ndofs {
        dof_state.write(batch_ids.mbi(batch_id, v_base + lane as usize), velocity);
    }
}
