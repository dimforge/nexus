//! Multibody joint limit / motor constraints.
//!
//! Each constraint targets a single generalized DOF and is solved with PGS
//! sweeps. Per-multibody, all constraint slots are scanned (`kind == 0` ones
//! are skipped).

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use crate::dynamics::ConstraintSoftness;
use crate::dynamics::joint::SPATIAL_DIM;
use crate::utils::BatchIndices;
use crate::utils::linalg::{MatSlice, lu_solve_in_place};
use crate::{DIM, MAX_FLT};

use super::types::{
    MultibodyInfo, MultibodyJointConstraint, MultibodyLinkStatic, MultibodyLinkWorkspace,
};

/// Compute joint motor parameters mirroring rapier's `JointMotor::motor_params`.
#[inline]
fn motor_params(motor: &crate::dynamics::joint::JointMotor, dt: f32) -> (f32, f32, f32, f32, f32) {
    // Returns (erp_inv_dt, cfm_coeff, cfm_gain, target_vel_clamp_inv_dt, max_impulse).
    let inv_dt = if dt != 0.0 { 1.0 / dt } else { 0.0 };
    let mp = motor.motor_params(dt);
    (
        mp.erp_inv_dt,
        mp.cfm_coeff,
        mp.cfm_gain,
        inv_dt,
        mp.max_impulse,
    )
}

/// Solve `M · x = e_d` in place (writes `x` into `dst[0..n]`). Uses the same
/// LU factor + pivots produced by `gpu_mb_lu_decompose`.
#[inline]
fn lu_solve_unit(
    buf_m: &[f32],
    m: MatSlice,
    buf_pivots: &[u32],
    pivots_offset: usize,
    dst: &mut [f32],
    dst_offset: usize,
    dof_id: u32,
) {
    let n = m.rows;
    // dst[0..n] := e_{dof_id}  (then permuted by lu_solve_in_place).
    for i in 0..n {
        dst[dst_offset + i as usize] = if i == dof_id { 1.0 } else { 0.0 };
    }
    lu_solve_in_place(buf_m, m, buf_pivots, pivots_offset, dst, dst_offset);
}

/// PGS sweep body — shared between `gpu_mb_solve_joint_constraints` and
/// the fused init+solve / remove-bias+solve kernels. Writes back `cons`
/// before subtracting `delta · column` from `v`.
#[inline]
fn solve_joint_constraints_body(
    multibody_info: &[MultibodyInfo],
    joint_constraints: &mut [MultibodyJointConstraint],
    joint_constraint_columns: &[f32],
    dof_state: &mut [f32],
    batch_id: u32,
    mb_idx: u32,
    batch_ids: &BatchIndices,
) {
    let mb = batch_ids
        .mb_batch(batch_id, multibody_info)
        .read(mb_idx as usize);
    let ndofs = mb.ndofs;
    if ndofs == 0 || mb.max_constraints == 0 {
        return;
    }
    let v_base = batch_ids.dof_start(batch_id) + mb.first_dof as usize;
    let cons_base = batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let col_base = batch_ids.mb_joint_constraint_columns_start(batch_id)
        + (mb.first_constraint as usize) * dofs_stride;

    for s in 0..mb.max_constraints {
        let mut cons = joint_constraints.read(cons_base + s as usize);
        if cons.kind == 0 {
            continue;
        }

        let v_d = dof_state.read(v_base + cons.dof_id as usize);
        let rhs_total = v_d + cons.rhs;
        let raw_imp = cons.impulse + cons.inv_lhs * (rhs_total - cons.cfm_gain * cons.impulse);
        let mut new_imp = raw_imp;
        if new_imp < cons.impulse_lo {
            new_imp = cons.impulse_lo;
        }
        if new_imp > cons.impulse_hi {
            new_imp = cons.impulse_hi;
        }
        let delta = new_imp - cons.impulse;
        cons.impulse = new_imp;
        joint_constraints.write(cons_base + s as usize, cons);

        for i in 0..ndofs {
            let v_idx = v_base + i as usize;
            let cur = dof_state.read(v_idx);
            let col =
                joint_constraint_columns.read(col_base + (s as usize) * dofs_stride + i as usize);
            dof_state.write(v_idx, cur - delta * col);
        }
    }
}

#[inline]
fn init_joint_constraints_body(
    multibody_info: &[MultibodyInfo],
    links_static: &[MultibodyLinkStatic],
    links_workspace: &[MultibodyLinkWorkspace],
    mass_matrices: &[f32],
    lu_pivots: &[u32],
    joint_constraints: &mut [MultibodyJointConstraint],
    joint_constraint_columns: &mut [f32],
    batch_id: u32,
    mb_idx: u32,
    dt: f32,
    joint_erp_inv_dt: f32,
    joint_cfm_coeff: f32,
    batch_ids: &BatchIndices,
) {
    let mb = batch_ids
        .mb_batch(batch_id, multibody_info)
        .read(mb_idx as usize);
    let num_links = mb.num_links;
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        return;
    }
    let mb_mm_base = batch_ids.mm_start(batch_id) + mb.mass_matrix_offset as usize;
    let piv_offset = batch_ids.dof_start(batch_id) + mb.first_dof as usize;
    let cons_base = batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;
    // One column of M⁻¹ per constraint slot — `dof_batch_capacity` floats
    // per slot (only the first `ndofs` of each are meaningful, but we use
    // the batch-wide max as the stride to match the host allocation
    // `cons_col_cap = cons_cap * dofs_cap` and to avoid two multibodies
    // with different ndofs stomping on each other's columns).
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let col_base = batch_ids.mb_joint_constraint_columns_start(batch_id)
        + (mb.first_constraint as usize) * dofs_stride;

    let stat_slice = batch_ids
        .mb_links_batch(batch_id, links_static)
        .offset(mb.first_link as usize);
    let ws_slice = batch_ids
        .mb_links_batch(batch_id, links_workspace)
        .offset(mb.first_link as usize);
    let m = MatSlice::dense(mb_mm_base, ndofs, ndofs);

    for s in 0..mb.max_constraints {
        let mut cz: MultibodyJointConstraint = joint_constraints.read(cons_base + s as usize);
        cz.kind = 0;
        cz.impulse = 0.0;
        joint_constraints.write(cons_base + s as usize, cz);
    }

    let inv_dt = if dt != 0.0 { 1.0 / dt } else { 0.0 };

    let mut slot = 0u32;
    for k in 0..num_links {
        let stat = &stat_slice[k as usize];
        let ws = &ws_slice[k as usize];
        let locked = stat.data.locked_axes;
        let limit_axes = stat.data.limit_axes & !locked;
        let motor_axes = stat.data.motor_axes & !locked;
        if limit_axes == 0 && motor_axes == 0 {
            continue;
        }
        if stat.kinematic != 0 {
            continue;
        }

        // Walk free axes in DOF order, mirroring `MultibodyJoint::velocity_constraints`.
        // `curr_free_dof` tracks the position within this joint's slice of the
        // multibody's generalized-velocity vector; the absolute index is
        // `stat.assembly_id + curr_free_dof`.
        let mut curr_free_dof = 0u32;

        // Linear DOFs first.
        for axis in 0..DIM {
            if (locked & (1 << axis)) != 0 {
                continue;
            }
            let abs_dof = stat.assembly_id + curr_free_dof;
            let curr_pos = ws.coords.read(axis as usize);

            if (motor_axes & (1 << axis)) != 0 {
                let has_limits = (limit_axes & (1 << axis)) != 0;
                let limit_min = stat.data.limits[axis as usize].min;
                let limit_max = stat.data.limits[axis as usize].max;
                emit_motor_constraint(
                    joint_constraints,
                    joint_constraint_columns,
                    cons_base,
                    col_base,
                    dofs_stride,
                    slot,
                    abs_dof,
                    ndofs,
                    curr_pos,
                    inv_dt,
                    dt,
                    &stat.data.motors[axis as usize],
                    has_limits,
                    limit_min,
                    limit_max,
                    mass_matrices,
                    m,
                    lu_pivots,
                    piv_offset,
                );
                slot += 1;
            }
            if (limit_axes & (1 << axis)) != 0 {
                emit_limit_constraint(
                    joint_constraints,
                    joint_constraint_columns,
                    cons_base,
                    col_base,
                    dofs_stride,
                    slot,
                    abs_dof,
                    ndofs,
                    curr_pos,
                    [
                        stat.data.limits[axis as usize].min,
                        stat.data.limits[axis as usize].max,
                    ],
                    joint_erp_inv_dt,
                    joint_cfm_coeff,
                    mass_matrices,
                    m,
                    lu_pivots,
                    piv_offset,
                );
                slot += 1;
            }
            curr_free_dof += 1;
        }

        // Angular DOFs.
        for axis in DIM..(SPATIAL_DIM as u32) {
            if (locked & (1 << axis)) != 0 {
                continue;
            }
            let abs_dof = stat.assembly_id + curr_free_dof;
            let curr_pos = ws.coords.read(axis as usize);

            if (limit_axes & (1 << axis)) != 0 {
                emit_limit_constraint(
                    joint_constraints,
                    joint_constraint_columns,
                    cons_base,
                    col_base,
                    dofs_stride,
                    slot,
                    abs_dof,
                    ndofs,
                    curr_pos,
                    [
                        stat.data.limits[axis as usize].min,
                        stat.data.limits[axis as usize].max,
                    ],
                    joint_erp_inv_dt,
                    joint_cfm_coeff,
                    mass_matrices,
                    m,
                    lu_pivots,
                    piv_offset,
                );
                slot += 1;
            }
            if (motor_axes & (1 << axis)) != 0 {
                let has_limits = (limit_axes & (1 << axis)) != 0;
                let limit_min = stat.data.limits[axis as usize].min;
                let limit_max = stat.data.limits[axis as usize].max;
                emit_motor_constraint(
                    joint_constraints,
                    joint_constraint_columns,
                    cons_base,
                    col_base,
                    dofs_stride,
                    slot,
                    abs_dof,
                    ndofs,
                    curr_pos,
                    inv_dt,
                    dt,
                    &stat.data.motors[axis as usize],
                    has_limits,
                    limit_min,
                    limit_max,
                    mass_matrices,
                    m,
                    lu_pivots,
                    piv_offset,
                );
                slot += 1;
            }
            curr_free_dof += 1;
        }
    }
}

/// Solve `M · column = e_{dof_id}` (writes the M⁻¹ column) and return the raw
/// `lhs = column[dof_id]` for J = e_{dof_id}.
#[inline]
fn compute_constraint_column(
    joint_constraint_columns: &mut [f32],
    col_base: usize,
    slot: u32,
    dofs_stride: usize,
    ndofs: u32,
    dof_id: u32,
    mass_matrices: &[f32],
    m: MatSlice,
    lu_pivots: &[u32],
    piv_offset: usize,
) -> f32 {
    let _ = ndofs;
    let col_offset = col_base + (slot as usize) * dofs_stride;
    lu_solve_unit(
        mass_matrices,
        m,
        lu_pivots,
        piv_offset,
        joint_constraint_columns,
        col_offset,
        dof_id,
    );
    joint_constraint_columns.read(col_offset + dof_id as usize)
}

/// `1 / x`, or 0 when `x == 0` — matches rapier's `crate::utils::inv`.
#[inline]
fn inv(x: f32) -> f32 {
    if x != 0.0 { 1.0 / x } else { 0.0 }
}

/// Initialize a single limit constraint slot. Mirrors rapier's
/// `unit_joint_limit_constraint`.
#[inline]
fn emit_limit_constraint(
    joint_constraints: &mut [MultibodyJointConstraint],
    joint_constraint_columns: &mut [f32],
    cons_base: usize,
    col_base: usize,
    dofs_stride: usize,
    slot: u32,
    dof_id: u32,
    ndofs: u32,
    curr_pos: f32,
    limits: [f32; 2],
    erp_inv_dt: f32,
    cfm_coeff: f32,
    mass_matrices: &[f32],
    m: MatSlice,
    lu_pivots: &[u32],
    piv_offset: usize,
) {
    // rapier (`limit_*` builder): erp_inv_dt = joint.softness.erp_inv_dt(dt),
    // cfm_coeff = joint.softness.cfm_coeff(dt), cfm_gain = 0 — configurable via
    // `joint_natural_frequency` / `joint_damping_ratio` (defaults make this
    // near-rigid, matching the old hardcoded `1/dt`).
    let min_enabled = curr_pos < limits[0];
    let max_enabled = limits[1] < curr_pos;
    let lo_excess = (limits[0] - curr_pos).max(0.0);
    let hi_excess = (curr_pos - limits[1]).max(0.0);
    let rhs_bias = (hi_excess - lo_excess) * erp_inv_dt;
    let rhs_wo_bias = 0.0f32;

    let lhs = compute_constraint_column(
        joint_constraint_columns,
        col_base,
        slot,
        dofs_stride,
        ndofs,
        dof_id,
        mass_matrices,
        m,
        lu_pivots,
        piv_offset,
    );
    // rapier `finalize_generic_constraints` (the multibody-internal constraints
    // ARE finalized, after `unit_joint_limit_constraint` sets the preliminary
    // 1/lhs): cfm_gain = lhs·cfm_coeff + cfm_gain_init; inv_lhs = 1/(lhs +
    // cfm_gain). The PGS sweep then applies `cfm_gain` directly. For limits
    // cfm_gain_init = 0 and (near-rigid) cfm_coeff ≈ 2e-6, so this is ~1/lhs —
    // but we replicate the formula exactly so it's correct for any softness.
    let cfm_gain = lhs * cfm_coeff;
    let inv_lhs = inv(lhs + cfm_gain);

    let max_neg_impulse = if min_enabled { -MAX_FLT } else { 0.0 };
    let max_pos_impulse = if max_enabled { MAX_FLT } else { 0.0 };

    let cons = MultibodyJointConstraint {
        dof_id,
        kind: 1,
        _kind_extra: 0,
        _pad0: 0,
        rhs: rhs_wo_bias + rhs_bias,
        rhs_wo_bias,
        inv_lhs,
        impulse: 0.0,
        impulse_lo: max_neg_impulse,
        impulse_hi: max_pos_impulse,
        cfm_coeff,
        cfm_gain,
    };
    joint_constraints.write(cons_base + slot as usize, cons);
}

/// Initialize a single motor constraint slot. Mirrors rapier's
/// `unit_joint_motor_constraint`. `has_limits` + `(limit_min, limit_max)` flatten
/// rapier's `Option<[Real; 2]>` parameter (rust-gpu can't represent enums).
#[inline]
fn emit_motor_constraint(
    joint_constraints: &mut [MultibodyJointConstraint],
    joint_constraint_columns: &mut [f32],
    cons_base: usize,
    col_base: usize,
    dofs_stride: usize,
    slot: u32,
    dof_id: u32,
    ndofs: u32,
    curr_pos: f32,
    inv_dt: f32,
    dt: f32,
    motor: &crate::dynamics::joint::JointMotor,
    has_limits: bool,
    limit_min: f32,
    limit_max: f32,
    mass_matrices: &[f32],
    m: MatSlice,
    lu_pivots: &[u32],
    piv_offset: usize,
) {
    let (erp_inv_dt, cfm_coeff, cfm_gain, _, max_impulse) = motor_params(motor, dt);

    let mut rhs_wo_bias = 0.0f32;
    if erp_inv_dt != 0.0 {
        rhs_wo_bias += (curr_pos - motor.target_pos) * erp_inv_dt;
    }

    let mut target_vel = motor.target_vel;
    if has_limits {
        let lo = (limit_min - curr_pos) * inv_dt;
        let hi = (limit_max - curr_pos) * inv_dt;
        if target_vel < lo {
            target_vel = lo;
        }
        if target_vel > hi {
            target_vel = hi;
        }
    }
    rhs_wo_bias += -target_vel;

    let lhs = compute_constraint_column(
        joint_constraint_columns,
        col_base,
        slot,
        dofs_stride,
        ndofs,
        dof_id,
        mass_matrices,
        m,
        lu_pivots,
        piv_offset,
    );
    // rapier `finalize_generic_constraints` (run after `unit_joint_motor_
    // constraint` sets the preliminary 1/lhs): cfm_gain = lhs·cfm_coeff +
    // cfm_gain_init; inv_lhs = 1/(lhs + cfm_gain). The PGS sweep then applies
    // `cfm_gain` (`impulse + inv_lhs·(rhs - cfm_gain·impulse)`). This is the
    // ACCELERATION-based motor's compliance: for an acceleration-based position
    // servo (`<position kp>`), `cfm_coeff` is large (∝ 1/(dt²·stiffness)), so the
    // fold dominates and makes the effective gain inertia-independent — without
    // it the servo is far too weak and the robot sags. `cfm_gain` here is
    // `motor_params.cfm_gain` (nonzero only for force-based motors).
    let cfm_gain = lhs * cfm_coeff + cfm_gain;
    let inv_lhs = inv(lhs + cfm_gain);

    let cons = MultibodyJointConstraint {
        dof_id,
        kind: 2,
        _kind_extra: 0,
        _pad0: 0,
        rhs: rhs_wo_bias,
        rhs_wo_bias,
        inv_lhs,
        impulse: 0.0,
        impulse_lo: -max_impulse,
        impulse_hi: max_impulse,
        cfm_coeff,
        cfm_gain,
    };
    joint_constraints.write(cons_base + slot as usize, cons);
}

/// Initialize the multibody's joint-limit / joint-motor unit constraints.
///
/// For each link, scans every free DOF that has either `limit_axes` or `motor_axes`
/// set, and emits one `MultibodyJointConstraint` per active limit and one per
/// active motor (rapier emits these separately even when both are on the same axis).
///
/// Must run after `gpu_mb_lu_decompose` — the LU factors of `M` are used to compute
/// the per-constraint M⁻¹ column and effective inverse mass.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_init_joint_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    links_workspace: &[MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] mass_matrices: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] lu_pivots: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)]
    joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)]
    joint_constraint_columns: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 7)] softness: &ConstraintSoftness,
    #[spirv(uniform, descriptor_set = 0, binding = 8)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let mb_idx = invocation_id.x;
    let num_mb = batch_ids.multibodies_len;
    if mb_idx >= num_mb {
        return;
    }
    init_joint_constraints_body(
        multibody_info,
        links_static,
        links_workspace,
        mass_matrices,
        lu_pivots,
        joint_constraints,
        joint_constraint_columns,
        batch_id,
        mb_idx,
        softness.dt,
        softness.joint_erp_inv_dt,
        softness.joint_cfm_coeff,
        batch_ids,
    );
}

/// One PGS sweep: iterates the multibody's active limit/motor constraints and
/// updates `dof_velocities` in place. Mirrors rapier's `JointConstraint::solve_generic`
/// for a 1-DOF jacobian.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_solve_joint_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    joint_constraint_columns: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dof_state: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let mb_idx = invocation_id.x;
    let num_mb = batch_ids.multibodies_len;
    if mb_idx >= num_mb {
        return;
    }
    solve_joint_constraints_body(
        multibody_info,
        joint_constraints,
        joint_constraint_columns,
        dof_state,
        batch_id,
        mb_idx,
        batch_ids,
    );
}

/// Fused `remove_bias + solve_without_bias` for joint constraints — runs once
/// per substep at the end of the substep, after position integration. Drops
/// one threads(1) dispatch per substep.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_remove_solve_joint_no_bias(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] joint_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dof_state: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let mb_idx = invocation_id.x;
    let num_mb = batch_ids.multibodies_len;
    if mb_idx >= num_mb {
        return;
    }

    let mb = batch_ids
        .mb_batch(batch_id, multibody_info)
        .read(mb_idx as usize);
    let cons_base = batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;

    // Inlined `remove_bias`: replace `rhs` with `rhs_wo_bias` for active slots.
    for s in 0..mb.max_constraints {
        let mut cons = joint_constraints.read(cons_base + s as usize);
        if cons.kind == 0 {
            continue;
        }
        cons.rhs = cons.rhs_wo_bias;
        joint_constraints.write(cons_base + s as usize, cons);
    }

    solve_joint_constraints_body(
        multibody_info,
        joint_constraints,
        joint_constraint_columns,
        dof_state,
        batch_id,
        mb_idx,
        batch_ids,
    );
}
