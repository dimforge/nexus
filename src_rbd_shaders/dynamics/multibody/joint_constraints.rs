//! Multibody joint limit / motor constraints.
//!
//! Each constraint targets a single generalized DOF and is solved with PGS
//! sweeps. Per-multibody, all constraint slots are scanned (`kind == 0` ones
//! are skipped).

use glamx::Vec4;
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::iter::StepRng;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::control_barrier;

use crate::dynamics::ConstraintSoftness;
use crate::dynamics::joint::SPATIAL_DIM;
use crate::utils::BatchIndices;
use crate::utils::linalg::{MatSlice, VSlice, lu_solve_in_place};
use crate::{DIM, MAX_FLT};

use super::types::{
    MB_JOINT_KIND_COUPLING, MB_JOINT_KIND_FRICTION, MB_JOINT_KIND_LIMIT,
    MB_JOINT_KIND_LIMIT_INACTIVE, MB_JOINT_KIND_MOTOR, MbDofCoupling, MultibodyInfo,
    MultibodyJointConstraint, MultibodyLinkStatic,
};
use super::ws_soa::{WsAddr, ws_coord};

/// Per-batch stride of the actuator-delay state: `[tick, k, prev_target x
/// links_batch_capacity]`.
#[inline]
fn motor_delay_stride(batch_ids: &BatchIndices) -> usize {
    2 + batch_ids.links_batch_capacity as usize
}

/// The motor position target to track this substep.
///
/// `tick` counts the physics steps begun since the host last refreshed the
/// delay state, incremented once per step by `gpu_mb_delay_tick`; it is stable
/// for a whole step, so every substep of a step agrees. While it is at most the
/// batch's delay `k`, the motor tracks the previous control step's target
/// instead of the current one, modelling actuator latency with no mid-step host
/// writes. An all-zero delay buffer (the default) leaves `target` untouched:
/// the first step's `tick` is 1, already past `k = 0`.
#[inline]
fn delayed_motor_target(
    motor_delay_state: &[f32],
    batch_ids: &BatchIndices,
    batch_id: u32,
    link_id: u32,
    target: f32,
) -> f32 {
    let base = batch_id as usize * motor_delay_stride(batch_ids);
    let tick = motor_delay_state.read(base);
    let delay_k = motor_delay_state.read(base + 1);
    if tick <= delay_k {
        motor_delay_state.read(base + 2 + link_id as usize)
    } else {
        target
    }
}

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

/// Solve `M · x = J` in place (writes `x` into `dst[0..n]`) where `J =
/// e_{dof_id} − coeff·e_{dof2_id}` (a plain unit rhs when `coeff == 0`).
/// Uses the same LU factor + pivots produced by `gpu_mb_lu_decompose`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn lu_solve_unit(
    buf_m: &[f32],
    m: MatSlice,
    buf_pivots: &[u32],
    piv: VSlice,
    dst: &mut [f32],
    dst_offset: usize,
    dof_id: u32,
    dof2_id: u32,
    coeff: f32,
) {
    let n = m.rows;
    // dst[0..n] := e_{dof_id} − coeff·e_{dof2_id} (then permuted by
    // lu_solve_in_place).
    for i in 0..n {
        let mut v = if i == dof_id { 1.0 } else { 0.0 };
        if i == dof2_id {
            v -= coeff;
        }
        dst[dst_offset + i as usize] = v;
    }
    lu_solve_in_place(buf_m, m, buf_pivots, piv, dst, VSlice::dense(dst_offset));
}

/// Serially writes the metadata of every active limit/motor constraint slot.
#[inline]
#[allow(clippy::too_many_arguments)]
fn emit_joint_constraints(
    links_static: &[MultibodyLinkStatic],
    links_workspace: &[Vec4],
    dof_couplings: &[MbDofCoupling],
    joint_constraints: &mut [MultibodyJointConstraint],
    dof_state: &[f32],
    mb: &MultibodyInfo,
    cons_base: usize,
    batch_id: u32,
    dt: f32,
    joint_erp_inv_dt: f32,
    joint_cfm_coeff: f32,
    motor_delay_state: &[f32],
    batch_ids: &BatchIndices,
) {
    let num_links = mb.num_links;

    let stat_slice = batch_ids
        .ib(batch_id, links_static)
        .offset(mb.first_link as usize);
    let wa = WsAddr::new(mb.first_link as usize, batch_ids.num_batches, batch_id);

    let inv_dt = if dt != 0.0 { 1.0 / dt } else { 0.0 };

    let mut slot = 0u32;
    for k in 0..num_links {
        let stat = &stat_slice[k as usize];
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
            let curr_pos = ws_coord(links_workspace, wa, k, axis);

            if (motor_axes & (1 << axis)) != 0 {
                let has_limits = (limit_axes & (1 << axis)) != 0;
                let limit_min = stat.data.limits.read(axis as usize).min;
                let limit_max = stat.data.limits.read(axis as usize).max;
                let cons = build_motor_constraint(
                    abs_dof,
                    k,
                    axis,
                    curr_pos,
                    inv_dt,
                    dt,
                    stat.data.motors.at(axis as usize),
                    delayed_motor_target(
                        motor_delay_state,
                        batch_ids,
                        batch_id,
                        mb.first_link + k,
                        stat.data.motors.read(axis as usize).target_pos,
                    ),
                    has_limits,
                    limit_min,
                    limit_max,
                );
                if slot < mb.max_constraints {
                    joint_constraints.write(cons_base + slot as usize, cons);
                    slot += 1;
                }
            }
            if (limit_axes & (1 << axis)) != 0 {
                let cons = build_limit_constraint(
                    abs_dof,
                    k,
                    axis,
                    curr_pos,
                    [
                        stat.data.limits.read(axis as usize).min,
                        stat.data.limits.read(axis as usize).max,
                    ],
                    joint_erp_inv_dt,
                    joint_cfm_coeff,
                );
                if slot < mb.max_constraints {
                    joint_constraints.write(cons_base + slot as usize, cons);
                    slot += 1;
                }
            }
            curr_free_dof += 1;
        }

        // Angular DOFs.
        for axis in DIM..(SPATIAL_DIM as u32) {
            if (locked & (1 << axis)) != 0 {
                continue;
            }
            let abs_dof = stat.assembly_id + curr_free_dof;
            let curr_pos = ws_coord(links_workspace, wa, k, axis);

            if (limit_axes & (1 << axis)) != 0 {
                let cons = build_limit_constraint(
                    abs_dof,
                    k,
                    axis,
                    curr_pos,
                    [
                        stat.data.limits.read(axis as usize).min,
                        stat.data.limits.read(axis as usize).max,
                    ],
                    joint_erp_inv_dt,
                    joint_cfm_coeff,
                );
                if slot < mb.max_constraints {
                    joint_constraints.write(cons_base + slot as usize, cons);
                    slot += 1;
                }
            }
            if (motor_axes & (1 << axis)) != 0 {
                let has_limits = (limit_axes & (1 << axis)) != 0;
                let limit_min = stat.data.limits.read(axis as usize).min;
                let limit_max = stat.data.limits.read(axis as usize).max;
                let cons = build_motor_constraint(
                    abs_dof,
                    k,
                    axis,
                    curr_pos,
                    inv_dt,
                    dt,
                    stat.data.motors.at(axis as usize),
                    delayed_motor_target(
                        motor_delay_state,
                        batch_ids,
                        batch_id,
                        mb.first_link + k,
                        stat.data.motors.read(axis as usize).target_pos,
                    ),
                    has_limits,
                    limit_min,
                    limit_max,
                );
                if slot < mb.max_constraints {
                    joint_constraints.write(cons_base + slot as usize, cons);
                    slot += 1;
                }
            }
            curr_free_dof += 1;
        }
    }

    // DoF couplings: one bilateral row per coupling, solved among the joint
    // constraints (rapier's `coupling_velocity_constraints`).
    let coupling_base = batch_ids.mb_dof_couplings_start(batch_id) + mb.first_coupling as usize;
    for c in 0..mb.num_couplings {
        let coupling = dof_couplings.read(coupling_base + c as usize);
        let q1 = ws_coord(
            links_workspace,
            wa,
            coupling.link_axis1 & 0xffff,
            coupling.link_axis1 >> 16,
        );
        let q2 = ws_coord(
            links_workspace,
            wa,
            coupling.link_axis2 & 0xffff,
            coupling.link_axis2 >> 16,
        );
        let cons = build_coupling_constraint(&coupling, q1, q2, joint_erp_inv_dt);
        if slot < mb.max_constraints {
            joint_constraints.write(cons_base + slot as usize, cons);
            slot += 1;
        }
    }

    // Joint dry friction (MJCF `frictionloss`): one box-bounded row per DoF
    // that has a non-zero loss. Purely DoF-indexed, so no link walk is needed.
    // The frictionloss section is all-zero unless the host reserved the extra
    // slots through `RbdState::set_dof_frictionloss`, so this emits nothing
    // (and cannot overflow `max_constraints`) by default.
    let dof_cap = batch_ids.dof_batch_capacity as usize;
    let frictionloss_slice = batch_ids
        .ib(batch_id, dof_state)
        .offset(6 * dof_cap + mb.first_dof as usize);
    let kin_mask_slice = batch_ids
        .ib(batch_id, dof_state)
        .offset(5 * dof_cap + mb.first_dof as usize);
    for d in 0..mb.ndofs {
        let fl = frictionloss_slice.read(d as usize);
        // Kinematic DoFs have a prescribed velocity; a friction row would
        // fight it.
        if fl > 0.0 && kin_mask_slice.read(d as usize) == 0.0 && slot < mb.max_constraints {
            let cons = build_friction_constraint(d, fl, dt, joint_cfm_coeff);
            joint_constraints.write(cons_base + slot as usize, cons);
            slot += 1;
        }
    }
}

/// Solve `M · column = J` (writes the `M⁻¹·J` column) and return the raw
/// `lhs = J·M⁻¹·J`, for `J = e_{dof_id} − coeff·e_{dof2_id}` (`coeff == 0`
/// for limit / motor rows).
#[inline]
#[allow(clippy::too_many_arguments)]
fn compute_constraint_column(
    joint_constraint_columns: &mut [f32],
    col_base: usize,
    slot: u32,
    dofs_stride: usize,
    ndofs: u32,
    dof_id: u32,
    dof2_id: u32,
    coeff: f32,
    mass_matrices: &[f32],
    m: MatSlice,
    lu_pivots: &[u32],
    piv: VSlice,
) -> f32 {
    let _ = ndofs;
    let col_offset = col_base + (slot as usize) * dofs_stride;
    lu_solve_unit(
        mass_matrices,
        m,
        lu_pivots,
        piv,
        joint_constraint_columns,
        col_offset,
        dof_id,
        dof2_id,
        coeff,
    );
    joint_constraint_columns.read(col_offset + dof_id as usize)
        - coeff * joint_constraint_columns.read(col_offset + dof2_id as usize)
}

/// `1 / x`, or 0 when `x == 0` — matches rapier's `crate::utils::inv`.
#[inline]
fn inv(x: f32) -> f32 {
    if x != 0.0 { 1.0 / x } else { 0.0 }
}

/// Initialize a single limit constraint slot.
#[inline]
#[allow(clippy::too_many_arguments)]
fn build_limit_constraint(
    dof_id: u32,
    link_id: u32,
    axis: u32,
    curr_pos: f32,
    limits: [f32; 2],
    erp_inv_dt: f32,
    cfm_coeff: f32,
) -> MultibodyJointConstraint {
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

    let max_neg_impulse = if min_enabled { -MAX_FLT } else { 0.0 };
    let max_pos_impulse = if max_enabled { MAX_FLT } else { 0.0 };

    let kind = if min_enabled || max_enabled {
        MB_JOINT_KIND_LIMIT
    } else {
        // Inactive this substep: the solve skips it, the finalize stage still
        // back-solves its column for later refreshes.
        MB_JOINT_KIND_LIMIT_INACTIVE
    };

    MultibodyJointConstraint {
        dof_id,
        kind,
        _kind_extra: link_id | (axis << 16),
        dof2_id: 0,
        rhs: rhs_wo_bias + rhs_bias,
        rhs_wo_bias,
        inv_lhs: 0.0,
        impulse: 0.0,
        impulse_lo: max_neg_impulse,
        impulse_hi: max_pos_impulse,
        cfm_coeff,
        // This will be calculated in the finalize (orthogonalization) step.
        cfm_gain: 0.0,
        coupling_coeff: 0.0,
        coupling_offset: 0.0,
        _kind_extra2: 0,
        _pad1: 0,
    }
}

/// Initialize a single DoF-coupling constraint slot. Mirrors rapier's
/// `Multibody::coupling_velocity_constraints`: the coupling `q2 = coeff·q1 +
/// offset` is one bilateral constraint with jacobian `J = e_{g2} −
/// coeff·e_{g1}` and a rhs pulling the position drift back to zero.
#[inline]
fn build_coupling_constraint(
    coupling: &MbDofCoupling,
    q1: f32,
    q2: f32,
    erp_inv_dt: f32,
) -> MultibodyJointConstraint {
    let drift = q2 - coupling.coeff * q1 - coupling.offset;

    MultibodyJointConstraint {
        dof_id: coupling.dof2,
        kind: MB_JOINT_KIND_COUPLING,
        _kind_extra: coupling.link_axis2,
        dof2_id: coupling.dof1,
        rhs: drift * erp_inv_dt,
        rhs_wo_bias: 0.0,
        inv_lhs: 0.0,
        impulse: 0.0,
        impulse_lo: -MAX_FLT,
        impulse_hi: MAX_FLT,
        cfm_coeff: 0.0,
        cfm_gain: 0.0,
        coupling_coeff: coupling.coeff,
        coupling_offset: coupling.offset,
        _kind_extra2: coupling.link_axis1,
        _pad1: 0,
    }
}

/// Initialize a single dry-friction constraint slot (MJCF `frictionloss`).
///
/// The row has no position residual: it simply drives the DoF velocity to zero
/// with an impulse clamped to `±frictionloss·dt`, which is MuJoCo's friction
/// loss (a load-independent force bound, unlike Coulomb friction).
///
/// `cfm_coeff` is the shared joint softness (rapier's `joint.softness.cfm_coeff(dt)`).
#[inline]
fn build_friction_constraint(
    dof_id: u32,
    frictionloss: f32,
    dt: f32,
    cfm_coeff: f32,
) -> MultibodyJointConstraint {
    let max_impulse = frictionloss * dt;

    MultibodyJointConstraint {
        dof_id,
        kind: MB_JOINT_KIND_FRICTION,
        _kind_extra: 0,
        dof2_id: 0,
        rhs: 0.0,
        rhs_wo_bias: 0.0,
        inv_lhs: 0.0,
        impulse: 0.0,
        impulse_lo: -max_impulse,
        impulse_hi: max_impulse,
        cfm_coeff,
        // Folded with the row's `lhs` by the finalize stage.
        cfm_gain: 0.0,
        coupling_coeff: 0.0,
        coupling_offset: 0.0,
        _kind_extra2: 0,
        _pad1: 0,
    }
}

/// Initialize a single motor constraint slot..
#[inline]
#[allow(clippy::too_many_arguments)]
fn build_motor_constraint(
    dof_id: u32,
    link_id: u32,
    axis: u32,
    curr_pos: f32,
    inv_dt: f32,
    dt: f32,
    motor: &crate::dynamics::joint::JointMotor,
    // The position target to track, which actuator delay may pull from a
    // previous control step (see `delayed_motor_target`).
    target_pos: f32,
    has_limits: bool,
    limit_min: f32,
    limit_max: f32,
) -> MultibodyJointConstraint {
    let (erp_inv_dt, cfm_coeff, cfm_gain, _, max_impulse) = motor_params(motor, dt);

    let mut rhs_wo_bias = 0.0f32;
    if erp_inv_dt != 0.0 {
        rhs_wo_bias += (curr_pos - target_pos) * erp_inv_dt;
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

    MultibodyJointConstraint {
        dof_id,
        kind: MB_JOINT_KIND_MOTOR,
        _kind_extra: link_id | (axis << 16),
        dof2_id: 0,
        rhs: rhs_wo_bias,
        rhs_wo_bias,
        inv_lhs: 0.0,
        impulse: 0.0,
        impulse_lo: -max_impulse,
        impulse_hi: max_impulse,
        cfm_coeff,
        cfm_gain,
        coupling_coeff: 0.0,
        coupling_offset: 0.0,
        _kind_extra2: 0,
        _pad1: 0,
    }
}

/// Initialize the multibody's joint-limit / joint-motor unit constraints.
///
/// For each link, scans every free DOF that has either `limit_axes` or `motor_axes`
/// set, and emits one `MultibodyJointConstraint` per active limit and one per
/// active motor (rapier emits these separately even when both are on the same axis).
///
/// Must run after `gpu_mb_lu_decompose` — the LU factors of `M` are used to compute
/// the per-constraint M⁻¹ column and effective inverse mass.
///
/// One 64-lane workgroup per (multibody, batch), in two stages:
///   1. lane-parallel: zero all constraint slots;
///   2. lane 0: the serial link walk emitting constraint metadata (cheap).
///
/// The M⁻¹-column back-solve is a separate dispatch
/// ([`gpu_mb_finalize_joint_constraints`]), so each pass fits 8 storage buffers.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_init_joint_constraints(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] links_workspace: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dof_couplings: &[MbDofCoupling],
    // Actuator-delay state, per-batch `[tick, k, prev_target x links]`. Zeroed
    // (the default) means no delay; see `delayed_motor_target`.
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] motor_delay_state: &[f32],
    // Packed per-DoF sections; only the kinematic mask (5) and frictionloss
    // (6) ones are read here.
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] dof_state: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 7)] softness: &ConstraintSoftness,
    #[spirv(uniform, descriptor_set = 0, binding = 8)] batch_ids: &BatchIndices,
) {
    const LANES: u32 = 64;

    // One workgroup per (multibody, batch): grid `[mbs · LANES, batches, 1]`.
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

    let mb = batch_ids.ib(batch_id, multibody_info).read(slot as usize);
    let ndofs = mb.ndofs;
    // Uniform per workgroup: every lane of this group returns together.
    #[cfg(not(feature = "web-compat"))]
    if ndofs == 0 {
        return;
    }
    let active = in_range && ndofs != 0;

    let cons_base = batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;

    // Stage 1: lane-parallel slot reset.
    if active {
        for s in StepRng::new(lane..mb.max_constraints, LANES) {
            let mut cz: MultibodyJointConstraint = joint_constraints.read(cons_base + s as usize);
            cz.kind = 0;
            cz.impulse = 0.0;
            joint_constraints.write(cons_base + s as usize, cz);
        }
    }

    control_barrier::<
        { khal_std::memory::Scope::Workgroup as u32 },
        { khal_std::memory::Scope::QueueFamily as u32 },
        {
            khal_std::memory::Semantics::UNIFORM_MEMORY.bits()
                | khal_std::memory::Semantics::ACQUIRE_RELEASE.bits()
        },
    >();

    // Stage 2: serial metadata emission on lane 0.
    if active && lane == 0 {
        emit_joint_constraints(
            links_static,
            links_workspace,
            dof_couplings,
            joint_constraints,
            dof_state,
            &mb,
            cons_base,
            batch_id,
            softness.dt,
            softness.joint_erp_inv_dt,
            softness.joint_cfm_coeff,
            motor_delay_state,
            batch_ids,
        );
    }
}

/// Back-solves one M⁻¹ column per emitted joint-constraint slot and applies
/// rapier's `finalize_generic_constraints`.
///
/// Split from [`gpu_mb_init_joint_constraints`] so that each pass stays within
/// the 8-storage-buffer WebGPU limit. Must run after it, and after
/// `gpu_mb_lu_decompose`, whose LU factors of `M` it consumes.
///
/// One 64-lane workgroup per (multibody, batch); lanes stride the slots.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_finalize_joint_constraints(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    joint_constraint_columns: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] mass_matrices: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] lu_pivots: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
) {
    const LANES: u32 = 64;

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

    let mb = batch_ids.ib(batch_id, multibody_info).read(slot as usize);
    let ndofs = mb.ndofs;
    #[cfg(not(feature = "web-compat"))]
    if ndofs == 0 {
        return;
    }
    let active = in_range && ndofs != 0;

    let piv = VSlice::dense(batch_ids.mb_region(batch_id, mb.first_dof, ndofs));
    let cons_base = batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let col_base = batch_ids.mb_joint_constraint_columns_start(batch_id)
        + (mb.first_constraint as usize) * dofs_stride;
    let m = MatSlice::dense(
        batch_ids.mb_region(batch_id, mb.mass_matrix_offset, ndofs * ndofs),
        ndofs,
        ndofs,
    );

    if active {
        for s in StepRng::new(lane..mb.max_constraints, LANES) {
            let mut cons = joint_constraints.read(cons_base + s as usize);
            if cons.kind == 0 {
                continue;
            }
            let lhs = compute_constraint_column(
                joint_constraint_columns,
                col_base,
                s,
                dofs_stride,
                ndofs,
                cons.dof_id,
                cons.dof2_id,
                cons.coupling_coeff,
                mass_matrices,
                m,
                lu_pivots,
                piv,
            );
            let cfm_gain = lhs * cons.cfm_coeff + cons.cfm_gain;
            cons.cfm_gain = cfm_gain;
            cons.inv_lhs = inv(lhs + cfm_gain);
            joint_constraints.write(cons_base + s as usize, cons);
        }
    }
}

/// Per-substep refresh of the joint limit / motor slots, the cheap alternative
/// to a full rebuild.
///
/// One 64-lane workgroup per (multibody, batch); lanes stride the slots.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_refresh_joint_constraints(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] links_workspace: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    joint_constraints: &mut [MultibodyJointConstraint],
    // Actuator-delay state; see `gpu_mb_init_joint_constraints`.
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] motor_delay_state: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] softness: &ConstraintSoftness,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] batch_ids: &BatchIndices,
) {
    const LANES: u32 = 64;
    let batch_id = workgroup_id.y;
    let mb_idx = workgroup_id.x;
    let lane = local_id.x;
    if mb_idx >= batch_ids.multibodies_len {
        return;
    }

    let mb = batch_ids.ib(batch_id, multibody_info).read(mb_idx as usize);
    if mb.ndofs == 0 || mb.max_constraints == 0 {
        return;
    }
    let cons_base = batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;
    let stat_slice = batch_ids
        .ib(batch_id, links_static)
        .offset(mb.first_link as usize);
    let wa = WsAddr::new(mb.first_link as usize, batch_ids.num_batches, batch_id);

    let dt = softness.dt;
    let inv_dt = if dt != 0.0 { 1.0 / dt } else { 0.0 };

    for s in StepRng::new(lane..mb.max_constraints, LANES) {
        let old = joint_constraints.read(cons_base + s as usize);
        // Friction rows are per-step constants except for the accumulated
        // impulse, which must restart from zero so the `±frictionloss·dt`
        // bound applies per substep rather than per step.
        if old.kind == MB_JOINT_KIND_FRICTION {
            let mut fresh = old;
            fresh.impulse = 0.0;
            joint_constraints.write(cons_base + s as usize, fresh);
            continue;
        }
        // Coupling rows are per-step constants; inactive slots stay inactive.
        if old.kind != MB_JOINT_KIND_MOTOR
            && old.kind != MB_JOINT_KIND_LIMIT
            && old.kind != MB_JOINT_KIND_LIMIT_INACTIVE
        {
            continue;
        }
        let link_id = old._kind_extra & 0xffff;
        let axis = old._kind_extra >> 16;
        let stat = &stat_slice[link_id as usize];
        let curr_pos = ws_coord(links_workspace, wa, link_id, axis);
        let limit_min = stat.data.limits.read(axis as usize).min;
        let limit_max = stat.data.limits.read(axis as usize).max;

        // Rebuild the per-substep fields with the same formulas the full
        // emission uses, then graft back the per-step constants (the
        // column-derived `inv_lhs` and the folded `cfm_gain`).
        let mut fresh = if old.kind == MB_JOINT_KIND_MOTOR {
            let locked = stat.data.locked_axes;
            let has_limits = (stat.data.limit_axes & !locked & (1 << axis)) != 0;
            build_motor_constraint(
                old.dof_id,
                link_id,
                axis,
                curr_pos,
                inv_dt,
                dt,
                stat.data.motors.at(axis as usize),
                delayed_motor_target(
                    motor_delay_state,
                    batch_ids,
                    batch_id,
                    mb.first_link + link_id,
                    stat.data.motors.read(axis as usize).target_pos,
                ),
                has_limits,
                limit_min,
                limit_max,
            )
        } else {
            build_limit_constraint(
                old.dof_id,
                link_id,
                axis,
                curr_pos,
                [limit_min, limit_max],
                softness.joint_erp_inv_dt,
                softness.joint_cfm_coeff,
            )
        };
        fresh.inv_lhs = old.inv_lhs;
        fresh.cfm_gain = old.cfm_gain;
        joint_constraints.write(cons_base + s as usize, fresh);
    }
}
