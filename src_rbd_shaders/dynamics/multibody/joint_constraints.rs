//! Multibody joint limit / motor constraints.
//!
//! Each constraint targets a single generalized DOF and is solved with PGS
//! sweeps. Per-multibody, all constraint slots are scanned (`kind == 0` ones
//! are skipped).

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::iter::StepRng;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::control_barrier;

use crate::dynamics::ConstraintSoftness;
use crate::dynamics::joint::SPATIAL_DIM;
use crate::utils::BatchIndices;
use crate::utils::linalg::{MatSlice, lu_solve_in_place};
use crate::{DIM, MAX_FLT};

use super::types::{
    MB_JOINT_KIND_INACTIVE, MB_JOINT_KIND_LIMIT, MB_JOINT_KIND_LIMIT_INACTIVE,
    MB_JOINT_KIND_MOTOR, MultibodyInfo, MultibodyJointConstraint, MultibodyLinkStatic,
    MultibodyLinkWorkspace,
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

/// Serial (lane-0) emission walk: writes the metadata of every active
/// limit/motor constraint slot. The expensive M⁻¹-column back-solves happen
/// afterwards, lane-parallel, in `gpu_mb_init_joint_constraints`' finalize
/// stage. Slot zeroing also happens there (lane-parallel, before this walk).
#[inline]
fn emit_joint_constraints(
    links_static: &[MultibodyLinkStatic],
    links_workspace: &[MultibodyLinkWorkspace],
    joint_constraints: &mut [MultibodyJointConstraint],
    mb: &MultibodyInfo,
    cons_base: usize,
    batch_id: u32,
    dt: f32,
    joint_erp_inv_dt: f32,
    joint_cfm_coeff: f32,
    batch_ids: &BatchIndices,
) {
    let num_links = mb.num_links;

    let stat_slice = batch_ids
        .mb_links_batch(batch_id, links_static)
        .offset(mb.first_link as usize);
    let ws_slice = batch_ids
        .mb_links_batch(batch_id, links_workspace)
        .offset(mb.first_link as usize);

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
                let cons = build_motor_constraint(
                    abs_dof,
                    k,
                    axis,
                    curr_pos,
                    inv_dt,
                    dt,
                    &stat.data.motors[axis as usize],
                    has_limits,
                    limit_min,
                    limit_max,
                );
                joint_constraints.write(cons_base + slot as usize, cons);
                slot += 1;
            }
            if (limit_axes & (1 << axis)) != 0 {
                let cons = build_limit_constraint(
                    abs_dof,
                    k,
                    axis,
                    curr_pos,
                    [
                        stat.data.limits[axis as usize].min,
                        stat.data.limits[axis as usize].max,
                    ],
                    joint_erp_inv_dt,
                    joint_cfm_coeff,
                );
                joint_constraints.write(cons_base + slot as usize, cons);
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
                let cons = build_limit_constraint(
                    abs_dof,
                    k,
                    axis,
                    curr_pos,
                    [
                        stat.data.limits[axis as usize].min,
                        stat.data.limits[axis as usize].max,
                    ],
                    joint_erp_inv_dt,
                    joint_cfm_coeff,
                );
                joint_constraints.write(cons_base + slot as usize, cons);
                slot += 1;
            }
            if (motor_axes & (1 << axis)) != 0 {
                let has_limits = (limit_axes & (1 << axis)) != 0;
                let limit_min = stat.data.limits[axis as usize].min;
                let limit_max = stat.data.limits[axis as usize].max;
                let cons = build_motor_constraint(
                    abs_dof,
                    k,
                    axis,
                    curr_pos,
                    inv_dt,
                    dt,
                    &stat.data.motors[axis as usize],
                    has_limits,
                    limit_min,
                    limit_max,
                );
                joint_constraints.write(cons_base + slot as usize, cons);
                slot += 1;
            }
            curr_free_dof += 1;
        }
    }
}

/// Per-substep joint-constraint refresh — the explicit-coriolis fast path.
///
/// With explicit coriolis the mass-matrix LU (and therefore every slot's M⁻¹
/// column, `inv_lhs` and folded `cfm_gain`) is a per-step constant: only the
/// rhs (from the integrated joint positions), the limit activity and the
/// accumulated impulse change per substep. This kernel recomputes exactly
/// those from the slot's stashed (link, axis) — the full emission walk and
/// the back-solves run once per step instead of once per substep.
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    links_workspace: &[MultibodyLinkWorkspace],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    joint_constraints: &mut [MultibodyJointConstraint],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] softness: &ConstraintSoftness,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
) {
    const LANES: u32 = 64;
    let batch_id = workgroup_id.y;
    let mb_idx = workgroup_id.x;
    let lane = local_id.x;
    let num_mb = batch_ids.multibodies_len;
    if mb_idx >= num_mb {
        return;
    }

    let mb = batch_ids
        .mb_batch(batch_id, multibody_info)
        .read(mb_idx as usize);
    if mb.ndofs == 0 || mb.max_constraints == 0 {
        return;
    }
    let cons_base = batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;

    let stat_slice = batch_ids
        .mb_links_batch(batch_id, links_static)
        .offset(mb.first_link as usize);
    let ws_slice = batch_ids
        .mb_links_batch(batch_id, links_workspace)
        .offset(mb.first_link as usize);

    let dt = softness.dt;
    let inv_dt = if dt != 0.0 { 1.0 / dt } else { 0.0 };

    for s in StepRng::new(lane..mb.max_constraints, LANES) {
        let old = joint_constraints.read(cons_base + s as usize);
        if old.kind == MB_JOINT_KIND_INACTIVE {
            continue;
        }
        let link_id = old._kind_extra & 0xffff;
        let axis = old._kind_extra >> 16;
        let stat = &stat_slice[link_id as usize];
        let ws = &ws_slice[link_id as usize];
        let curr_pos = ws.coords.read(axis as usize);

        // Rebuild the per-substep fields with the SAME formulas as the full
        // emission, then graft the per-step constants (column-derived
        // `inv_lhs` and folded `cfm_gain`) from the existing slot.
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
                &stat.data.motors[axis as usize],
                has_limits,
                stat.data.limits[axis as usize].min,
                stat.data.limits[axis as usize].max,
            )
        } else {
            build_limit_constraint(
                old.dof_id,
                link_id,
                axis,
                curr_pos,
                [
                    stat.data.limits[axis as usize].min,
                    stat.data.limits[axis as usize].max,
                ],
                softness.joint_erp_inv_dt,
                softness.joint_cfm_coeff,
            )
        };
        fresh.inv_lhs = old.inv_lhs;
        fresh.cfm_gain = old.cfm_gain;
        joint_constraints.write(cons_base + s as usize, fresh);
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
///
/// Emits METADATA ONLY: `inv_lhs` is left 0 and `cfm_gain` holds the
/// pre-fold gain (0 for limits); the lane-parallel finalize stage of
/// `gpu_mb_init_joint_constraints` back-solves the M⁻¹ column and applies
/// rapier's `finalize_generic_constraints` fold.
///
/// Inactive limits are emitted too (with `MB_JOINT_KIND_LIMIT_INACTIVE`, and
/// the link/axis packed into `_kind_extra`) so their columns exist when a
/// later substep's refresh activates them.
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
        _pad0: 0,
        rhs: rhs_wo_bias + rhs_bias,
        rhs_wo_bias,
        inv_lhs: 0.0,
        impulse: 0.0,
        impulse_lo: max_neg_impulse,
        impulse_hi: max_pos_impulse,
        cfm_coeff,
        // Pre-fold gain (`cfm_gain_init`); the finalize stage replaces this
        // with `lhs·cfm_coeff + cfm_gain_init`.
        cfm_gain: 0.0,
    }
}

/// Initialize a single motor constraint slot. Mirrors rapier's
/// `unit_joint_motor_constraint`. `has_limits` + `(limit_min, limit_max)` flatten
/// rapier's `Option<[Real; 2]>` parameter (rust-gpu can't represent enums).
///
/// Emits METADATA ONLY: `inv_lhs` is left 0 and `cfm_gain` holds the
/// pre-fold gain (`motor_params.cfm_gain`, nonzero only for force-based
/// motors); the lane-parallel finalize stage of
/// `gpu_mb_init_joint_constraints` back-solves the M⁻¹ column and applies
/// rapier's `finalize_generic_constraints` fold.
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
    has_limits: bool,
    limit_min: f32,
    limit_max: f32,
) -> MultibodyJointConstraint {
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

    MultibodyJointConstraint {
        dof_id,
        kind: MB_JOINT_KIND_MOTOR,
        _kind_extra: link_id | (axis << 16),
        _pad0: 0,
        rhs: rhs_wo_bias,
        rhs_wo_bias,
        inv_lhs: 0.0,
        impulse: 0.0,
        impulse_lo: -max_impulse,
        impulse_hi: max_impulse,
        cfm_coeff,
        // Pre-fold gain (`cfm_gain_init`); the finalize stage replaces this
        // with `lhs·cfm_coeff + cfm_gain_init`.
        cfm_gain,
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
/// One 64-lane workgroup per (multibody, batch), in three stages:
///   1. lane-parallel: zero all constraint slots;
///   2. lane 0: the serial link walk emitting constraint metadata (cheap);
///   3. lane-parallel: one M⁻¹-column LU back-solve per emitted slot plus
///      rapier's `finalize_generic_constraints` cfm fold — this is the
///      expensive part (`O(ndofs²)` storage traffic per constraint) that used
///      to run sequentially on a single thread.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_init_joint_constraints(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
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
    const LANES: u32 = 64;

    // One workgroup per (multibody, batch) — grid `[mbs · LANES, batches, 1]`.
    let batch_id = workgroup_id.y;
    let mb_idx = workgroup_id.x;
    let lane = local_id.x;
    let num_mb = batch_ids.multibodies_len;
    if mb_idx >= num_mb {
        return;
    }

    let mb = batch_ids
        .mb_batch(batch_id, multibody_info)
        .read(mb_idx as usize);
    let ndofs = mb.ndofs;
    // Uniform per workgroup: every lane of this group returns together.
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
    let m = MatSlice::dense(mb_mm_base, ndofs, ndofs);

    // Stage 1 — lane-parallel slot reset.
    for s in StepRng::new(lane..mb.max_constraints, LANES) {
        let mut cz: MultibodyJointConstraint = joint_constraints.read(cons_base + s as usize);
        cz.kind = 0;
        cz.impulse = 0.0;
        joint_constraints.write(cons_base + s as usize, cz);
    }

    // All-storage-memory barrier reached uniformly by every lane. QueueFamily
    // scope + UNIFORM_MEMORY covers storage buffers (same recipe as the LBVH
    // builder); workgroup execution scope orders the stages within this group.
    control_barrier::<
        { khal_std::memory::Scope::Workgroup as u32 },
        { khal_std::memory::Scope::QueueFamily as u32 },
        {
            khal_std::memory::Semantics::UNIFORM_MEMORY.bits()
                | khal_std::memory::Semantics::ACQUIRE_RELEASE.bits()
        },
    >();

    // Stage 2 — serial metadata emission on lane 0.
    if lane == 0 {
        emit_joint_constraints(
            links_static,
            links_workspace,
            joint_constraints,
            &mb,
            cons_base,
            batch_id,
            softness.dt,
            softness.joint_erp_inv_dt,
            softness.joint_cfm_coeff,
            batch_ids,
        );
    }

    control_barrier::<
        { khal_std::memory::Scope::Workgroup as u32 },
        { khal_std::memory::Scope::QueueFamily as u32 },
        {
            khal_std::memory::Semantics::UNIFORM_MEMORY.bits()
                | khal_std::memory::Semantics::ACQUIRE_RELEASE.bits()
        },
    >();

    // Stage 3 — lane-parallel finalize: back-solve the M⁻¹ column of each
    // emitted slot and fold the CFM (rapier's `finalize_generic_constraints`:
    // `cfm_gain = lhs·cfm_coeff + cfm_gain_init; inv_lhs = 1/(lhs + cfm_gain)`.
    // For an acceleration-based position servo (`<position kp>`), `cfm_coeff`
    // is large (∝ 1/(dt²·stiffness)), so the fold dominates and makes the
    // effective gain inertia-independent — without it the servo is far too
    // weak and the robot sags).
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
            mass_matrices,
            m,
            lu_pivots,
            piv_offset,
        );
        let cfm_gain = lhs * cons.cfm_coeff + cons.cfm_gain;
        cons.cfm_gain = cfm_gain;
        cons.inv_lhs = inv(lhs + cfm_gain);
        joint_constraints.write(cons_base + s as usize, cons);
    }
}

// The PGS sweeps over these constraints live in `gpu_mb_solve_constraints`
// (see `solve_constraints.rs`): one fused joint+contact sweep per substep
// phase, with the bias removal folded in as a `use_bias` uniform.
