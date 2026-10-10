//! Native Metal substep loop for small multibodies, one per environment.
//!
//! Runs every substep phase of the explicit-Coriolis solver (velocity update,
//! joint refresh, contact warmstart, biased sweeps, position update and
//! stabilization sweep) in one dispatch instead of one dispatch per phase, so
//! an environment's state is loaded once per step rather than once per phase.
//! A SIMD group holds four environments in partitions of eight lanes: lane `p`
//! owns DOFs `4p..4p + 4` (one quad of each padded joint column), links
//! `p + 8b` and joint slots `p + 8j`. Generalized velocities and coordinates
//! stay in registers; joint rows stay in their storage buffer.
//! The update order and arithmetic match the per-phase kernels, so results are
//! bit-identical to them.

use super::joint_constraints::{build_limit_constraint, motor_rhs_wo_bias};
use super::link_static_soa::LinkStatics;
use super::solve_constraints::cap_friction;
use super::types::*;
use super::ws_soa::{WS_COORDS, WS_JOINT_ROT, WsAddr};
use crate::dynamics::{ConstraintSoftness, Velocity, decode_bias_mode};
use crate::utils::BatchIndices;
use crate::{DIM, gdot};
use crate::{Vector, rotation_from_scaled_axis, rotation_renormalize_fast};
use glamx::{Quat, Vec4};
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::{subgroup_f_add, workgroup_memory_barrier_with_group_sync};
use parry::math::VectorExt;

/// Lanes per environment.
const LANES: u32 = 8;
/// Environments per SIMD group.
const PARTITIONS: u32 = 32 / LANES;

#[inline(always)]
fn shuffle(value: f32, lane: u32) -> f32 {
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

/// One link's generalized coordinates (`coords[0..6]`).
#[derive(Copy, Clone)]
struct Coords {
    q0: Vec4,
    q1: Vec4,
}

impl Coords {
    #[inline(always)]
    fn get(&self, i: u32) -> f32 {
        match i {
            0 => self.q0.x,
            1 => self.q0.y,
            2 => self.q0.z,
            3 => self.q0.w,
            4 => self.q1.x,
            _ => self.q1.y,
        }
    }

    #[inline(always)]
    fn set(&mut self, i: u32, v: f32) {
        match i {
            0 => self.q0.x = v,
            1 => self.q0.y = v,
            2 => self.q0.z = v,
            3 => self.q0.w = v,
            4 => self.q1.x = v,
            _ => self.q1.y = v,
        }
    }
}

/// Unrolls `$body` over a lane's four bands (DOFs, links).
macro_rules! bands {
    ($b:ident, $body:block) => {{
        {
            let $b = 0u32;
            $body
        }
        {
            let $b = 1u32;
            $body
        }
        {
            let $b = 2u32;
            $body
        }
        {
            let $b = 3u32;
            $body
        }
    }};
}

#[inline(always)]
fn band_get(v: Vec4, b: u32) -> f32 {
    match b {
        0 => v.x,
        1 => v.y,
        2 => v.z,
        _ => v.w,
    }
}

#[inline(always)]
fn band_set(v: &mut Vec4, b: u32, x: f32) {
    match b {
        0 => v.x = x,
        1 => v.y = x,
        2 => v.z = x,
        _ => v.w = x,
    }
}

/// DOF of band `b` of partition lane `p`.
#[inline(always)]
fn lane_dof(p: u32, b: u32) -> u32 {
    4 * p + b
}

/// Value of `dof`, read from its owner lane in the partition at `base`.
/// Every lane of the partition must request the same `dof`: the band is
/// selected before the shuffle.
#[inline(always)]
fn dof_value(v: Vec4, base: u32, dof: u32) -> f32 {
    shuffle(band_get(v, dof % 4), base + dof / 4)
}

/// Sum of `value` over the partition at `base` (masked subgroup sums keep the
/// partitions apart).
#[inline(always)]
fn partition_sum(value: f32, base: u32) -> f32 {
    let mut total = 0.0;
    for part in 0..PARTITIONS {
        let s = subgroup_f_add(if base == part * LANES { value } else { 0.0 });
        if base == part * LANES {
            total = s;
        }
    }
    total
}

/// Storage writes by other lanes of this SIMD group become visible.
#[inline(always)]
fn storage_barrier() {
    khal_std::sync::control_barrier::<2, 2, 0x148>();
}

/// Whether the sweeps solve a joint row of this kind.
#[inline(always)]
fn is_solved(kind: u32) -> bool {
    kind == MB_JOINT_KIND_LIMIT
        || kind == MB_JOINT_KIND_MOTOR
        || kind == MB_JOINT_KIND_COUPLING
        || kind == MB_JOINT_KIND_FRICTION
}

/// This lane's quad of one joint slot's padded column (zero past `ndofs`).
#[inline(always)]
fn load_column(joint_columns: &[Vec4], slot: usize, quads: u32, p: u32) -> Vec4 {
    if p < quads {
        joint_columns.read(slot * quads as usize + p as usize)
    } else {
        Vec4::ZERO
    }
}

/// One projected Gauss-Seidel update of a solved joint row.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn solve_joint_row(
    mut cons: MultibodyJointConstraint,
    col: Vec4,
    s: usize,
    p: u32,
    base: u32,
    use_bias: bool,
    writes: bool,
    velocity: &mut Vec4,
    joint_constraints: &mut [MultibodyJointConstraint],
    jcons_base: usize,
) {
    cons.impulse = shuffle(cons.impulse, base);
    let rhs = if use_bias { cons.rhs } else { cons.rhs_wo_bias };
    let mut v_d = dof_value(*velocity, base, cons.dof_id);
    if cons.kind == MB_JOINT_KIND_COUPLING {
        v_d -= cons.coupling_coeff * dof_value(*velocity, base, cons.dof2_id);
    }
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
    if p == 0 && writes {
        joint_constraints.at_mut(jcons_base + s).impulse = new_imp;
    }
    *velocity -= col * delta;
}

/// One packed PGS sweep: the solved joint rows, then the contact rows (same
/// order and arithmetic as `solve_packed8`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn sweep_packed(
    p: u32,
    base: u32,
    ndofs: u32,
    max_constraints: u32,
    velocity: &mut Vec4,
    mode: u32,
    writes: bool,
    joint_constraints: &mut [MultibodyJointConstraint],
    joint_columns: &[Vec4],
    quads: u32,
    jcons_base: usize,
    stride: usize,
    contact_constraints: &mut [MultibodyContactConstraint],
    contact_jac_cols: &[f32],
    ccons_base: usize,
    contact_count: u32,
    solver_vels: &mut [Velocity],
) {
    let (use_bias, solve_friction) = decode_bias_mode(mode);
    for s in 0..max_constraints.min(64) as usize {
        let cons = joint_constraints.read(jcons_base + s);
        if !is_solved(cons.kind) {
            continue;
        }
        let col = load_column(joint_columns, jcons_base + s, quads, p);
        solve_joint_row(
            cons,
            col,
            s,
            p,
            base,
            use_bias,
            writes,
            velocity,
            joint_constraints,
            jcons_base,
        );
    }
    // No barrier: each sweep takes the impulse from lane `p == 0`, which
    // wrote it (see the shuffle above).

    let cjc_base = ccons_base * 2 * stride;
    for s in 0..contact_count {
        let cons_idx = ccons_base + s as usize;
        let cons = contact_constraints.read(cons_idx);
        let is_tangent = cons.kind == MB_CONTACT_KIND_TANGENT;
        if is_tangent && !solve_friction {
            continue;
        }
        if is_tangent && s != cons.normal_constraint_slot + 1 {
            continue;
        }
        let has_pair = is_tangent;
        let jac_offset = cjc_base + s as usize * 2 * stride;
        let jac_offset2 = jac_offset + 2 * stride;
        let mut product0 = 0.0;
        let mut product1 = 0.0;
        bands!(b, {
            let dof = lane_dof(p, b);
            if dof < ndofs {
                product0 +=
                    contact_jac_cols.read(jac_offset + dof as usize) * band_get(*velocity, b);
                if has_pair {
                    product1 +=
                        contact_jac_cols.read(jac_offset2 + dof as usize) * band_get(*velocity, b);
                }
            }
        });
        let mut j_dot_v0 = partition_sum(product0, base);
        let mut j_dot_v1 = partition_sum(product1, base);
        let mut delta0 = 0.0;
        let mut delta1 = 0.0;
        if p == 0 {
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
                    j_dot_v1 += cons2.lin_jac.dot(free.linear) + gdot(cons2.ang_jac, free.angular);
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
            if writes {
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
        }
        let delta0 = shuffle(delta0, base);
        let delta1 = shuffle(delta1, base);
        bands!(b, {
            let dof = lane_dof(p, b);
            if dof < ndofs {
                let mut v = band_get(*velocity, b);
                if delta0 != 0.0 {
                    v += delta0 * contact_jac_cols.read(jac_offset + stride + dof as usize);
                }
                if has_pair && delta1 != 0.0 {
                    v += delta1 * contact_jac_cols.read(jac_offset2 + stride + dof as usize);
                }
                band_set(velocity, b, v);
            }
        });
    }
}

/// Same rebuild as `gpu_mb_refresh_joint_constraints` for one slot, from its
/// refresh parameters (`JointRefreshParams`) instead of the link data.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn refresh_slot(
    joint_constraints: &mut [MultibodyJointConstraint],
    index: usize,
    pa: Vec4,
    pb: Vec4,
    shared_vals: &[f32; 128],
    sb: usize,
    inv_dt: f32,
    softness: &ConstraintSoftness,
) {
    let meta = pb.y.to_bits();
    let kind = meta & 0xff;
    let dof = (meta >> 8) & 0xff;
    let curr_pos = shared_vals[sb + dof.min(31) as usize];
    if kind == MB_JOINT_KIND_MOTOR {
        let rhs = motor_rhs_wo_bias(
            curr_pos,
            pa.x,
            pa.y,
            pa.z,
            inv_dt,
            (meta >> 16) & 1 != 0,
            pa.w,
            pb.x,
        );
        let row = joint_constraints.at_mut(index);
        row.rhs = rhs;
        row.rhs_wo_bias = rhs;
        row.impulse = 0.0;
    } else if kind == MB_JOINT_KIND_LIMIT {
        let fresh = build_limit_constraint(
            dof,
            0,
            0,
            curr_pos,
            [pa.w, pb.x],
            softness.joint_erp_inv_dt,
            softness.joint_cfm_coeff,
        );
        let row = joint_constraints.at_mut(index);
        row.kind = fresh.kind;
        row.rhs = fresh.rhs;
        row.rhs_wo_bias = fresh.rhs_wo_bias;
        row.impulse = 0.0;
        row.impulse_lo = fresh.impulse_lo;
        row.impulse_hi = fresh.impulse_hi;
    } else if kind == MB_JOINT_KIND_FRICTION {
        joint_constraints.at_mut(index).impulse = 0.0;
    }
}

/// The substep loop (see the module docs).
///
/// Expects the first substep's velocity update and constraint build (joint
/// emission + finalize with padded columns, contact build, warmstart transfer,
/// restitution seed) to have run already; it starts at that substep's contact
/// warmstart. Requires one multibody per environment with at most 32 DOFs, 32
/// links and 64 joint slots, and a 32-wide subgroup; the host checks all of
/// this.
#[spirv_bindgen]
#[spirv(compute(threads(32)))]
pub fn gpu_mb_substeps_packed(
    #[spirv(workgroup_id)] wid: khal_std::glamx::UVec3,
    #[spirv(local_invocation_id)] lid: khal_std::glamx::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_static: &[glamx::UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] links_workspace: &mut [Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    joint_constraints: &mut [MultibodyJointConstraint],
    // Columns padded to whole quads (`gpu_mb_finalize_joint_simd`'s padded
    // layout): lane `p` reads quad `p`.
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] joint_columns: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] contact_jac_cols: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] gen_accelerations: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 8)] batch_ids: &BatchIndices,
    #[spirv(uniform, descriptor_set = 0, binding = 9)] softness: &ConstraintSoftness,
    #[spirv(uniform, descriptor_set = 0, binding = 10)] dt_uniform: &f32,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] solver_vels: &mut [Velocity],
    #[spirv(uniform, descriptor_set = 1, binding = 3)] num_substeps: &u32,
    #[spirv(uniform, descriptor_set = 1, binding = 4)] num_iterations: &u32,
    #[spirv(uniform, descriptor_set = 1, binding = 5)] bias_mode: &u32,
    #[spirv(uniform, descriptor_set = 1, binding = 6)] warmstart: &u32,
    // Two quads per joint slot (`JointRefreshParams`), written by the joint
    // build pass.
    #[spirv(storage_buffer, descriptor_set = 1, binding = 7)] refresh_params: &[Vec4],
    #[spirv(workgroup)] shared_vals: &mut [f32; 128],
    #[spirv(workgroup)] shared_words: &mut [u32; 128],
) {
    let lane = lid.x;
    let p = lane % LANES;
    let base = lane - p;
    let part = (lane / LANES) as usize;
    let num_batches = batch_ids.num_batches;
    if num_batches == 0 || batch_ids.multibodies_len == 0 {
        return;
    }
    let dt = *dt_uniform;
    let num_iterations = *num_iterations;
    let bias_mode = *bias_mode;
    // Every lane reaches the barriers: a partition past the last environment
    // replays the last one without writing.
    let env = wid.x * PARTITIONS + lane / LANES;
    let writes = env < num_batches;
    let batch_id = if writes { env } else { num_batches - 1 };
    let mb = multibody_info.read(batch_ids.mbi(batch_id, 0));
    let ndofs = mb.ndofs;
    let num_links = mb.num_links;
    let max_constraints = mb.max_constraints;
    let stride = batch_ids.dof_batch_capacity as usize;
    let quads = batch_ids.mb_max_ndofs.div_ceil(4);
    let inv_dt = if softness.dt != 0.0 {
        1.0 / softness.dt
    } else {
        0.0
    };
    let acc_base = batch_ids.mb_region(batch_id, mb.first_dof, ndofs);
    let jcons_base = batch_ids.mb_joint_constraints_start(batch_id) + mb.first_constraint as usize;
    let ccons_base = mb.contact_constraint_start as usize;
    let cjc_base = ccons_base * 2 * stride;
    let contact_count = mb.contact_constraint_count;
    let statics = LinkStatics::new(links_static, mb.first_link as usize, num_batches, batch_id);
    let wa = WsAddr::new(mb.first_link as usize, num_batches, batch_id);
    // This partition's 32-entry window of the shared arrays.
    let sb = part * 32;

    let mut velocity = Vec4::ZERO;
    let mut acceleration = Vec4::ZERO;
    bands!(b, {
        let dof = lane_dof(p, b);
        if dof < ndofs {
            band_set(
                &mut velocity,
                b,
                dof_state.read(batch_ids.mbi(batch_id, mb.first_dof as usize + dof as usize)),
            );
            band_set(
                &mut acceleration,
                b,
                gen_accelerations.read(acc_base + dof as usize),
            );
        }
    });

    // The position update advances each free coordinate by its DOF velocity
    // (`gpu_mb_integrate`), so the coordinates are tracked per DOF here: link
    // lanes publish them with the coordinate each DOF advances. Only links with
    // three free angular axes accumulate a rotation every substep.
    let mut rot3_links = 0u32;
    bands!(b, {
        let link = p + b * LANES;
        if link < num_links {
            let stat = statics.at(link as usize);
            let locked = stat.locked_axes();
            let first_dof = stat.assembly_id();
            let coords = Coords {
                q0: links_workspace.read(wa.at(link, WS_COORDS)),
                q1: links_workspace.read(wa.at(link, WS_COORDS + 1)),
            };
            let num_ang = 3 - ((locked >> DIM) & 0x7).count_ones();
            if num_ang == 3 {
                rot3_links |= 1 << b;
            }
            let mut c = 0u32;
            for axis in 0..6u32 {
                if (locked & (1 << axis)) == 0 {
                    let dof = first_dof + c;
                    // Two free angular axes: `gpu_mb_integrate` leaves them.
                    let advances = axis < DIM || num_ang != 2;
                    if dof < 32 {
                        shared_vals[sb + dof as usize] = coords.get(axis);
                        shared_words[sb + dof as usize] = advances as u32;
                    }
                    c += 1;
                }
            }
        }
    });
    workgroup_memory_barrier_with_group_sync();
    let mut q = Vec4::ZERO;
    let mut advances = 0u32;
    bands!(b, {
        let dof = lane_dof(p, b);
        if dof < ndofs {
            band_set(&mut q, b, shared_vals[sb + dof as usize]);
            advances |= shared_words[sb + dof as usize] << b;
        }
    });
    let any_rot3 = khal_std::sync::subgroup_f_max(rot3_links as f32) > 0.0;

    let slot_rounds = max_constraints.min(64).div_ceil(LANES);
    for substep in 0..*num_substeps {
        if substep > 0 {
            // P1: `v += a · dt`.
            bands!(b, {
                if lane_dof(p, b) < ndofs {
                    let v = band_get(velocity, b) + band_get(acceleration, b) * dt;
                    band_set(&mut velocity, b, v);
                }
            });
            // P2: joint rhs / limit activity from the integrated coordinates.
            workgroup_memory_barrier_with_group_sync();
            bands!(b, {
                let dof = lane_dof(p, b);
                if dof < ndofs {
                    shared_vals[sb + dof as usize] = band_get(q, b);
                }
            });
            workgroup_memory_barrier_with_group_sync();
            if writes {
                for j in 0..slot_rounds {
                    let s = p + j * LANES;
                    if s < max_constraints {
                        let index = jcons_base + s as usize;
                        refresh_slot(
                            joint_constraints,
                            index,
                            refresh_params.read(2 * index),
                            refresh_params.read(2 * index + 1),
                            shared_vals,
                            sb,
                            inv_dt,
                            softness,
                        );
                    }
                }
            }
            storage_barrier();
        }

        // Contact warmstart: re-apply the accumulated impulses.
        if *warmstart != 0 {
            for s in 0..contact_count {
                let cons = contact_constraints.read(ccons_base + s as usize);
                let imp = cons.impulse;
                if imp != 0.0 {
                    let col_offset = cjc_base + (s as usize) * 2 * stride + stride;
                    bands!(b, {
                        let dof = lane_dof(p, b);
                        if dof < ndofs {
                            let v = band_get(velocity, b)
                                + imp * contact_jac_cols.read(col_offset + dof as usize);
                            band_set(&mut velocity, b, v);
                        }
                    });
                    let is_self = cons.free_body_id == u32::MAX;
                    if p == 0 && !is_self && writes {
                        let free = solver_vels.read(cons.free_body_id as usize);
                        let mut new_free = free;
                        new_free.linear += cons.lin_jac * (cons.free_body_im * imp);
                        new_free.angular += cons.ii_ang_jac * imp;
                        solver_vels.write(cons.free_body_id as usize, new_free);
                    }
                }
            }
            storage_barrier();
        }

        // P3: biased sweeps, then P4 (position update) and P5 (the
        // stabilization sweep) as the last pass.
        for k in 0..num_iterations + 1 {
            let last = k == num_iterations;
            if last {
                bands!(b, {
                    if (advances & (1 << b)) != 0 {
                        let x = band_get(q, b) + band_get(velocity, b) * dt;
                        band_set(&mut q, b, x);
                    }
                });
                if any_rot3 {
                    workgroup_memory_barrier_with_group_sync();
                    bands!(b, {
                        let dof = lane_dof(p, b);
                        if dof < ndofs {
                            shared_vals[sb + dof as usize] = band_get(velocity, b);
                        }
                    });
                    workgroup_memory_barrier_with_group_sync();
                    bands!(b, {
                        let link = p + b * LANES;
                        if (rot3_links & (1 << b)) != 0 && writes {
                            let stat = statics.at(link as usize);
                            let locked = stat.locked_axes();
                            // Angular DOFs follow the free linear ones.
                            let first = stat.assembly_id() + 3 - (locked & 0x7).count_ones();
                            let vx = shared_vals[sb + first.min(31) as usize];
                            let vy = shared_vals[sb + (first + 1).min(31) as usize];
                            let vz = shared_vals[sb + (first + 2).min(31) as usize];
                            let disp = rotation_from_scaled_axis(Vector::new(vx, vy, vz) * dt);
                            let r = links_workspace.read(wa.at(link, WS_JOINT_ROT));
                            let joint_rot = rotation_renormalize_fast(
                                disp * Quat::from_xyzw(r.x, r.y, r.z, r.w),
                            );
                            links_workspace.write(
                                wa.at(link, WS_JOINT_ROT),
                                Vec4::new(joint_rot.x, joint_rot.y, joint_rot.z, joint_rot.w),
                            );
                        }
                    });
                }
            }
            sweep_packed(
                p,
                base,
                ndofs,
                max_constraints,
                &mut velocity,
                if last { 0 } else { bias_mode },
                writes,
                joint_constraints,
                joint_columns,
                quads,
                jcons_base,
                stride,
                contact_constraints,
                contact_jac_cols,
                ccons_base,
                contact_count,
                solver_vels,
            );
        }
    }

    // Write the coordinates back, plus the rotation of single-angle joints,
    // which `gpu_mb_integrate` recomputes from the final angle.
    workgroup_memory_barrier_with_group_sync();
    bands!(b, {
        let dof = lane_dof(p, b);
        if dof < ndofs {
            shared_vals[sb + dof as usize] = band_get(q, b);
        }
    });
    workgroup_memory_barrier_with_group_sync();
    if writes {
        bands!(b, {
            let dof = lane_dof(p, b);
            if dof < ndofs {
                dof_state.write(
                    batch_ids.mbi(batch_id, mb.first_dof as usize + dof as usize),
                    band_get(velocity, b),
                );
            }
        });
        bands!(b, {
            let link = p + b * LANES;
            if link < num_links {
                let stat = statics.at(link as usize);
                let locked = stat.locked_axes();
                let first_dof = stat.assembly_id();
                let mut coords = Coords {
                    q0: links_workspace.read(wa.at(link, WS_COORDS)),
                    q1: links_workspace.read(wa.at(link, WS_COORDS + 1)),
                };
                let mut c = 0u32;
                for axis in 0..6u32 {
                    if (locked & (1 << axis)) == 0 {
                        coords.set(axis, shared_vals[sb + (first_dof + c).min(31) as usize]);
                        c += 1;
                    }
                }
                links_workspace.write(wa.at(link, WS_COORDS), coords.q0);
                links_workspace.write(wa.at(link, WS_COORDS + 1), coords.q1);
                let ang_locked = (locked >> DIM) & 0x7;
                if 3 - ang_locked.count_ones() == 1 {
                    let dof_id = (!ang_locked & 0x7).trailing_zeros();
                    let r = rotation_from_scaled_axis(Vector::ith(
                        dof_id as usize,
                        coords.get(3 + dof_id),
                    ));
                    links_workspace.write(wa.at(link, WS_JOINT_ROT), Vec4::new(r.x, r.y, r.z, r.w));
                }
            }
        });
    }
}
