//! Multibody contact constraints.
//!
//! Mirrors rapier's `RigidBodyMultibodyContactConstraint` flow for contacts
//! where one or both sides are a multibody link. Each contact point produces
//! one normal (non-penetration) slot plus `DIM-1` Coulomb-friction tangent
//! slots; the tangent slots clamp their impulse to `±μ · normal_impulse` at
//! solve time (independent per-tangent clamp — i.e. box friction; rapier's
//! circular-cone joint clamp via `cap_magnitude` is a future refinement).
//!
//! Pipeline, called once per substep from `apply_substep`:
//!   1. `gpu_mb_init_contact_constraints`
//!   2. `gpu_mb_finalize_contact_constraints`
//!   3. `gpu_mb_solve_contact_constraints`
//!   4. `gpu_mb_remove_contact_constraint_bias`

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use crate::dynamics::ConstraintSoftness;
use crate::dynamics::body::{Velocity, WorldMassProperties};
use crate::dynamics::joint::SPATIAL_DIM;
use crate::queries::IndexedManifold;
use crate::utils::BatchIndices;
use crate::utils::linalg::{MatSlice, lu_solve_in_place};
use crate::{ANG_DIM, AngVector, DIM, Pose, Vector, gcross, gdot};

use super::types::{
    CONTACT_CONSTRAINTS_PER_POINT, MAX_MB_CONTACT_CONSTRAINTS_PER_MB, MAX_MB_CONTACTS_PER_MB,
    MB_CONTACT_KIND_NORMAL, MB_CONTACT_KIND_TANGENT, MultibodyContactConstraint, MultibodyInfo,
};

#[cfg(feature = "dim2")]
use glamx::Vec2;
#[cfg(feature = "dim3")]
use glamx::Vec3;

/// Compute an arbitrary unit vector orthogonal to `v` (assumed unit length).
/// Mirrors rapier's `OrthonormalBasis::orthonormal_vector` fallback used when
/// the relative tangent velocity is too small to drive friction direction
/// selection.
#[cfg(feature = "dim3")]
#[inline]
fn orthonormal_vector(v: Vec3) -> Vec3 {
    let sign = if v.z < 0.0 { -1.0 } else { 1.0 };
    let a = -1.0 / (sign + v.z);
    let b = v.x * v.y * a;
    Vec3::new(b, sign + v.y * v.y * a, -v.y)
}

#[cfg(feature = "dim2")]
#[inline]
fn orthonormal_vector(v: Vec2) -> Vec2 {
    Vec2::new(-v.y, v.x)
}

/// Read the `link_id`-th column block of the multibody's body jacobian and
/// project it through the per-side `(unit_force, unit_torque)` pair,
/// **adding** the resulting `Jᵀ` row to `out_jacs[col_offset ..]` (so two
/// calls accumulate — used by self-collisions, which combine the two
/// touched links into a single net `Jᵀ` row).
#[inline]
fn fill_contact_jac_row(
    body_jacobians: &[f32],
    mb_jac_base: usize,
    ndofs: u32,
    link_id: u32,
    unit_force: Vector,
    unit_torque: AngVector,
    out_jacs: &mut [f32],
    col_offset: usize,
    accumulate: bool,
) {
    // Per-link SPATIAL_DIM × ndofs jacobian (rows 0..DIM = J_v, rows
    // DIM..SPATIAL_DIM = J_w).
    let link_jac_base = mb_jac_base + (link_id as usize) * SPATIAL_DIM * (ndofs as usize);
    let link_j = MatSlice::dense(link_jac_base, SPATIAL_DIM as u32, ndofs);
    let (link_j_v, link_j_w) = link_j.rows_range_pair(0, DIM, DIM, ANG_DIM);
    for j in 0..ndofs {
        // Linear contribution: `unit_force · J_v[:, j]`.
        let dot;
        #[cfg(feature = "dim3")]
        {
            let jv0 = body_jacobians.read(link_j_v.idx(0, j));
            let jv1 = body_jacobians.read(link_j_v.idx(1, j));
            let jv2 = body_jacobians.read(link_j_v.idx(2, j));
            let jw0 = body_jacobians.read(link_j_w.idx(0, j));
            let jw1 = body_jacobians.read(link_j_w.idx(1, j));
            let jw2 = body_jacobians.read(link_j_w.idx(2, j));
            dot = unit_force.x * jv0
                + unit_force.y * jv1
                + unit_force.z * jv2
                + unit_torque.x * jw0
                + unit_torque.y * jw1
                + unit_torque.z * jw2;
        }
        #[cfg(feature = "dim2")]
        {
            let jv0 = body_jacobians.read(link_j_v.idx(0, j));
            let jv1 = body_jacobians.read(link_j_v.idx(1, j));
            let jw0 = body_jacobians.read(link_j_w.idx(0, j));
            dot = unit_force.x * jv0 + unit_force.y * jv1 + unit_torque * jw0;
        }
        let prev = if accumulate {
            out_jacs.read(col_offset + j as usize)
        } else {
            0.0
        };
        out_jacs.write(col_offset + j as usize, prev + dot);
    }
}

/// Pack the per-link world-space contact point into the constraint.
///
/// Pass 1: scans every contact in `contacts[batch]` and, for each contact
/// point touching a link of this multibody, emits a normal-direction
/// `MultibodyContactConstraint` and writes the multibody-side `Jᵀ` row into
/// `contact_constraint_jacs`. Multibody-multibody contacts (each side a
/// different multibody) are not handled — such contacts are skipped.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_init_contact_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    multibody_info: &mut [MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] body_jacobians: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] body_to_link: &[[u32; 2]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] contact_constraint_jacs: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] softness: &ConstraintSoftness,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] mprops: &[WorldMassProperties],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] contacts: &[IndexedManifold],
    #[spirv(uniform, descriptor_set = 0, binding = 6)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let mb_idx = invocation_id.x;
    // Soft-constraint coefficients (rapier TGS-soft), precomputed on the host.
    // The old path used a rigid `erp = 1/dt` with zero CFM, which overshoots
    // penetration recovery (~14× too stiff for the defaults) and jitters.
    let inv_dt = softness.inv_dt;
    let erp_inv_dt = softness.erp_inv_dt;
    let allowed_lin_err = softness.allowed_lin_err;
    let max_corr_velocity = softness.max_corr_velocity;
    let cfm_factor = softness.cfm_factor;

    let mb_start = batch_ids.mb_start(batch_id);
    let cons_start = batch_ids.mb_contact_constraints_start(batch_id);
    let col_start = batch_ids.mb_contact_constraint_columns_start(batch_id);
    let colliders_start = batch_ids.coll_start(batch_id);
    // `body_to_link` is laid out with stride = colliders_batch_capacity.
    let b2l_start = colliders_start;

    // Per-multibody early-out: padding multibody slots have `ndofs == 0`,
    // which we use here as the sentinel (replaces the `num_multibodies`
    // storage binding the kernel used to read).
    let mut mb = multibody_info.read(mb_start + mb_idx as usize);
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        mb.contact_constraint_count = 0;
        multibody_info.write(mb_start + mb_idx as usize, mb);
        return;
    }
    let mb_jac_base = batch_ids.jac_start(batch_id) + mb.jacobian_offset as usize;
    let cons_base = cons_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize);
    // Each constraint slot reserves `dof_batch_capacity` floats in the
    // column buffer (matches the allocation in `from_rapier` and avoids any
    // overlap between multibodies of differing `ndofs`).
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let col_base =
        col_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize) * dofs_stride;

    let contacts_slice = batch_ids.contact_batch(batch_id, contacts);
    // Iterate to capacity (instead of reading a `contacts_len` storage buffer):
    // empty slots have `contact.len == 0` and are skipped. Drops one binding to
    // fit the 8-storage-buffer WebGPU limit; matches `gpu_solver_init_constraints`.
    let n_contacts = batch_ids.contacts_batch_capacity;
    let mut count = 0u32;

    for ci in 0..n_contacts {
        if count >= MAX_MB_CONTACTS_PER_MB {
            break;
        }
        let im = contacts_slice[ci as usize];
        if im.contact.len == 0 {
            continue;
        }
        let id1 = im.colliders.x;
        let id2 = im.colliders.y;
        let b1 = im.bodies.x;
        let b2 = im.bodies.y;

        let l1 = body_to_link.read(b2l_start + b1 as usize);
        let l2 = body_to_link.read(b2l_start + b2 as usize);
        let mb_on_1 = l1[0] == mb_idx;
        let mb_on_2 = l2[0] == mb_idx;

        if !mb_on_1 && !mb_on_2 {
            continue;
        }
        // Inter-multibody contacts (each side is a DIFFERENT multibody) are
        // not yet handled — skip them. Self-collisions (both sides on this
        // SAME multibody) are handled below.
        if l1[0] != u32::MAX && l2[0] != u32::MAX && l1[0] != l2[0] {
            continue;
        }

        let is_self = mb_on_1 && mb_on_2;
        // Honor rapier's `Multibody::self_contacts_enabled` (MJCF
        // `DISABLE_SELF_CONTACTS`): skip contacts between two links of the same
        // multibody when self-contacts are disabled.
        if is_self && mb.self_contacts_enabled == 0 {
            continue;
        }
        let (mb_link_id_a, mb_link_id_b, free_body_id) = if is_self {
            (l1[1], l2[1], u32::MAX)
        } else if mb_on_1 {
            (l1[1], u32::MAX, b2)
        } else {
            (l2[1], u32::MAX, b1)
        };

        // Skip degenerate self-contacts on the same link.
        if is_self && mb_link_id_a == mb_link_id_b {
            continue;
        }

        let pose1 = poses.read(colliders_start + id1 as usize);
        let pose2 = poses.read(colliders_start + id2 as usize);
        let world_normal = pose1.rotation * im.contact.normal_a;
        let lin_jac = if is_self || mb_on_1 {
            world_normal
        } else {
            -world_normal
        };
        let mb_normal = -lin_jac;

        let free_mp = if is_self {
            WorldMassProperties::default()
        } else {
            mprops.read(colliders_start + free_body_id as usize)
        };
        let free_im = if is_self { 0.0 } else { free_mp.inv_mass.x };

        // Multibody-link origins come from the collider poses buffer instead
        // of `links_workspace` (which we no longer bind in this kernel — see
        // binding-count cap discussion). This uses the contacting collider's
        // world translation as a proxy for its link's world origin. It is exact
        // when the collider sits at the link origin (the historical one-collider
        // case); for a link carrying a collider offset from its origin the
        // angular lever arm `pt_world - link_origin_a` is off by that offset.
        // Binding `body_poses` here to use the true link origin would exceed the
        // 10-storage-buffer cap — left as a follow-up. `mb_link_id_a` always
        // corresponds to side `id1`/`b1` when `mb_on_1 || is_self`, and to
        // `id2`/`b2` otherwise. `mb_link_id_b` is only used in the self-contact
        // case where it corresponds to `id2`.
        let link_origin_a = if is_self || mb_on_1 {
            pose1.translation
        } else {
            pose2.translation
        };
        let link_origin_b_default = link_origin_a;

        for k in 0..im.contact.len {
            // One contact point produces 1 normal + (DIM-1) friction slots.
            if count + CONTACT_CONSTRAINTS_PER_POINT > MAX_MB_CONTACT_CONSTRAINTS_PER_MB {
                break;
            }
            let pt_local = im.contact.points_a.read(k as usize).pt;
            let dist = im.contact.points_a.read(k as usize).dist;
            let pt_world = pose1 * (pt_local + im.contact.normal_a * (dist * 0.5));

            // Tangent basis — matches rapier's fallback path
            // (`OrthonormalBasis::orthonormal_vector(force_dir1)` then
            // `dir1.cross(tangent1)`). Velocity-driven tangent selection
            // (rapier's preferred path when `|tangent_relvel|` is large) is
            // skipped for now — the fallback is correct, just less optimal.
            let mb_tangent0 = orthonormal_vector(mb_normal);
            #[cfg(feature = "dim3")]
            let mb_tangent1 = mb_normal.cross(mb_tangent0);

            // A-side (link `mb_link_id_a`, rapier's body 1): impulse along
            // `force_dir1 = -world_normal_a = mb_normal`.
            let shift_a = pt_world - link_origin_a;
            let torque_a_normal = gcross(shift_a, mb_normal);
            let torque_a_t0 = gcross(shift_a, mb_tangent0);
            #[cfg(feature = "dim3")]
            let torque_a_t1 = gcross(shift_a, mb_tangent1);

            let rhs_bias = (erp_inv_dt * (dist + allowed_lin_err)).clamp(-max_corr_velocity, 0.0);
            let rhs_wo_bias = if dist > 0.0 { dist * inv_dt } else { 0.0 };

            let normal_slot = count;
            let normal_col_offset = col_base + (normal_slot as usize) * dofs_stride;
            // Warmstart: preserve the accumulated impulse from the previous
            // substep (same contact slot — within a frame the manifolds are
            // fixed). `gpu_mb_reset_contact_warmstart` zeroes these once per
            // frame so the first substep starts cold.
            let warmstart_normal_impulse = contact_constraints
                .read(cons_base + normal_slot as usize)
                .impulse;

            fill_contact_jac_row(
                body_jacobians,
                mb_jac_base,
                ndofs,
                mb_link_id_a,
                mb_normal,
                torque_a_normal,
                contact_constraint_jacs,
                normal_col_offset,
                false,
            );

            // B-side fold-in for self-contacts, free body for the rest. The
            // ang_jac fields below describe the FREE body side; for self
            // contacts they collapse to zero because both sides are folded
            // into `J_mb` already.
            let (ang_jac_normal, ii_ang_jac_normal) = if is_self {
                // Self-contact: B-side link is the collider at `id2`, so its
                // world pose is `pose2` (already loaded). Avoids a
                // `links_workspace` binding.
                let link_origin_b = pose2.translation;
                let _ = link_origin_b_default;
                let shift_b = pt_world - link_origin_b;
                let torque_b_normal = gcross(shift_b, lin_jac);
                fill_contact_jac_row(
                    body_jacobians,
                    mb_jac_base,
                    ndofs,
                    mb_link_id_b,
                    lin_jac,
                    torque_b_normal,
                    contact_constraint_jacs,
                    normal_col_offset,
                    true,
                );
                #[cfg(feature = "dim3")]
                {
                    (AngVector::ZERO, AngVector::ZERO)
                }
                #[cfg(feature = "dim2")]
                {
                    (0.0f32, 0.0f32)
                }
            } else {
                let _ = link_origin_b_default;
                let free_shift = pt_world - free_mp.com;
                let aj = gcross(free_shift, lin_jac);
                let iiaj = free_mp.inv_inertia_mul(aj);
                (aj, iiaj)
            };

            // Normal constraint slot.
            #[cfg(feature = "dim3")]
            let normal_cons = MultibodyContactConstraint {
                multibody_id: mb_idx,
                link_id: mb_link_id_a,
                kind: MB_CONTACT_KIND_NORMAL,
                free_body_id,
                free_body_im: free_im,
                friction_coeff: im.friction,
                normal_constraint_slot: normal_slot,
                _pad0: 0,
                lin_jac,
                _pad1: 0,
                ang_jac: ang_jac_normal,
                _pad2: 0,
                ii_ang_jac: ii_ang_jac_normal,
                _pad3: 0,
                inv_lhs: 0.0,
                rhs: rhs_wo_bias + rhs_bias,
                rhs_wo_bias,
                impulse: warmstart_normal_impulse,
                cfm_factor,
                _unused_cfm: 0.0,
                _pad4: [0; 2],
            };
            #[cfg(feature = "dim2")]
            let normal_cons = MultibodyContactConstraint {
                multibody_id: mb_idx,
                link_id: mb_link_id_a,
                kind: MB_CONTACT_KIND_NORMAL,
                free_body_id,
                free_body_im: free_im,
                ang_jac: ang_jac_normal,
                ii_ang_jac: ii_ang_jac_normal,
                friction_coeff: im.friction,
                normal_constraint_slot: normal_slot,
                _pad0: [0; 1],
                lin_jac,
                inv_lhs: 0.0,
                rhs: rhs_wo_bias + rhs_bias,
                rhs_wo_bias,
                impulse: warmstart_normal_impulse,
                cfm_factor,
                _unused_cfm: 0.0,
            };
            contact_constraints.write(cons_base + normal_slot as usize, normal_cons);
            count += 1;

            // Friction tangent constraints — same contact point, tangent
            // direction. The MB-side `Jᵀ` row is written into the next slab
            // column; the free-side jacobians are stored on the constraint.
            // Limit `±μ · normal_impulse` is computed at solve time by
            // looking up `cons[normal_constraint_slot].impulse`.
            for tang_idx in 0..(CONTACT_CONSTRAINTS_PER_POINT - 1) {
                let mb_tangent = if tang_idx == 0 {
                    mb_tangent0
                } else {
                    #[cfg(feature = "dim3")]
                    {
                        mb_tangent1
                    }
                    #[cfg(feature = "dim2")]
                    {
                        // Unreachable in 2D (loop count = 0).
                        mb_tangent0
                    }
                };
                let torque_a_tang = if tang_idx == 0 {
                    torque_a_t0
                } else {
                    #[cfg(feature = "dim3")]
                    {
                        torque_a_t1
                    }
                    #[cfg(feature = "dim2")]
                    {
                        torque_a_t0
                    }
                };
                let free_tangent = -mb_tangent;
                let tang_slot = count;
                let tang_col_offset = col_base + (tang_slot as usize) * dofs_stride;
                // Warmstart: preserve the accumulated tangent impulse (see the
                // normal slot above).
                let warmstart_tang_impulse = contact_constraints
                    .read(cons_base + tang_slot as usize)
                    .impulse;

                fill_contact_jac_row(
                    body_jacobians,
                    mb_jac_base,
                    ndofs,
                    mb_link_id_a,
                    mb_tangent,
                    torque_a_tang,
                    contact_constraint_jacs,
                    tang_col_offset,
                    false,
                );

                let (ang_jac_tang, ii_ang_jac_tang) = if is_self {
                    let link_origin_b = pose2.translation;
                    let shift_b = pt_world - link_origin_b;
                    let torque_b_tang = gcross(shift_b, free_tangent);
                    fill_contact_jac_row(
                        body_jacobians,
                        mb_jac_base,
                        ndofs,
                        mb_link_id_b,
                        free_tangent,
                        torque_b_tang,
                        contact_constraint_jacs,
                        tang_col_offset,
                        true,
                    );
                    #[cfg(feature = "dim3")]
                    {
                        (AngVector::ZERO, AngVector::ZERO)
                    }
                    #[cfg(feature = "dim2")]
                    {
                        (0.0f32, 0.0f32)
                    }
                } else {
                    let free_shift = pt_world - free_mp.com;
                    let aj = gcross(free_shift, free_tangent);
                    let iiaj = free_mp.inv_inertia_mul(aj);
                    (aj, iiaj)
                };

                // No surface velocity (TODO: conveyor belts) → rhs = 0.
                #[cfg(feature = "dim3")]
                let tang_cons = MultibodyContactConstraint {
                    multibody_id: mb_idx,
                    link_id: mb_link_id_a,
                    kind: MB_CONTACT_KIND_TANGENT,
                    free_body_id,
                    free_body_im: free_im,
                    friction_coeff: im.friction,
                    normal_constraint_slot: normal_slot,
                    _pad0: 0,
                    lin_jac: free_tangent,
                    _pad1: 0,
                    ang_jac: ang_jac_tang,
                    _pad2: 0,
                    ii_ang_jac: ii_ang_jac_tang,
                    _pad3: 0,
                    inv_lhs: 0.0,
                    rhs: 0.0,
                    rhs_wo_bias: 0.0,
                    impulse: warmstart_tang_impulse,
                    cfm_factor,
                    _unused_cfm: 0.0,
                    _pad4: [0; 2],
                };
                #[cfg(feature = "dim2")]
                let tang_cons = MultibodyContactConstraint {
                    multibody_id: mb_idx,
                    link_id: mb_link_id_a,
                    kind: MB_CONTACT_KIND_TANGENT,
                    free_body_id,
                    free_body_im: free_im,
                    ang_jac: ang_jac_tang,
                    ii_ang_jac: ii_ang_jac_tang,
                    friction_coeff: im.friction,
                    normal_constraint_slot: normal_slot,
                    _pad0: [0; 1],
                    lin_jac: free_tangent,
                    inv_lhs: 0.0,
                    rhs: 0.0,
                    rhs_wo_bias: 0.0,
                    impulse: warmstart_tang_impulse,
                    cfm_factor,
                    _unused_cfm: 0.0,
                };
                contact_constraints.write(cons_base + tang_slot as usize, tang_cons);
                count += 1;
            }
        }
    }

    // The solve / finalize / remove-bias kernels iterate `0..count` so we
    // don't need to mark surplus slots inactive — they're never read.
    mb.contact_constraint_count = count;
    multibody_info.write(mb_start + mb_idx as usize, mb);
}

/// Zero the accumulated impulse of every contact-constraint slot for each
/// multibody. Called ONCE per visible frame (from `init_step`, before the
/// substep loop) so the first substep's warmstart starts cold; within a frame
/// `gpu_mb_init_contact_constraints` preserves the impulse across substeps and
/// `gpu_mb_warmstart_contact_constraints` re-applies it each substep.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_reset_contact_warmstart(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let mb_idx = invocation_id.x;
    let num_mb = batch_ids.multibodies_len;
    if mb_idx >= num_mb {
        return;
    }
    let mb_start = batch_ids.mb_start(batch_id);
    let cons_start = batch_ids.mb_contact_constraints_start(batch_id);
    let mb = multibody_info.read(mb_start + mb_idx as usize);
    if mb.ndofs == 0 {
        return;
    }
    let cons_base = cons_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize);
    // Zero to capacity (the per-frame contact count isn't known here, and last
    // frame's count may be smaller than this frame's).
    for s in 0..MAX_MB_CONTACT_CONSTRAINTS_PER_MB {
        let mut cons = contact_constraints.read(cons_base + s as usize);
        cons.impulse = 0.0;
        contact_constraints.write(cons_base + s as usize, cons);
    }
}

/// Warmstart: re-apply each active contact constraint's accumulated `impulse`
/// to the multibody generalized velocities (`dof_state`) and the free-body
/// solver velocities. Applies the FULL accumulated impulse (no `rhs` term, no
/// clamping).
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_warmstart_contact_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    contact_constraints: &[MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contact_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] solver_vels: &mut [Velocity],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let mb_idx = invocation_id.x;
    let num_mb = batch_ids.multibodies_len;
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_ids.mb_start(batch_id);
    let cons_start = batch_ids.mb_contact_constraints_start(batch_id);
    let col_start = batch_ids.mb_contact_constraint_columns_start(batch_id);
    let colliders_start = batch_ids.coll_start(batch_id);

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        return;
    }
    let v_base = batch_ids.dof_start(batch_id) + mb.first_dof as usize;
    let cons_base = cons_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize);
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let col_base =
        col_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize) * dofs_stride;

    let count = mb.contact_constraint_count;
    for s in 0..count {
        let cons = contact_constraints.read(cons_base + s as usize);
        let imp = cons.impulse;
        if imp != 0.0 {
            let col_offset = col_base + (s as usize) * dofs_stride;
            // Multibody side: v += impulse · column (column = M⁻¹ Jᵀ).
            for i in 0..ndofs {
                let v_idx = v_base + i as usize;
                let cur = dof_state.read(v_idx);
                let col = contact_constraint_columns.read(col_offset + i as usize);
                dof_state.write(v_idx, cur + imp * col);
            }
            // Free body side (skipped for self-contacts).
            let is_self = cons.free_body_id == u32::MAX;
            if !is_self {
                let free = solver_vels.read(colliders_start + cons.free_body_id as usize);
                let mut new_free = free;
                new_free.linear += cons.lin_jac * (cons.free_body_im * imp);
                new_free.angular += cons.ii_ang_jac * imp;
                solver_vels.write(colliders_start + cons.free_body_id as usize, new_free);
            }
        }
    }
}

/// Pass 2: for each emitted constraint, LU back-solve `M · column = Jᵀ`
/// (the row produced by the init kernel) and set `inv_lhs = 1 / (Jᵀ ·
/// column + free_body_inv_r)`.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_finalize_contact_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mass_matrices: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] lu_pivots: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] contact_constraint_jacs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)]
    contact_constraint_columns: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 6)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let mb_idx = invocation_id.x;
    let num_mb = batch_ids.multibodies_len;
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_ids.mb_start(batch_id);
    let cons_start = batch_ids.mb_contact_constraints_start(batch_id);
    let col_start = batch_ids.mb_contact_constraint_columns_start(batch_id);

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        return;
    }
    let mb_mm_base = batch_ids.mm_start(batch_id) + mb.mass_matrix_offset as usize;
    let piv_offset = batch_ids.dof_start(batch_id) + mb.first_dof as usize;
    let cons_base = cons_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize);
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let col_base =
        col_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize) * dofs_stride;

    let m = MatSlice::dense(mb_mm_base, ndofs, ndofs);
    let count = mb.contact_constraint_count;

    for s in 0..count {
        let col_offset = col_base + (s as usize) * dofs_stride;
        // The LU back-solve read-modify-writes the column O(n²) times (permute
        // + forward + backward substitution). Doing that in the GLOBAL
        // `contact_constraint_columns` buffer costs an L2 round-trip on every
        // step of the dependence chain — GPU L1 is write-evict for global
        // stores, so each write is not seen by the next dependent read until it
        // reaches L2. Hold the column in a per-thread LOCAL array instead
        // (registers / L1-backed local memory), and write it out once at the
        // end. Same arithmetic and iteration order → bit-identical result.
        let mut col = [0.0f32; 64];
        // 1) J^T row into the local column.
        for i in 0..ndofs {
            col[i as usize] = contact_constraint_jacs.read(col_offset + i as usize);
        }
        // 2) Solve M · column = J^T in place in the local vector.
        lu_solve_in_place(mass_matrices, m, lu_pivots, piv_offset, &mut col, 0);
        // 3) Write the finished column out + inv_r_mb = J · column in one pass.
        let mut inv_r_mb = 0.0f32;
        for i in 0..ndofs {
            let c = col[i as usize];
            contact_constraint_columns.write(col_offset + i as usize, c);
            inv_r_mb += contact_constraint_jacs.read(col_offset + i as usize) * c;
        }
        // 4) Add free body's contribution: im (since lin_jac is unit) +
        //    ang_jac · ii_ang_jac. For self-contacts the B-side is folded into
        //    `J_mb`, so there's no free-body term.
        let mut cons = contact_constraints.read(cons_base + s as usize);
        let is_self = cons.free_body_id == u32::MAX;
        let inv_r_free = if is_self {
            0.0
        } else {
            cons.free_body_im + gdot(cons.ang_jac, cons.ii_ang_jac)
        };
        let total = inv_r_mb + inv_r_free;
        cons.inv_lhs = if total > 0.0 { 1.0 / total } else { 0.0 };
        contact_constraints.write(cons_base + s as usize, cons);
    }
}

/// One PGS sweep over the multibody's active contact constraints. Updates
/// the multibody's `dof_velocities` and the free body's `solver_vels`.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_solve_contact_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contact_constraint_jacs: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] contact_constraint_columns: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] solver_vels: &mut [Velocity],
    #[spirv(uniform, descriptor_set = 0, binding = 6)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let mb_idx = invocation_id.x;
    let num_mb = batch_ids.multibodies_len;
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_ids.mb_start(batch_id);
    let cons_start = batch_ids.mb_contact_constraints_start(batch_id);
    let col_start = batch_ids.mb_contact_constraint_columns_start(batch_id);
    let colliders_start = batch_ids.coll_start(batch_id);

    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        return;
    }
    let v_base = batch_ids.dof_start(batch_id) + mb.first_dof as usize;
    let cons_base = cons_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize);
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let col_base =
        col_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize) * dofs_stride;

    let count = mb.contact_constraint_count;
    for s in 0..count {
        let mut cons = contact_constraints.read(cons_base + s as usize);
        let col_offset = col_base + (s as usize) * dofs_stride;

        // J · u = J_mb · v_mb_dofs + J_free · v_free.
        let is_self = cons.free_body_id == u32::MAX;
        let mut j_dot_v = 0.0f32;
        for i in 0..ndofs {
            let j = contact_constraint_jacs.read(col_offset + i as usize);
            let v = dof_state.read(v_base + i as usize);
            j_dot_v += j * v;
        }
        let free = if is_self {
            Velocity::default()
        } else {
            solver_vels.read(colliders_start + cons.free_body_id as usize)
        };
        if !is_self {
            j_dot_v += cons.lin_jac.dot(free.linear) + gdot(cons.ang_jac, free.angular);
        }

        let rhs_total = j_dot_v + cons.rhs;
        // CFM-factor form (rapier's `*ContactConstraintNormalPart::generic_solve`):
        // `new = cfm_factor · (impulse − r · Δvel)`. `cfm_factor < 1` provides the
        // soft-constraint compliance that keeps resting contacts from jittering.
        let raw_imp = cons.cfm_factor * (cons.impulse - cons.inv_lhs * rhs_total);

        // Normal: clamp to ≥ 0 (no separation impulse). Friction tangent:
        // clamp to `±μ · normal_impulse` — looks up the paired normal slot
        // for the current accumulated impulse. Mirrors rapier's
        // `ContactConstraintNormalPart::generic_solve` /
        // `ContactConstraintTangentPart::generic_solve` (independent
        // per-tangent clamp, i.e. box friction; rapier's circular-cone
        // joint clamp is a future refinement).
        let new_imp = if cons.kind == MB_CONTACT_KIND_TANGENT {
            let normal = contact_constraints.read(cons_base + cons.normal_constraint_slot as usize);
            let limit = cons.friction_coeff * normal.impulse;
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
        let delta = new_imp - cons.impulse;
        cons.impulse = new_imp;
        contact_constraints.write(cons_base + s as usize, cons);

        if delta != 0.0 {
            for i in 0..ndofs {
                let v_idx = v_base + i as usize;
                let cur = dof_state.read(v_idx);
                let col = contact_constraint_columns.read(col_offset + i as usize);
                dof_state.write(v_idx, cur + delta * col);
            }
            if !is_self {
                let mut new_free = free;
                new_free.linear += cons.lin_jac * (cons.free_body_im * delta);
                new_free.angular += cons.ii_ang_jac * delta;
                solver_vels.write(colliders_start + cons.free_body_id as usize, new_free);
            }
        }
    }
}

/// Strip the positional bias from each active contact constraint's `rhs`,
/// matching `gpu_mb_remove_joint_constraint_bias`.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_remove_contact_constraint_bias(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] multibody_info: &[MultibodyInfo],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let mb_idx = invocation_id.x;
    let num_mb = batch_ids.multibodies_len;
    if mb_idx >= num_mb {
        return;
    }

    let mb_start = batch_ids.mb_start(batch_id);
    let cons_start = batch_ids.mb_contact_constraints_start(batch_id);
    let cons_base = cons_start + (mb_idx as usize) * (MAX_MB_CONTACT_CONSTRAINTS_PER_MB as usize);
    let mb = multibody_info.read(mb_start + mb_idx as usize);
    let count = mb.contact_constraint_count;

    for s in 0..count {
        let mut cons = contact_constraints.read(cons_base + s as usize);
        if cons.kind == 0 {
            continue;
        }
        cons.rhs = cons.rhs_wo_bias;
        contact_constraints.write(cons_base + s as usize, cons);
    }
}
