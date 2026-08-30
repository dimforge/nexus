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

use glamx::Vec4;
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::iter::StepRng;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::{atomic_add_u32, atomic_load_u32, workgroup_memory_barrier_with_group_sync};

use crate::broad_phase::ContactPlan;
use crate::dynamics::ConstraintSoftness;
use crate::dynamics::body::{Velocity, WorldMassProperties};
use crate::dynamics::joint::SPATIAL_DIM;
use crate::queries::IndexedManifold;
use crate::utils::BatchIndices;
use crate::utils::linalg::{MAX_MB_DOFS, MatSlice, VSlice, lu_solve_in_place};
use crate::{ANG_DIM, AngVector, DIM, Pose, Vector, gcross, gdot};

use super::types::{
    CONTACT_CONSTRAINTS_PER_POINT, MB_CONS_SLOT_RESERVE, MB_CONTACT_KIND_NORMAL,
    MB_CONTACT_KIND_TANGENT, MbContactIndexEntry, MultibodyContactConstraint, MultibodyInfo,
    MultibodyLinkStatic,
};

const MB_SWEEP_WG: u32 = 64;
use super::utils::zero_kinematic_dofs;
use super::ws_soa::{WS_LTW, WS_WORLD_COM, WsAddr, ws_pose, ws_vec};

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
#[allow(clippy::too_many_arguments)]
#[inline]
fn fill_contact_jac_row(
    body_jacobians: &[f32],
    jac0: usize,
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
    let link_j = MatSlice::dense(
        jac0 + (link_id as usize) * SPATIAL_DIM * (ndofs as usize),
        SPATIAL_DIM as u32,
        ndofs,
    );
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

struct MbContactOwner {
    mb: u32,
    batch: u32,
    link_a: u32,
    link_b: u32,
    mb_on_first: u32,
}

const MB_OWNER_SKIP: MbContactOwner = MbContactOwner {
    mb: u32::MAX,
    batch: 0,
    link_a: u32::MAX,
    link_b: u32::MAX,
    mb_on_first: 0,
};

#[inline(always)]
fn mb_contact_owner(
    im: &IndexedManifold,
    body_to_link: &[[u32; 2]],
    multibody_info: &[MultibodyInfo],
    batch_ids: &BatchIndices,
) -> MbContactOwner {
    if im.contact.len == 0 {
        return MB_OWNER_SKIP;
    }
    let l1 = body_to_link.read(im.bodies.x as usize);
    let l2 = body_to_link.read(im.bodies.y as usize);
    if l1[0] == u32::MAX && l2[0] == u32::MAX {
        return MB_OWNER_SKIP;
    }
    if l1[0] != u32::MAX && l2[0] != u32::MAX && l1[0] != l2[0] {
        return MB_OWNER_SKIP;
    }
    let is_self = l1[0] != u32::MAX && l2[0] != u32::MAX;
    if is_self && l1[1] == l2[1] {
        return MB_OWNER_SKIP;
    }
    let mb_on_first = l1[0] != u32::MAX;
    let owner = if mb_on_first { l1[0] } else { l2[0] };
    let batch = batch_ids.collider_batch(im.bodies.x);
    let mb = multibody_info.read(batch_ids.mbi(batch, owner as usize));
    if mb.ndofs == 0 {
        return MB_OWNER_SKIP;
    }
    if is_self && mb.self_contacts_enabled == 0 {
        return MB_OWNER_SKIP;
    }
    MbContactOwner {
        mb: owner,
        batch,
        link_a: if mb_on_first { l1[1] } else { l2[1] },
        link_b: if is_self { l2[1] } else { u32::MAX },
        mb_on_first: mb_on_first as u32,
    }
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_count_contact_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] contacts: &[IndexedManifold],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] body_to_link: &[[u32; 2]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] mb_cons_counts: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] mb_index_counts: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] contact_plan: &ContactPlan,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * MB_SWEEP_WG;
    let total = contact_plan.bound;
    for t in StepRng::new(invocation_id.x..total, num_threads) {
        let im = contacts.read(t as usize);
        let owner = mb_contact_owner(&im, body_to_link, multibody_info, batch_ids);
        if owner.mb == u32::MAX {
            continue;
        }
        let slot = batch_ids.mbi(owner.batch, owner.mb as usize);
        let demand = im.contact.len * CONTACT_CONSTRAINTS_PER_POINT;
        atomic_add_u32(mb_cons_counts.at_mut(slot), demand);
        atomic_add_u32(mb_index_counts.at_mut(slot), 1);
    }
}

#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_mb_cons_offsets_scan(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    multibody_info: &mut [MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mb_cons_counts: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] mb_index_counts: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] mb_cons_demand: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
) {
    let total_infos = batch_ids.multibodies_len * batch_ids.num_batches;
    let capacity = batch_ids.mb_contact_constraints_capacity;

    let mut demand = 0u32;
    let mut reserved_total = 0u32;
    for i in 0..total_infos {
        let count = atomic_load_u32(mb_cons_counts.at_mut(i as usize));
        demand += count;
        reserved_total += count.min(MB_CONS_SLOT_RESERVE);
    }
    #[allow(clippy::implicit_saturating_sub)]
    let mut extra_budget = if capacity > reserved_total {
        capacity - reserved_total
    } else {
        0
    };

    let mut acc = 0u32;
    let mut index_acc = 0u32;
    for i in 0..total_infos {
        let mut mb = multibody_info.read(i as usize);
        let count = atomic_load_u32(mb_cons_counts.at_mut(i as usize));
        let reserve = count.min(MB_CONS_SLOT_RESERVE);
        let extra = (count - reserve).min(extra_budget);
        extra_budget -= extra;
        let start = acc.min(capacity);
        let avail = (reserve + extra).min(capacity - start);
        mb.contact_constraint_start = start;
        mb.contact_constraint_count = avail;
        acc = start + avail;

        mb.contact_index_start = index_acc;
        mb.contact_index_len = atomic_load_u32(mb_index_counts.at_mut(i as usize));
        index_acc += mb.contact_index_len;

        multibody_info.write(i as usize, mb);
        mb_cons_counts.write(i as usize, 0);
        mb_index_counts.write(i as usize, 0);
    }
    mb_cons_demand.write(0, demand);
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_scatter_contact_index(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] contacts: &[IndexedManifold],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] body_to_link: &[[u32; 2]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] mb_index_counts: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)]
    mb_contact_index: &mut [MbContactIndexEntry],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] contact_plan: &ContactPlan,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * MB_SWEEP_WG;
    let total = contact_plan.bound;
    for t in StepRng::new(invocation_id.x..total, num_threads) {
        let im = contacts.read(t as usize);
        let owner = mb_contact_owner(&im, body_to_link, multibody_info, batch_ids);
        if owner.mb == u32::MAX {
            continue;
        }
        let slot = batch_ids.mbi(owner.batch, owner.mb as usize);
        let mb = multibody_info.read(slot);
        let pos = atomic_add_u32(mb_index_counts.at_mut(slot), 1);
        mb_contact_index.write(
            (mb.contact_index_start + pos) as usize,
            MbContactIndexEntry {
                contact_slot: t,
                link_a: owner.link_a,
                link_b: owner.link_b,
                mb_on_first: owner.mb_on_first,
            },
        );
    }
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_save_prev_cons_bounds(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    multibody_info: &mut [MultibodyInfo],
    #[spirv(uniform, descriptor_set = 0, binding = 1)] batch_ids: &BatchIndices,
) {
    let num_mb = batch_ids.multibodies_len;
    if invocation_id.x >= num_mb * batch_ids.num_batches {
        return;
    }
    let batch_id = invocation_id.x / num_mb;
    let mb_idx = invocation_id.x % num_mb;
    let mut mb = multibody_info.read(batch_ids.mbi(batch_id, mb_idx as usize));
    mb.old_contact_constraint_start = mb.contact_constraint_start;
    mb.old_contact_constraint_count = mb.contact_constraint_count;
    multibody_info.write(batch_ids.mbi(batch_id, mb_idx as usize), mb);
}

/// Pack the per-link world-space contact point into the constraint.
///
/// Pass 1: scans every contact in `contacts[batch]` and, for each contact
/// point touching a link of this multibody, emits a normal-direction
/// `MultibodyContactConstraint` plus its friction slots. The multibody-side
/// `Jᵀ` rows are assembled later, by the finalize pass. Multibody-multibody
/// contacts (each side a different multibody) are not handled — such contacts
/// are skipped.
/// One 64-lane workgroup per (multibody, batch).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_init_contact_constraints(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    multibody_info: &mut [MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] links_workspace: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    mb_contact_index: &[MbContactIndexEntry],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] softness: &ConstraintSoftness,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] mprops: &[WorldMassProperties],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] contacts: &[IndexedManifold],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 3)] solver_body_poses: &[Pose],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] first_substep: &u32,
) {
    // Only active multibody slots are visited now; the `ndofs == 0` sentinel
    // below is kept for all-locked (zero-dof) multibodies. Padding slots past
    // `multibodies_len` are never read (every consumer guards on it).
    let num_mb = batch_ids.multibodies_len;
    let batch_id = workgroup_id.y;
    let mb_idx = workgroup_id.x;
    let lane = local_id.x;
    if mb_idx >= num_mb {
        return;
    }

    let inv_dt = softness.inv_dt;
    let max_corr_velocity = softness.max_corr_velocity;
    let warmstart_coeff = softness.warmstart_coefficient;
    // On the first substep of a step the anchors are (re)frozen from the fresh
    // manifold, afterwards they are updated normally.
    let freeze_anchors = *first_substep != 0;

    // Per-multibody early-out: padding multibody slots have `ndofs == 0`,
    // which we use here as the sentinel (replaces the `num_multibodies`
    // storage binding the kernel used to read).
    let mut mb = multibody_info.read(batch_ids.mbi(batch_id, mb_idx as usize));
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        // Uniform per workgroup: every lane returns together.
        if lane == 0 {
            mb.contact_constraint_count = 0;
            multibody_info.write(batch_ids.mbi(batch_id, mb_idx as usize), mb);
        }
        return;
    }
    let cons_base = mb.contact_constraint_start as usize;
    let avail = mb.contact_constraint_count;
    let wa = WsAddr::new(mb.first_link as usize, batch_ids.num_batches, batch_id);

    let idx_base = mb.contact_index_start as usize;
    let n_entries = mb.contact_index_len;
    let mut count = 0u32;

    for ci in 0..n_entries {
        let entry = mb_contact_index.read(idx_base + ci as usize);
        let im = contacts.read(entry.contact_slot as usize);
        let demand = im.contact.len * CONTACT_CONSTRAINTS_PER_POINT;
        if count + demand > avail {
            break;
        }
        let id1 = im.colliders.x;
        let b1 = im.bodies.x;
        let b2 = im.bodies.y;

        let mb_on_1 = entry.mb_on_first != 0;
        let is_self = entry.link_b != u32::MAX;
        let (mb_link_id_a, mb_link_id_b, free_body_id) = if is_self {
            (entry.link_a, entry.link_b, u32::MAX)
        } else if mb_on_1 {
            (entry.link_a, u32::MAX, b2)
        } else {
            (entry.link_a, u32::MAX, b1)
        };

        let pose1 = poses.read(id1 as usize);
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
            mprops.read(free_body_id as usize)
        };
        let free_im = if is_self { 0.0 } else { free_mp.inv_mass.x };

        let contact_is_static = !is_self && free_mp.inv_mass == Vector::ZERO;
        let erp_inv_dt = if contact_is_static {
            softness.static_erp_inv_dt
        } else {
            softness.erp_inv_dt
        };
        let cfm_factor = if contact_is_static {
            softness.static_cfm_factor
        } else {
            softness.cfm_factor
        };

        // The body jacobian of a link measures its velocity at the link's
        // center of mass, so every torque arm is taken from there.
        let com_a = ws_vec(links_workspace, wa, mb_link_id_a, WS_WORLD_COM);
        let pose_a = ws_pose(links_workspace, wa, mb_link_id_a, WS_LTW);
        // Frame the other side's anchor lives in: the second link for a
        // self-contact, the free body's center-of-mass solver pose otherwise.
        let pose_b = if is_self {
            ws_pose(links_workspace, wa, mb_link_id_b, WS_LTW)
        } else {
            solver_body_poses.read(free_body_id as usize)
        };

        for k in 0..im.contact.len {
            // One contact point produces 1 normal + (DIM-1) friction slots.
            if count + CONTACT_CONSTRAINTS_PER_POINT > avail {
                break;
            }
            let normal_slot = count;
            let prev = contact_constraints.read(cons_base + normal_slot as usize);

            // Re-resolve both anchors through the current poses, then track the
            // separation as their drift along the contact normal.
            let (local_p1, local_p2, base_dist) = if freeze_anchors {
                let pt_local = im.contact.points_a.read(k as usize).pt;
                let d = im.contact.points_a.read(k as usize).dist;
                let pt = pose1 * (pt_local + im.contact.normal_a * (d * 0.5));
                (pose_a.inverse() * pt, pose_b.inverse() * pt, d)
            } else {
                (prev.local_p1, prev.local_p2, prev.base_dist)
            };
            let p1 = pose_a * local_p1;
            let p2 = pose_b * local_p2;
            let pt_world = p1;
            let dist = base_dist + (p1 - p2).dot(mb_normal);

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
            let shift_a = pt_world - com_a;
            let torque_a_normal = gcross(shift_a, mb_normal);
            let torque_a_t0 = gcross(shift_a, mb_tangent0);
            #[cfg(feature = "dim3")]
            let torque_a_t1 = gcross(shift_a, mb_tangent1);

            let rhs_bias = (erp_inv_dt * dist).clamp(-max_corr_velocity, 0.0);
            let rhs_wo_bias = if dist > 0.0 { dist * inv_dt } else { 0.0 };

            let warmstart_normal_impulse = if freeze_anchors {
                0.0
            } else {
                prev.impulse * warmstart_coeff
            };

            // B-side fold-in for self-contacts, free body for the rest. The
            // ang_jac fields below describe the FREE body side; for self
            // contacts they collapse to zero because both sides are folded
            // into `J_mb` already.
            let (torque_b_normal, ang_jac_normal, ii_ang_jac_normal) = if is_self {
                let shift_b = p2 - ws_vec(links_workspace, wa, mb_link_id_b, WS_WORLD_COM);
                #[cfg(feature = "dim3")]
                {
                    (gcross(shift_b, lin_jac), AngVector::ZERO, AngVector::ZERO)
                }
                #[cfg(feature = "dim2")]
                {
                    (gcross(shift_b, lin_jac), 0.0f32, 0.0f32)
                }
            } else {
                let free_shift = p2 - pose_b.translation;
                let aj = gcross(free_shift, lin_jac);
                let iiaj = free_mp.inv_inertia_mul(aj);
                #[cfg(feature = "dim3")]
                {
                    (AngVector::ZERO, aj, iiaj)
                }
                #[cfg(feature = "dim2")]
                {
                    (0.0f32, aj, iiaj)
                }
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
                link_id_b: mb_link_id_b,
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
                restitution_seed: prev.restitution_seed,
                restitution: im.restitution,
                _pad4: 0,
                torque_a: torque_a_normal,
                _pad5: 0,
                torque_b: torque_b_normal,
                _pad6: 0,
                local_p1,
                base_dist,
                local_p2,
                _pad7: 0,
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
                link_id_b: mb_link_id_b,
                lin_jac,
                inv_lhs: 0.0,
                rhs: rhs_wo_bias + rhs_bias,
                rhs_wo_bias,
                impulse: warmstart_normal_impulse,
                cfm_factor,
                restitution_seed: prev.restitution_seed,
                restitution: im.restitution,
                torque_a: torque_a_normal,
                torque_b: torque_b_normal,
                local_p1,
                local_p2,
                base_dist,
                _pad1: [0.0; 2],
            };
            if lane == 0 {
                contact_constraints.write(cons_base + normal_slot as usize, normal_cons);
            }
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
                // Warmstart: preserve the accumulated tangent impulse (see the
                // normal slot above; lane 0 only).
                let tang_prev = contact_constraints.read(cons_base + tang_slot as usize);
                let warmstart_tang_impulse = if freeze_anchors {
                    0.0
                } else {
                    tang_prev.impulse * warmstart_coeff
                };

                let (torque_b_tang, ang_jac_tang, ii_ang_jac_tang) = if is_self {
                    let shift_b = p2 - ws_vec(links_workspace, wa, mb_link_id_b, WS_WORLD_COM);
                    #[cfg(feature = "dim3")]
                    {
                        (
                            gcross(shift_b, free_tangent),
                            AngVector::ZERO,
                            AngVector::ZERO,
                        )
                    }
                    #[cfg(feature = "dim2")]
                    {
                        (gcross(shift_b, free_tangent), 0.0f32, 0.0f32)
                    }
                } else {
                    let free_shift = p2 - pose_b.translation;
                    let aj = gcross(free_shift, free_tangent);
                    let iiaj = free_mp.inv_inertia_mul(aj);
                    #[cfg(feature = "dim3")]
                    {
                        (AngVector::ZERO, aj, iiaj)
                    }
                    #[cfg(feature = "dim2")]
                    {
                        (0.0f32, aj, iiaj)
                    }
                };

                // Positional bias along the tangent: pull the two anchors back
                // together so friction sticks instead of drifting. No surface
                // velocity yet (TODO: conveyor belts), so `rhs_wo_bias` is 0.
                let tang_bias = (p1 - p2).dot(mb_tangent) * inv_dt;
                #[cfg(feature = "dim3")]
                let tang_cons = MultibodyContactConstraint {
                    multibody_id: mb_idx,
                    link_id: mb_link_id_a,
                    kind: MB_CONTACT_KIND_TANGENT,
                    free_body_id,
                    free_body_im: free_im,
                    friction_coeff: im.friction,
                    normal_constraint_slot: normal_slot,
                    link_id_b: mb_link_id_b,
                    lin_jac: free_tangent,
                    _pad1: 0,
                    ang_jac: ang_jac_tang,
                    _pad2: 0,
                    ii_ang_jac: ii_ang_jac_tang,
                    _pad3: 0,
                    inv_lhs: 0.0,
                    rhs: tang_bias,
                    rhs_wo_bias: 0.0,
                    impulse: warmstart_tang_impulse,
                    cfm_factor,
                    restitution_seed: tang_prev.restitution_seed,
                    restitution: im.restitution,
                    _pad4: 0,
                    torque_a: torque_a_tang,
                    _pad5: 0,
                    torque_b: torque_b_tang,
                    _pad6: 0,
                    local_p1,
                    base_dist,
                    local_p2,
                    _pad7: 0,
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
                    link_id_b: mb_link_id_b,
                    lin_jac: free_tangent,
                    inv_lhs: 0.0,
                    rhs: tang_bias,
                    rhs_wo_bias: 0.0,
                    impulse: warmstart_tang_impulse,
                    cfm_factor,
                    restitution_seed: tang_prev.restitution_seed,
                    restitution: im.restitution,
                    torque_a: torque_a_tang,
                    torque_b: torque_b_tang,
                    local_p1,
                    local_p2,
                    base_dist,
                    _pad1: [0.0; 2],
                };
                if lane == 0 {
                    contact_constraints.write(cons_base + tang_slot as usize, tang_cons);
                }
                count += 1;
            }
        }
    }

    if lane == 0 {
        mb.contact_constraint_count = count;
        multibody_info.write(batch_ids.mbi(batch_id, mb_idx as usize), mb);
    }
}

///
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_snapshot_contact_warmstart(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    contact_constraints: &[MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    old_contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] batch_ids: &BatchIndices,
) {
    let i = invocation_id.x;
    if i < batch_ids.mb_contact_constraints_capacity {
        old_contact_constraints.write(i as usize, contact_constraints.read(i as usize));
    }
}

/// Warmstart: re-apply each active contact constraint's accumulated `impulse`
/// to the multibody generalized velocities (`dof_state`) and the free-body
/// solver velocities. Applies the FULL accumulated impulse (no `rhs` term, no
/// clamping).
///
/// One 64-lane workgroup per (multibody, batch).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_warmstart_contact_constraints(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    contact_constraints: &[MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contact_jac_cols: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] solver_vels: &mut [Velocity],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
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
    if ndofs == 0 {
        return;
    }
    let v_base = mb.first_dof as usize;
    let cons_base = mb.contact_constraint_start as usize;
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let jc_base = cons_base * 2 * dofs_stride;

    let count = mb.contact_constraint_count;
    // No accumulated impulses to re-apply: skip the dof round-trip.
    if count == 0 {
        return;
    }

    // This lane's DOF velocity, accumulated in a register across every
    // constraint.
    let mut v_lane = if lane < ndofs {
        dof_state.read(batch_ids.mbi(batch_id, v_base + lane as usize))
    } else {
        0.0
    };
    for s in 0..count {
        let cons = contact_constraints.read(cons_base + s as usize);
        let imp = cons.impulse;
        if imp != 0.0 {
            let col_offset = jc_base + (s as usize) * 2 * dofs_stride + dofs_stride;
            // Multibody side: v += impulse · column (column = M⁻¹ Jᵀ).
            if lane < ndofs {
                let col = contact_jac_cols.read(col_offset + lane as usize);
                v_lane += imp * col;
            }
            // Free body side (skipped for self-contacts).
            let is_self = cons.free_body_id == u32::MAX;
            if lane == 0 && !is_self {
                let free = solver_vels.read(cons.free_body_id as usize);
                let mut new_free = free;
                new_free.linear += cons.lin_jac * (cons.free_body_im * imp);
                new_free.angular += cons.ii_ang_jac * imp;
                solver_vels.write(cons.free_body_id as usize, new_free);
            }
        }
    }

    if lane < ndofs {
        dof_state.write(batch_ids.mbi(batch_id, v_base + lane as usize), v_lane);
    }
}

/// Pass 2: for each emitted constraint, build its multibody-side `Jᵀ` row,
/// LU back-solve `M · column = Jᵀ` and set `inv_lhs = 1 / (Jᵀ · column +
/// free_body_inv_r)`.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_finalize_contact_constraints(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mass_matrices: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] lu_pivots: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] contact_jac_cols: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)]
    links_static: &[MultibodyLinkStatic],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] body_jacobians: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 7)] batch_ids: &BatchIndices,
) {
    const LANES: u32 = 64;
    let batch_id = workgroup_id.y;
    let mb_idx = workgroup_id.x;
    let lane = local_id.x;
    let num_mb = batch_ids.multibodies_len;
    if mb_idx >= num_mb {
        return;
    }

    let mb = multibody_info.read(batch_ids.mbi(batch_id, mb_idx as usize));
    let ndofs = mb.ndofs;
    if ndofs == 0 {
        return;
    }
    let jac0 = batch_ids.mb_region(
        batch_id,
        mb.jacobian_offset,
        mb.num_links * SPATIAL_DIM as u32 * ndofs,
    );
    let piv = VSlice::dense(batch_ids.mb_region(batch_id, mb.first_dof, ndofs));
    let cons_base = mb.contact_constraint_start as usize;
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let jc_base = cons_base * 2 * dofs_stride;

    let m = MatSlice::dense(
        batch_ids.mb_region(batch_id, mb.mass_matrix_offset, ndofs * ndofs),
        ndofs,
        ndofs,
    );
    let count = mb.contact_constraint_count;
    let stat_slice = batch_ids
        .ib(batch_id, links_static)
        .offset(mb.first_link as usize);

    for s in StepRng::new(lane..count, LANES) {
        let jac_offset = jc_base + (s as usize) * 2 * dofs_stride;
        let col_offset = jac_offset + dofs_stride;
        let mut cons = contact_constraints.read(cons_base + s as usize);
        let is_self = cons.free_body_id == u32::MAX;

        // 1) Build the multibody-side Jᵀ row from the stored contact wrench,
        //    folding both touched links in for a self-contact.
        fill_contact_jac_row(
            body_jacobians,
            jac0,
            ndofs,
            cons.link_id,
            -cons.lin_jac,
            cons.torque_a,
            contact_jac_cols,
            jac_offset,
            false,
        );
        if is_self {
            fill_contact_jac_row(
                body_jacobians,
                jac0,
                ndofs,
                cons.link_id_b,
                cons.lin_jac,
                cons.torque_b,
                contact_jac_cols,
                jac_offset,
                true,
            );
        }

        // 2) Copy J^T row into the column buffer (it'll be overwritten by the
        //    LU solve with the M⁻¹·Jᵀ result).
        for i in 0..ndofs {
            let v = contact_jac_cols.read(jac_offset + i as usize);
            contact_jac_cols.write(col_offset + i as usize, v);
        }
        // 3) Solve M · column = J^T  (in place).
        lu_solve_in_place(
            mass_matrices,
            m,
            lu_pivots,
            piv,
            contact_jac_cols,
            VSlice::dense(col_offset),
        );
        // 3b) Kinematic dofs are user-driven: the impulse must not move them.
        zero_kinematic_dofs(contact_jac_cols, col_offset, &stat_slice, mb.num_links);
        // 4) inv_r_mb = J · column.
        let mut inv_r_mb = 0.0f32;
        for i in 0..ndofs {
            let j = contact_jac_cols.read(jac_offset + i as usize);
            let c = contact_jac_cols.read(col_offset + i as usize);
            inv_r_mb += j * c;
        }
        // 5) Add free body's contribution: im (since lin_jac is unit) +
        //    ang_jac · ii_ang_jac. For self-contacts the B-side is folded into
        //    `J_mb`, so there's no free-body term.
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

/// Carries the accumulated contact impulses of the previous frame over to this
/// frame's freshly built slots. A point is matched by the pair of links (or
/// link and free body) it touches plus the proximity of both frozen local
/// anchors; friction is re-projected through world space onto the new tangent
/// basis. Runs once per frame, right after the first build.
///
/// One 64-lane workgroup per (multibody, batch).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_transfer_contact_warmstart(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    old_contact_constraints: &[MultibodyContactConstraint],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
    #[spirv(uniform, descriptor_set = 0, binding = 4)] softness: &ConstraintSoftness,
) {
    const LANES: u32 = 64;
    // Anchors this far apart (in each side's own frame) are taken to be the
    // same contact point from one frame to the next.
    const MATCH_DIST: f32 = 1.0e-1;

    let batch_id = workgroup_id.y;
    let mb_idx = workgroup_id.x;
    let lane = local_id.x;
    if mb_idx >= batch_ids.multibodies_len {
        return;
    }

    let mb = multibody_info.read(batch_ids.mbi(batch_id, mb_idx as usize));
    let count = mb.contact_constraint_count;
    if mb.ndofs == 0 || count == 0 {
        return;
    }

    let cons_base = mb.contact_constraint_start as usize;
    let old_base = mb.old_contact_constraint_start as usize;
    let old_count = if mb.old_contact_constraint_start + mb.old_contact_constraint_count
        <= batch_ids.mb_contact_constraints_capacity
    {
        mb.old_contact_constraint_count
    } else {
        0
    };
    let sq_threshold = MATCH_DIST * MATCH_DIST;
    let warmstart_coeff = softness.warmstart_coefficient;

    // Each lane owns whole contact points (their normal slot), so the tangent
    // writes below stay disjoint.
    for s in StepRng::new(lane..count, LANES) {
        let mut cons = contact_constraints.read(cons_base + s as usize);
        if cons.kind != MB_CONTACT_KIND_NORMAL {
            continue;
        }

        for j in 0..old_count {
            let old = old_contact_constraints.read(old_base + j as usize);
            if old.kind != MB_CONTACT_KIND_NORMAL
                || old.link_id != cons.link_id
                || old.link_id_b != cons.link_id_b
                || old.free_body_id != cons.free_body_id
            {
                continue;
            }
            let d1 = old.local_p1 - cons.local_p1;
            let d2 = old.local_p2 - cons.local_p2;
            if d1.dot(d1) >= sq_threshold || d2.dot(d2) >= sq_threshold {
                continue;
            }

            cons.impulse = old.impulse * warmstart_coeff;
            contact_constraints.write(cons_base + s as usize, cons);

            // Friction rows follow their normal row contiguously.
            #[cfg(feature = "dim3")]
            {
                let old_t0 = old_contact_constraints.read(old_base + (j + 1) as usize);
                let old_t1 = old_contact_constraints.read(old_base + (j + 2) as usize);
                let world = (-old_t0.lin_jac * old_t0.impulse - old_t1.lin_jac * old_t1.impulse)
                    * warmstart_coeff;

                let mut new_t0 = contact_constraints.read(cons_base + (s + 1) as usize);
                let mut new_t1 = contact_constraints.read(cons_base + (s + 2) as usize);
                new_t0.impulse = world.dot(-new_t0.lin_jac);
                new_t1.impulse = world.dot(-new_t1.lin_jac);
                contact_constraints.write(cons_base + (s + 1) as usize, new_t0);
                contact_constraints.write(cons_base + (s + 2) as usize, new_t1);
            }
            #[cfg(feature = "dim2")]
            {
                let old_t0 = old_contact_constraints.read(old_base + (j + 1) as usize);
                let mut new_t0 = contact_constraints.read(cons_base + (s + 1) as usize);
                new_t0.impulse = old_t0.impulse * warmstart_coeff;
                contact_constraints.write(cons_base + (s + 1) as usize, new_t0);
            }
            break;
        }
    }
}

/// Captures each bouncy contact point's approaching normal velocity at the
/// start of the step, so [`gpu_mb_apply_contact_restitution`] can drive the
/// point back to it once every substep is done. Dispatched once per step,
/// right after the first `finalize`.
///
/// One 64-lane workgroup per (multibody, batch).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_seed_contact_restitution(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contact_jac_cols: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] dof_state: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] solver_vels: &[Velocity],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
) {
    const LANES: u32 = 64;
    let batch_id = workgroup_id.y;
    let mb_idx = workgroup_id.x;
    let lane = local_id.x;
    if mb_idx >= batch_ids.multibodies_len {
        return;
    }

    let mb = multibody_info.read(batch_ids.mbi(batch_id, mb_idx as usize));
    let ndofs = mb.ndofs;
    let count = mb.contact_constraint_count;
    if ndofs == 0 || count == 0 {
        return;
    }

    let v_base = mb.first_dof as usize;
    let cons_base = mb.contact_constraint_start as usize;
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let jc_base = cons_base * 2 * dofs_stride;

    for s in StepRng::new(lane..count, LANES) {
        let mut cons = contact_constraints.read(cons_base + s as usize);
        if cons.kind != MB_CONTACT_KIND_NORMAL {
            continue;
        }
        let jac_off = jc_base + (s as usize) * 2 * dofs_stride;
        let mut j_dot_v = 0.0f32;
        for i in 0..ndofs {
            j_dot_v += contact_jac_cols.read(jac_off + i as usize)
                * dof_state.read(batch_ids.mbi(batch_id, v_base + i as usize));
        }
        if cons.free_body_id != u32::MAX {
            let free = solver_vels.read(cons.free_body_id as usize);
            j_dot_v += cons.lin_jac.dot(free.linear) + gdot(cons.ang_jac, free.angular);
        }

        // A fresh contact bounces whenever restitution is non-zero; one that
        // survived the previous step is resting unless restitution is maximal.
        let is_new = cons.impulse == 0.0;
        let bouncy = if is_new {
            cons.restitution > 0.0
        } else {
            cons.restitution >= 1.0
        };
        cons.restitution_seed = if bouncy {
            cons.restitution * j_dot_v
        } else {
            0.0
        };
        contact_constraints.write(cons_base + s as usize, cons);
    }
}

/// End-of-step restitution pass: drives every bouncy point that carried an
/// impulse back to its seeded approach velocity, by re-running the normal
/// solve with `rhs = restitution_seed` and no CFM.
///
/// One 64-lane workgroup per (multibody, batch).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_mb_apply_contact_restitution(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] multibody_info: &[MultibodyInfo],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    contact_constraints: &mut [MultibodyContactConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contact_jac_cols: &[f32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
    #[spirv(uniform, descriptor_set = 0, binding = 4)] max_contact_constraints: &u32,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] dof_state: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] solver_vels: &mut [Velocity],
    #[spirv(workgroup)] dof_v: &mut [f32; MAX_MB_DOFS],
    #[spirv(workgroup)] scratch: &mut [f32; 64],
    #[spirv(workgroup)] delta_shared: &mut f32,
) {
    let batch_id = workgroup_id.y;
    let mb_idx = workgroup_id.x;
    let lane = local_id.x;
    let in_range = mb_idx < batch_ids.multibodies_len;
    #[cfg(not(feature = "web-compat"))]
    if !in_range {
        return;
    }
    let slot = if in_range { mb_idx } else { 0 };

    let mb = multibody_info.read(batch_ids.mbi(batch_id, slot as usize));
    let ndofs = mb.ndofs;
    let count = mb.contact_constraint_count;
    #[cfg(not(feature = "web-compat"))]
    if ndofs == 0 || count == 0 {
        return;
    }
    let active = in_range && ndofs != 0 && count != 0;

    let v_base = mb.first_dof as usize;
    let cons_base = mb.contact_constraint_start as usize;
    let dofs_stride = batch_ids.dof_batch_capacity as usize;
    let jc_base = cons_base * 2 * dofs_stride;

    if active && lane < ndofs {
        dof_v.write(
            lane as usize,
            dof_state.read(batch_ids.mbi(batch_id, v_base + lane as usize)),
        );
    }
    workgroup_memory_barrier_with_group_sync();

    #[cfg(feature = "web-compat")]
    let contact_sweep_len = *max_contact_constraints;
    #[cfg(not(feature = "web-compat"))]
    let contact_sweep_len = count;
    #[cfg(not(feature = "web-compat"))]
    let _ = max_contact_constraints;

    for s in 0..contact_sweep_len {
        let slot_active = active && s < count;
        let cons_idx = if slot_active {
            cons_base + s as usize
        } else {
            0
        };
        let cons = contact_constraints.read(cons_idx);
        // Only approaching, load-bearing points bounce.
        let solve = slot_active
            && cons.kind == MB_CONTACT_KIND_NORMAL
            && cons.restitution_seed < 0.0
            && cons.impulse > 0.0;
        #[cfg(not(feature = "web-compat"))]
        if !solve {
            continue;
        }
        let jac_offset = jc_base + (s as usize) * 2 * dofs_stride;
        let is_self = cons.free_body_id == u32::MAX;

        if solve {
            scratch.write(
                lane as usize,
                if lane < ndofs {
                    contact_jac_cols.read(jac_offset + lane as usize) * dof_v.read(lane as usize)
                } else {
                    0.0
                },
            );
        }
        workgroup_memory_barrier_with_group_sync();

        if solve && lane == 0 {
            let mut j_dot_v = 0.0f32;
            for i in 0..ndofs {
                j_dot_v += scratch.read(i as usize);
            }
            let free = if is_self {
                Velocity::default()
            } else {
                solver_vels.read(cons.free_body_id as usize)
            };
            if !is_self {
                j_dot_v += cons.lin_jac.dot(free.linear) + gdot(cons.ang_jac, free.angular);
            }

            let raw = cons.impulse - cons.inv_lhs * (j_dot_v + cons.restitution_seed);
            let new_imp = if raw < 0.0 { 0.0 } else { raw };
            let delta = new_imp - cons.impulse;
            *delta_shared = delta;

            let mut updated = cons;
            updated.impulse = new_imp;
            contact_constraints.write(cons_base + s as usize, updated);

            if delta != 0.0 && !is_self {
                let mut new_free = free;
                new_free.linear += cons.lin_jac * (cons.free_body_im * delta);
                new_free.angular += cons.ii_ang_jac * delta;
                solver_vels.write(cons.free_body_id as usize, new_free);
            }
        }
        workgroup_memory_barrier_with_group_sync();

        let delta = *delta_shared;
        if solve && delta != 0.0 && lane < ndofs {
            let col = contact_jac_cols.read(jac_offset + dofs_stride + lane as usize);
            dof_v.write(lane as usize, dof_v.read(lane as usize) + delta * col);
        }
        workgroup_memory_barrier_with_group_sync();
    }

    if active && lane < ndofs {
        dof_state.write(
            batch_ids.mbi(batch_id, v_base + lane as usize),
            dof_v.read(lane as usize),
        );
    }
}
