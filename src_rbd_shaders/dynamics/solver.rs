//! Solver compute shader kernels
//!
//! This module contains the actual GPU compute shader entry points for the physics solver.

use crate::broad_phase::ContactPlan;
use khal_std::glamx::UVec3;
use khal_std::macros::{spirv, spirv_bindgen};

use crate::{AngVector, Pose, Vector};
use khal_std::{index::MaybeIndexUnchecked, iter::StepRng, sync::atomic_add_u32};

use super::body::{LocalMassProperties, Velocity, WorldMassProperties};
use super::mass_splitting::{HubCounts, NOT_A_HUB};
use super::sim_params::RbdSimParams;

use crate::queries::IndexedManifold;
use crate::utils::{BatchIndices, Slice, SliceMut};

const WORKGROUP_SIZE: u32 = 64;

/// Counts, per body, how many constraints touch it (the adjacency lists of mass splitting,
/// graph coloring and the warmstart gather).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_solver_count_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] contacts: &[IndexedManifold],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] body_constraint_counts: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] body_is_multibody: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] mprops: &[WorldMassProperties],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] contact_plan: &ContactPlan,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;

    // Flat over all contacts of all batches: ids are global, counts cumulative.
    let total = contact_plan.bound;
    let contacts = Slice(contacts, 0);
    let mut body_constraint_counts = SliceMut(body_constraint_counts, 0);
    let body_is_multibody = Slice(body_is_multibody, 0);
    let mprops = Slice(mprops, 0);

    for i in StepRng::new(invocation_id.x..total, num_threads) {
        let im = &contacts[i as usize];
        if im.contact.len == 0 {
            continue;
        }
        // Multibody-owned manifolds have no rigid-body constraint (see
        // `gpu_init_contact_links`).
        if touches_multibody(im, &body_is_multibody) {
            continue;
        }
        let body1 = im.bodies.x;
        let body2 = im.bodies.y;

        // Count toward the body's slot (only free bodies are left here, whose
        // graph group is themselves). A body is "active" for the
        // graph-coloring graph if it's a free dynamic body (inv_mass != 0).
        if mprops[body1 as usize].inv_mass != Vector::ZERO {
            atomic_add_u32(&mut body_constraint_counts[body1 as usize], 1);
        }
        if mprops[body2 as usize].inv_mass != Vector::ZERO {
            atomic_add_u32(&mut body_constraint_counts[body2 as usize], 1);
        }
    }
}

/// `true` when either body of the manifold is a multibody link: its contacts
/// are solved by the multibody solver, never by the rigid-body one.
#[inline(always)]
fn touches_multibody(im: &IndexedManifold, body_is_multibody: &Slice<'_, u32>) -> bool {
    body_is_multibody[im.bodies.x as usize] != 0 || body_is_multibody[im.bodies.y as usize] != 0
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_solver_sort_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] body_constraint_counts: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mprops: &[WorldMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contacts: &[IndexedManifold],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] contact_plan: &ContactPlan,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] body_constraint_ids: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] body_is_multibody: &[u32],
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;

    let total = contact_plan.bound;
    let contacts = Slice(contacts, 0);
    let mut body_constraint_counts = SliceMut(body_constraint_counts, 0);
    let body_is_multibody = Slice(body_is_multibody, 0);
    let mprops = Slice(mprops, 0);
    let mut body_constraint_ids = SliceMut(body_constraint_ids, 0);

    for i in StepRng::new(invocation_id.x..total, num_threads) {
        let im = &contacts[i as usize];
        // Same filter as `gpu_solver_count_constraints`.
        if im.contact.len == 0 || touches_multibody(im, &body_is_multibody) {
            continue;
        }
        let body1 = im.bodies.x as usize;
        let body2 = im.bodies.y as usize;

        if mprops[body1].inv_mass != Vector::ZERO {
            let id1 = atomic_add_u32(&mut body_constraint_counts[body1], 1);
            body_constraint_ids[id1 as usize] = i;
        }
        if mprops[body2].inv_mass != Vector::ZERO {
            let id2 = atomic_add_u32(&mut body_constraint_counts[body2], 1);
            body_constraint_ids[id2 as usize] = i;
        }
    }
}

/// Cleans up solver state and initializes solver velocities.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_solver_cleanup(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] body_constraint_counts: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] vels: &[Velocity],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] mprops: &[WorldMassProperties],
    #[spirv(uniform, descriptor_set = 1, binding = 2)] batch_ids: &BatchIndices,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 3)] hub_first_slot: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 4)] hub_counts: &mut HubCounts,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let num_slots = batch_ids.colliders_batch_capacity * batch_ids.num_batches;

    if invocation_id.x == 0 {
        hub_counts.hubs = 0;
        hub_counts.slots = 0;
    }

    for i in StepRng::new(invocation_id.x..num_slots, num_threads) {
        let idx = i as usize;
        body_constraint_counts.write(idx, 0);
        hub_first_slot.write(idx, NOT_A_HUB);

        // HACK: to handle static bodies.
        if mprops.at(idx).inv_mass != Vector::ZERO {
            solver_vels.at_mut(idx).linear = vels.at(idx).linear;
            solver_vels.at_mut(idx).angular = vels.at(idx).angular;
        } else {
            solver_vels.at_mut(idx).linear = Vector::ZERO;
            solver_vels.at_mut(idx).angular = AngVector::default();
        }
    }
}

/// Initializes solver velocity increments (gravity, external forces).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_init_solver_vels_inc(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] solver_vels_inc: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mprops: &[WorldMassProperties],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] params: &RbdSimParams,
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
    #[spirv(uniform, descriptor_set = 0, binding = 4)] gravity: &glamx::Vec4,
) {
    let i = invocation_id.x;

    let num_bodies = batch_ids.bodies_len * batch_ids.num_batches;

    if i < num_bodies {
        let idx = i as usize;
        solver_vels_inc.at_mut(idx).linear = Vector::ZERO;
        solver_vels_inc.at_mut(idx).angular = AngVector::default();

        // TODO: this isn't a very pretty way of detecting static bodies.
        if mprops.at(idx).inv_mass != Vector::ZERO {
            // TODO: this currently only handles gravity (no user forces yet).
            #[cfg(feature = "dim3")]
            let g = Vector::new(gravity.x, gravity.y, gravity.z);
            #[cfg(feature = "dim2")]
            let g = Vector::new(gravity.x, gravity.y);
            solver_vels_inc.at_mut(idx).linear = g * params.dt;
        }
    }
}

/// Applies solver velocity increments to solver velocities.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_apply_solver_vels_inc(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] solver_vels_inc: &[Velocity],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] batch_ids: &BatchIndices,
) {
    let i = invocation_id.x;

    let num_bodies = batch_ids.bodies_len * batch_ids.num_batches;

    if i < num_bodies {
        let idx = i as usize;
        solver_vels.at_mut(idx).linear += solver_vels_inc.at(idx).linear;
        solver_vels.at_mut(idx).angular += solver_vels_inc.at(idx).angular;
    }
}

/// Integrates velocity to update poses.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_integrate_linearized(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] poses: &mut [Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] solver_vels: &mut [Velocity],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] params: &RbdSimParams,
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
) {
    let i = invocation_id.x;

    let num_bodies = batch_ids.bodies_len * batch_ids.num_batches;

    if i < num_bodies {
        let idx = i as usize;
        let mut vels = solver_vels.read(idx);

        let max_lin = params.max_linear_velocity();
        let lin_norm = vels.linear.length();
        if lin_norm > max_lin {
            vels.linear *= max_lin / lin_norm;
        }

        let max_ang = params.max_angular_velocity();
        #[cfg(feature = "dim2")]
        if vels.angular.abs() > max_ang {
            // Explicit sign select rather than `signum`: `f32::signum` compiles to
            // a comparison against a NaN constant, and naga rejects a NaN literal
            // outright, so the whole module fails to translate at pipeline
            // creation. The guard above rules out zero, so the two cases suffice.
            vels.angular = if vels.angular > 0.0 {
                max_ang
            } else {
                -max_ang
            };
        }
        #[cfg(feature = "dim3")]
        {
            let ang_norm = vels.angular.length();
            if ang_norm > max_ang {
                vels.angular *= max_ang / ang_norm;
            }
        }

        solver_vels.write(idx, vels);
        let pose = poses.at_mut(idx);
        vels.integrate_linearized(params.dt, &mut pose.translation, &mut pose.rotation);
    }
}

/// Initializes the solver-bodies' COM-centered poses from the body world poses.
///
/// `solver_body_pose = body_pose.prepend_translation(local_com)`. Mirrors
/// rapier's `SolverBodies::copy_from`.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_init_solver_bodies(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] body_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    local_mprops: &[LocalMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] solver_body_poses: &mut [Pose],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
) {
    let i = invocation_id.x;

    let num_bodies = batch_ids.bodies_len * batch_ids.num_batches;

    if i < num_bodies {
        let idx = i as usize;
        solver_body_poses.write(
            idx,
            body_poses
                .read(idx)
                .prepend_translation(local_mprops.at(idx).com),
        );
    }
}

/// Finalizes solver by copying solver velocities back to body velocities and
/// converting the COM-centered solver poses back to body-origin poses.
///
/// `body_pose = solver_body_pose.prepend_translation(-local_com)`. Mirrors
/// rapier's `velocity_solver::writeback_bodies`.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_solver_finalize(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] solver_vels: &[Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] body_poses: &mut [Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] solver_body_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)]
    local_mprops: &[LocalMassProperties],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
) {
    let i = invocation_id.x;

    let num_bodies = batch_ids.bodies_len * batch_ids.num_batches;

    if i < num_bodies {
        let idx = i as usize;
        vels.at_mut(idx).linear = solver_vels.at(idx).linear;
        vels.at_mut(idx).angular = solver_vels.at(idx).angular;
        body_poses.write(
            idx,
            solver_body_poses
                .read(idx)
                .prepend_translation(-local_mprops.at(idx).com),
        );
    }
}
