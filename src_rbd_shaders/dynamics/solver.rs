//! Solver compute shader kernels
//!
//! This module contains the actual GPU compute shader entry points for the physics solver.

use khal_std::glamx::UVec3;
use khal_std::macros::{spirv, spirv_bindgen};

use crate::{AngVector, Pose, Vector};
use khal_std::{index::MaybeIndexUnchecked, iter::StepRng, sync::atomic_add_u32};

use super::body::{LocalMassProperties, Velocity, WorldMassProperties};
use super::constraint::{TwoBodyConstraint, TwoBodyConstraintBuilder};
use super::sim_params::RbdSimParams;
use super::solver_utils::warmstart_body;

use crate::queries::IndexedManifold;
use crate::utils::{BatchIndices, Slice, SliceMut};

const WORKGROUP_SIZE: u32 = 64;

/// Initializes constraints from contact manifolds.
///
/// Split into two passes to stay within WebGPU's 8-storage-buffer per-stage
/// limit: this pass builds the per-contact constraint/builder;
/// `gpu_solver_count_constraints` does the per-body-group constraint counting.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_solver_init_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] contacts: &[IndexedManifold],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    constraints: &mut [TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)]
    constraint_builders: &mut [TwoBodyConstraintBuilder],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] collider_world_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] solver_body_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] vels: &[Velocity],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 3)] mprops: &[WorldMassProperties],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 4)] all_params: &[RbdSimParams],
    #[spirv(uniform, descriptor_set = 1, binding = 5)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;
    let params = all_params.at(batch_id as usize);

    let contacts = batch_ids.contact_batch(batch_id, contacts);
    let mut constraints = batch_ids.contact_batch_mut(batch_id, constraints);
    let mut constraint_builders = batch_ids.contact_batch_mut(batch_id, constraint_builders);
    let collider_world_poses = batch_ids.coll_batch(batch_id, collider_world_poses);
    let solver_body_poses = batch_ids.coll_batch(batch_id, solver_body_poses);
    let vels = batch_ids.coll_batch(batch_id, vels);
    let mprops = batch_ids.coll_batch(batch_id, mprops);
    // Iterating to `cap` (instead of `contacts_len[batch]`) lets us drop the
    // `contacts_len` storage binding. Empty / unused contact slots have
    // `contact.len == 0` and are skipped — narrow-phase zero-initialises the
    // buffer so the sentinel is reliable.
    let cap = batch_ids.contacts_batch_capacity;

    for i in StepRng::new(invocation_id.x..cap, num_threads) {
        let im = &contacts[i as usize];
        if im.contact.len == 0 {
            continue;
        }
        im.contact_to_constraint(
            &mprops,
            &collider_world_poses,
            &solver_body_poses,
            &vels,
            params,
            &mut constraints[i as usize],
            &mut constraint_builders[i as usize],
        );
    }
}

/// Companion pass to `gpu_solver_init_constraints`: counts, per body-group, how
/// many constraints touch each body (used to size the graph-coloring graph).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_solver_count_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] contacts: &[IndexedManifold],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] body_constraint_counts: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] body_group: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] mprops: &[WorldMassProperties],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;

    let contacts = batch_ids.contact_batch(batch_id, contacts);
    let mut body_constraint_counts = batch_ids.coll_batch_mut(batch_id, body_constraint_counts);
    let body_group = batch_ids.coll_batch(batch_id, body_group);
    let mprops = batch_ids.coll_batch(batch_id, mprops);
    let cap = batch_ids.contacts_batch_capacity;

    for i in StepRng::new(invocation_id.x..cap, num_threads) {
        let im = &contacts[i as usize];
        if im.contact.len == 0 {
            continue;
        }

        let body1 = im.bodies.x;
        let body2 = im.bodies.y;
        let group1 = body_group[body1 as usize];
        let group2 = body_group[body2 as usize];

        // Count toward the body's GROUP slot. A body is "active" for the
        // graph-coloring graph if it's a free dynamic body (inv_mass != 0) OR
        // it's part of a multibody (group != self — the multibody handles its
        // own dynamics but its bodies still need correct coloring so contacts
        // touching different links of the same multibody never share a color).
        let is_mb1 = group1 != body1;
        if mprops[body1 as usize].inv_mass != Vector::ZERO || is_mb1 {
            atomic_add_u32(&mut body_constraint_counts[group1 as usize], 1);
        }
        let is_mb2 = group2 != body2;
        if mprops[body2 as usize].inv_mass != Vector::ZERO || is_mb2 {
            atomic_add_u32(&mut body_constraint_counts[group2 as usize], 1);
        }
    }
}

/// Updates constraints for a new substep.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_solver_update_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    constraints: &mut [TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    constraint_builders: &[TwoBodyConstraintBuilder],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contacts_len: &[u32],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] solver_body_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] all_params: &[RbdSimParams],
    #[spirv(uniform, descriptor_set = 1, binding = 2)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;
    let params = all_params.at(batch_id as usize);

    let mut constraints = batch_ids.contact_batch_mut(batch_id, constraints);
    let constraint_builders = batch_ids.contact_batch(batch_id, constraint_builders);
    let solver_body_poses = batch_ids.coll_batch(batch_id, solver_body_poses);
    let len = contacts_len.read(batch_id as usize);

    for i in StepRng::new(invocation_id.x..len, num_threads) {
        constraints[i as usize].update_constraint(
            &constraint_builders[i as usize],
            &solver_body_poses,
            params,
        );
    }
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_solver_sort_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] body_constraint_counts: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mprops: &[WorldMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contacts: &[IndexedManifold],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] contacts_len: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] body_constraint_ids: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] body_group: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 6)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;
    let bci_start = batch_id as usize * 2 * batch_ids.contacts_batch_capacity as usize;

    let contacts = batch_ids.contact_batch(batch_id, contacts);
    let mut body_constraint_counts = batch_ids.coll_batch_mut(batch_id, body_constraint_counts);
    let body_group = batch_ids.coll_batch(batch_id, body_group);
    let mprops = batch_ids.coll_batch(batch_id, mprops);
    let mut body_constraint_ids = SliceMut(body_constraint_ids, bci_start);
    let len = contacts_len.read(batch_id as usize);

    for i in StepRng::new(invocation_id.x..len, num_threads) {
        let body1 = contacts[i as usize].bodies.x as usize;
        let body2 = contacts[i as usize].bodies.y as usize;
        let group1 = body_group[body1] as usize;
        let group2 = body_group[body2] as usize;

        let is_mb1 = group1 != body1;
        if mprops[body1].inv_mass != Vector::ZERO || is_mb1 {
            let id1 = atomic_add_u32(&mut body_constraint_counts[group1], 1);
            body_constraint_ids[id1 as usize] = i;
        }

        let is_mb2 = group2 != body2;
        if mprops[body2].inv_mass != Vector::ZERO || is_mb2 {
            let id2 = atomic_add_u32(&mut body_constraint_counts[group2], 1);
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
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;
    let num_bodies = batch_ids.colliders_len;

    let mut body_constraint_counts = batch_ids.coll_batch_mut(batch_id, body_constraint_counts);
    let mut solver_vels = batch_ids.coll_batch_mut(batch_id, solver_vels);
    let vels = batch_ids.coll_batch(batch_id, vels);
    let mprops = batch_ids.coll_batch(batch_id, mprops);

    for i in StepRng::new(invocation_id.x..num_bodies, num_threads) {
        let idx = i as usize;
        body_constraint_counts[idx] = 0;

        // HACK: to handle static bodies.
        if mprops[idx].inv_mass != Vector::ZERO {
            solver_vels[idx].linear = vels[idx].linear;
            solver_vels[idx].angular = vels[idx].angular;
        } else {
            solver_vels[idx].linear = Vector::ZERO;
            solver_vels[idx].angular = AngVector::default();
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] all_params: &[RbdSimParams],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let params = all_params.at(batch_id as usize);
    let i = invocation_id.x;

    let num_bodies = batch_ids.bodies_len;
    let mut solver_vels_inc = batch_ids.coll_batch_mut(batch_id, solver_vels_inc);
    let mprops = batch_ids.coll_batch(batch_id, mprops);

    if i < num_bodies {
        let idx = i as usize;
        solver_vels_inc[idx].linear = Vector::ZERO;
        solver_vels_inc[idx].angular = AngVector::default();

        // TODO: this isn't a very pretty way of detecting static bodies.
        if mprops[idx].inv_mass != Vector::ZERO {
            // TODO: this currently only handles gravity.
            // TODO: make the gravity configurable
            let gravity = Vector::Y * -9.81;
            solver_vels_inc[idx].linear = gravity * params.dt;
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
    let batch_id = invocation_id.y;
    let i = invocation_id.x;

    let num_bodies = batch_ids.bodies_len;
    let mut solver_vels = batch_ids.coll_batch_mut(batch_id, solver_vels);
    let solver_vels_inc = batch_ids.coll_batch(batch_id, solver_vels_inc);

    if i < num_bodies {
        let idx = i as usize;
        solver_vels[idx].linear += solver_vels_inc[idx].linear;
        solver_vels[idx].angular += solver_vels_inc[idx].angular;
    }
}

/// Applies warmstart impulses without graph coloring (gather-style per body).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_warmstart_without_colors(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] body_constraint_counts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] body_constraint_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] constraints: &[TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] solver_vels: &mut [Velocity],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;
    let bci_start = batch_id as usize * 2 * batch_ids.contacts_batch_capacity as usize;
    let num_bodies = batch_ids.bodies_len;

    let body_constraint_counts = batch_ids.coll_batch(batch_id, body_constraint_counts);
    let body_constraint_ids = Slice(body_constraint_ids, bci_start);
    let constraints = batch_ids.contact_batch(batch_id, constraints);
    let mut solver_vels = batch_ids.coll_batch_mut(batch_id, solver_vels);

    for body_id in StepRng::new(invocation_id.x..num_bodies, num_threads) {
        let mut solver_vel = solver_vels[body_id as usize];
        warmstart_body(
            body_id,
            &body_constraint_counts,
            &body_constraint_ids,
            &constraints,
            &mut solver_vel,
        );
        solver_vels[body_id as usize] = solver_vel;
    }
}

/// Applies warmstart impulses with graph coloring (scatter-style per constraint).
///
/// Iterates only the current color's bucket of `color_sorted_ids` (built by
/// the `gpu_color_buckets_*` passes) instead of scanning every constraint and
/// filtering by color.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_warmstart(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] constraints: &[TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] color_starts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] color_sorted_ids: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] curr_color: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;
    let stride = batch_ids.solver_color_buckets_stride;

    let constraints = batch_ids.contact_batch(batch_id, constraints);
    let color_sorted_ids = batch_ids.contact_batch(batch_id, color_sorted_ids);
    let mut solver_vels = batch_ids.coll_batch_mut(batch_id, solver_vels);
    let color = *curr_color;

    let bucket = (batch_id * stride + color) as usize;
    let start = color_starts.read(bucket);
    let end = color_starts.read(bucket + 1);

    for k in StepRng::new(start + invocation_id.x..end, num_threads) {
        let i = color_sorted_ids[k as usize];
        let constraint = &constraints[i as usize];
        let solver_id1 = constraint.solver_body_a as usize;
        let solver_id2 = constraint.solver_body_b as usize;

        let mut solver_vel1 = solver_vels[solver_id1];
        let mut solver_vel2 = solver_vels[solver_id2];

        constraint.warmstart_constraint(&mut solver_vel1, &mut solver_vel2);

        solver_vels[solver_id1] = solver_vel1;
        solver_vels[solver_id2] = solver_vel2;
    }
}

/// Main constraint solver iteration kernel (Projected Gauss-Seidel).
///
/// Iterates only the current color's bucket of `color_sorted_ids` (built by
/// the `gpu_color_buckets_*` passes) instead of scanning every constraint and
/// filtering by color.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_step_gauss_seidel(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    constraints: &mut [TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] color_starts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] color_sorted_ids: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] curr_color: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;
    let stride = batch_ids.solver_color_buckets_stride;

    let mut constraints = batch_ids.contact_batch_mut(batch_id, constraints);
    let color_sorted_ids = batch_ids.contact_batch(batch_id, color_sorted_ids);
    let mut solver_vels = batch_ids.coll_batch_mut(batch_id, solver_vels);
    let color = *curr_color;

    let bucket = (batch_id * stride + color) as usize;
    let start = color_starts.read(bucket);
    let end = color_starts.read(bucket + 1);

    for k in StepRng::new(start + invocation_id.x..end, num_threads) {
        let i = color_sorted_ids[k as usize];
        let solver_id1 = constraints[i as usize].solver_body_a as usize;
        let solver_id2 = constraints[i as usize].solver_body_b as usize;

        let mut solver_vel1 = solver_vels[solver_id1];
        let mut solver_vel2 = solver_vels[solver_id2];

        constraints[i as usize]
            .solve_constraint_gauss_seidel(&mut solver_vel1, &mut solver_vel2);

        solver_vels[solver_id1] = solver_vel1;
        solver_vels[solver_id2] = solver_vel2;
    }
}

/// Integrates velocity to update poses.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_integrate_linearized(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] poses: &mut [Pose],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] solver_vels: &[Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] all_params: &[RbdSimParams],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let params = all_params.at(batch_id as usize);
    let i = invocation_id.x;

    let num_bodies = batch_ids.bodies_len;
    let mut poses = batch_ids.coll_batch_mut(batch_id, poses);
    let solver_vels = batch_ids.coll_batch(batch_id, solver_vels);

    if i < num_bodies {
        let idx = i as usize;
        let vels = &solver_vels[idx];
        let pose = &mut poses[idx];
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
    let batch_id = invocation_id.y;
    let i = invocation_id.x;

    let num_bodies = batch_ids.bodies_len;
    let body_poses = batch_ids.coll_batch(batch_id, body_poses);
    let local_mprops = batch_ids.coll_batch(batch_id, local_mprops);
    let mut solver_body_poses = batch_ids.coll_batch_mut(batch_id, solver_body_poses);

    if i < num_bodies {
        let idx = i as usize;
        solver_body_poses[idx] = body_poses[idx].prepend_translation(local_mprops[idx].com);
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
    let batch_id = invocation_id.y;
    let i = invocation_id.x;

    let num_bodies = batch_ids.bodies_len;
    let mut vels = batch_ids.coll_batch_mut(batch_id, vels);
    let solver_vels = batch_ids.coll_batch(batch_id, solver_vels);
    let mut body_poses = batch_ids.coll_batch_mut(batch_id, body_poses);
    let solver_body_poses = batch_ids.coll_batch(batch_id, solver_body_poses);
    let local_mprops = batch_ids.coll_batch(batch_id, local_mprops);

    if i < num_bodies {
        let idx = i as usize;
        vels[idx].linear = solver_vels[idx].linear;
        vels[idx].angular = solver_vels[idx].angular;
        body_poses[idx] = solver_body_poses[idx].prepend_translation(-local_mprops[idx].com);
    }
}

/// Removes CFM and bias from constraints for the final substep iteration.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_remove_cfm_and_bias_kernel(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    constraints: &mut [TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] contacts_len: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] batch_ids: &BatchIndices,
) {
    let batch_id = invocation_id.y;
    let i = invocation_id.x;

    let mut constraints = batch_ids.contact_batch_mut(batch_id, constraints);
    let len = contacts_len.read(batch_id as usize);

    if i < len {
        constraints[i as usize].remove_cfm_and_bias();
    }
}
