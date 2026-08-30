//! Solver compute shader kernels
//!
//! This module contains the actual GPU compute shader entry points for the physics solver.

use crate::broad_phase::ContactPlan;
use khal_std::glamx::UVec3;
use khal_std::macros::{spirv, spirv_bindgen};

use crate::{AngVector, Pose, Vector};
use khal_std::{
    index::MaybeIndexUnchecked,
    iter::StepRng,
    sync::{atomic_add_u32, control_barrier},
};

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
    #[spirv(uniform, descriptor_set = 0, binding = 3)] contact_plan: &ContactPlan,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] collider_world_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 1)] solver_body_poses: &[Pose],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 2)] vels: &[Velocity],
    #[spirv(storage_buffer, descriptor_set = 1, binding = 3)] mprops: &[WorldMassProperties],
    #[spirv(uniform, descriptor_set = 1, binding = 4)] params: &RbdSimParams,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;

    let total = contact_plan.bound;
    let collider_world_poses = Slice(collider_world_poses, 0);
    let solver_body_poses = Slice(solver_body_poses, 0);
    let vels = Slice(vels, 0);
    let mprops = Slice(mprops, 0);

    for i in StepRng::new(invocation_id.x..total, num_threads) {
        let i = i as usize;
        let im = contacts.at(i);
        if im.contact.len == 0 {
            // Gap or inert slot: clear the (stale) constraint so every flat
            // consumer skips it.
            constraints.at_mut(i).len = 0;
            continue;
        }
        im.contact_to_constraint(
            &mprops,
            &collider_world_poses,
            &solver_body_poses,
            &vels,
            params,
            constraints.at_mut(i),
            constraint_builders.at_mut(i),
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
    #[spirv(uniform, descriptor_set = 0, binding = 4)] contact_plan: &ContactPlan,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;

    // Flat over all contacts of all batches: ids are global, counts cumulative.
    let total = contact_plan.bound;
    let contacts = Slice(contacts, 0);
    let mut body_constraint_counts = SliceMut(body_constraint_counts, 0);
    let body_group = Slice(body_group, 0);
    let mprops = Slice(mprops, 0);

    for i in StepRng::new(invocation_id.x..total, num_threads) {
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
    #[spirv(uniform, descriptor_set = 0, binding = 2)] contact_plan: &ContactPlan,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] solver_body_poses: &[Pose],
    #[spirv(uniform, descriptor_set = 1, binding = 1)] params: &RbdSimParams,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;

    let total = contact_plan.bound;
    let mut constraints = SliceMut(constraints, 0);
    let constraint_builders = Slice(constraint_builders, 0);
    let solver_body_poses = Slice(solver_body_poses, 0);

    for i in StepRng::new(invocation_id.x..total, num_threads) {
        if constraints[i as usize].len == 0 {
            // Gap / inert slot.
            continue;
        }
        constraints[i as usize].update_constraint(
            &constraint_builders[i as usize],
            &solver_body_poses,
            params,
        );
    }
}

/// Relax-pass refresh: recomputes the unbiased normal rhs of every contact
/// constraint from the current (post-integration) solver poses.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_solver_refresh_rhs_wo_bias(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    constraints: &mut [TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)]
    constraint_builders: &[TwoBodyConstraintBuilder],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] contact_plan: &ContactPlan,
    #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] solver_body_poses: &[Pose],
    #[spirv(uniform, descriptor_set = 1, binding = 1)] params: &RbdSimParams,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;

    let total = contact_plan.bound;
    let mut constraints = SliceMut(constraints, 0);
    let constraint_builders = Slice(constraint_builders, 0);
    let solver_body_poses = Slice(solver_body_poses, 0);

    for i in StepRng::new(invocation_id.x..total, num_threads) {
        if constraints[i as usize].len == 0 {
            // Gap / inert slot.
            continue;
        }
        constraints[i as usize].refresh_rhs_wo_bias(
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
    #[spirv(uniform, descriptor_set = 0, binding = 3)] contact_plan: &ContactPlan,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] body_constraint_ids: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] body_group: &[u32],
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;

    let total = contact_plan.bound;
    let contacts = Slice(contacts, 0);
    let mut body_constraint_counts = SliceMut(body_constraint_counts, 0);
    let body_group = Slice(body_group, 0);
    let mprops = Slice(mprops, 0);
    let mut body_constraint_ids = SliceMut(body_constraint_ids, 0);

    for i in StepRng::new(invocation_id.x..total, num_threads) {
        if contacts[i as usize].contact.len == 0 {
            continue;
        }
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
    let num_slots = batch_ids.colliders_batch_capacity * batch_ids.num_batches;

    for i in StepRng::new(invocation_id.x..num_slots, num_threads) {
        let idx = i as usize;
        body_constraint_counts.write(idx, 0);

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
    let num_bodies = batch_ids.bodies_len * batch_ids.num_batches;

    let body_constraint_counts = Slice(body_constraint_counts, 0);
    let body_constraint_ids = Slice(body_constraint_ids, 0);
    let constraints = Slice(constraints, 0);
    let mut solver_vels = SliceMut(solver_vels, 0);

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
    let nb = batch_ids.num_batches;

    let constraints = Slice(constraints, 0);
    let color_sorted_ids = Slice(color_sorted_ids, 0);
    let mut solver_vels = SliceMut(solver_vels, 0);
    let color = *curr_color;

    // Buckets are color-major (`color * num_batches + batch`) and the buffer
    // holds post-scatter exclusive ENDS, so color `c` (over every batch) spans
    // `[ends[c*nb - 1], ends[(c+1)*nb - 1])`. `c >= 1` keeps the index valid.
    let start = color_starts.read((color * nb - 1) as usize);
    let end = color_starts.read(((color + 1) * nb - 1) as usize);

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
    #[spirv(uniform, descriptor_set = 0, binding = 6)] use_bias: &u32,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let nb = batch_ids.num_batches;

    let mut constraints = SliceMut(constraints, 0);
    let color_sorted_ids = Slice(color_sorted_ids, 0);
    let mut solver_vels = SliceMut(solver_vels, 0);
    let color = *curr_color;
    let use_bias = *use_bias != 0;

    // Color-major bucket ends; see `gpu_warmstart`.
    let start = color_starts.read((color * nb - 1) as usize);
    let end = color_starts.read(((color + 1) * nb - 1) as usize);

    for k in StepRng::new(start + invocation_id.x..end, num_threads) {
        let i = color_sorted_ids[k as usize];
        let solver_id1 = constraints[i as usize].solver_body_a as usize;
        let solver_id2 = constraints[i as usize].solver_body_b as usize;

        let mut solver_vel1 = solver_vels[solver_id1];
        let mut solver_vel2 = solver_vels[solver_id2];

        constraints[i as usize].solve_constraint_gauss_seidel(
            &mut solver_vel1,
            &mut solver_vel2,
            use_bias,
        );

        solver_vels[solver_id1] = solver_vel1;
        solver_vels[solver_id2] = solver_vel2;
    }
}

/// Fused colored warmstart: only one 64-lane workgroup per batch walks every color
/// bucket.
///
/// Used for small scenes where the contact count is small wrt. the environment count.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_warmstart_fused(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] constraints: &[TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] color_starts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] color_sorted_ids: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] num_colors: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
) {
    let lane = invocation_id.x;
    let batch_id = invocation_id.y;
    let nb = batch_ids.num_batches;

    let constraints = Slice(constraints, 0);
    let color_sorted_ids = Slice(color_sorted_ids, 0);
    let mut solver_vels = SliceMut(solver_vels, 0);
    let num_colors = *num_colors;

    for color in 1..=num_colors {
        // This batch's bucket for `color` (color-major layout, post-scatter
        // exclusive ends; the index is >= num_batches >= 1 for color >= 1).
        let bucket = (color * nb + batch_id) as usize;
        let start = color_starts.read(bucket - 1);
        let end = color_starts.read(bucket);
        #[cfg(not(feature = "web-compat"))]
        if start == end {
            // Empty color.
            continue;
        }

        if start != end {
            for k in StepRng::new(start + lane..end, WORKGROUP_SIZE) {
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

        control_barrier::<
            { khal_std::memory::Scope::Workgroup as u32 },
            { khal_std::memory::Scope::QueueFamily as u32 },
            {
                khal_std::memory::Semantics::UNIFORM_MEMORY.bits()
                    | khal_std::memory::Semantics::ACQUIRE_RELEASE.bits()
            },
        >();
    }
}

/// Fused colored Gauss-Seidel sweep: only one 64-lane workgroup per batch walks
/// every color bucket with a storage barrier between colors.
///
/// Used for small scenes where the contact count is small wrt. the environment count.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_step_gauss_seidel_fused(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    constraints: &mut [TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] color_starts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] color_sorted_ids: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] num_colors: &u32,
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] use_bias: &u32,
) {
    let lane = invocation_id.x;
    let batch_id = invocation_id.y;
    let nb = batch_ids.num_batches;

    let mut constraints = SliceMut(constraints, 0);
    let color_sorted_ids = Slice(color_sorted_ids, 0);
    let mut solver_vels = SliceMut(solver_vels, 0);
    let num_colors = *num_colors;
    let use_bias = *use_bias != 0;

    for color in 1..=num_colors {
        // Empty-color skip: see `gpu_warmstart_fused`.
        let bucket = (color * nb + batch_id) as usize;
        let start = color_starts.read(bucket - 1);
        let end = color_starts.read(bucket);
        #[cfg(not(feature = "web-compat"))]
        if start == end {
            continue;
        }

        if start != end {
            for k in StepRng::new(start + lane..end, WORKGROUP_SIZE) {
                let i = color_sorted_ids[k as usize];
                let solver_id1 = constraints[i as usize].solver_body_a as usize;
                let solver_id2 = constraints[i as usize].solver_body_b as usize;

                let mut solver_vel1 = solver_vels[solver_id1];
                let mut solver_vel2 = solver_vels[solver_id2];

                constraints[i as usize].solve_constraint_gauss_seidel(
                    &mut solver_vel1,
                    &mut solver_vel2,
                    use_bias,
                );

                solver_vels[solver_id1] = solver_vel1;
                solver_vels[solver_id2] = solver_vel2;
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
