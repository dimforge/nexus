//! Mass splitting for bodies with many contacts ("hubs").
//!
//! Graph coloring assigns different colors to constraints sharing a body, so a body touching
//! `n` others needs at least `n` colors. A heavy ball ploughing through a brick wall touches
//! hundreds of bricks at once, which blows up the color count (and every colored sweep).
//!
//! Instead, a hub body is split into one sub-body per contact (Tonge et al. 2012, "Mass
//! splitting for jitter-free parallel rigid body simulation"): each sub-body has `1/n`-th of
//! the mass and inertia, owns a private velocity slot past the regular bodies in the solver
//! velocity buffer, and its constraints are colored ignoring the hub. Before each sweep the hub
//! velocity is scattered to its sub-bodies, and after the sweep their velocities are averaged
//! back into the hub, which applies the sum of every contact impulse to the real body.

use crate::broad_phase::ContactPlan;
use crate::utils::{BatchIndices, Slice, SliceMut};
use crate::{AngVector, Vector, gdot};
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::iter::StepRng;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::{atomic_add_u32, workgroup_memory_barrier_with_group_sync};

use super::body::{Velocity, WorldMassProperties};
use super::constraint::{SUB_LEN, TwoBodyConstraint};

const WORKGROUP_SIZE: u32 = 64;

/// A body is split when more constraints than this touch it.
pub const HUB_MIN_CONSTRAINTS: u32 = 32;
/// Marks a body that isn't split in `hub_first_slot`.
pub const NOT_A_HUB: u32 = u32::MAX;

/// Index of the number of hubs in the `hub_counts` buffer.
pub const HUB_COUNT_HUBS: usize = 0;
/// Index of the number of allocated sub-body slots in the `hub_counts` buffer.
pub const HUB_COUNT_SLOTS: usize = 1;
/// Index of the first sub-body velocity slot (the number of regular body slots).
pub const HUB_COUNT_BASE: usize = 2;
/// Index of the sub-body slot capacity.
pub const HUB_COUNT_POOL: usize = 3;
/// Length of the `hub_counts` buffer.
pub const HUB_COUNTS_LEN: usize = 4;

/// Number of workgroups averaging the sub-bodies back into their hubs (at most).
pub const HUB_AVERAGE_WORKGROUPS: u32 = 256;

/// The constraint range of `body` in the cumulative per-body constraint counts.
#[inline(always)]
fn constraint_range(counts: &Slice<u32>, body: u32) -> (u32, u32) {
    let first = if body != 0 {
        counts[body as usize - 1]
    } else {
        0
    };
    (first, counts[body as usize])
}

/// Picks the hubs and reserves their sub-body slots.
///
/// Runs after the per-body constraint lists are built (`body_constraint_counts` holds each
/// body's cumulative end). `hub_first_slot` and the hub counters must have been reset.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_hub_assign(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] body_constraint_counts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mprops: &[WorldMassProperties],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] body_group: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] hub_first_slot: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] hub_list: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] hub_counts: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 6)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let num_slots = batch_ids.colliders_batch_capacity * batch_ids.num_batches;
    let counts = Slice(body_constraint_counts, 0);
    let pool = hub_counts.read(HUB_COUNT_POOL);
    let max_hubs = pool / HUB_MIN_CONSTRAINTS + 1;

    for body in StepRng::new(invocation_id.x..num_slots, num_threads) {
        let (first, last) = constraint_range(&counts, body);
        let n = last - first;
        // Multibody links are colored as their multibody's group and never split.
        let splittable = n > HUB_MIN_CONSTRAINTS
            && body_group.read(body as usize) == body
            && mprops.at(body as usize).inv_mass != Vector::ZERO;

        if splittable {
            let s0 = atomic_add_u32(hub_counts.at_mut(HUB_COUNT_SLOTS), n);
            if s0 + n <= pool {
                let k = atomic_add_u32(hub_counts.at_mut(HUB_COUNT_HUBS), 1);
                if k < max_hubs {
                    hub_first_slot.write(body as usize, s0);
                    hub_list.write(k as usize, body);
                }
            }
        }
    }
}

/// Points each hub constraint at its sub-body velocity slot and scales the hub side's inverse
/// mass and inertia by the number of sub-bodies.
///
/// One thread per entry of the per-body constraint lists.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_hub_split_constraints(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] body_constraint_counts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] body_constraint_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] hub_first_slot: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)]
    constraints: &mut [TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] hub_slot_body: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] hub_slot_constraint: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] hub_counts: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 7)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let num_slots = batch_ids.colliders_batch_capacity * batch_ids.num_batches;
    let counts = Slice(body_constraint_counts, 0);
    let base = hub_counts.read(HUB_COUNT_BASE);
    let num_entries = counts[num_slots as usize - 1];

    for entry in StepRng::new(invocation_id.x..num_entries, num_threads) {
        // The body owning this entry: the first body whose cumulative end exceeds it.
        let mut lo = 0u32;
        let mut hi = num_slots - 1;
        for _ in 0..32u32 {
            if lo < hi {
                let mid = (lo + hi) / 2;
                if counts[mid as usize] > entry {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
        }
        let body = lo;

        let s0 = hub_first_slot.read(body as usize);
        if s0 == NOT_A_HUB {
            continue;
        }

        let (first, last) = constraint_range(&counts, body);
        let n = (last - first) as f32;
        let slot = s0 + (entry - first);
        let cid = body_constraint_ids.read(entry as usize);
        let constraint = constraints.at_mut(cid as usize);
        let len = constraint.len as usize;

        if constraint.solver_body_a == body {
            constraint.vel_slot_a = base + slot;
            constraint.im_a *= n;
            for k in 0..len {
                let element = constraint.elements.at_mut(k);
                element.normal_part.ii_torque_dir_a *= n;
                for j in 0..SUB_LEN {
                    element.tangent_part.ii_torque_dir_a.at_mut(j).0 *= n;
                }
            }
        } else {
            constraint.vel_slot_b = base + slot;
            constraint.im_b *= n;
            for k in 0..len {
                let element = constraint.elements.at_mut(k);
                element.normal_part.ii_torque_dir_b *= n;
                for j in 0..SUB_LEN {
                    element.tangent_part.ii_torque_dir_b.at_mut(j).0 *= n;
                }
            }
        }

        hub_slot_body.write(slot as usize, body);
        hub_slot_constraint.write(slot as usize, cid);
    }
}

#[inline(always)]
fn inv(x: f32) -> f32 {
    if x == 0.0 { 0.0 } else { 1.0 / x }
}

/// Recomputes the effective masses of the constraints touching a split hub, and sizes the
/// scatter and average dispatches.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_hub_update_effective_masses(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
    constraints: &mut [TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] hub_counts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] slots_indirect: &mut [[u32; 3]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] average_indirect: &mut [[u32; 3]],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] contact_plan: &ContactPlan,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;

    if invocation_id.x == 0 {
        let pool = hub_counts.read(HUB_COUNT_POOL);
        let slots = hub_counts.read(HUB_COUNT_SLOTS).min(pool);
        let max_hubs = pool / HUB_MIN_CONSTRAINTS + 1;
        let hubs = hub_counts.read(HUB_COUNT_HUBS).min(max_hubs);
        slots_indirect.write(0, [slots.div_ceil(WORKGROUP_SIZE), 1, 1]);
        average_indirect.write(0, [hubs.min(HUB_AVERAGE_WORKGROUPS), 1, 1]);
    }

    for i in StepRng::new(invocation_id.x..contact_plan.bound, num_threads) {
        let c = constraints.at_mut(i as usize);
        if c.len == 0 || (c.vel_slot_a == c.solver_body_a && c.vel_slot_b == c.solver_body_b) {
            continue;
        }

        let dir = c.dir_a;
        let imsum = c.im_a + c.im_b;
        #[cfg(feature = "dim3")]
        let tangents = [c.tangent_a, dir.cross(c.tangent_a)];

        for k in 0..(c.len as usize) {
            let e = c.elements.at_mut(k);
            let n = &mut e.normal_part;
            n.r = inv(dir.dot(imsum * dir)
                + gdot(n.ii_torque_dir_a, n.torque_dir_a)
                + gdot(n.ii_torque_dir_b, n.torque_dir_b));

            let t = &mut e.tangent_part;
            #[cfg(feature = "dim2")]
            {
                let tangent = crate::Vector::new(-dir.y, dir.x);
                let r = tangent.dot(imsum * tangent)
                    + gdot(t.ii_torque_dir_a.at(0).0, t.torque_dir_a.at(0).0)
                    + gdot(t.ii_torque_dir_b.at(0).0, t.torque_dir_b.at(0).0);
                t.r.write(0, inv(r));
            }
            #[cfg(feature = "dim3")]
            {
                for j in 0..SUB_LEN {
                    let tj = tangents.read(j);
                    let r = tj.dot(imsum * tj)
                        + gdot(t.ii_torque_dir_a.at(j).0, t.torque_dir_a.at(j).0)
                        + gdot(t.ii_torque_dir_b.at(j).0, t.torque_dir_b.at(j).0);
                    t.r.write(j, r);
                }
                let cross = 2.0
                    * (t.torque_dir_a.at(0).0.dot(t.ii_torque_dir_a.at(1).0)
                        + t.torque_dir_b.at(0).0.dot(t.ii_torque_dir_b.at(1).0));
                t.r.write(2, cross);
            }
        }
    }
}

/// Copies each hub's velocity into its sub-body slots.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_hub_scatter(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] hub_slot_body: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] hub_counts: &[u32],
) {
    let slot = invocation_id.x;
    let slots = hub_counts
        .read(HUB_COUNT_SLOTS)
        .min(hub_counts.read(HUB_COUNT_POOL));
    if slot < slots {
        let base = hub_counts.read(HUB_COUNT_BASE);
        let body = hub_slot_body.read(slot as usize);
        let vel = solver_vels.read(body as usize);
        solver_vels.write((base + slot) as usize, vel);
    }
}

/// Applies the warmstart impulse of each hub constraint to its sub-body (the hub's own
/// warmstart gather skips them).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_hub_warmstart(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] constraints: &[TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] hub_slot_constraint: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] hub_counts: &[u32],
) {
    let slot = invocation_id.x;
    let slots = hub_counts
        .read(HUB_COUNT_SLOTS)
        .min(hub_counts.read(HUB_COUNT_POOL));
    if slot < slots {
        let vel_slot = hub_counts.read(HUB_COUNT_BASE) + slot;
        let constraint = constraints.at(hub_slot_constraint.read(slot as usize) as usize);
        let mut vel = solver_vels.read(vel_slot as usize);
        let mut other = Velocity::default();
        if constraint.vel_slot_a == vel_slot {
            constraint.warmstart_constraint(&mut vel, &mut other);
        } else {
            constraint.warmstart_constraint(&mut other, &mut vel);
        }
        solver_vels.write(vel_slot as usize, vel);
    }
}

/// Averages the sub-body velocities back into their hub. One workgroup per hub.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_hub_average(
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] hub_list: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] hub_first_slot: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] body_constraint_counts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] hub_counts: &[u32],
    #[spirv(workgroup)] linear_sums: &mut [Vector; 64],
    #[spirv(workgroup)] angular_sums: &mut [AngVector; 64],
) {
    let lane = local_id.x as usize;
    let counts = Slice(body_constraint_counts, 0);
    let pool = hub_counts.read(HUB_COUNT_POOL);
    let base = hub_counts.read(HUB_COUNT_BASE);
    let max_hubs = pool / HUB_MIN_CONSTRAINTS + 1;
    let num_hubs = hub_counts.read(HUB_COUNT_HUBS).min(max_hubs);
    let mut solver_vels = SliceMut(solver_vels, 0);

    // Uniform across the workgroup, so the barriers below are reached by every lane.
    for k in StepRng::new(workgroup_id.x..num_hubs, num_workgroups.x) {
        let body = hub_list.read(k as usize);
        let s0 = hub_first_slot.read(body as usize);
        let (first, last) = constraint_range(&counts, body);
        let n = last - first;

        let mut linear = Vector::ZERO;
        let mut angular = AngVector::default();
        for s in StepRng::new(lane as u32..n, WORKGROUP_SIZE) {
            let vel = solver_vels[(base + s0 + s) as usize];
            linear += vel.linear;
            angular += vel.angular;
        }
        linear_sums.write(lane, linear);
        angular_sums.write(lane, angular);

        let mut stride = 32usize;
        for _ in 0..6u32 {
            workgroup_memory_barrier_with_group_sync();
            if lane < stride {
                let l = linear_sums.read(lane) + linear_sums.read(lane + stride);
                let a = angular_sums.read(lane) + angular_sums.read(lane + stride);
                linear_sums.write(lane, l);
                angular_sums.write(lane, a);
            }
            stride /= 2;
        }
        workgroup_memory_barrier_with_group_sync();

        if lane == 0 && n > 0 {
            let inv_n = 1.0 / n as f32;
            solver_vels[body as usize].linear = linear_sums.read(0) * inv_n;
            solver_vels[body as usize].angular = angular_sums.read(0) * inv_n;
        }
        // The shared sums are rewritten by the next hub.
        workgroup_memory_barrier_with_group_sync();
    }
}
