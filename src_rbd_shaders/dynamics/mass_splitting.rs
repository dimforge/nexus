//! Mass splitting for bodies with many contacts ("hubs").
//!
//! Graph coloring assigns different colors to constraints sharing a body, so a body touching
//! `n` others needs at least `n` colors. A heavy ball ploughing through a brick wall touches
//! hundreds of bricks at once, which blows up the color count (and every colored dispatch).
//!
//! Instead, a hub body is split into one sub-body per contact (Tonge et al. 2012, "Mass
//! splitting for jitter-free parallel rigid body simulation"): each sub-body has `1/n`-th of
//! the mass and inertia, owns a private velocity slot past the regular bodies in the solver
//! velocity buffer, and its constraints are colored ignoring the hub. Before each iteration the hub
//! velocity is scattered to its sub-bodies, and after the iteration their velocities are averaged
//! back into the hub, which applies the sum of every contact impulse to the real body.

use crate::utils::{BatchIndices, Slice, SliceMut};
use crate::{AngVector, Vector};
use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::iter::StepRng;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::{atomic_add_u32, workgroup_memory_barrier_with_group_sync};

use super::ContactLink;
use super::body::{Velocity, WorldMassProperties};

const WORKGROUP_SIZE: u32 = 64;

/// A body is split when more constraints than this touch it.
pub const HUB_MIN_CONSTRAINTS: u32 = 32;
/// Marks a body that isn't split in `hub_first_slot`.
pub const NOT_A_HUB: u32 = u32::MAX;

/// The counters shared by the mass splitting kernels.
#[derive(Copy, Clone, Default, PartialEq, Debug)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct HubCounts {
    /// Number of hubs picked this step, possibly past the hub list capacity.
    pub hubs: u32,
    /// Number of sub-body slots requested this step, possibly past `pool`.
    pub slots: u32,
    /// First sub-body velocity slot (the number of regular body slots).
    pub base: u32,
    /// Sub-body slot capacity.
    pub pool: u32,
}

impl HubCounts {
    /// Capacity of the hub list.
    #[inline(always)]
    pub fn max_hubs(&self) -> u32 {
        self.pool / HUB_MIN_CONSTRAINTS + 1
    }

    /// Number of hubs recorded in the hub list.
    #[inline(always)]
    pub fn live_hubs(&self) -> u32 {
        self.hubs.min(self.max_hubs())
    }

    /// Number of sub-body slots in use.
    #[inline(always)]
    pub fn live_slots(&self) -> u32 {
        self.slots.min(self.pool)
    }
}

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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] hub_counts: &mut HubCounts,
    #[spirv(uniform, descriptor_set = 0, binding = 6)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let num_slots = batch_ids.colliders_batch_capacity * batch_ids.num_batches;
    let counts = Slice(body_constraint_counts, 0);
    let pool = hub_counts.pool;
    let max_hubs = hub_counts.max_hubs();

    for body in StepRng::new(invocation_id.x..num_slots, num_threads) {
        let (first, last) = constraint_range(&counts, body);
        let n = last - first;
        // Multibody links are colored as their multibody's group and never split.
        let splittable = n > HUB_MIN_CONSTRAINTS
            && body_group.read(body as usize) == body
            && mprops.at(body as usize).inv_mass != Vector::ZERO;

        if splittable {
            let s0 = atomic_add_u32(&mut hub_counts.slots, n);
            if s0 + n <= pool {
                let k = atomic_add_u32(&mut hub_counts.hubs, 1);
                if k < max_hubs {
                    hub_first_slot.write(body as usize, s0);
                    hub_list.write(k as usize, body);
                }
            }
        }
    }
}

/// Points each hub constraint at its sub-body velocity slot, and scales the hub side's inverse
/// mass and inertia by the number of sub-bodies (applied when the constraints are built).
///
/// One workgroup per hub, striding over its adjacency entries. Ordinary bodies
/// never enter this pass; no per-contact binary search of body ranges is needed.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_hub_split_constraints(
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] body_constraint_counts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] body_constraint_ids: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] hub_first_slot: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] constraints: &mut [ContactLink],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] hub_slot_body: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] hub_slot_constraint: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] hub_counts: &HubCounts,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] hub_list: &[u32],
) {
    let counts = Slice(body_constraint_counts, 0);
    let base = hub_counts.base;
    let hubs = hub_counts.live_hubs();
    for hub in StepRng::new(workgroup_id.x..hubs, num_workgroups.x) {
        let body = hub_list.read(hub as usize);
        let s0 = hub_first_slot.read(body as usize);
        let (first, last) = constraint_range(&counts, body);
        for entry in StepRng::new(first + local_id.x..last, WORKGROUP_SIZE) {
            let n = (last - first) as f32;
            let slot = s0 + (entry - first);
            let cid = body_constraint_ids.read(entry as usize);
            let constraint = constraints.at_mut(cid as usize);

            if constraint.solver_body_a == body {
                constraint.vel_slot_a = base + slot;
                constraint.mass_scale_a = n;
            } else {
                constraint.vel_slot_b = base + slot;
                constraint.mass_scale_b = n;
            }

            hub_slot_body.write(slot as usize, body);
            hub_slot_constraint.write(slot as usize, cid);
        }
    }
}

/// Sizes hub-only dispatches after assignment, including the zero-hub case.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_hub_dispatch(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] hub_counts: &HubCounts,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] slots_indirect: &mut [[u32; 3]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] average_indirect: &mut [[u32; 3]],
) {
    let slots = hub_counts.live_slots();
    let hubs = hub_counts.live_hubs();
    slots_indirect.write(0, [slots.div_ceil(WORKGROUP_SIZE), 1, 1]);
    average_indirect.write(0, [hubs.min(HUB_AVERAGE_WORKGROUPS), 1, 1]);
}

/// Copies each hub's velocity into its sub-body slots.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_hub_scatter(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] solver_vels: &mut [Velocity],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] hub_slot_body: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] hub_counts: &HubCounts,
) {
    let slot = invocation_id.x;
    if slot < hub_counts.live_slots() {
        let base = hub_counts.base;
        let body = hub_slot_body.read(slot as usize);
        let vel = solver_vels.read(body as usize);
        solver_vels.write((base + slot) as usize, vel);
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
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] hub_counts: &HubCounts,
    #[spirv(workgroup)] linear_sums: &mut [Vector; 64],
    #[spirv(workgroup)] angular_sums: &mut [AngVector; 64],
) {
    let lane = local_id.x as usize;
    let counts = Slice(body_constraint_counts, 0);
    let base = hub_counts.base;
    let num_hubs = hub_counts.live_hubs();
    let mut solver_vels = SliceMut(solver_vels, 0);

    // Uniform across the workgroup, so the barriers below are reached by every lane. The WGSL
    // uniformity analysis doesn't trust `workgroup_id`, so on the web the rounds are bounded by
    // `max_hubs` (read-only storage) and the out-of-range hubs run them inactive.
    #[cfg(not(feature = "web-compat"))]
    #[allow(clippy::implicit_saturating_sub)]
    let num_rounds = if num_hubs > workgroup_id.x {
        (num_hubs - workgroup_id.x).div_ceil(num_workgroups.x)
    } else {
        0
    };
    #[cfg(feature = "web-compat")]
    let num_rounds = hub_counts.max_hubs().div_ceil(HUB_AVERAGE_WORKGROUPS);

    for round in 0..num_rounds {
        let k = workgroup_id.x + round * num_workgroups.x;
        let active = k < num_hubs;
        let body = if active { hub_list.read(k as usize) } else { 0 };
        let s0 = if active {
            hub_first_slot.read(body as usize)
        } else {
            0
        };
        let (first, last) = if active {
            constraint_range(&counts, body)
        } else {
            (0, 0)
        };
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

#[cfg(all(test, not(target_arch_is_gpu)))]
#[path = "../tests/hub_split.rs"]
mod tests;
