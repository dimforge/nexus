//! Bucket-sort of contact constraints by graph-coloring color.
//!
//! After the per-step (global) coloring converges, the constraint indices are
//! bucket-sorted by `(color, batch)` into `color_sorted_ids`. Buckets are laid
//! out color-major (`bucket = color * num_batches + batch`, buffer length
//! `solver_color_buckets_stride * num_batches`), so one color's constraints
//! are contiguous across every batch (per-color solver sweeps) while each
//! `(color, batch)` cell stays contiguous too (fused per-batch sweeps).

use crate::broad_phase::ContactPlan;
use khal_std::glamx::UVec3;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::{index::MaybeIndexUnchecked, iter::StepRng, sync::atomic_add_u32};

use super::constraint::TwoBodyConstraint;
use crate::utils::{BatchIndices, Slice};

const WORKGROUP_SIZE: u32 = 64;

/// Zeroes the `(color, batch)` bucket counts (flat 1-D grid over the whole
/// bucket buffer).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_color_buckets_reset(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] color_buckets: &mut [u32],
) {
    let i = invocation_id.x as usize;
    if i < color_buckets.len() {
        color_buckets.write(i, 0);
    }
}

/// Counts how many constraints fall in each `(color, batch)` bucket.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_color_buckets_count(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] constraints_colors: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] constraints: &[TwoBodyConstraint],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] contact_plan: &ContactPlan,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] color_buckets: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let nb = batch_ids.num_batches;
    let stride = batch_ids.solver_color_buckets_stride;
    let total = contact_plan.bound;
    let constraints = Slice(constraints, 0);

    for i in StepRng::new(invocation_id.x..total, num_threads) {
        let color = constraints_colors.read(i as usize);
        // Color 0 (uncolored / gap slots) is never swept; colors past the
        // swept range (bounded coloring didn't converge) are dropped. They
        // were never solved before either. Skipping color 0 also keeps the
        // stale body ids of gap slots from being dereferenced.
        if color != 0 && color < stride - 1 {
            let batch = batch_ids.collider_batch(constraints[i as usize].solver_body_a);
            atomic_add_u32(color_buckets.at_mut((color * nb + batch) as usize), 1);
        }
    }
}

/// Scatters each constraint index into its `(color, batch)` bucket. The bucket
/// buffer holds the scanned exclusive starts, used as cursors; after this pass
/// every entry is its bucket's exclusive end.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_color_buckets_scatter(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] constraints_colors: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] constraints: &[TwoBodyConstraint],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] contact_plan: &ContactPlan,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] color_buckets: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] color_sorted_ids: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let nb = batch_ids.num_batches;
    let stride = batch_ids.solver_color_buckets_stride;
    let total = contact_plan.bound;
    let constraints = Slice(constraints, 0);

    for i in StepRng::new(invocation_id.x..total, num_threads) {
        let color = constraints_colors.read(i as usize);
        if color != 0 && color < stride - 1 {
            let batch = batch_ids.collider_batch(constraints[i as usize].solver_body_a);
            let dst = atomic_add_u32(color_buckets.at_mut((color * nb + batch) as usize), 1);
            color_sorted_ids.write(dst as usize, i);
        }
    }
}
