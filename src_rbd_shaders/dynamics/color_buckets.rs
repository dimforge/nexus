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
use khal_std::sync::workgroup_memory_barrier_with_group_sync;
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

/// Index of the largest color bucket (all batches) in the color stats buffer.
pub const COLOR_STATS_MAX_BUCKET: usize = 0;
/// Index of the highest non-empty color in the color stats buffer.
pub const COLOR_STATS_MAX_COLOR: usize = 1;

/// Sizes the colored sweeps from the color buckets: their grid only needs to cover the
/// largest color (the sweeps stride over their bucket), not every contact. Also records the
/// highest color in use, read back to adapt the color budget. One workgroup.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_color_sweep_grid(
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] color_buckets: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] color_stats: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] sweep_indirect: &mut [[u32; 3]],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
    #[spirv(workgroup)] max_sizes: &mut [u32; 64],
    #[spirv(workgroup)] max_colors: &mut [u32; 64],
) {
    let lane = local_id.x as usize;
    let nb = batch_ids.num_batches;
    // Swept colors are `1..=stride - 2`; color `c` spans `[ends[c*nb - 1], ends[(c+1)*nb - 1])`.
    let num_colors = batch_ids.solver_color_buckets_stride - 2;

    let mut max_size = 0u32;
    let mut max_color = 0u32;
    for color in StepRng::new(1 + lane as u32..num_colors + 1, WORKGROUP_SIZE) {
        let start = color_buckets.read((color * nb - 1) as usize);
        let end = color_buckets.read(((color + 1) * nb - 1) as usize);
        let size = end - start;
        max_size = max_size.max(size);
        if size > 0 {
            max_color = max_color.max(color);
        }
    }
    max_sizes.write(lane, max_size);
    max_colors.write(lane, max_color);

    let mut stride = 32usize;
    for _ in 0..6u32 {
        workgroup_memory_barrier_with_group_sync();
        if lane < stride {
            let s = max_sizes.read(lane).max(max_sizes.read(lane + stride));
            let c = max_colors.read(lane).max(max_colors.read(lane + stride));
            max_sizes.write(lane, s);
            max_colors.write(lane, c);
        }
        stride /= 2;
    }
    workgroup_memory_barrier_with_group_sync();

    if lane == 0 {
        let max_size = max_sizes.read(0);
        color_stats.write(COLOR_STATS_MAX_BUCKET, max_size);
        color_stats.write(COLOR_STATS_MAX_COLOR, max_colors.read(0));
        sweep_indirect.write(0, [max_size.div_ceil(WORKGROUP_SIZE), 1, 1]);
    }
}
