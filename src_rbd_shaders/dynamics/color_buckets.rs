//! Bucket-sort of contact constraints by graph-coloring color.
//!
//! After the per-step (global) coloring converges, the constraint links are
//! bucket-sorted by `(color, batch)` into `sorted_links`. Buckets are laid
//! out color-major (`bucket = color * num_batches + batch`, buffer length
//! `solver_color_buckets_stride * num_batches`), so one color's constraints
//! are contiguous across every batch (per-color solver dispatches) while each
//! `(color, batch)` cell stays contiguous too (fused per-batch dispatches).
//!
//! The position of a constraint in `sorted_links` is its index in the contact tiles.
//! Active constraints the bounded coloring left uncolored (or past the solved colors) are never
//! solved, but still need a place in the tiles for their warmstart: they go to the last color
//! bucket.

use crate::broad_phase::ContactPlan;
use crunchy::unroll;
use khal_std::glamx::UVec3;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::atomic_add_u32_workgroup;
use khal_std::sync::workgroup_memory_barrier_with_group_sync;
use khal_std::{index::MaybeIndexUnchecked, iter::StepRng, sync::atomic_add_u32};

use super::ContactLink;
use crate::utils::{BatchIndices, Slice};

const WORKGROUP_SIZE: u32 = 64;

/// The bucket color of a constraint: its color if it is solved, the last color bucket if it is
/// active but not solved, or 0 for gap, inert and multibody-owned slots (no bucket).
#[inline(always)]
fn bucket_color(color: u32, len: u32, stride: u32) -> u32 {
    if len == 0 {
        0
    } else if color != 0 && color < stride - 1 {
        color
    } else {
        stride - 1
    }
}

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

/// Counts how many constraints fall in each `(color, batch)` bucket, and marks every sorted
/// link as inactive before the scatter fills them.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_color_buckets_count(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] constraints_colors: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] constraints: &[ContactLink],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] contact_plan: &ContactPlan,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] color_buckets: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] sorted_links: &mut [ContactLink],
    #[spirv(workgroup)] local_counts: &mut [u32; 128],
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let nb = batch_ids.num_batches;
    let stride = batch_ids.solver_color_buckets_stride;
    let total = contact_plan.bound;
    let constraints = Slice(constraints, 0);

    // Naga lowers num_workgroups through a private global, so browser WGSL
    // validation cannot prove this condition uniform. Keep every barrier outside
    // the branch. Run the generic path last: its checked batch division can lower
    // to an early exit, which would also make subsequent barriers non-uniform.
    // Fallback lanes initialize an empty histogram and reach every barrier. The last bucket,
    // `stride - 1`, must fit the 128-entry histogram.
    let use_histogram = nb == 1 && stride <= 128 && num_threads >= total;
    let lane = local_id.x as usize;
    local_counts.write(lane, 0);
    local_counts.write(lane + 64, 0);
    workgroup_memory_barrier_with_group_sync();
    if use_histogram && invocation_id.x < total {
        let i = invocation_id.x as usize;
        let color = bucket_color(constraints_colors.read(i), constraints[i].len, stride);
        if color != 0 {
            atomic_add_u32_workgroup(local_counts.at_mut(color as usize), 1);
        }
    }
    workgroup_memory_barrier_with_group_sync();
    crunchy::unroll! { for k in 0..2 {
        let bucket = lane + k * 64;
        let size = local_counts.read(bucket);
        if size != 0 {
            atomic_add_u32(color_buckets.at_mut(bucket), size);
        }
    } }

    if !use_histogram {
        for i in StepRng::new(invocation_id.x..total, num_threads) {
            let link = &constraints[i as usize];
            // Skipping the slots without bucket also keeps the stale body ids of gap slots
            // from being dereferenced.
            let color = bucket_color(constraints_colors.read(i as usize), link.len, stride);
            if color != 0 {
                let batch = batch_ids.collider_batch(link.solver_body_a);
                atomic_add_u32(color_buckets.at_mut((color * nb + batch) as usize), 1);
            }
        }
    }
    for i in StepRng::new(invocation_id.x..total, num_threads) {
        sorted_links.at_mut(i as usize).len = 0;
    }
}

/// Scatters each constraint link into its `(color, batch)` bucket. The bucket
/// buffer holds the scanned exclusive starts, used as cursors; after this pass
/// every entry is its bucket's exclusive end.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_color_buckets_scatter(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] constraints_colors: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] constraints: &[ContactLink],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] contact_plan: &ContactPlan,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] color_buckets: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] sorted_links: &mut [ContactLink],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] constraint_indices: &mut [u32],
    #[spirv(workgroup)] local_counts: &mut [u32; 128],
    #[spirv(workgroup)] local_starts: &mut [u32; 128],
) {
    let nb = batch_ids.num_batches;
    let stride = batch_ids.solver_color_buckets_stride;
    let total = contact_plan.bound;
    let lane = local_id.x as usize;
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    // Match the count fast path's bounds. Reserving one contiguous span per local
    // color reduces global contention and keeps nearby input constraints together.
    // As in count, fallback lanes must reach all barriers for WGSL uniformity.
    let use_histogram = nb == 1 && stride <= 128 && num_threads >= total;
    local_counts.write(lane, 0);
    local_counts.write(lane + 64, 0);
    workgroup_memory_barrier_with_group_sync();
    let color = if use_histogram && gid.x < total {
        let i = gid.x as usize;
        bucket_color(constraints_colors.read(i), constraints.at(i).len, stride)
    } else {
        0
    };
    let valid = color != 0;
    let rank = if valid {
        atomic_add_u32_workgroup(local_counts.at_mut(color as usize), 1)
    } else {
        0
    };
    workgroup_memory_barrier_with_group_sync();
    crunchy::unroll! { for k in 0..2 {
        let bucket = lane + k * 64;
        let size = local_counts.read(bucket);
        if size != 0 {
            let start = atomic_add_u32(color_buckets.at_mut(bucket), size);
            local_starts.write(bucket, start);
        }
    } }
    workgroup_memory_barrier_with_group_sync();
    if valid {
        let index = local_starts.read(color as usize) + rank;
        sorted_links.write(index as usize, constraints.read(gid.x as usize));
        constraint_indices.write(gid.x as usize, index);
    }

    if !use_histogram {
        for i in StepRng::new(gid.x..total, num_threads) {
            let link = constraints.at(i as usize);
            let color = bucket_color(constraints_colors.read(i as usize), link.len, stride);
            if color != 0 {
                let batch = batch_ids.collider_batch(link.solver_body_a);
                let dst = atomic_add_u32(color_buckets.at_mut((color * nb + batch) as usize), 1);
                sorted_links.write(dst as usize, *link);
                constraint_indices.write(i as usize, dst);
            }
        }
    }
}

/// Index of the largest color bucket (all batches) in the color stats buffer.
pub const COLOR_STATS_MAX_BUCKET: usize = 0;
/// Index of the highest non-empty color in the color stats buffer.
pub const COLOR_STATS_MAX_COLOR: usize = 1;

/// Sizes the colored dispatches from the color buckets: their grid only needs to cover the
/// largest color (the solve kernels stride over their bucket), not every contact. Also records the
/// highest color in use, read back to adapt the color budget. One workgroup.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_color_dispatch_grid(
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] color_buckets: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] color_stats: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] dispatch_indirect: &mut [[u32; 3]],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
    #[spirv(workgroup)] max_sizes: &mut [u32; 64],
    #[spirv(workgroup)] max_colors: &mut [u32; 64],
) {
    let lane = local_id.x as usize;
    let nb = batch_ids.num_batches;
    // Swept colors are `1..=stride - 2`; color `c` spans `[ends[c*nb - 1], ends[(c+1)*nb - 1])`.
    let num_colors = batch_ids.solver_color_buckets_stride - 2;

    // Per-color sizes are CPU scheduling hints for subsequent steps. They never bound
    // the actual dispatch: every dispatch traverses its current GPU bucket with a strided for.
    let color = lane as u32;
    let size = if color > 0 && color <= num_colors {
        color_buckets.read(((color + 1) * nb - 1) as usize)
            - color_buckets.read((color * nb - 1) as usize)
    } else {
        0
    };
    color_stats.write(2 + lane, size);

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
        // Every color shares this grid. Limit the empty workgroups in the smaller buckets;
        // the solve kernels stride over large buckets, so no constraints are skipped.
        dispatch_indirect.write(0, [max_size.div_ceil(WORKGROUP_SIZE).min(256), 1, 1]);
    }
}
