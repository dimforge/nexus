//! Bucket-sort of contact constraints by graph-coloring color.
//!
//! After the per-step coloring converges, the constraint indices are
//! bucket-sorted by color (`color_sorted_ids`), with per-batch, per-color
//! exclusive prefix sums (`color_starts`). Each colored solver sweep then
//! iterates only its own bucket — one coalesced index read per constraint —
//! instead of scanning the whole constraint buffer and filtering by color,
//! which cost O(num_colors × num_constraints) reads per sweep.
//!
//! Buffer layout: `color_counts` / `color_starts` / `color_cursors` are flat
//! `[num_batches × stride]` arrays with `stride =
//! BatchIndices::solver_color_buckets_stride` (= `max_colors + 3`, so
//! `starts[c + 1]` is always in bounds for every swept color).
//! `color_sorted_ids` shares the contacts layout (stride
//! `contacts_batch_capacity`).

use khal_std::glamx::UVec3;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::{index::MaybeIndexUnchecked, iter::StepRng, sync::atomic_add_u32};

use crate::utils::BatchIndices;

const WORKGROUP_SIZE: u32 = 64;

/// Zeroes the per-batch per-color constraint counts.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_color_buckets_reset(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] color_counts: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 1)] batch_ids: &BatchIndices,
) {
    let stride = batch_ids.solver_color_buckets_stride;
    let batch_id = invocation_id.y;
    let i = invocation_id.x;

    if i < stride {
        color_counts.write((batch_id * stride + i) as usize, 0);
    }
}

/// Counts, per batch, how many constraints hold each color.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_color_buckets_count(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] constraints_colors: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] contacts_len: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] color_counts: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;
    let stride = batch_ids.solver_color_buckets_stride;

    let constraints_colors = batch_ids.contact_batch(batch_id, constraints_colors);
    let len = contacts_len
        .read(batch_id as usize)
        .min(batch_ids.contacts_batch_capacity);

    for i in StepRng::new(invocation_id.x..len, num_threads) {
        let color = constraints_colors[i as usize];
        // Colors past the swept range (can happen if the bounded coloring
        // didn't converge) are dropped — they were never solved before either.
        if color < stride - 1 {
            atomic_add_u32(color_counts.at_mut((batch_id * stride + color) as usize), 1);
        }
    }
}

/// Per-batch serial exclusive prefix sum over the (tiny) per-color counts,
/// producing bucket start offsets. Also seeds the scatter cursors.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_color_buckets_scan(
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] color_counts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] color_starts: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] color_cursors: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] batch_ids: &BatchIndices,
) {
    let stride = batch_ids.solver_color_buckets_stride;
    let batch_id = workgroup_id.y;
    let base = (batch_id * stride) as usize;

    let mut acc = 0u32;
    for c in 0..stride as usize {
        color_starts.write(base + c, acc);
        color_cursors.write(base + c, acc);
        acc += color_counts.read(base + c);
    }
}

/// Scatters each constraint index into its color's bucket.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_color_buckets_scatter(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] constraints_colors: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] contacts_len: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] color_cursors: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] color_sorted_ids: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let batch_id = invocation_id.y;
    let stride = batch_ids.solver_color_buckets_stride;

    let constraints_colors = batch_ids.contact_batch(batch_id, constraints_colors);
    let mut color_sorted_ids = batch_ids.contact_batch_mut(batch_id, color_sorted_ids);
    let len = contacts_len
        .read(batch_id as usize)
        .min(batch_ids.contacts_batch_capacity);

    for i in StepRng::new(invocation_id.x..len, num_threads) {
        let color = constraints_colors[i as usize];
        if color < stride - 1 {
            let dst =
                atomic_add_u32(color_cursors.at_mut((batch_id * stride + color) as usize), 1);
            color_sorted_ids[dst as usize] = i;
        }
    }
}
