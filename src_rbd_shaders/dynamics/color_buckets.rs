//! Bucket-sort of contact constraints by graph-coloring color.
//!
//! After the per-step coloring converges, the constraint indices are
//! bucket-sorted by color (`color_sorted_ids`, contacts layout) with
//! per-batch per-color exclusive prefix sums (`color_starts`), so each
//! colored solver sweep iterates only its own bucket instead of scanning the
//! whole constraint buffer. The count/start/cursor buffers are flat
//! `[num_batches × stride]` arrays with `stride =
//! BatchIndices::solver_color_buckets_stride` (= `max_colors + 3`, keeping
//! `starts[c + 1]` in bounds for every swept color).

use khal_std::glamx::UVec3;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::{index::MaybeIndexUnchecked, iter::StepRng, sync::atomic_add_u32};

use super::constraint::TwoBodyConstraint;
use crate::utils::{BatchIndices, Slice};

const WORKGROUP_SIZE: u32 = 64;

/// Zeroes the per-batch per-color constraint counts.
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

/// Counts, per batch, how many constraints hold each color.
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_color_buckets_count(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] constraints_colors: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] constraints: &[TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contact_offsets: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] color_buckets: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 4)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let nb = batch_ids.num_batches;
    let stride = batch_ids.solver_color_buckets_stride;
    let total = contact_offsets.read(batch_ids.num_batches as usize);
    let constraints = Slice(constraints, 0);

    for i in StepRng::new(invocation_id.x..total, num_threads) {
        let color = constraints_colors.read(i as usize);
        if color != 0 && color < stride - 1 {
            let batch = batch_ids.collider_batch(constraints[i as usize].solver_body_a);
            atomic_add_u32(color_buckets.at_mut((color * nb + batch) as usize), 1);
        }
    }
}

#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_color_buckets_scatter(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(num_workgroups)] num_workgroups: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] constraints_colors: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] constraints: &[TwoBodyConstraint],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] contact_offsets: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] color_buckets: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] color_sorted_ids: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 5)] batch_ids: &BatchIndices,
) {
    let num_threads = num_workgroups.x * WORKGROUP_SIZE;
    let nb = batch_ids.num_batches;
    let stride = batch_ids.solver_color_buckets_stride;
    let total = contact_offsets.read(batch_ids.num_batches as usize);
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
