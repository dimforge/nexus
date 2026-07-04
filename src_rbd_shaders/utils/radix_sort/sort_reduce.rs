//! Radix Sort Reduce Kernel
//!
//! Reduces per-workgroup histograms to compute partial sums. This is the second
//! pass in the radix sort pipeline.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

use super::sorting::{BIN_COUNT, BLOCK_SIZE, ELEMENTS_PER_THREAD, WG, div_ceil};
use crate::utils::radix_sort::SortUniforms;
use crate::{udiv, umod};

/// Radix sort reduce kernel.
///
/// Reduces per-workgroup histogram counts to compute partial sums.
#[spirv_bindgen]
#[spirv(compute(threads(256)))]
pub fn gpu_sort_reduce(
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup_id)] gid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] config: &SortUniforms,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] num_keys_arr: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] counts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] reduced: &mut [u32],
    #[spirv(workgroup)] sums: &mut [u32; 256],
) {
    let group_id = gid.x;
    let batch_id = gid.y;
    let num_keys = num_keys_arr.read(batch_id as usize);
    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);
    let max_wgs_per_batch = div_ceil(config.max_keys_per_batch, BLOCK_SIZE);
    let num_reduce_wgs = BIN_COUNT * div_ceil(num_wgs, BLOCK_SIZE);

    let counts_offset = batch_id * BIN_COUNT * max_wgs_per_batch;
    let reduced_offset = batch_id * BLOCK_SIZE;

    // Filter-out out of bounds workgroups but don’t
    // just early-exit to keep uniform control flow
    // wrt. the barriers (for web compatibility).
    let active = group_id < num_reduce_wgs;
    let num_reduce_wg_per_bin = num_reduce_wgs / BIN_COUNT;

    let mut sum = 0u32;

    if active {
        let bin_id = udiv(group_id, num_reduce_wg_per_bin);
        let bin_offset = bin_id * num_wgs;
        let base_index = umod(group_id, num_reduce_wg_per_bin) * BLOCK_SIZE;

        for i in 0..ELEMENTS_PER_THREAD {
            let data_index = base_index + i * WG + local_id.x;
            if data_index < num_wgs {
                sum = sum
                    .wrapping_add(counts.read((counts_offset + bin_offset + data_index) as usize));
            }
        }
    }

    sums.write(local_id.x as usize, sum);

    // Parallel reduction
    for i in 0..8u32 {
        workgroup_memory_barrier_with_group_sync();
        if local_id.x < ((WG / 2) >> i) {
            sum = sum.wrapping_add(sums.read((local_id.x + ((WG / 2) >> i)) as usize));
            sums.write(local_id.x as usize, sum);
        }
    }

    if local_id.x == 0 && active {
        reduced.write((reduced_offset + group_id) as usize, sum);
    }
}
