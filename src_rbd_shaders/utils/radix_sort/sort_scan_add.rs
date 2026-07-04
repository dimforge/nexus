//! Radix Sort Scan Add Kernel
//!
//! Adds scan results back to per-workgroup counts.
//! This completes the prefix sum computation across all workgroups.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

use super::sorting::{BIN_COUNT, BLOCK_SIZE, ELEMENTS_PER_THREAD, WG, div_ceil};
use crate::utils::radix_sort::SortUniforms;
use crate::{udiv, umod};

/// Radix sort scan add kernel.
///
/// Adds scan results to per-workgroup counts to complete prefix sum.
#[spirv_bindgen]
#[spirv(compute(threads(256)))]
pub fn gpu_sort_scan_add(
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup_id)] gid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] config: &SortUniforms,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] num_keys_arr: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] reduced: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] counts: &mut [u32],
    #[spirv(workgroup)] sums: &mut [u32; 256],
    #[spirv(workgroup)] lds: &mut [[u32; 256]; 4],
) {
    let group_id = gid.x;
    let batch_id = gid.y;
    let num_keys = num_keys_arr.read(batch_id as usize);
    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);
    let max_wgs_per_batch = div_ceil(config.max_keys_per_batch, BLOCK_SIZE);
    let num_reduce_wgs = BIN_COUNT * div_ceil(num_wgs, BLOCK_SIZE);

    let counts_offset = batch_id * BIN_COUNT * max_wgs_per_batch;
    let reduced_offset = batch_id * BLOCK_SIZE;

    let active = group_id < num_reduce_wgs;
    let num_reduce_wg_per_bin = num_reduce_wgs / BIN_COUNT;

    // Load with transposition
    // Inactive workgroups zero-fill shared memory so they participate in barriers
    // without affecting results.
    if active {
        let bin_id = udiv(group_id, num_reduce_wg_per_bin);
        let bin_offset = bin_id * num_wgs;
        let base_index = umod(group_id, num_reduce_wg_per_bin) * ELEMENTS_PER_THREAD * WG;

        let counts_len = counts.len() as u32;
        for i in 0..ELEMENTS_PER_THREAD {
            let data_index = base_index + i * WG + local_id.x;
            let col = (i * WG + local_id.x) / ELEMENTS_PER_THREAD;
            let row = (i * WG + local_id.x) % ELEMENTS_PER_THREAD;
            let global_index = counts_offset + bin_offset + data_index;
            // Read 0 if we are out of bounds. We don't just rely on robustness since
            // rustgpu automatically inserts early-exiting bound checks that would break it.
            let value = if global_index < counts_len {
                counts.read(global_index as usize)
            } else {
                0
            };
            lds.at_mut(row as usize).write(col as usize, value);
        }
    } else {
        for i in 0..ELEMENTS_PER_THREAD {
            let col = (i * WG + local_id.x) / ELEMENTS_PER_THREAD;
            let row = (i * WG + local_id.x) % ELEMENTS_PER_THREAD;
            lds.at_mut(row as usize).write(col as usize, 0);
        }
    }

    workgroup_memory_barrier_with_group_sync();

    // Local prefix sum
    let mut sum = 0u32;
    for i in 0..ELEMENTS_PER_THREAD {
        let tmp = lds.at(i as usize).read(local_id.x as usize);
        lds.at_mut(i as usize).write(local_id.x as usize, sum);
        sum = sum.wrapping_add(tmp);
    }

    // Workgroup prefix sum
    sums.write(local_id.x as usize, sum);
    for i in 0..8u32 {
        workgroup_memory_barrier_with_group_sync();
        if local_id.x >= (1 << i) {
            sum = sum.wrapping_add(sums.read((local_id.x - (1 << i)) as usize));
        }
        workgroup_memory_barrier_with_group_sync();
        sums.write(local_id.x as usize, sum);
    }

    workgroup_memory_barrier_with_group_sync();

    if active {
        // Add global offset from reduced array
        sum = reduced.read((reduced_offset + group_id) as usize);
        if local_id.x > 0 {
            sum = sum.wrapping_add(sums.read((local_id.x - 1) as usize));
        }

        for i in 0..ELEMENTS_PER_THREAD {
            let x = lds.at_mut(i as usize).read(local_id.x as usize);
            lds.at_mut(i as usize)
                .write(local_id.x as usize, x.wrapping_add(sum));
        }
    }

    // lds now contains exclusive prefix sum
    // Note: storing inclusive might be slightly cheaper here
    workgroup_memory_barrier_with_group_sync();
    if active {
        let bin_id = udiv(group_id, num_reduce_wg_per_bin);

        let bin_offset = bin_id * num_wgs;
        let base_index = umod(group_id, num_reduce_wg_per_bin) * ELEMENTS_PER_THREAD * WG;

        // Write back with transposition
        for i in 0..ELEMENTS_PER_THREAD {
            let data_index = base_index + i * WG + local_id.x;
            let col = (i * WG + local_id.x) / ELEMENTS_PER_THREAD;
            let row = (i * WG + local_id.x) % ELEMENTS_PER_THREAD;
            if data_index < num_wgs {
                counts.write(
                    (counts_offset + bin_offset + data_index) as usize,
                    lds.at(row as usize).read(col as usize),
                );
            }
        }
    }
}
