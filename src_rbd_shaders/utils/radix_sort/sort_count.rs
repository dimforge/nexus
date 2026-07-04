//! Radix Sort Count (Histogram) Kernel
//!
//! First pass of radix sort: computes per-workgroup histograms for the current
//! 4-bit digit.

use khal_std::glamx::UVec3;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

use khal_std::index::MaybeIndexUnchecked;
use khal_std::sync::atomic_add_u32_workgroup;

use super::sorting::{BIN_COUNT, BLOCK_SIZE, ELEMENTS_PER_THREAD, SortUniforms, WG, div_ceil};

/// Radix sort count (histogram) kernel.
///
/// Computes per-workgroup histograms for the current 4-bit digit.
#[spirv_bindgen]
#[spirv(compute(threads(256)))]
pub fn gpu_sort_count(
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup_id)] gid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] config: &SortUniforms,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] num_keys_arr: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] src: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] counts: &mut [u32],
    #[spirv(workgroup)] histogram: &mut [u32; 16],
) {
    let group_id = gid.x;
    let batch_id = gid.y;
    let num_keys = num_keys_arr.read(batch_id as usize);
    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);
    let max_wgs_per_batch = div_ceil(config.max_keys_per_batch, BLOCK_SIZE);
    let key_offset = batch_id * config.max_keys_per_batch;
    let counts_offset = batch_id * BIN_COUNT * max_wgs_per_batch;

    // Filter-out out of bounds workgroups but don’t
    // just early-exit to keep uniform control flow
    // wrt. the barriers (for web compatibility).
    let active = group_id < num_wgs;

    if active && local_id.x < BIN_COUNT {
        histogram.write(local_id.x as usize, 0);
    }

    workgroup_memory_barrier_with_group_sync();

    if active {
        let wg_block_start = BLOCK_SIZE * group_id;
        let shift_bit = config.shift;
        let mut data_index = wg_block_start + local_id.x;

        for _ in 0..ELEMENTS_PER_THREAD {
            if data_index < num_keys {
                let local_key = (src.read((key_offset + data_index) as usize) >> shift_bit) & 0xf;
                atomic_add_u32_workgroup(histogram.at_mut(local_key as usize), 1);
            }
            data_index += WG;
        }
    }

    workgroup_memory_barrier_with_group_sync();

    if active && local_id.x < BIN_COUNT {
        counts.write(
            (counts_offset + local_id.x * num_wgs + group_id) as usize,
            histogram.read(local_id.x as usize),
        );
    }
}
