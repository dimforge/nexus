//! Radix Sort Scatter Kernel
//!
//! Final pass of radix sort: scatters keys and values to their sorted positions.

use khal_std::glamx::UVec3;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

use khal_std::{iter::StepRng, sync::atomic_add_u32_workgroup};

use khal_std::index::MaybeIndexUnchecked;

use super::sorting::{
    BIN_COUNT, BITS_PER_PASS, BLOCK_SIZE, ELEMENTS_PER_THREAD, SortUniforms, WG, div_ceil,
};

/// Radix sort scatter kernel.
///
/// Scatters keys and values to their sorted positions using computed prefix sums.
/// When `config.has_aux != 0`, also scatters an auxiliary buffer (used for batch_ids
/// in flattened batched sort).
#[spirv_bindgen]
#[spirv(compute(threads(256)))]
pub fn gpu_sort_scatter(
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(workgroup_id)] gid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] config: &SortUniforms,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] num_keys_arr: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] src: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] values: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] counts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] out: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] out_values: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] aux: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] out_aux: &mut [u32],
    #[spirv(workgroup)] lds_scratch: &mut [u32; 256],
    #[spirv(workgroup)] bin_offset_cache: &mut [u32; 256],
    #[spirv(workgroup)] local_histogram: &mut [u32; 16],
) {
    let group_id = gid.x;
    let batch_id = gid.y;
    let local_id = lid.x;
    let has_aux = config.has_aux;

    // SAFETY: all indices are bounded by the algorithm's structure:
    // - local_id < 256 (workgroup size)
    // - key_index < 16 (masked to 4 bits)
    // - key_offset < 256 (masked to 8 bits)
    // - data_index/total_offset are checked against num_keys before use
    // - storage buffer sizes match the algorithm's requirements
    // Keeping the bounds checks apparently makes the shader too complex and results in
    // a DeviceLost error on some NVidia graphics cards.
    let num_keys = num_keys_arr.read(batch_id as usize);
    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);
    let max_wgs_per_batch = div_ceil(config.max_keys_per_batch, BLOCK_SIZE);

    let key_offset = batch_id * config.max_keys_per_batch;
    let counts_offset = batch_id * BIN_COUNT * max_wgs_per_batch;

    // Filter-out out of bounds workgroups but don't
    // just early-exit to keep uniform control flow
    // wrt. the barriers (for web compatibility).
    let active = group_id < num_wgs;

    // Load global histogram offsets for this workgroup
    if local_id < BIN_COUNT && active {
        bin_offset_cache.write(
            local_id as usize,
            counts.read((counts_offset + local_id * num_wgs + group_id) as usize),
        );
    }
    workgroup_memory_barrier_with_group_sync();

    let wg_block_start = BLOCK_SIZE * group_id;
    let mut data_index = wg_block_start + local_id;

    for _ in 0..ELEMENTS_PER_THREAD {
        // Reset local histogram
        if local_id < BIN_COUNT {
            local_histogram.write(local_id as usize, 0);
        }

        let mut local_key = !0u32;
        let mut local_value = 0u32;
        let mut local_aux = 0u32;

        if active && data_index < num_keys {
            local_key = src.read((key_offset + data_index) as usize);
            local_value = values.read((key_offset + data_index) as usize);
            if has_aux != 0 {
                local_aux = aux.read((key_offset + data_index) as usize);
            }
        }

        // Hierarchical sorting: 2 passes of 2-bit sorts
        for bit_shift in StepRng::new(0..BITS_PER_PASS, 2) {
            let key_index = (local_key >> config.shift) & 0xf;
            let bit_key = (key_index >> bit_shift) & 3;
            let mut packed_histogram = 1u32 << (bit_key * 8);

            // Workgroup prefix sum
            let mut sum = packed_histogram;
            lds_scratch.write(local_id as usize, sum);
            for i in 0..8u32 {
                workgroup_memory_barrier_with_group_sync();
                if local_id >= (1 << i) {
                    sum = sum.wrapping_add(lds_scratch.read((local_id - (1 << i)) as usize));
                }
                workgroup_memory_barrier_with_group_sync();
                lds_scratch.write(local_id as usize, sum);
            }
            workgroup_memory_barrier_with_group_sync();

            packed_histogram = lds_scratch.read((WG - 1) as usize);
            packed_histogram = (packed_histogram << 8)
                .wrapping_add(packed_histogram << 16)
                .wrapping_add(packed_histogram << 24);
            let mut local_sum = packed_histogram;
            if local_id > 0 {
                local_sum = local_sum.wrapping_add(lds_scratch.read((local_id - 1) as usize));
            }
            let key_offset = (local_sum >> (bit_key * 8)) & 0xff;

            // Ensure all threads finished reading prefix sum results from
            // lds_scratch before we reuse it for the rearrangement below.
            workgroup_memory_barrier_with_group_sync();

            // Rearrange keys (reusing lds_scratch since prefix sum data is no longer needed)
            lds_scratch.write(key_offset as usize, local_key);
            workgroup_memory_barrier_with_group_sync();
            local_key = lds_scratch.read(local_id as usize);
            workgroup_memory_barrier_with_group_sync();
            // Rearrange values
            lds_scratch.write(key_offset as usize, local_value);
            workgroup_memory_barrier_with_group_sync();
            local_value = lds_scratch.read(local_id as usize);
            workgroup_memory_barrier_with_group_sync();
            // Rearrange aux (conditional — uniform branch, all threads take same path)
            if has_aux != 0 {
                lds_scratch.write(key_offset as usize, local_aux);
                workgroup_memory_barrier_with_group_sync();
                local_aux = lds_scratch.read(local_id as usize);
                workgroup_memory_barrier_with_group_sync();
            }
        }

        // Update local histogram
        let key_index = (local_key >> config.shift) & 0xf;
        atomic_add_u32_workgroup(local_histogram.at_mut(key_index as usize), 1);
        workgroup_memory_barrier_with_group_sync();

        // Compute histogram prefix sum
        let mut histogram_local_sum = 0u32;
        if local_id < BIN_COUNT {
            histogram_local_sum = local_histogram.read(local_id as usize);
        }

        let mut histogram_prefix_sum = histogram_local_sum;
        if local_id < BIN_COUNT {
            lds_scratch.write(local_id as usize, histogram_prefix_sum);
        }

        for i in 0..4u32 {
            workgroup_memory_barrier_with_group_sync();
            if local_id >= (1 << i) && local_id < BIN_COUNT {
                histogram_prefix_sum = histogram_prefix_sum
                    .wrapping_add(lds_scratch.read((local_id - (1 << i)) as usize));
            }
            workgroup_memory_barrier_with_group_sync();
            if local_id < BIN_COUNT {
                lds_scratch.write(local_id as usize, histogram_prefix_sum);
            }
        }
        let global_offset = if active {
            bin_offset_cache.read(key_index as usize)
        } else {
            0
        };

        workgroup_memory_barrier_with_group_sync();

        // Compute output position and write
        if active {
            let mut local_offset = local_id;
            if key_index > 0 {
                local_offset =
                    local_offset.wrapping_sub(lds_scratch.read((key_index - 1) as usize));
            }
            let total_offset = global_offset + local_offset;

            if total_offset < num_keys {
                out.write((key_offset + total_offset) as usize, local_key);
                out_values.write((key_offset + total_offset) as usize, local_value);
                if has_aux != 0 {
                    out_aux.write((key_offset + total_offset) as usize, local_aux);
                }
            }

            // Update offsets for next iteration
            if local_id < BIN_COUNT {
                let curr = bin_offset_cache.read(local_id as usize);
                let hist = local_histogram.read(local_id as usize);
                bin_offset_cache.write(local_id as usize, curr.wrapping_add(hist));
            }
        }
        workgroup_memory_barrier_with_group_sync();
        data_index += WG;
    }
}
