//! Radix Sort Scan Kernel
//!
//! Computes exclusive prefix sums of the reduced histogram counts.
//! This determines the global offsets for each bin.

use khal_std::glamx::UVec3;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

use khal_std::index::MaybeIndexUnchecked;

use super::sorting::{BIN_COUNT, BLOCK_SIZE, ELEMENTS_PER_THREAD, WG, div_ceil};

/// Radix sort scan kernel.
///
/// Computes exclusive prefix sums of reduced histogram counts.
#[spirv_bindgen]
#[spirv(compute(threads(256)))]
pub fn gpu_sort_scan(
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup_id)] workgroup_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] num_keys_arr: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] reduced: &mut [u32],
    #[spirv(workgroup)] sums: &mut [u32; 256],
    #[spirv(workgroup)] lds: &mut [[u32; 256]; 4],
) {
    let batch_id = workgroup_id.y;
    let num_keys = num_keys_arr.read(batch_id as usize);
    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);
    let num_reduce_wgs = BIN_COUNT * div_ceil(num_wgs, BLOCK_SIZE);
    let reduced_offset = batch_id * BLOCK_SIZE;

    // Load with transposition
    for i in 0..ELEMENTS_PER_THREAD {
        let data_index = i * WG + local_id.x;
        let col = (i * WG + local_id.x) / ELEMENTS_PER_THREAD;
        let row = (i * WG + local_id.x) % ELEMENTS_PER_THREAD;
        lds.at_mut(row as usize).write(
            col as usize,
            reduced.read((reduced_offset + data_index) as usize),
        );
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

    sum = 0;
    if local_id.x > 0 {
        sum = sums.read((local_id.x - 1) as usize);
    }

    for i in 0..ELEMENTS_PER_THREAD {
        let elt = lds.at_mut(i as usize).read(local_id.x as usize);
        lds.at_mut(i as usize)
            .write(local_id.x as usize, elt.wrapping_add(sum));
    }

    // lds now contains exclusive prefix sum
    workgroup_memory_barrier_with_group_sync();

    // Write back with transposition
    for i in 0..ELEMENTS_PER_THREAD {
        let data_index = i * WG + local_id.x;
        let col = (i * WG + local_id.x) / ELEMENTS_PER_THREAD;
        let row = (i * WG + local_id.x) % ELEMENTS_PER_THREAD;
        if data_index < num_reduce_wgs {
            reduced.write(
                (reduced_offset + data_index) as usize,
                lds.at(row as usize).read(col as usize),
            );
        }
    }
}
