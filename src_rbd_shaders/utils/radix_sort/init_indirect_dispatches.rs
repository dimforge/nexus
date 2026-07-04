//! Initialize Indirect Dispatch Arguments for Radix Sort
//!
//! Sets up the indirect dispatch arguments based on the number of keys to sort.

use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};

use super::sorting::{BIN_COUNT, BLOCK_SIZE, div_ceil};

/// GPU kernel to initialize indirect dispatch arguments for radix sort.
#[spirv_bindgen]
#[spirv(compute(threads(1)))]
pub fn gpu_init_sort_dispatch(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] num_keys_arr: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] sort_dispatch: &mut [u32; 3],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] reduce_dispatch: &mut [u32; 3],
) {
    let mut num_keys = 0;
    for i in 0..num_keys_arr.len() {
        num_keys = num_keys.max(num_keys_arr[i]);
    }

    let num_batches = num_keys_arr.len() as u32;
    let num_wgs = div_ceil(num_keys, BLOCK_SIZE);
    let reduce_wgs = BIN_COUNT * div_ceil(num_wgs, BLOCK_SIZE);

    sort_dispatch.write(0, num_wgs);
    sort_dispatch.write(1, num_batches);
    sort_dispatch.write(2, 1);
    reduce_dispatch.write(0, reduce_wgs);
    reduce_dispatch.write(1, num_batches);
    reduce_dispatch.write(2, 1);
}
