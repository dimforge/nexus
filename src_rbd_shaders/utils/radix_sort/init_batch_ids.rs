//! Initialize Batch IDs and Key/Value Buffers for Flattened Batched Radix Sort
//!
//! Prepares the flattened sort by:
//! 1. Setting `batch_ids[i] = i / per_batch_size` for all elements
//! 2. Copying active keys/values from input to output
//! 3. Writing sentinel keys (0xFFFFFFFF) for inactive elements so they sort
//!    to the end of their batch

use khal_std::glamx::UVec3;
use khal_std::macros::{spirv, spirv_bindgen};

use khal_std::index::MaybeIndexUnchecked;

use super::sorting::SortUniforms;

/// GPU kernel to initialize batch IDs and copy keys/values with sentinel handling.
///
/// `config.max_keys_per_batch` is the per-batch allocated size (not the flattened total).
/// `config.shift` is repurposed to carry `num_batches` (shift is unused by this kernel).
/// `num_keys_arr[batch_id]` gives the number of active keys in each batch.
/// Elements beyond the active count get sentinel keys (0xFFFFFFFF).
#[spirv_bindgen]
#[spirv(compute(threads(256)))]
pub fn gpu_init_sort_batched(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] config: &SortUniforms,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] num_keys_arr: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] input_keys: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] input_values: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] out_keys: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] out_values: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] batch_ids: &mut [u32],
) {
    let i = gid.x;
    let per_batch = config.max_keys_per_batch;
    let num_batches = config.shift; // repurposed: shift carries num_batches for this kernel
    let total = num_batches * per_batch;

    if i >= total {
        return;
    }

    let batch_id = i / per_batch;
    let local_id = i - batch_id * per_batch;
    let num_keys = num_keys_arr.read(batch_id as usize);

    batch_ids.write(i as usize, batch_id);

    if local_id < num_keys {
        out_keys.write(i as usize, input_keys.read(i as usize));
        out_values.write(i as usize, input_values.read(i as usize));
    } else {
        out_keys.write(i as usize, !0u32);
        out_values.write(i as usize, 0);
    }
}
