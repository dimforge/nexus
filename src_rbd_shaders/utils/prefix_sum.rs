//! Prefix sum compute shader kernels
//!
//! GPU compute shader entry points for parallel exclusive prefix sum: given an
//! input array `[a0, a1, a2, ...]`, produces `[0, a0, a0+a1, a0+a1+a2, ...]`.

use khal_std::glamx::UVec3;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

use khal_std::index::MaybeIndexUnchecked;

/// Workgroup size: number of elements processed per workgroup.
pub const WORKGROUP_SIZE: usize = 256;

/// Performs exclusive prefix sum on a segment of the data array.
///
/// Each workgroup processes `WORKGROUP_SIZE` elements.
#[spirv_bindgen]
#[spirv(compute(threads(256)))]
pub fn gpu_prefix_sum_sweep(
    #[spirv(local_invocation_id)] thread_id: UVec3,
    #[spirv(workgroup_id)] block_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] aux: &mut [u32],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] batch_stride: &u32,
    #[spirv(workgroup)] workspace: &mut [u32; 256],
) {
    let batch_id = block_id.y as usize;
    let bid = block_id.x as usize;
    let tid = thread_id.x as usize;
    let data_len = *batch_stride as usize;
    let data_offset = batch_id * data_len;
    let aux_stride = (data_len + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    let aux_offset = batch_id * aux_stride;
    // Global index for this thread's element within the batch
    let elt_id = tid + bid * WORKGROUP_SIZE;

    // Phase 0: Load data into shared memory
    if elt_id < data_len {
        workspace.write(tid, data.read(data_offset + elt_id));
    } else {
        // Pad with zeros for out-of-bounds threads
        workspace.write(tid, 0);
    }

    // Phase 1: Up-sweep (reduce) - build tree of partial sums
    // After this phase, workspace[WORKGROUP_SIZE-1] contains the total sum.
    // NOTE: we always process the full WORKGROUP_SIZE (a power of two) instead of
    //       rounding up the actual block length. Out-of-bounds elements are zero-padded
    //       so the result is correct, and using a compile-time constant avoids non-uniform
    //       control flow that breaks workgroup barriers on WebGPU.
    {
        let mut d = WORKGROUP_SIZE / 2;
        let mut offset = 1usize;
        // log2(256) = 8 iterations
        for _ in 0..8u32 {
            workgroup_memory_barrier_with_group_sync();
            if tid < d {
                let ia = tid * 2 * offset + offset - 1;
                let ib = (tid * 2 + 1) * offset + offset - 1;

                let sum = workspace.read(ia) + workspace.read(ib);
                workspace.write(ib, sum);
            }

            d /= 2;
            offset *= 2;
        }
    }

    // Thread 0 saves the total sum and clears the root for down-sweep
    if tid == 0 {
        let total_sum = workspace.read(WORKGROUP_SIZE - 1);
        aux.write(aux_offset + bid, total_sum);
        workspace.write(WORKGROUP_SIZE - 1, 0);
    }

    // Phase 2: Down-sweep - propagate partial sums down the tree
    // Transforms the tree into an exclusive scan
    {
        let mut d = 1usize;
        let mut offset = WORKGROUP_SIZE / 2;
        // log2(256) = 8 iterations
        for _ in 0..8u32 {
            workgroup_memory_barrier_with_group_sync();
            if tid < d {
                let ia = tid * 2 * offset + offset - 1;
                let ib = (tid * 2 + 1) * offset + offset - 1;

                let a = workspace.read(ia);
                let b = workspace.read(ib);

                workspace.write(ia, b);
                workspace.write(ib, a + b);
            }

            d *= 2;
            offset /= 2;
        }
    }

    // Synchronize before writing results
    workgroup_memory_barrier_with_group_sync();

    // Write results back to global memory
    if elt_id < data_len {
        data.write(data_offset + elt_id, workspace.read(tid));
    }
}

/// Adds per-block offsets to complete a multi-block prefix sum.
///
/// Run after [`gpu_prefix_sum_sweep`] to fold each block's cumulative offset
/// into its elements.
#[spirv_bindgen]
#[spirv(compute(threads(256)))]
pub fn gpu_add_data_grp(
    #[spirv(global_invocation_id)] thread_id: UVec3,
    #[spirv(workgroup_id)] block_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] aux: &[u32],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] batch_stride: &u32,
) {
    let batch_id = block_id.y as usize;
    let tid = thread_id.x as usize;
    let bid = block_id.x as usize;
    let data_len = *batch_stride as usize;
    let data_offset = batch_id * data_len;
    let aux_stride = (data_len + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    let aux_offset = batch_id * aux_stride;

    if tid < data_len {
        // Add the cumulative sum from all previous blocks
        *data.at_mut(data_offset + tid) += aux.read(aux_offset + bid);
    }
}
