//! Small-batch segmented sort: a one-launch replacement for the multi-pass
//! radix cascade when the per-batch capacity is small.
//!
//! One workgroup per batch loads the batch's keys into shared memory,
//! bitonic-sorts them, and writes the result — no batch-id passes, no
//! ping-pong/histogram workspace. Bitonic networks are not stable, so the
//! comparator orders by `(key, original_index)`; the output is therefore
//! bit-identical to the stable radix path's. Capacity: `SMALL_SORT_MAX`
//! elements per batch, padded to a power of two with `u32::MAX` sentinels.

use khal_std::glamx::UVec3;
use khal_std::index::MaybeIndexUnchecked;
use khal_std::macros::{spirv, spirv_bindgen};
use khal_std::sync::workgroup_memory_barrier_with_group_sync;

use super::sorting::SortUniforms;

/// Maximum per-batch capacity handled by the small sort (shared-memory bound).
pub const SMALL_SORT_MAX: u32 = 128;
/// Workgroup width: each thread owns ≤ 2 slots of the padded array.
pub const SMALL_SORT_WG: u32 = 64;

/// Segmented bitonic sort: one workgroup per batch.
///
/// Reuses [`SortUniforms`]: `max_keys_per_batch` is the per-batch capacity
/// (`shift` / `has_aux` are unused). Like the radix path, only the first
/// `n_sort[batch]` elements of each batch participate; output slots past the
/// count are left untouched (consumers never read past the count).
#[spirv_bindgen]
#[spirv(compute(threads(64)))]
pub fn gpu_sort_small(
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(workgroup_id)] gid: UVec3,
    #[spirv(uniform, descriptor_set = 0, binding = 0)] config: &SortUniforms,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] n_sort: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] src: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] values: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] out: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] out_values: &mut [u32],
    #[spirv(workgroup)] keys: &mut [u32; 128],
    #[spirv(workgroup)] idxs: &mut [u32; 128],
) {
    let batch = gid.x;
    let tid = lid.x;
    let n = n_sort
        .read(batch as usize)
        .min(config.max_keys_per_batch)
        .min(SMALL_SORT_MAX);
    let base = (batch * config.max_keys_per_batch) as usize;
    if n == 0 {
        return;
    }

    // Padded (power-of-two) problem size.
    let mut p = 1u32;
    while p < n {
        p <<= 1;
    }

    // Load keys (+ identity index) into shared memory; pad with MAX sentinels.
    let mut i = tid;
    while i < p {
        let k = if i < n {
            src.read(base + i as usize)
        } else {
            u32::MAX
        };
        keys.write(i as usize, k);
        idxs.write(i as usize, i);
        i += SMALL_SORT_WG;
    }
    workgroup_memory_barrier_with_group_sync();

    // Bitonic sort, ascending on (key, original_index) — the index tiebreak
    // makes it stable.
    let mut k = 2u32;
    while k <= p {
        let mut j = k >> 1;
        while j > 0 {
            let mut i = tid;
            while i < p {
                let partner = i ^ j;
                if partner > i {
                    let ascending = (i & k) == 0;
                    let (ka, ia) = (keys.read(i as usize), idxs.read(i as usize));
                    let (kb, ib) = (keys.read(partner as usize), idxs.read(partner as usize));
                    // Lexicographic (key, index) "a greater than b".
                    let a_gt_b = ka > kb || (ka == kb && ia > ib);
                    if a_gt_b == ascending {
                        keys.write(i as usize, kb);
                        idxs.write(i as usize, ib);
                        keys.write(partner as usize, ka);
                        idxs.write(partner as usize, ia);
                    }
                }
                i += SMALL_SORT_WG;
            }
            workgroup_memory_barrier_with_group_sync();
            j >>= 1;
        }
        k <<= 1;
    }

    // Write back: keys directly, values gathered through the sorted index.
    let mut i = tid;
    while i < n {
        out.write(base + i as usize, keys.read(i as usize));
        let src_idx = idxs.read(i as usize) as usize;
        out_values.write(base + i as usize, values.read(base + src_idx));
        i += SMALL_SORT_WG;
    }
}
