//! GPU Radix Sort - Common Constants and Utilities
//!
//! Shared constants for the GPU radix sort implementation. The sort is stable
//! (preserves relative order of equal keys) and sorts both keys and associated
//! values.
//!
//! Mostly copied from [brush](https://github.com/ArthurBrussee/brush)
//! (Apache-2.0 license).

/// Workgroup size (threads per workgroup).
pub const WG: u32 = 256;

/// Number of bits processed per radix sort pass.
pub const BITS_PER_PASS: u32 = 4;

/// Bit mask for extracting digit (2^BITS_PER_PASS - 1).
pub const DIGIT_MASK: u32 = (1 << BITS_PER_PASS) - 1;

/// Number of histogram bins (2^BITS_PER_PASS).
pub const BIN_COUNT: u32 = 1 << BITS_PER_PASS;

/// Total histogram size across all threads in a workgroup.
pub const HISTOGRAM_SIZE: u32 = WG * BIN_COUNT;

/// Number of elements each thread processes.
pub const ELEMENTS_PER_THREAD: u32 = 4;

/// Total elements processed by one workgroup.
pub const BLOCK_SIZE: u32 = WG * ELEMENTS_PER_THREAD;

/// Total number of sort passes for 32-bit keys.
pub const NUM_PASSES: u32 = 32 / BITS_PER_PASS;

/// Radix sort configuration uniforms.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct SortUniforms {
    /// Bit shift amount for this sort pass (0, 4, 8, 12, ..., 28).
    pub shift: u32,
    pub max_keys_per_batch: u32,
    /// When non-zero, the scatter kernel also rearranges an auxiliary buffer
    /// alongside keys and values (used for batch_ids in flattened batched sort).
    pub has_aux: u32,
}

/// Integer division with ceiling (rounds up).
#[inline]
pub fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

/// Computes the number of workgroups needed for a given number of keys.
#[inline]
pub fn num_workgroups(num_keys: u32) -> u32 {
    div_ceil(num_keys, BLOCK_SIZE)
}

/// Extracts the 4-bit digit from a key at the given shift position.
#[inline]
pub fn extract_digit(key: u32, shift: u32) -> u32 {
    (key >> shift) & 0xF
}
