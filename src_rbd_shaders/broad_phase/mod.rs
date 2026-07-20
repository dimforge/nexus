//! Broad phase collision detection algorithms.
//!
//! Algorithms for quickly pruning non-colliding pairs:
//! - Brute force (O(n^2), for small scenes)
//! - LBVH (Linear Bounding Volume Hierarchy, for large scenes)

// Data structures and algorithms
mod brute_force;
mod lbvh;

// GPU compute shader kernels
mod narrow_phase;

use glamx::UVec2;
// Re-export non-spirv items explicitly to avoid ambiguous glob re-exports.
// The div_ceil functions have different signatures (u32 vs i32) so we pick one.
// Spirv-only items (functions and generated structs) are re-exported via glob.
pub use brute_force::*;
pub use lbvh::*;
#[cfg(feature = "dim2")]
pub use lbvh::{expand_bits_2d, morton_2d};
pub use narrow_phase::*;

#[derive(Copy, Clone, PartialEq, Eq, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct CollisionPair {
    pub colliders: UVec2,
}
