//! Utility functions and data structures for GPU-accelerated algorithms.
//!
//! This module provides general-purpose GPU algorithms that support the collision
//! detection and physics simulation pipelines.

pub use prefix_sum::{GpuPrefixSum, PrefixSumWorkspace};
pub use radix_sort::{RadixSort, RadixSortWorkspace};

mod prefix_sum;
mod radix_sort;
