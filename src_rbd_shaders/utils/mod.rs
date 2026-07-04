//! Utility algorithms.

mod basis;
mod indices;
pub mod linalg; // TODO: this should be moved to vortx
pub mod prefix_sum;
pub mod radix_sort;
mod slice;

pub use basis::orthonormal_basis3;
pub use indices::BatchIndices;
pub use slice::{Slice, SliceMut};

/// Division with ceiling (signed).
pub fn div_ceil(x: i32, y: i32) -> i32 {
    (x + y - 1) / y
}

/// Division with ceiling (unsigned).
pub fn udiv_ceil(x: u32, y: u32) -> u32 {
    (x + y - 1) / y
}
