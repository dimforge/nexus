//! Point projection queries.

use crate::math::Point;

/// Result of a point projection query, GPU-compatible layout.
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GpuProjectionResult {
    /// The projected point on the shape's surface.
    ///
    /// This is the point on the shape's boundary that is closest to the query point.
    pub point: Point,
    /// Non-zero if the query point was inside the shape.
    pub is_inside: u32,

    #[cfg(feature = "dim2")]
    /// Padding for 2D builds to maintain alignment (unused).
    pub padding: u32,
}
