//! Polyline Shape Module
//!
//! Geometric operations for polylines (line strips). A polyline uses a BVH for
//! efficient queries.

use crate::PaddedVector;
use crate::bounding_volumes::Aabb;
use crate::shapes::segment::Segment;
use khal_std::index::MaybeIndexUnchecked;

/// A polyline (connected line segments) with BVH acceleration structure.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Polyline {
    /// Index of the root AABB in the vertex buffer.
    pub bvh_vtx_root_id: u32,
    /// The root AABB left-child index in the index buffer.
    pub bvh_idx_root_id: u32,
    /// The number of BVH nodes. Segment indices are stored after the last BVH node.
    pub bvh_node_len: u32,
    /// Root AABB of the polyline.
    pub root_aabb: Aabb,
}

/// BVH node indices for tree traversal.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct BvhIdx {
    /// Index to enter (left child). If 0xffffffff, this is a leaf node.
    pub entry_index: u32,
    /// Index to exit (skip to sibling or parent's sibling).
    pub exit_index: u32,
    /// Index of the shape (segment) in the polyline. Only valid for leaf nodes.
    pub shape_index: u32,
}

impl Polyline {
    /// Creates a new polyline.
    #[inline]
    pub fn new(
        bvh_vtx_root_id: u32,
        bvh_idx_root_id: u32,
        bvh_node_len: u32,
        root_aabb: Aabb,
    ) -> Self {
        Self {
            bvh_vtx_root_id,
            bvh_idx_root_id,
            bvh_node_len,
            root_aabb,
        }
    }

    /// Computes the AABB of a polyline.
    pub fn aabb(&self) -> Aabb {
        self.root_aabb
    }

    /// Gets the AABB of a BVH node.
    #[inline]
    pub fn bvh_node_aabb(&self, node_id: u32, vertices: &[PaddedVector]) -> Aabb {
        // Multiply by 2 since there are two values per AABB (min/max).
        let vid = (self.bvh_vtx_root_id + node_id * 2) as usize;
        Aabb::new(*vertices.read(vid), *vertices.read(vid + 1))
    }

    /// Gets the BVH node indices for tree traversal.
    #[inline]
    pub fn bvh_node_idx(&self, node_id: u32, indices: &[u32]) -> BvhIdx {
        let base_id = (self.bvh_idx_root_id + node_id * 3) as usize;
        BvhIdx {
            entry_index: indices.read(base_id),
            exit_index: indices.read(base_id + 1),
            shape_index: indices.read(base_id + 2),
        }
    }

    /// Gets a segment from the polyline by its index.
    #[inline]
    pub fn segment(&self, seg_id: u32, vertices: &[PaddedVector], indices: &[u32]) -> Segment {
        let base_id = (self.bvh_idx_root_id + self.bvh_node_len * 3 + seg_id * 2) as usize;
        let base_vid = (self.bvh_vtx_root_id + self.bvh_node_len * 2) as usize;
        let a = *vertices.read(base_vid + indices.read(base_id) as usize);
        let b = *vertices.read(base_vid + indices.read(base_id + 1) as usize);
        Segment::new(a, b)
    }
}
