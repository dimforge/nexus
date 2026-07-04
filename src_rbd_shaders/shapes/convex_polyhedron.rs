//! Convex Polyhedron Shape Module
//!
//! This module provides geometric operations for convex polyhedra.
//! A convex polyhedron is defined by its vertex indices into a shared vertex buffer.

use crate::bounding_volumes::Aabb;
use crate::queries::PolygonalFeature;
use crate::{MAX_FLT, PaddedVector, Vector};
use khal_std::index::MaybeIndexUnchecked;

/// A convex polyhedron defined by vertex and face indices.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct ConvexPolyhedron {
    /// First vertex index.
    pub first_vtx_id: u32,
    /// End vertex index (exclusive).
    pub end_vtx_id: u32,
    /// First face index (3D only).
    pub first_face_id: u32,
    /// End face index (exclusive, 3D only).
    pub end_face_id: u32,
}

impl ConvexPolyhedron {
    /// Creates a new convex polyhedron with the given vertex range.
    #[inline]
    pub fn new(first_vtx_id: u32, end_vtx_id: u32, first_face_id: u32, end_face_id: u32) -> Self {
        Self {
            first_vtx_id,
            end_vtx_id,
            first_face_id,
            end_face_id,
        }
    }

    // TODO: cache the AABB (for example as the first two entries of the shape's index buffer)
    //       so it doesn't get recomputed at each frame?
    /// Computes the AABB of a convex polyhedron.
    pub fn aabb(&self, vertices: &[PaddedVector]) -> Aabb {
        let mut mins = *vertices.read(self.first_vtx_id as usize);
        let mut maxs = *vertices.read(self.first_vtx_id as usize);

        for i in self.first_vtx_id..self.end_vtx_id {
            mins = mins.min(*vertices.read(i as usize));
            maxs = maxs.max(*vertices.read(i as usize));
        }

        Aabb::new(mins, maxs)
    }

    /// Computes the local support point of a convex polyhedron.
    pub fn local_support_point(&self, vertices: &[PaddedVector], dir: Vector) -> Vector {
        let mut best_dot = -MAX_FLT;
        let mut best = Vector::default();

        for i in self.first_vtx_id..self.end_vtx_id {
            let val = vertices.at(i as usize).dot(dir);
            if val > best_dot {
                best_dot = val;
                best = *vertices.read(i as usize);
            }
        }

        best
    }

    /// Computes the support face of a 2D convex polygon.
    #[cfg(feature = "dim2")]
    pub fn support_face(&self, vertices: &[PaddedVector], dir: Vector) -> PolygonalFeature {
        use glamx::Vec2;

        let mut result = PolygonalFeature::default();
        let mut best = glamx::UVec2::ZERO;
        let mut best_dot = -MAX_FLT;
        let num_vertices = self.end_vtx_id - self.first_vtx_id;

        for i in 0..num_vertices {
            let j = (i + 1) % num_vertices;
            let a = *vertices.read((self.first_vtx_id + i) as usize);
            let b = *vertices.read((self.first_vtx_id + j) as usize);
            let ab = b - a;
            // CounterClockWise 2D normal.
            let n = Vec2::new(ab.y, -ab.x);
            let n_len = n.length();

            if n_len != 0.0 {
                let val = (n / n_len).dot(dir);
                if val > best_dot {
                    best_dot = val;
                    best = glamx::UVec2::new(i, j);
                }
            }
        }

        result
            .vertices
            .write(0, *vertices.read((self.first_vtx_id + best.x) as usize));
        result
            .vertices
            .write(1, *vertices.read((self.first_vtx_id + best.y) as usize));
        result.num_vertices = 2;
        result
    }

    /// Computes the support face of a 3D convex polyhedron.
    #[cfg(feature = "dim3")]
    pub fn support_face(
        &self,
        vertices: &[PaddedVector],
        indices: &[u32],
        dir: Vector,
    ) -> PolygonalFeature {
        let mut result = PolygonalFeature::default();
        let mut best = glamx::UVec3::ZERO;
        let mut best_dot = -MAX_FLT;
        let base_vid = glamx::UVec3::splat(self.first_vtx_id);

        // NOTE: we use fixed-size for loops to avoid miscompilation issues of while loops on MacOs.
        let num_faces = (self.end_face_id - self.first_face_id) / 3;
        for face_idx in 0..num_faces {
            let i = self.first_face_id + face_idx * 3;
            let vids = base_vid
                + glamx::UVec3::new(
                    indices.read(i as usize),
                    indices.read((i + 1) as usize),
                    indices.read((i + 2) as usize),
                );
            let a = *vertices.read(vids.x as usize);
            let b = *vertices.read(vids.y as usize);
            let c = *vertices.read(vids.z as usize);
            let ab = b - a;
            let ac = c - a;
            let n = ab.cross(ac);
            let n_len = n.length();

            if n_len != 0.0 {
                let val = (n / n_len).dot(dir);
                if val > best_dot {
                    best_dot = val;
                    best = vids;
                }
            }
        }

        result.vertices.write(0, *vertices.read(best.x as usize));
        result.vertices.write(1, *vertices.read(best.y as usize));
        result.vertices.write(2, *vertices.read(best.z as usize));
        result.num_vertices = 3;
        result
    }
}
