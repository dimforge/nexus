//! Triangle Mesh Shape Module
//!
//! Geometric operations for triangle meshes. A trimesh uses a BVH for efficient
//! queries.

use crate::bounding_volumes::Aabb;
use crate::shapes::triangle::Triangle;
use crate::{PaddedVector, Vector};
use khal_std::index::MaybeIndexUnchecked;

/// A triangle mesh with BVH acceleration structure.
#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch_is_gpu), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct TriMesh {
    /// Index of the root AABB in the vertex buffer.
    pub bvh_vtx_root_id: u32,
    /// The root AABB left-child index in the index buffer.
    pub bvh_idx_root_id: u32,
    /// The number of BVH nodes. Triangle indices are stored after the last BVH node.
    pub bvh_node_len: u32,
    pub num_triangles: u32, // TODO: is this really needed?
    pub num_vertices: u32,
    /// Root AABB of the mesh.
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
    /// Index of the shape (triangle) in the mesh. Only valid for leaf nodes.
    pub shape_index: u32,
}

impl TriMesh {
    /// Creates a new triangle mesh.
    #[inline]
    pub fn new(
        bvh_vtx_root_id: u32,
        bvh_idx_root_id: u32,
        bvh_node_len: u32,
        num_triangles: u32,
        num_vertices: u32,
        root_aabb: Aabb,
    ) -> Self {
        Self {
            bvh_vtx_root_id,
            bvh_idx_root_id,
            bvh_node_len,
            num_triangles,
            num_vertices,
            root_aabb,
        }
    }

    /// Computes the AABB of a trimesh.
    pub fn aabb(&self) -> Aabb {
        self.root_aabb
    }

    /// Gets the AABB of a BVH node.
    #[inline]
    pub fn bvh_node_aabb(&self, vertices: &[PaddedVector], node_id: u32) -> Aabb {
        // Multiply by 2 since there are two values per AABB (min/max).
        let vid = (self.bvh_vtx_root_id + node_id * 2) as usize;
        Aabb::new(*vertices.read(vid), *vertices.read(vid + 1))
    }

    /// Gets the BVH node indices for tree traversal.
    #[inline]
    pub fn bvh_node_idx(&self, indices: &[u32], node_id: u32) -> BvhIdx {
        let base_id = (self.bvh_idx_root_id + node_id * 3) as usize;
        BvhIdx {
            entry_index: indices.read(base_id),
            exit_index: indices.read(base_id + 1),
            shape_index: indices.read(base_id + 2),
        }
    }

    /// Gets the vertex indices for a triangle.
    #[inline]
    pub fn triangle_vids(&self, idx: &[u32], tri_id: u32) -> UVec3 {
        let base_id = (self.bvh_idx_root_id + self.bvh_node_len * 3 + tri_id * 3) as usize;
        let base_vid = self.bvh_vtx_root_id + self.bvh_node_len * 2;
        let a = base_vid + idx.read(base_id);
        let b = base_vid + idx.read(base_id + 1);
        let c = base_vid + idx.read(base_id + 2);
        UVec3::new(a, b, c)
    }

    /// Gets a triangle from the mesh by its index.
    #[inline]
    pub fn triangle(&self, idx: &[u32], vtx: &[PaddedVector], tri_id: u32) -> Triangle {
        let vids = self.triangle_vids(idx, tri_id);
        let a = vtx.read(vids.x as usize).0;
        let b = vtx.read(vids.y as usize).0;
        let c = vtx.read(vids.z as usize).0;
        Triangle::new(a, b, c)
    }

    /// Gets the pseudo-normal for a feature on a triangle.
    ///
    /// Pseudo-normals are used to determine inside/outside status during
    /// point projection onto a mesh.
    #[inline]
    pub fn pseudo_normal(
        &self,
        idx: &[u32],
        vtx: &[PaddedVector],
        tri_id: u32,
        feat_type: u32,
        feat_id: u32,
    ) -> Vector {
        if feat_type == FEATURE_VERTEX {
            let vids = self.triangle_vids(idx, tri_id);
            let vid = match feat_id {
                0 => vids.x,
                1 => vids.y,
                _ => vids.z,
            };
            return vtx.read((vid + self.num_vertices) as usize).0;
        }

        if feat_type == FEATURE_EDGE {
            let base_vid = self.bvh_vtx_root_id
                // Two points per BVH node.
                + self.bvh_node_len * 2
                // Triangle vertices.
                + self.num_vertices
                // One pseudo-normal per vertex.
                + self.num_vertices
                // Three pseudo-normals per triangle (one per edge).
                + tri_id * 3;
            return vtx.read((base_vid + feat_id) as usize).0;
        }

        #[cfg(feature = "dim3")]
        if feat_type == FEATURE_SOLID || feat_type == FEATURE_FACE {
            let tri = self.triangle(idx, vtx, tri_id);
            let ab = tri.b - tri.a;
            let ac = tri.c - tri.a;
            return ab.cross(ac);
        }

        Vector::ZERO
    }

    /// Projects a local point onto the triangle mesh and returns the
    /// projection result with inside/outside information.
    pub fn project_local_point(
        &self,
        idx: &[u32],
        vtx: &[PaddedVector],
        pt: Vector,
        max_dist: f32,
    ) -> (ProjectionResult, bool) {
        let mut curr = 0u32;
        let mut best = max_dist;
        let mut best_proj = ProjectionWithLocation::solid(pt);
        let mut best_tri_id = u32::MAX;

        for _ in 0..self.bvh_node_len {
            if curr >= self.bvh_node_len {
                break;
            }
            let node_idx = self.bvh_node_idx(idx, curr);
            if node_idx.entry_index == 0xFFFFFFFF {
                // This is a leaf.
                let tri = self.triangle(idx, vtx, node_idx.shape_index);
                let proj = triangle_project_local_point_and_get_location(&tri, pt);
                let dist = (proj.point - pt).length();
                if dist <= best {
                    best = dist;
                    best_proj = proj;
                    best_tri_id = node_idx.shape_index;
                }

                // Continue traversal.
                curr = node_idx.exit_index;
            } else {
                let aabb = self.bvh_node_aabb(vtx, curr);
                // HACK: convert to parry AABB. We should just use the same AABB type.
                use crate::parry::query::PointQuery;
                let paabb = parry::bounding_volume::Aabb::new(aabb.mins, aabb.maxs);
                let dist = paabb.distance_to_local_point(pt, true);
                if dist <= best {
                    curr = node_idx.entry_index;
                } else {
                    curr = node_idx.exit_index;
                }
            }
        }

        if best_tri_id != u32::MAX {
            let pn =
                self.pseudo_normal(idx, vtx, best_tri_id, best_proj.feature_type, best_proj.id);
            let is_inside = pn.dot(pt - best_proj.point) <= 0.0;
            (ProjectionResult::new(best_proj.point, is_inside), true)
        } else {
            (ProjectionResult::new(Vector::ZERO, false), false)
        }
    }
}

/*
 * UVec3 import for vertex index triples.
 */
use crate::queries::{FEATURE_EDGE, FEATURE_VERTEX, ProjectionResult, ProjectionWithLocation};
#[cfg(feature = "dim3")]
use crate::queries::{FEATURE_FACE, FEATURE_SOLID};
use glamx::{UVec3, Vec2, Vec3};

/*
 * Triangle point projection with location information.
 * TODO: everything below can probably be shared with parry
 */

struct ProjectionInfo {
    feature: u32,
    params: Vec3,
}

/*
 * Constants for edge/face voronoi region identification.
 */
const AB: u32 = 0;
const AC: u32 = 1;
const BC: u32 = 2;
const FACE_CW: u32 = 3;
const FACE_CCW: u32 = 4;

/// Checks on which edge voronoi region the point is.
#[inline]
fn stable_check_edges_voronoi(
    ab: Vector,
    ac: Vector,
    bc: Vector,
    ap: Vector,
    bp: Vector,
    _cp: Vector,
    ab_ap: f32,
    ab_bp: f32,
    ac_ap: f32,
    ac_cp: f32,
    ac_bp: f32,
    ab_cp: f32,
) -> ProjectionInfo {
    #[cfg(feature = "dim3")]
    {
        let n = ab.cross(ac);
        let vc = n.dot(ab.cross(ap));
        if vc < 0.0 && ab_ap >= 0.0 && ab_bp <= 0.0 {
            return ProjectionInfo {
                feature: AB,
                params: Vec3::ZERO,
            };
        }

        let vb = -n.dot(ac.cross(_cp));
        if vb < 0.0 && ac_ap >= 0.0 && ac_cp <= 0.0 {
            return ProjectionInfo {
                feature: AC,
                params: Vec3::ZERO,
            };
        }

        let va = n.dot(bc.cross(bp));
        if va < 0.0 && ac_bp - ab_bp >= 0.0 && ab_cp - ac_cp >= 0.0 {
            return ProjectionInfo {
                feature: BC,
                params: Vec3::ZERO,
            };
        }

        if n.dot(ap) >= 0.0 {
            ProjectionInfo {
                feature: FACE_CW,
                params: Vec3::new(va, vb, vc),
            }
        } else {
            ProjectionInfo {
                feature: FACE_CCW,
                params: Vec3::new(va, vb, vc),
            }
        }
    }

    #[cfg(feature = "dim2")]
    {
        let _ = (
            ab, ac, bc, ap, bp, _cp, ab_ap, ab_bp, ac_ap, ac_cp, ac_bp, ab_cp,
        );
        // TODO: 2D case not fully implemented
        ProjectionInfo {
            feature: FACE_CW,
            params: Vec3::ZERO,
        }
    }
}

/// Projects a point onto a triangle and returns location information
/// (which feature — vertex, edge, or face — the projection lies on).
fn triangle_project_local_point_and_get_location(
    shape: &Triangle,
    pt: Vector,
) -> ProjectionWithLocation {
    let a = shape.a;
    let b = shape.b;
    let c = shape.c;

    let ab = b - a;
    let ac = c - a;
    let ap = pt - a;

    let ab_ap = ab.dot(ap);
    let ac_ap = ac.dot(ap);

    if ab_ap <= 0.0 && ac_ap <= 0.0 {
        // Voronoi region of `a`.
        return ProjectionWithLocation::vertex(a, 0, false);
    }

    let bp = pt - b;
    let ab_bp = ab.dot(bp);
    let ac_bp = ac.dot(bp);

    if ab_bp >= 0.0 && ac_bp <= ab_bp {
        // Voronoi region of `b`.
        return ProjectionWithLocation::vertex(b, 1, false);
    }

    let cp = pt - c;
    let ab_cp = ab.dot(cp);
    let ac_cp = ac.dot(cp);

    if ac_cp >= 0.0 && ab_cp <= ac_cp {
        // Voronoi region of `c`.
        return ProjectionWithLocation::vertex(c, 2, false);
    }

    let bc = c - b;
    let proj = stable_check_edges_voronoi(
        ab, ac, bc, ap, bp, cp, ab_ap, ab_bp, ac_ap, ac_cp, ac_bp, ab_cp,
    );

    match proj.feature {
        AB => {
            // Voronoi region of `ab`.
            let v = ab_ap / ab.dot(ab);
            let bcoords = Vec2::new(1.0 - v, v);
            let res = a + ab * v;
            return ProjectionWithLocation::edge(res, bcoords, 0, false);
        }
        AC => {
            // Voronoi region of `ac`.
            let w = ac_ap / ac.dot(ac);
            let bcoords = Vec2::new(1.0 - w, w);
            let res = a + ac * w;
            return ProjectionWithLocation::edge(res, bcoords, 2, false);
        }
        BC => {
            // Voronoi region of `bc`.
            let w = bc.dot(bp) / bc.dot(bc);
            let bcoords = Vec2::new(1.0 - w, w);
            let res = b + bc * w;
            return ProjectionWithLocation::edge(res, bcoords, 1, false);
        }
        FACE_CW | FACE_CCW => {
            // Voronoi region of the face.
            // NOTE: in some cases, numerical instability
            // may result in the denominator being zero
            // when the triangle is nearly degenerate.
            if proj.params.x + proj.params.y + proj.params.z != 0.0 {
                let denom = 1.0 / (proj.params.x + proj.params.y + proj.params.z);
                let v = proj.params.y * denom;
                let w = proj.params.z * denom;
                let bcoords = Vec3::new(1.0 - v - w, v, w);
                let res = a + ab * v + ac * w;
                return ProjectionWithLocation::face(res, bcoords, proj.feature, false);
            }
        }
        _ => { /* fall through to solid case */ }
    }

    // Special treatment if we work in 2D because in this case we really
    // are inside of the object.
    // NOTE: this should never be reached in 3D.
    ProjectionWithLocation::solid(pt)
}
