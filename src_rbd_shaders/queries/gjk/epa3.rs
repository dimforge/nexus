//! EPA (Expanding Polytope Algorithm) Module (3D)
//!
//! This module provides the EPA algorithm for computing penetration depth
//! and contact normal for intersecting convex shapes in 3D.

use crate::queries::gjk::VoronoiSimplex;
use crate::queries::gjk::cso_point::{CsoPoint, EPS_TOL, FLT_EPS};
use crate::queries::gjk::gjk::cso_point_from_shapes;
use crate::queries::projection;
use crate::shapes::{Shape, Triangle};
use crate::{MAX_FLT, Pad, Pose};
use glamx::Vec3;
use khal_std::index::MaybeIndexUnchecked;

const PADDING: u32 = 0;

// TODO: find the ideal values.
const MAX_VERTICES_LEN: usize = 32;
const MAX_FACES_LEN: usize = 64;
const MAX_SILHOUETTE_LEN: usize = 32;
const MAX_HEAP_LEN: usize = 64;
const MAX_STACK_LEN: usize = 32;

#[derive(Clone, Copy, Default)]
struct FaceId {
    id: u32,
    neg_dist: f32,
}

struct OptionFaceId {
    face_id: FaceId,
    valid: bool,
}

fn face_id_new(id: u32, neg_dist: f32) -> OptionFaceId {
    if neg_dist > EPS_TOL {
        OptionFaceId {
            face_id: FaceId::default(),
            valid: false,
        }
    } else {
        OptionFaceId {
            face_id: FaceId { id, neg_dist },
            valid: true,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Face {
    pts: [u32; 4], // We only need 3 elements, but 4 are needed for Spv -> WGSL conversion.
    adj: [u32; 4], // We only need 3 elements, but 4 are needed for Spv -> WGSL conversion.
    normal: Vec3,
    deleted: bool,
    bcoords: Vec3,
    // Explicit padding for the Spv -> WGSL struct layout.
    _padding: u32,
}

struct FaceWithProj {
    face: Face,
    proj_inside: bool,
}

#[derive(Clone, Copy, Default)]
struct SilhouetteEdge {
    face_id: u32,
    opp_pt_id: u32,
}

/// EPA result containing closest points and contact normal.
#[derive(Clone, Copy, Default)]
pub struct EpaResult {
    pub pt_a: Vec3,
    pub pt_b: Vec3,
    pub normal: Vec3,
    pub valid: bool,
}

impl EpaResult {
    pub fn new(pt_a: Vec3, pt_b: Vec3, normal: Vec3, valid: bool) -> Self {
        Self {
            pt_a,
            pt_b,
            normal,
            valid,
        }
    }
}

fn none() -> EpaResult {
    EpaResult::new(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, false)
}

/// EPA state for 3D collision detection.
pub struct Epa3 {
    vertices: [CsoPoint; MAX_VERTICES_LEN],
    vertices_len: usize,
    faces: [Face; MAX_FACES_LEN],
    faces_len: usize,
    silhouette: [SilhouetteEdge; MAX_SILHOUETTE_LEN],
    silhouette_len: usize,
    heap: [FaceId; MAX_HEAP_LEN],
    heap_len: usize,
}

impl Default for Epa3 {
    fn default() -> Self {
        Self {
            vertices: [CsoPoint::default(); MAX_VERTICES_LEN],
            vertices_len: 0,
            faces: [Face::default(); MAX_FACES_LEN],
            faces_len: 0,
            silhouette: [SilhouetteEdge::default(); MAX_SILHOUETTE_LEN],
            silhouette_len: 0,
            heap: [FaceId::default(); MAX_HEAP_LEN],
            heap_len: 0,
        }
    }
}

impl Epa3 {
    fn heap_best_index(&self) -> usize {
        let mut best_id = 0;

        for i in 0..self.heap_len {
            if self.heap.read(i).neg_dist > self.heap.read(best_id).neg_dist {
                best_id = i;
            }
        }

        best_id
    }

    fn heap_peek(&self) -> FaceId {
        self.heap.read(self.heap_best_index())
    }

    fn heap_pop(&mut self) -> FaceId {
        let i = self.heap_best_index();
        let result = self.heap.read(i);

        if self.heap_len != 0 {
            self.heap_len -= 1;
            self.heap.write(i, self.heap.read(self.heap_len)); // Swap-remove.
        }

        result
    }

    fn heap_push(&mut self, elt: FaceId) -> bool {
        if self.heap_len != MAX_HEAP_LEN {
            self.heap.write(self.heap_len, elt);
            self.heap_len += 1;
            true
        } else {
            false
        }
    }

    fn face_new_with_proj(&self, bcoords: Vec3, pts: [u32; 4], adj: [u32; 4]) -> Face {
        let n = ccw_face_normal(
            self.vertices.at(pts.read(0) as usize).point,
            self.vertices.at(pts.read(1) as usize).point,
            self.vertices.at(pts.read(2) as usize).point,
        );
        Face {
            pts,
            adj,
            normal: n,
            bcoords,
            deleted: false,
            _padding: PADDING,
        }
    }

    fn face_new(&self, pts: [u32; 4], adj: [u32; 4]) -> FaceWithProj {
        let tri = Triangle::new(
            self.vertices.at(pts.read(0) as usize).point,
            self.vertices.at(pts.read(1) as usize).point,
            self.vertices.at(pts.read(2) as usize).point,
        );
        let proj = tri.project_local_point_and_get_location(Vec3::ZERO, true);

        match proj.feature_type {
            projection::FEATURE_VERTEX | projection::FEATURE_EDGE => {
                let eps_tol = FLT_EPS * 100.0;
                let proj_inside = proj.inside || proj.point.dot(proj.point) < eps_tol * eps_tol;
                let bcoords = proj.barycentric_coordinates();
                FaceWithProj {
                    face: self.face_new_with_proj(bcoords, pts, adj),
                    proj_inside,
                }
            }
            projection::FEATURE_FACE => FaceWithProj {
                face: self.face_new_with_proj(proj.bcoords, pts, adj),
                proj_inside: true,
            },
            _ => FaceWithProj {
                face: self.face_new_with_proj(Vec3::ZERO, pts, adj),
                proj_inside: false,
            },
        }
    }

    fn face_closest_points(&self, face: &Face) -> [Vec3; 2] {
        [
            self.vertices.at(face.pts.read(0) as usize).orig_a * face.bcoords.x
                + self.vertices.at(face.pts.read(1) as usize).orig_a * face.bcoords.y
                + self.vertices.at(face.pts.read(2) as usize).orig_a * face.bcoords.z,
            self.vertices.at(face.pts.read(0) as usize).orig_b * face.bcoords.x
                + self.vertices.at(face.pts.read(1) as usize).orig_b * face.bcoords.y
                + self.vertices.at(face.pts.read(2) as usize).orig_b * face.bcoords.z,
        ]
    }

    fn next_ccw_pt_id(&self, face: &Face, id: u32) -> u32 {
        if face.pts.read(0) == id {
            1
        } else if face.pts.read(1) == id {
            2
        } else {
            0
        }
    }

    fn can_be_seen_by(&self, face: &Face, point: u32, opp_pt_id: u32) -> bool {
        let p0 = self
            .vertices
            .at(face.pts.read(opp_pt_id as usize) as usize)
            .point;
        let p1 = self
            .vertices
            .at(face.pts.read(((opp_pt_id + 1) % 3) as usize) as usize)
            .point;
        let p2 = self
            .vertices
            .at(face.pts.read(((opp_pt_id + 2) % 3) as usize) as usize)
            .point;
        let pt = self.vertices.at(point as usize).point;

        (pt - p0).dot(face.normal) >= -EPS_TOL
            || Triangle::is_affinely_dependent(p1, p2, pt, FLT_EPS * 100.0)
    }

    fn compute_silhouette(&mut self, point: u32, id: u32, opp_pt_id: u32) -> bool {
        let mut stack = [SilhouetteEdge::default(); MAX_STACK_LEN];
        let mut stack_len = 1usize;
        stack.write(
            0,
            SilhouetteEdge {
                face_id: id,
                opp_pt_id,
            },
        );

        // NOTE: we use fixed-size for loops to avoid miscompilation issues of while loops on MacOs.
        for _ in 0..MAX_STACK_LEN {
            if stack_len == 0 {
                break;
            }

            stack_len -= 1;
            let edge = stack.read(stack_len);

            if !self.faces.at(edge.face_id as usize).deleted {
                if !self.can_be_seen_by(self.faces.at(edge.face_id as usize), point, edge.opp_pt_id)
                {
                    if self.silhouette_len < MAX_STACK_LEN {
                        self.silhouette.write(
                            self.silhouette_len,
                            SilhouetteEdge {
                                face_id: edge.face_id,
                                opp_pt_id: edge.opp_pt_id,
                            },
                        );
                        self.silhouette_len += 1;
                    } else {
                        return false;
                    }
                } else {
                    self.faces.at_mut(edge.face_id as usize).deleted = true;

                    let adj_pt_id1 = (edge.opp_pt_id + 2) % 3;
                    let adj_pt_id2 = edge.opp_pt_id;

                    let adj1 = self
                        .faces
                        .at(edge.face_id as usize)
                        .adj
                        .read(adj_pt_id1 as usize);
                    let adj2 = self
                        .faces
                        .at(edge.face_id as usize)
                        .adj
                        .read(adj_pt_id2 as usize);

                    let adj_opp_pt_id1 = self.next_ccw_pt_id(
                        self.faces.at(adj1 as usize),
                        self.faces
                            .at(edge.face_id as usize)
                            .pts
                            .read(adj_pt_id1 as usize),
                    );
                    let adj_opp_pt_id2 = self.next_ccw_pt_id(
                        self.faces.at(adj2 as usize),
                        self.faces
                            .at(edge.face_id as usize)
                            .pts
                            .read(adj_pt_id2 as usize),
                    );

                    stack.write(
                        stack_len,
                        SilhouetteEdge {
                            face_id: adj2,
                            opp_pt_id: adj_opp_pt_id2,
                        },
                    );
                    stack_len += 1;

                    if stack_len < MAX_STACK_LEN {
                        stack.write(
                            stack_len,
                            SilhouetteEdge {
                                face_id: adj1,
                                opp_pt_id: adj_opp_pt_id1,
                            },
                        );
                        stack_len += 1;
                    } else {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Computes closest points using EPA algorithm.
    pub fn closest_points(
        &mut self,
        pos12: Pose,
        g1: &Shape,
        g2: &Shape,
        simplex: &VoronoiSimplex,
        vertices: &[Pad<crate::Vector, u32>],
    ) -> EpaResult {
        let _eps = FLT_EPS;
        let _eps_tol = _eps * 100.0;

        // Reset buffers.
        self.vertices_len = 0;
        self.faces_len = 0;
        self.silhouette_len = 0;
        self.heap_len = 0;

        // Initialization.
        for i in 0..=simplex.dim {
            self.vertices
                .write(self.vertices_len, simplex.vertices.read(i as usize));
            self.vertices_len += 1;
        }

        if simplex.dim == 0 {
            let n = Vec3::new(0.0, 1.0, 0.0);
            return EpaResult::new(Vec3::ZERO, Vec3::ZERO, n, true);
        } else if simplex.dim == 3 {
            let dp1 = self.vertices.at(1).point - self.vertices.at(0).point;
            let dp2 = self.vertices.at(2).point - self.vertices.at(0).point;
            let dp3 = self.vertices.at(3).point - self.vertices.at(0).point;

            if dp1.cross(dp2).dot(dp3) > 0.0 {
                // Swap 1, 2
                let tmp = self.vertices.read(1);
                self.vertices.write(1, self.vertices.read(2));
                self.vertices.write(2, tmp);
            }

            let pts1 = [0u32, 1, 2, PADDING];
            let pts2 = [1u32, 3, 2, PADDING];
            let pts3 = [0u32, 2, 3, PADDING];
            let pts4 = [0u32, 3, 1, PADDING];

            let adj1 = [3u32, 1, 2, PADDING];
            let adj2 = [3u32, 2, 0, PADDING];
            let adj3 = [0u32, 1, 3, PADDING];
            let adj4 = [2u32, 1, 0, PADDING];

            let face1 = self.face_new(pts1, adj1);
            let face2 = self.face_new(pts2, adj2);
            let face3 = self.face_new(pts3, adj3);
            let face4 = self.face_new(pts4, adj4);

            self.faces.write(0, face1.face);
            self.faces.write(1, face2.face);
            self.faces.write(2, face3.face);
            self.faces.write(3, face4.face);
            self.faces_len = 4;

            if face1.proj_inside {
                let dist1 = self.faces.at(0).normal.dot(self.vertices.at(0).point);
                let face_id = face_id_new(0, -dist1);

                if !face_id.valid {
                    return none();
                }

                self.heap_push(face_id.face_id);
            }

            if face2.proj_inside {
                let dist2 = self.faces.at(1).normal.dot(self.vertices.at(1).point);
                let face_id = face_id_new(1, -dist2);

                if !face_id.valid {
                    return none();
                }

                self.heap_push(face_id.face_id);
            }

            if face3.proj_inside {
                let dist3 = self.faces.at(2).normal.dot(self.vertices.at(2).point);
                let face_id = face_id_new(2, -dist3);

                if !face_id.valid {
                    return none();
                }

                self.heap_push(face_id.face_id);
            }

            if face4.proj_inside {
                let dist4 = self.faces.at(3).normal.dot(self.vertices.at(3).point);
                let face_id = face_id_new(3, -dist4);

                if !face_id.valid {
                    return none();
                }

                self.heap_push(face_id.face_id);
            }

            if !(face1.proj_inside || face2.proj_inside || face3.proj_inside || face4.proj_inside) {
                return none();
            }
        } else {
            if simplex.dim == 1 {
                let dpt = self.vertices.at(1).point - self.vertices.at(0).point;
                let basis = crate::utils::orthonormal_basis3(dpt);
                let cso_point_a = cso_point_from_shapes(pos12, g1, g2, basis.read(0), vertices);
                self.vertices.write(self.vertices_len, cso_point_a);
                self.vertices_len += 1;
            }

            let pts1 = [0u32, 1, 2, PADDING];
            let pts2 = [0u32, 2, 1, PADDING];

            let adj1 = [1u32, 1, 1, PADDING];
            let adj2 = [0u32, 0, 0, PADDING];

            self.faces.write(0, self.face_new(pts1, adj1).face);
            self.faces.write(1, self.face_new(pts2, adj2).face);
            self.faces_len = 2;

            self.heap_push(face_id_new(0, 0.0).face_id);
            self.heap_push(face_id_new(1, 0.0).face_id);
        }

        if self.heap_len == 0 {
            return none();
        }

        let mut max_dist = MAX_FLT;
        let mut best_face_id = self.heap_peek();
        let mut old_dist = 0.0;

        // Run the expansion.
        // NOTE: we use fixed-size for loops to avoid miscompilation issues of while loops on MacOs.
        for _ in 0..100u32 {
            if self.heap_len == 0 {
                break;
            }

            let face_id = self.heap_pop();
            let face = self.faces.read(face_id.id as usize);

            if face.deleted {
                continue;
            }

            let cso_point = cso_point_from_shapes(pos12, g1, g2, face.normal, vertices);
            let support_point_id = self.vertices_len as u32;

            if self.vertices_len != MAX_VERTICES_LEN {
                self.vertices.write(self.vertices_len, cso_point);
                self.vertices_len += 1;
            } else {
                return none();
            }

            let candidate_max_dist = cso_point.point.dot(face.normal);

            if candidate_max_dist < max_dist {
                best_face_id = face_id;
                max_dist = candidate_max_dist;
            }

            let curr_dist = -face_id.neg_dist;

            if max_dist - curr_dist < _eps_tol
                || ((curr_dist - old_dist).abs() < _eps && candidate_max_dist < max_dist)
            {
                let best_face = self.faces.at(best_face_id.id as usize);
                let points = self.face_closest_points(best_face);
                return EpaResult::new(points.read(0), points.read(1), best_face.normal, true);
            }

            old_dist = curr_dist;

            self.faces.at_mut(face_id.id as usize).deleted = true;

            let adj_opp_pt_id1 =
                self.next_ccw_pt_id(self.faces.at(face.adj.read(0) as usize), face.pts.read(0));
            let adj_opp_pt_id2 =
                self.next_ccw_pt_id(self.faces.at(face.adj.read(1) as usize), face.pts.read(1));
            let adj_opp_pt_id3 =
                self.next_ccw_pt_id(self.faces.at(face.adj.read(2) as usize), face.pts.read(2));

            if !self.compute_silhouette(support_point_id, face.adj.read(0), adj_opp_pt_id1) {
                return none();
            }
            if !self.compute_silhouette(support_point_id, face.adj.read(1), adj_opp_pt_id2) {
                return none();
            }
            if !self.compute_silhouette(support_point_id, face.adj.read(2), adj_opp_pt_id3) {
                return none();
            }

            let first_new_face_id = self.faces_len;

            if self.silhouette_len == 0 {
                return none();
            }

            for eid in 0..self.silhouette_len {
                let edge = self.silhouette.read(eid);
                if !self.faces.at(edge.face_id as usize).deleted {
                    let new_face_id = self.faces_len as u32;

                    let pt_id1 = self
                        .faces
                        .at(edge.face_id as usize)
                        .pts
                        .read(((edge.opp_pt_id + 2) % 3) as usize);
                    let pt_id2 = self
                        .faces
                        .at(edge.face_id as usize)
                        .pts
                        .read(((edge.opp_pt_id + 1) % 3) as usize);

                    let pts = [pt_id1, pt_id2, support_point_id, PADDING];
                    let adj = [edge.face_id, new_face_id + 1, new_face_id - 1, PADDING];
                    let new_face = self.face_new(pts, adj);

                    *self
                        .faces
                        .at_mut(edge.face_id as usize)
                        .adj
                        .at_mut(((edge.opp_pt_id + 1) % 3) as usize) = new_face_id;

                    if self.faces_len != MAX_FACES_LEN {
                        self.faces.write(self.faces_len, new_face.face);
                        self.faces_len += 1;
                    } else {
                        return none();
                    }

                    if new_face.proj_inside {
                        let pt = self
                            .vertices
                            .at(self.faces.at(new_face_id as usize).pts.read(0) as usize)
                            .point;
                        let dist = self.faces.at(new_face_id as usize).normal.dot(pt);
                        if dist < curr_dist {
                            let points = self.face_closest_points(&face);
                            return EpaResult::new(
                                points.read(0),
                                points.read(1),
                                face.normal,
                                true,
                            );
                        }

                        let to_push = face_id_new(new_face_id, -dist);

                        if !to_push.valid {
                            return none();
                        }

                        if !self.heap_push(to_push.face_id) {
                            return none();
                        }
                    }
                }
            }

            if first_new_face_id == self.faces_len {
                return none();
            }

            *self.faces.at_mut(first_new_face_id).adj.at_mut(2) = (self.faces_len - 1) as u32;
            *self.faces.at_mut(self.faces_len - 1).adj.at_mut(1) = first_new_face_id as u32;

            // Clear silhouette buffer.
            self.silhouette_len = 0;
        }

        let best_face = self.faces.at(best_face_id.id as usize);
        let points = self.face_closest_points(best_face);
        EpaResult::new(points.read(0), points.read(1), best_face.normal, true)
    }
}

fn ccw_face_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let res = ab.cross(ac);
    let res_length = res.length();
    if res_length > FLT_EPS {
        res / res_length
    } else {
        Vec3::ZERO
    }
}
