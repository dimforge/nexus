//! EPA (Expanding Polytope Algorithm) Module (2D)
//!
//! This module provides the EPA algorithm for computing penetration depth
//! and contact normal for intersecting convex shapes in 2D.

use crate::queries::gjk::cso_point::{CsoPoint, EPS_TOL, FLT_EPS};
use crate::queries::gjk::gjk::cso_point_from_shapes;
use crate::queries::gjk::voronoi_simplex2::VoronoiSimplex;
use crate::shapes::Shape;
use crate::{PaddedVector, Pose};
use glamx::Vec2;
use khal_std::index::MaybeIndexUnchecked;

// TODO: find the ideal values.
const MAX_VERTICES_LEN: usize = 32;
const MAX_FACES_LEN: usize = 32;
const MAX_HEAP_LEN: usize = 32;

#[derive(Clone, Copy, Default)]
struct FaceId {
    id: u32,
    neg_dist: f32,
}

struct OptionFaceId {
    face_id: FaceId,
    valid: bool,
}

impl OptionFaceId {
    fn new(id: u32, neg_dist: f32) -> OptionFaceId {
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
}

#[derive(Clone, Copy, Default)]
struct Face {
    pts: [u32; 2],
    normal: Vec2,
    proj: Vec2,
    bcoords: Vec2,
    deleted: bool,
    _padding: u32,
}

#[derive(Copy, Clone)]
struct FaceWithProj {
    face: Face,
    proj_inside: bool,
}

/// EPA result containing closest points and contact normal.
#[derive(Clone, Copy, Default)]
pub struct EpaResult {
    pub pt_a: Vec2,
    pub pt_b: Vec2,
    pub normal: Vec2,
    pub valid: bool,
}

impl EpaResult {
    pub fn new(pt_a: Vec2, pt_b: Vec2, normal: Vec2, valid: bool) -> Self {
        Self {
            pt_a,
            pt_b,
            normal,
            valid,
        }
    }

    fn none() -> EpaResult {
        EpaResult::new(Vec2::ZERO, Vec2::ZERO, Vec2::ZERO, false)
    }
}

/// EPA state for 2D collision detection.
pub struct Epa2 {
    vertices: [CsoPoint; MAX_VERTICES_LEN],
    vertices_len: usize,
    faces: [Face; MAX_FACES_LEN],
    faces_len: usize,
    heap: [FaceId; MAX_HEAP_LEN],
    heap_len: usize,
}

impl Default for Epa2 {
    fn default() -> Self {
        Self {
            vertices: [CsoPoint::default(); MAX_VERTICES_LEN],
            vertices_len: 0,
            faces: [Face::default(); MAX_FACES_LEN],
            faces_len: 0,
            heap: [FaceId::default(); MAX_HEAP_LEN],
            heap_len: 0,
        }
    }
}

impl Epa2 {
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

    fn face_new(&self, pts: [u32; 2]) -> FaceWithProj {
        let proj = project_origin(
            self.vertices.at(pts.read(0) as usize).point,
            self.vertices.at(pts.read(1) as usize).point,
        );
        if proj.valid {
            FaceWithProj {
                face: self.face_new_with_proj(proj.point, proj.bcoords, pts),
                proj_inside: true,
            }
        } else {
            FaceWithProj {
                face: self.face_new_with_proj(Vec2::ZERO, Vec2::ZERO, pts),
                proj_inside: false,
            }
        }
    }

    fn face_new_with_proj(&self, proj: Vec2, bcoords: Vec2, pts: [u32; 2]) -> Face {
        let n = ccw_face_normal(
            self.vertices.at(pts.read(0) as usize).point,
            self.vertices.at(pts.read(1) as usize).point,
        );
        Face {
            pts,
            normal: n,
            proj,
            bcoords,
            deleted: n == Vec2::ZERO,
            _padding: 0,
        }
    }

    fn face_closest_points(&self, face: &Face) -> [Vec2; 2] {
        [
            self.vertices.at(face.pts.read(0) as usize).orig_a * face.bcoords.x
                + self.vertices.at(face.pts.read(1) as usize).orig_a * face.bcoords.y,
            self.vertices.at(face.pts.read(0) as usize).orig_b * face.bcoords.x
                + self.vertices.at(face.pts.read(1) as usize).orig_b * face.bcoords.y,
        ]
    }

    /// Computes closest points using EPA algorithm.
    pub fn closest_points(
        &mut self,
        pose12: Pose,
        g1: &Shape,
        g2: &Shape,
        simplex: &VoronoiSimplex,
        vertices: &[PaddedVector],
    ) -> EpaResult {
        let _eps = FLT_EPS;
        let _eps_tol = _eps * 100.0;

        // Reset buffers.
        self.vertices_len = 0;
        self.faces_len = 0;
        self.heap_len = 0;

        // Initialization.
        for i in 0..=simplex.dim {
            self.vertices
                .write(self.vertices_len, simplex.vertices.read(i as usize));
            self.vertices_len += 1;
        }

        if simplex.dim == 0 {
            const MAX_ITERS: u32 = 100;

            // The contact is vertex-vertex.
            let mut n = Vec2::new(0.0, 1.0);

            // First, find a vector on the first vertex tangent cone.
            let orig1 = self.vertices.at(0).orig_a;
            for _ in 0..MAX_ITERS {
                let supp1 = g1.local_support_point(n, vertices);
                let mut tangent = supp1 - orig1;
                let tangent_len = tangent.length();

                if tangent_len > _eps_tol {
                    tangent /= tangent_len;
                    if n.dot(tangent) < _eps_tol {
                        break;
                    }

                    n = Vec2::new(-tangent.y, tangent.x);
                } else {
                    break;
                }
            }

            // Second, ensure the direction lies on the second vertex's tangent cone.
            let orig2 = self.vertices.at(0).orig_b;
            for _ in 0..MAX_ITERS {
                let supp2 = g2.support_point(pose12, -n, vertices);
                let mut tangent = supp2 - orig2;
                let tangent_len = tangent.length();

                if tangent_len > _eps_tol {
                    tangent /= tangent_len;
                    if (-n).dot(tangent) < _eps_tol {
                        break;
                    }

                    n = Vec2::new(-tangent.y, tangent.x);
                } else {
                    break;
                }
            }

            return EpaResult::new(Vec2::ZERO, Vec2::ZERO, n, true);
        } else if simplex.dim == 2 {
            let dp1 = self.vertices.at(1).point - self.vertices.at(0).point;
            let dp2 = self.vertices.at(2).point - self.vertices.at(0).point;

            if dp1.perp_dot(dp2) < 0.0 {
                let vtx1 = self.vertices.read(1);
                self.vertices.write(1, self.vertices.read(2));
                self.vertices.write(2, vtx1);
            }

            let pts1 = [0u32, 1];
            let pts2 = [1u32, 2];
            let pts3 = [2u32, 0];

            let face1 = self.face_new(pts1);
            let face2 = self.face_new(pts2);
            let face3 = self.face_new(pts3);

            self.faces.write(0, face1.face);
            self.faces.write(1, face2.face);
            self.faces.write(2, face3.face);
            self.faces_len = 3;

            if face1.proj_inside {
                let dist1 = self.faces.at(0).normal.dot(self.vertices.at(0).point);
                let face_id = OptionFaceId::new(0, -dist1);

                if !face_id.valid {
                    return EpaResult::none();
                }

                self.heap_push(face_id.face_id);
            }

            if face2.proj_inside {
                let dist2 = self.faces.at(1).normal.dot(self.vertices.at(1).point);
                let face_id = OptionFaceId::new(1, -dist2);

                if !face_id.valid {
                    return EpaResult::none();
                }

                self.heap_push(face_id.face_id);
            }

            if face3.proj_inside {
                let dist3 = self.faces.at(2).normal.dot(self.vertices.at(2).point);
                let face_id = OptionFaceId::new(2, -dist3);

                if !face_id.valid {
                    return EpaResult::none();
                }

                self.heap_push(face_id.face_id);
            }

            if !(face1.proj_inside || face2.proj_inside || face3.proj_inside) {
                return EpaResult::none();
            }
        } else {
            let pts1 = [0u32, 1];
            let pts2 = [1u32, 0];

            self.faces.write(
                0,
                self.face_new_with_proj(Vec2::ZERO, Vec2::new(1.0, 0.0), pts1),
            );
            self.faces.write(
                1,
                self.face_new_with_proj(Vec2::ZERO, Vec2::new(1.0, 0.0), pts2),
            );
            self.faces_len = 2;

            let dist1 = self.faces.at(0).normal.dot(self.vertices.at(0).point);
            let dist2 = self.faces.at(1).normal.dot(self.vertices.at(1).point);
            let fid1 = OptionFaceId::new(0, dist1);
            let fid2 = OptionFaceId::new(1, dist2);

            if !fid1.valid {
                return EpaResult::none();
            }
            if !fid2.valid {
                return EpaResult::none();
            }

            self.heap_push(fid1.face_id);
            self.heap_push(fid2.face_id);
        }

        if self.heap_len == 0 {
            return EpaResult::none();
        }

        let mut max_dist = 1.0e20;
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

            let cso_point = cso_point_from_shapes(pose12, g1, g2, face.normal, vertices);
            let support_point_id = self.vertices_len as u32;

            if self.vertices_len != MAX_VERTICES_LEN {
                self.vertices.write(self.vertices_len, cso_point);
                self.vertices_len += 1;
            } else {
                return EpaResult::none();
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

            let pts1 = [face.pts.read(0), support_point_id];
            let pts2 = [support_point_id, face.pts.read(1)];

            macro_rules! check_face {
                ($f: expr) => {
                    let f = $f;
                    if f.proj_inside {
                        let dist = f.face.normal.dot(f.face.proj);
                        if dist < curr_dist {
                            let cpts = self.face_closest_points(&f.face);
                            return EpaResult::new(cpts.read(0), cpts.read(1), f.face.normal, true);
                        }

                        if !f.face.deleted {
                            let new_fid = OptionFaceId::new(self.faces_len as u32, -dist);
                            if !new_fid.valid {
                                return EpaResult::none();
                            }

                            if !self.heap_push(new_fid.face_id) {
                                return EpaResult::none();
                            }
                        }
                    }

                    if self.faces_len != MAX_FACES_LEN {
                        self.faces.write(self.faces_len, f.face);
                        self.faces_len += 1;
                    } else {
                        return EpaResult::none();
                    }
                };
            }
            check_face!(self.face_new(pts1));
            check_face!(self.face_new(pts2));
        }

        let best_face = self.faces.at(best_face_id.id as usize);
        let points = self.face_closest_points(best_face);
        EpaResult::new(points.read(0), points.read(1), best_face.normal, true)
    }
}

struct ProjectOriginResult {
    point: Vec2,
    bcoords: Vec2,
    valid: bool,
}

fn project_origin(a: Vec2, b: Vec2) -> ProjectOriginResult {
    let ab = b - a;
    let ap = -a;
    let ab_ap = ab.dot(ap);
    let sqnab = ab.dot(ab);

    if sqnab == 0.0 {
        return ProjectOriginResult {
            point: Vec2::ZERO,
            bcoords: Vec2::ZERO,
            valid: false,
        };
    }

    if ab_ap < -EPS_TOL || ab_ap > sqnab + EPS_TOL {
        // Voronoï region of vertex 'a' or 'b'.
        ProjectOriginResult {
            point: Vec2::ZERO,
            bcoords: Vec2::ZERO,
            valid: false,
        }
    } else {
        // Voronoï region of the segment interior.
        let position_on_segment = ab_ap / sqnab;
        let res = a + ab * position_on_segment;

        ProjectOriginResult {
            point: res,
            bcoords: Vec2::new(1.0 - position_on_segment, position_on_segment),
            valid: true,
        }
    }
}

fn ccw_face_normal(a: Vec2, b: Vec2) -> Vec2 {
    let ab = b - a;
    let res = Vec2::new(ab.y, -ab.x);
    let res_length = res.length();
    if res_length > FLT_EPS {
        res / res_length
    } else {
        Vec2::ZERO
    }
}
