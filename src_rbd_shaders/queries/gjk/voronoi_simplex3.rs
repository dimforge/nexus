//! Voronoi Simplex Module (3D)
//!
//! A simplex of dimension up to 3 that uses Voronoï regions for computing point projections.

use crate::queries::gjk::cso_point::{CsoPoint, EPS_TOL};
use crate::queries::projection::{FEATURE_EDGE, FEATURE_FACE, FEATURE_VERTEX};
use crate::shapes::{Segment, Tetrahedron, Triangle};
use glamx::Vec3;
use khal_std::index::MaybeIndexUnchecked;

/// A simplex of dimension up to 3 that uses Voronoï regions for computing point projections.
#[derive(Clone, Copy)]
pub struct VoronoiSimplex {
    pub prev_vertices: [u32; 4],
    pub prev_proj: Vec3,
    pub prev_dim: u32,

    pub vertices: [CsoPoint; 4],
    pub proj: Vec3,
    pub dim: u32,
}

impl Default for VoronoiSimplex {
    fn default() -> Self {
        let origin = CsoPoint::origin();
        Self {
            prev_vertices: [0, 1, 2, 3],
            prev_proj: Vec3::ZERO,
            prev_dim: 0,
            vertices: [origin, origin, origin, origin],
            proj: Vec3::ZERO,
            dim: 0,
        }
    }
}

pub struct SimplexProjectionResult {
    pub simplex: VoronoiSimplex,
    pub point: Vec3,
}

impl VoronoiSimplex {
    /// Initializes a simplex with a single point.
    pub fn init(pt: CsoPoint) -> VoronoiSimplex {
        let origin = CsoPoint::origin();
        VoronoiSimplex {
            prev_vertices: [0, 1, 2, 3],
            prev_proj: Vec3::ZERO,
            prev_dim: 0,
            vertices: [pt, origin, origin, origin],
            proj: Vec3::ZERO,
            dim: 0,
        }
    }

    /// Add a point to this simplex.
    pub fn add_point(&self, pt: CsoPoint) -> VoronoiSimplex {
        if self.dim == 0 {
            let dpt = self.vertices.at(0).point - pt.point;
            if dpt.dot(dpt) < EPS_TOL {
                return *self;
            }
        } else if self.dim == 1 {
            let ab = self.vertices.at(1).point - self.vertices.at(0).point;
            let ac = pt.point - self.vertices.at(0).point;
            let ab_ac = ab.cross(ac);
            if ab_ac.dot(ab_ac) < EPS_TOL {
                return *self;
            }
        } else {
            // self.dim == 2
            let ab = self.vertices.at(1).point - self.vertices.at(0).point;
            let ac = self.vertices.at(2).point - self.vertices.at(0).point;
            let ap = pt.point - self.vertices.at(0).point;
            let ab_ac = ab.cross(ac);
            let n = ab_ac / ab_ac.length();

            if n.dot(ap).abs() < EPS_TOL {
                return *self;
            }
        }

        let mut result = *self;
        result.prev_dim = self.dim;
        result.prev_proj = self.proj;
        result.prev_vertices = [0, 1, 2, 3];
        result.dim += 1;
        result.vertices.write(result.dim as usize, pt);
        result
    }

    /// Projects the origin on the boundary of this simplex and reduces `self` the smallest subsimplex containing the origin.
    ///
    /// Returns the result of the projection or `Point::origin()` if the origin lies inside of the simplex.
    /// The state of the simplex before projection is saved, and can be retrieved using the methods prefixed
    /// by `prev_`.
    pub fn project_origin_and_reduce(&self) -> SimplexProjectionResult {
        let mut s = *self;

        match s.dim {
            0 => {
                s.proj.x = 1.0;
                SimplexProjectionResult {
                    simplex: s,
                    point: s.vertices.at(0).point,
                }
            }
            1 => {
                let seg = Segment::new(s.vertices.at(0).point, s.vertices.at(1).point);
                let proj = seg.project_local_point_and_get_location(Vec3::ZERO, true);

                match proj.feature_type {
                    FEATURE_VERTEX => {
                        if proj.id == 0 {
                            s.proj.x = 1.0;
                            s.dim = 0;
                        } else {
                            // Swap 0, 1
                            let tmp = s.vertices.read(0);
                            s.vertices.write(0, s.vertices.read(1));
                            s.vertices.write(1, tmp);
                            let tmp_prev = s.prev_vertices.read(0);
                            s.prev_vertices.write(0, s.prev_vertices.read(1));
                            s.prev_vertices.write(1, tmp_prev);

                            s.proj.x = 1.0;
                            s.dim = 0;
                        }
                    }
                    FEATURE_EDGE => {
                        s.proj.x = proj.bcoords.x;
                        s.proj.y = proj.bcoords.y;
                    }
                    _ => { /* unreachable */ }
                }

                SimplexProjectionResult {
                    simplex: s,
                    point: proj.point,
                }
            }
            2 => {
                let tri = Triangle::new(
                    s.vertices.at(0).point,
                    s.vertices.at(1).point,
                    s.vertices.at(2).point,
                );
                let proj = tri.project_local_point_and_get_location(Vec3::ZERO, true);

                match proj.feature_type {
                    FEATURE_VERTEX => {
                        let i = proj.id as usize;

                        // Swap 0, i
                        let tmp = s.vertices.read(0);
                        s.vertices.write(0, s.vertices.read(i));
                        s.vertices.write(i, tmp);
                        let tmp_prev = s.prev_vertices.read(0);
                        s.prev_vertices.write(0, s.prev_vertices.read(i));
                        s.prev_vertices.write(i, tmp_prev);

                        s.proj.x = 1.0;
                        s.dim = 0;
                    }
                    FEATURE_EDGE => {
                        if proj.id == 0 {
                            s.proj.x = proj.bcoords.x;
                            s.proj.y = proj.bcoords.y;
                            s.dim = 1;
                        } else if proj.id == 1 {
                            // Swap 0, 2
                            let tmp = s.vertices.read(0);
                            s.vertices.write(0, s.vertices.read(2));
                            s.vertices.write(2, tmp);
                            let tmp_prev = s.prev_vertices.read(0);
                            s.prev_vertices.write(0, s.prev_vertices.read(2));
                            s.prev_vertices.write(2, tmp_prev);

                            s.proj.x = proj.bcoords.y;
                            s.proj.y = proj.bcoords.x;
                            s.dim = 1;
                        } else {
                            // Swap 1, 2
                            let tmp = s.vertices.read(1);
                            s.vertices.write(1, s.vertices.read(2));
                            s.vertices.write(2, tmp);
                            let tmp_prev = s.prev_vertices.read(1);
                            s.prev_vertices.write(1, s.prev_vertices.read(2));
                            s.prev_vertices.write(2, tmp_prev);

                            s.proj.x = proj.bcoords.x;
                            s.proj.y = proj.bcoords.y;
                            s.dim = 1;
                        }
                    }
                    FEATURE_FACE => {
                        s.proj = proj.bcoords;
                    }
                    _ => { /* unreachable */ }
                }

                SimplexProjectionResult {
                    simplex: s,
                    point: proj.point,
                }
            }
            _ => {
                let tetra = Tetrahedron::new(
                    s.vertices.at(0).point,
                    s.vertices.at(1).point,
                    s.vertices.at(2).point,
                    s.vertices.at(3).point,
                );
                let proj = tetra.project_local_point_and_get_location(Vec3::ZERO, true);

                match proj.feature_type {
                    FEATURE_VERTEX => {
                        let i = proj.id as usize;
                        // Swap 0, i
                        let tmp = s.vertices.read(0);
                        s.vertices.write(0, s.vertices.read(i));
                        s.vertices.write(i, tmp);
                        let tmp_prev = s.prev_vertices.read(0);
                        s.prev_vertices.write(0, s.prev_vertices.read(i));
                        s.prev_vertices.write(i, tmp_prev);

                        s.proj.x = 1.0;
                        s.dim = 0;
                    }
                    FEATURE_EDGE => {
                        match proj.id {
                            0 => {
                                // ab
                            }
                            1 => {
                                // ac
                                // Swap 1, 2
                                let tmp = s.vertices.read(1);
                                s.vertices.write(1, s.vertices.read(2));
                                s.vertices.write(2, tmp);
                                let tmp_prev = s.prev_vertices.read(1);
                                s.prev_vertices.write(1, s.prev_vertices.read(2));
                                s.prev_vertices.write(2, tmp_prev);
                            }
                            2 => {
                                // ad
                                // Swap 1, 3
                                let tmp = s.vertices.read(1);
                                s.vertices.write(1, s.vertices.read(3));
                                s.vertices.write(3, tmp);
                                let tmp_prev = s.prev_vertices.read(1);
                                s.prev_vertices.write(1, s.prev_vertices.read(3));
                                s.prev_vertices.write(3, tmp_prev);
                            }
                            3 => {
                                // bc
                                // Swap 0, 2
                                let tmp = s.vertices.read(0);
                                s.vertices.write(0, s.vertices.read(2));
                                s.vertices.write(2, tmp);
                                let tmp_prev = s.prev_vertices.read(0);
                                s.prev_vertices.write(0, s.prev_vertices.read(2));
                                s.prev_vertices.write(2, tmp_prev);
                            }
                            4 => {
                                // bd
                                // Swap 0, 3
                                let tmp = s.vertices.read(0);
                                s.vertices.write(0, s.vertices.read(3));
                                s.vertices.write(3, tmp);
                                let tmp_prev = s.prev_vertices.read(0);
                                s.prev_vertices.write(0, s.prev_vertices.read(3));
                                s.prev_vertices.write(3, tmp_prev);
                            }
                            5 => {
                                // cd
                                // Swap 0, 2
                                let tmp = s.vertices.read(0);
                                s.vertices.write(0, s.vertices.read(2));
                                s.vertices.write(2, tmp);
                                let tmp_prev = s.prev_vertices.read(0);
                                s.prev_vertices.write(0, s.prev_vertices.read(2));
                                s.prev_vertices.write(2, tmp_prev);

                                // Swap 1, 3
                                let tmp_ = s.vertices.read(1);
                                s.vertices.write(1, s.vertices.read(3));
                                s.vertices.write(3, tmp_);
                                let tmp_prev_ = s.prev_vertices.read(1);
                                s.prev_vertices.write(1, s.prev_vertices.read(3));
                                s.prev_vertices.write(3, tmp_prev_);
                            }
                            _ => { /* unreachable */ }
                        }

                        match proj.id {
                            0 | 1 | 2 | 5 => {
                                s.proj.x = proj.bcoords.x;
                                s.proj.y = proj.bcoords.y;
                            }
                            3 | 4 => {
                                s.proj.x = proj.bcoords.y;
                                s.proj.y = proj.bcoords.x;
                            }
                            _ => { /* unreachable */ }
                        }
                        s.dim = 1;
                    }
                    FEATURE_FACE => {
                        match proj.id {
                            0 => {
                                // abc
                                s.proj = proj.bcoords;
                            }
                            1 => {
                                // abd
                                s.vertices.write(2, s.vertices.read(3));
                                s.proj = proj.bcoords;
                            }
                            2 => {
                                // acd
                                s.vertices.write(1, s.vertices.read(3));
                                s.proj.x = proj.bcoords.x;
                                s.proj.y = proj.bcoords.z;
                                s.proj.z = proj.bcoords.y;
                            }
                            3 => {
                                // bcd
                                s.vertices.write(0, s.vertices.read(3));
                                s.proj.x = proj.bcoords.z;
                                s.proj.y = proj.bcoords.x;
                                s.proj.z = proj.bcoords.y;
                            }
                            _ => { /* unreachable */ }
                        }

                        s.dim = 2;
                    }
                    _ => { /* unreachable */ }
                }

                SimplexProjectionResult {
                    simplex: s,
                    point: proj.point,
                }
            }
        }
    }
}
