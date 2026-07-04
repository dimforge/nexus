//! Voronoi Simplex Module (2D)
//!
//! A simplex of dimension up to 2 using Voronoï regions for computing point projections.

use crate::queries::gjk::cso_point::{CsoPoint, EPS_TOL};
use crate::queries::projection::{FEATURE_EDGE, FEATURE_VERTEX};
use crate::shapes::{Segment, Triangle};
use glamx::Vec2;
use khal_std::index::MaybeIndexUnchecked;

/// A simplex of dimension up to 2 using Voronoï regions for computing point projections.
#[derive(Clone, Copy)]
pub struct VoronoiSimplex {
    pub prev_vertices: [u32; 3],
    pub prev_proj: Vec2,
    pub prev_dim: u32,

    pub vertices: [CsoPoint; 3],
    pub proj: Vec2,
    pub dim: u32,
}

impl Default for VoronoiSimplex {
    fn default() -> Self {
        let origin = CsoPoint::origin();
        Self {
            prev_vertices: [0, 1, 2],
            prev_proj: Vec2::ZERO,
            prev_dim: 0,
            vertices: [origin, origin, origin],
            proj: Vec2::ZERO,
            dim: 0,
        }
    }
}

pub struct SimplexProjectionResult {
    pub simplex: VoronoiSimplex,
    pub point: Vec2,
}

impl VoronoiSimplex {
    /// Initializes a simplex with a single point.
    pub fn init(pt: CsoPoint) -> Self {
        let origin = CsoPoint::origin();
        VoronoiSimplex {
            prev_vertices: [0, 1, 2],
            prev_proj: Vec2::ZERO,
            prev_dim: 0,
            vertices: [pt, origin, origin],
            proj: Vec2::ZERO,
            dim: 0,
        }
    }

    /// Add a point to this simplex.
    pub fn add_point(&self, pt: CsoPoint) -> VoronoiSimplex {
        let mut result = *self;
        result.prev_dim = self.dim;
        result.prev_proj = self.proj;
        result.prev_vertices = [0, 1, 2];

        for i in 0..=self.dim {
            let dpt = self.vertices.at(i as usize).point - pt.point;
            if dpt.dot(dpt) < EPS_TOL {
                return *self;
            }
        }

        result.dim += 1;
        result.vertices.write(result.dim as usize, pt);
        result
    }

    /// Projects the origin on the boundary of this simplex and reduces `s` the smallest subsimplex containing the origin.
    ///
    /// Returns the result of the projection or Point::origin() if the origin lies inside of the simplex.
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
                let proj = seg.project_local_point_and_get_location(Vec2::ZERO, true);

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
            _ => {
                // case 2
                let tri = Triangle::new(
                    s.vertices.at(0).point,
                    s.vertices.at(1).point,
                    s.vertices.at(2).point,
                );
                let proj = tri.project_local_point_and_get_location(Vec2::ZERO, true);

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
