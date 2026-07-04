//! GJK (Gilbert-Johnson-Keerthi) Algorithm Module
//!
//! This module provides the GJK algorithm for computing the closest points
//! between convex shapes.

use crate::queries::gjk::VoronoiSimplex;
use crate::queries::gjk::cso_point::{CsoPoint, EPS_TOL, FLT_EPS};
use crate::shapes::Shape;
use crate::{DIM, MAX_FLT, PaddedVector, Pose, Vector};
use khal_std::index::MaybeIndexUnchecked;

/// GJK result status codes.
pub const INTERSECTION: u32 = 0;
pub const CLOSEST_POINTS: u32 = 1;
pub const PROXIMITY: u32 = 2;
pub const NO_INTERSECTION: u32 = 3;

/// Result of the GJK algorithm.
#[derive(Clone, Copy, Default)]
pub struct GjkResult {
    pub a: Vector,
    pub b: Vector,
    pub dir: Vector,
    pub status: u32,
}

impl GjkResult {
    pub fn new(status: u32, a: Vector, b: Vector, dir: Vector) -> Self {
        Self { status, a, b, dir }
    }
}

/// Creates a GJK result indicating intersection.
pub fn gjk_result_intersection() -> GjkResult {
    GjkResult::new(INTERSECTION, Vector::ZERO, Vector::ZERO, Vector::ZERO)
}

/// Creates a GJK result with closest points.
pub fn gjk_result_closest_points(a: Vector, b: Vector, dir: Vector) -> GjkResult {
    GjkResult::new(CLOSEST_POINTS, a, b, dir)
}

/// Creates a GJK result indicating proximity.
pub fn gjk_result_proximity(dir: Vector) -> GjkResult {
    GjkResult::new(PROXIMITY, Vector::ZERO, Vector::ZERO, dir)
}

/// Creates a GJK result indicating no intersection.
pub fn gjk_result_no_intersection(dir: Vector) -> GjkResult {
    GjkResult::new(NO_INTERSECTION, Vector::ZERO, Vector::ZERO, dir)
}

/// Computes closest points between two shapes using GJK algorithm.
pub fn closest_points(
    pose12: Pose,
    g1: &Shape,
    g2: &Shape,
    max_dist: f32,
    exact_dist: bool,
    simplex: &mut VoronoiSimplex,
    vertices: &[PaddedVector],
) -> GjkResult {
    let _eps = FLT_EPS;
    let _eps_tol: f32 = EPS_TOL;
    let _eps_rel: f32 = crate::sqrt(_eps_tol);

    // TODO: reset the simplex if it is empty?
    let mut proj = simplex.project_origin_and_reduce();

    let proj_len = proj.point.length();

    if proj_len == 0.0 {
        return gjk_result_intersection();
    }

    let mut old_dir = -proj.point / proj_len;
    let mut max_bound = MAX_FLT;
    let mut dir;

    for _ in 0..100 {
        let old_max_bound = max_bound;
        let proj_len = proj.point.length();

        if proj_len > EPS_TOL {
            dir = -proj.point / proj_len;
            max_bound = proj_len;
        } else {
            // The origin is on the simplex.
            *simplex = proj.simplex;
            return gjk_result_intersection();
        }

        if max_bound >= old_max_bound {
            if exact_dist {
                let pts = proj.simplex.result(true);
                return gjk_result_closest_points(pts.read(0), pts.read(1), old_dir);
            // upper bounds inconsistencies
            } else {
                return gjk_result_proximity(old_dir);
            }
        }

        let cso_point = cso_point_from_shapes(pose12, g1, g2, dir, vertices);
        let min_bound = -dir.dot(cso_point.point);

        if min_bound > max_dist {
            return gjk_result_no_intersection(dir);
        } else if !exact_dist && min_bound > 0.0 && max_bound <= max_dist {
            return gjk_result_proximity(old_dir);
        } else if max_bound - min_bound <= _eps_rel * max_bound {
            if exact_dist {
                let pts = proj.simplex.result(false);
                return gjk_result_closest_points(pts.read(0), pts.read(1), dir);
            // the distance found has a good enough precision
            } else {
                return gjk_result_proximity(dir);
            }
        }

        let dim_before_add = proj.simplex.dim;
        proj.simplex = proj.simplex.add_point(cso_point);

        // Check if we pushed the same support point twice.
        if dim_before_add == proj.simplex.dim {
            if exact_dist {
                let pts = proj.simplex.result(false);
                return gjk_result_closest_points(pts.read(0), pts.read(1), dir);
            } else {
                return gjk_result_proximity(dir);
            }
        }

        old_dir = dir;
        proj = proj.simplex.project_origin_and_reduce();

        if proj.simplex.dim == DIM {
            if min_bound >= EPS_TOL {
                if exact_dist {
                    let pts = proj.simplex.result(true);
                    return gjk_result_closest_points(pts.read(0), pts.read(1), old_dir);
                } else {
                    // NOTE: previous implementation used old_proj here.
                    return gjk_result_proximity(old_dir);
                }
            } else {
                *simplex = proj.simplex;
                return gjk_result_intersection(); // Point inside of the cso.
            }
        }
    }

    gjk_result_no_intersection(Vector::X)
}

impl VoronoiSimplex {
    /// Computes the closest points from the simplex.
    fn result(&self, prev: bool) -> [Vector; 2] {
        let mut a = Vector::ZERO;
        let mut b = Vector::ZERO;

        if prev {
            macro_rules! add_axis(
                ($i: expr, $x: ident) => {
                    let coord = self.prev_proj.$x;
                    let point = self.vertices.at(self.prev_vertices.read($i as usize) as usize);
                    a += point.orig_a * coord;
                    b += point.orig_b * coord;
                }
            );
            // NOTE: indexing a vec2/vec3 triggers a vulkan validation error (unless we
            //       enable some extra capabilities). So we unroll manually instead.
            add_axis!(0, x);
            if self.prev_dim >= 1 {
                add_axis!(1, y);
            }
            #[cfg(feature = "dim3")]
            if self.prev_dim >= 2 {
                add_axis!(2, z);
            }
        } else {
            macro_rules! add_axis(
                ($i: expr, $x: ident) => {
                    let coord = self.proj.$x;
                    let point = self.vertices.at($i as usize);
                    a += point.orig_a * coord;
                    b += point.orig_b * coord;
                }
            );
            add_axis!(0, x);
            if self.dim >= 1 {
                add_axis!(1, y);
            }
            #[cfg(feature = "dim3")]
            if self.dim >= 2 {
                add_axis!(2, z);
            }
        }

        [a, b]
    }
}

/// Computes the support point of the CSO of `g1` and `g2` toward the direction `dir`.
pub fn cso_point_from_shapes(
    pos12: Pose,
    g1: &Shape,
    g2: &Shape,
    dir: Vector,
    vertices: &[PaddedVector],
) -> CsoPoint {
    let sp1 = g1.local_support_point(dir, vertices);
    let sp2 = g2.support_point(pos12, -dir, vertices);
    CsoPoint::from_points(sp1, sp2)
}
