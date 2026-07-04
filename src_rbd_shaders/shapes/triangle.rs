//! Triangle Shape Module
//!
//! This module provides the triangle shape definition from its three vertices.

use crate::Vector;
use crate::bounding_volumes::Aabb;
use crate::queries::PolygonalFeature;
use crate::queries::ProjectionWithLocation;
use glamx::{Vec2, Vec3};
use khal_std::index::MaybeIndexUnchecked;

/// A triangle defined by three vertices.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Triangle {
    /// First vertex of the triangle.
    pub a: Vector,
    /// Second vertex of the triangle.
    pub b: Vector,
    /// Third vertex of the triangle.
    pub c: Vector,
}

impl Triangle {
    /// Creates a new triangle from three vertices.
    #[inline]
    pub fn new(a: Vector, b: Vector, c: Vector) -> Self {
        Self { a, b, c }
    }

    /// Computes the AABB of a triangle.
    pub fn aabb(&self) -> Aabb {
        let mins = self.a.min(self.b).min(self.c);
        let maxs = self.a.max(self.b).max(self.c);
        Aabb::new(mins, maxs)
    }

    /// Computes the local support point of a triangle.
    pub fn local_support_point(&self, dir: Vector) -> Vector {
        let da = dir.dot(self.a);
        let db = dir.dot(self.b);
        let dc = dir.dot(self.c);

        if da > db {
            if da > dc { self.a } else { self.c }
        } else if db > dc {
            self.b
        } else {
            self.c
        }
    }

    /// Computes the support face of a triangle (2D version).
    #[cfg(feature = "dim2")]
    pub fn support_face(&self, dir: Vector) -> PolygonalFeature {
        let mut result = PolygonalFeature::default();
        let tab = self.b - self.a;
        let tbc = self.c - self.b;
        let tca = self.a - self.c;
        // CCW normals
        let nab = dir.dot(Vec2::new(tab.y, -tab.x));
        let nbc = dir.dot(Vec2::new(tbc.y, -tbc.x));
        let nca = dir.dot(Vec2::new(tca.y, -tca.x));

        if nab > nbc {
            if nab > nca {
                // AB is the support face.
                result.vertices.write(0, self.a);
                result.vertices.write(1, self.b);
            } else {
                // CA is the support face.
                result.vertices.write(0, self.c);
                result.vertices.write(1, self.a);
            }
        } else if nbc > nca {
            // BC is the support face.
            result.vertices.write(0, self.b);
            result.vertices.write(1, self.c);
        } else {
            // CA is the support face.
            result.vertices.write(0, self.c);
            result.vertices.write(1, self.a);
        }

        result.num_vertices = 2;
        result
    }

    /// Computes the support face of a triangle (3D version).
    #[cfg(feature = "dim3")]
    pub fn support_face(&self, _dir: Vector) -> PolygonalFeature {
        // Just return the whole triangle.
        let mut result = PolygonalFeature::default();
        result.vertices.write(0, self.a);
        result.vertices.write(1, self.b);
        result.vertices.write(2, self.c);
        result.num_vertices = 3;
        result
    }

    /// Projects a point onto a triangle and returns location information.
    pub fn project_local_point_and_get_location(
        &self,
        pt: Vector,
        solid_flag: bool,
    ) -> ProjectionWithLocation {
        // To understand the ideas, consider reading the slides below
        // https://box2d.org/files/ErinCatto_GJK_GDC2010.pdf
        let a = self.a;
        let b = self.b;
        let c = self.c;

        let ab = b - a;
        let ac = c - a;
        let ap = pt - a;

        let ab_ap = ab.dot(ap);
        let ac_ap = ac.dot(ap);

        if ab_ap <= 0.0 && ac_ap <= 0.0 {
            // Voronoi region of `a`.
            let inside = is_proj_inside(pt, a);
            return ProjectionWithLocation::vertex(a, 0, inside);
        }

        let bp = pt - b;
        let ab_bp = ab.dot(bp);
        let ac_bp = ac.dot(bp);

        if ab_bp >= 0.0 && ac_bp <= ab_bp {
            // Voronoi region of `b`.
            let inside = is_proj_inside(pt, b);
            return ProjectionWithLocation::vertex(b, 1, inside);
        }

        let cp = pt - c;
        let ab_cp = ab.dot(cp);
        let ac_cp = ac.dot(cp);

        if ac_cp >= 0.0 && ab_cp <= ac_cp {
            // Voronoi region of `c`.
            let inside = is_proj_inside(pt, c);
            return ProjectionWithLocation::vertex(c, 2, inside);
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
                return ProjectionWithLocation::edge(res, bcoords, 0, is_proj_inside(pt, res));
            }
            AC => {
                // Voronoi region of `ac`.
                let w = ac_ap / ac.dot(ac);
                let bcoords = Vec2::new(1.0 - w, w);
                let res = a + ac * w;
                return ProjectionWithLocation::edge(res, bcoords, 2, is_proj_inside(pt, res));
            }
            BC => {
                // Voronoi region of `bc`.
                let w = bc.dot(bp) / bc.dot(bc);
                let bcoords = Vec2::new(1.0 - w, w);
                let res = b + bc * w;
                return ProjectionWithLocation::edge(res, bcoords, 1, is_proj_inside(pt, res));
            }
            // Voronoi region of the face. The guard fails when numerical
            // instability makes the denominator zero (nearly-degenerate
            // triangle); we then fall through to the solid case below.
            #[cfg(feature = "dim3")]
            FACE_CW | FACE_CCW if proj.params.x + proj.params.y + proj.params.z != 0.0 => {
                let denom = 1.0 / (proj.params.x + proj.params.y + proj.params.z);
                let v = proj.params.y * denom;
                let w = proj.params.z * denom;
                let bcoords = Vec3::new(1.0 - v - w, v, w);
                let res = a + ab * v + ac * w;
                return ProjectionWithLocation::face(
                    res,
                    bcoords,
                    proj.feature,
                    is_proj_inside(pt, res),
                );
            }
            _ => { /* FACE_INTERIOR, 2D only (implemented below) */ }
        }

        // Special treatment if we work in 2d because in this case we really are inside of the
        // object.
        if solid_flag {
            ProjectionWithLocation::solid(pt)
        } else {
            // We have to project on the closest edge.
            let v = ab_ap / (ab_ap - ab_bp); // proj on ab = a + ab * v
            let w = ac_ap / (ac_ap - ac_cp); // proj on ac = a + ac * w
            let u = (ac_bp - ab_bp) / (ac_bp - ab_bp + ab_cp - ac_cp); // proj on bc = b + bc * u

            let bc = c - b;
            let d_ab = ap.dot(ap) - (ab.dot(ab) * v * v);
            let d_ac = ap.dot(ap) - (ac.dot(ac) * w * w);
            let d_bc = bp.dot(bp) - (bc.dot(bc) * u * u);

            if d_ab < d_ac {
                if d_ab < d_bc {
                    // ab
                    let bcoords = Vec2::new(1.0 - v, v);
                    let proj = a + ab * v;
                    ProjectionWithLocation::edge(proj, bcoords, 0, true)
                } else {
                    // bc
                    let bcoords = Vec2::new(1.0 - u, u);
                    let proj = b + bc * u;
                    ProjectionWithLocation::edge(proj, bcoords, 1, true)
                }
            } else if d_ac < d_bc {
                // ac
                let bcoords = Vec2::new(1.0 - w, w);
                let proj = a + ac * w;
                ProjectionWithLocation::edge(proj, bcoords, 2, true)
            } else {
                // bc
                let bcoords = Vec2::new(1.0 - u, u);
                let proj = b + bc * u;
                ProjectionWithLocation::edge(proj, bcoords, 1, true)
            }
        }
    }
}

fn is_proj_inside(pt: Vector, proj: Vector) -> bool {
    #[cfg(feature = "dim2")]
    {
        proj == pt
    }
    #[cfg(feature = "dim3")]
    {
        // TODO: is this acceptable to assume the point is inside of the
        // triangle if it is close enough?
        crate::queries::relative_eq(proj, pt)
    }
}

const AB: u32 = 0;
const AC: u32 = 1;
const BC: u32 = 2;
const FACE_CW: u32 = 3;
// Only produced (and matched) by the 3D projection path.
#[cfg(feature = "dim3")]
const FACE_CCW: u32 = 4;

struct ProjectionInfo {
    feature: u32,
    // Read only in 3D; written by the shared construction code.
    #[cfg_attr(feature = "dim2", allow(dead_code))]
    params: Vec3,
}

/// Computes the 2D perpendicular dot product (cross product z-component).
#[cfg(feature = "dim2")]
pub fn perp(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

impl Triangle {
    /// Checks if three 3D points are affinely dependent (collinear).
    #[cfg(feature = "dim3")]
    pub fn is_affinely_dependent(p1: Vec3, p2: Vec3, p3: Vec3, eps: f32) -> bool {
        let p1p2 = p2 - p1;
        let p1p3 = p3 - p1;
        let c = p1p2.cross(p1p3);
        relative_eq_scalar(c.dot(c), 0.0, eps * eps)
    }
}

/// Approximately equal comparison for scalars.
#[cfg(feature = "dim3")]
fn relative_eq_scalar(a: f32, b: f32, eps: f32) -> bool {
    let abs_diff = (a - b).abs();

    // For when the numbers are really close together
    if abs_diff <= eps {
        return true;
    }

    let abs_a = a.abs();
    let abs_b = b.abs();

    // Use a relative difference comparison
    abs_diff <= abs_a.max(abs_b) * eps
}

/// Checks on which edge voronoi region the point is.
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
    #[cfg(feature = "dim2")]
    {
        let n = perp(ab, ac);
        let vc = n * perp(ab, ap);
        if vc < 0.0 && ab_ap >= 0.0 && ab_bp <= 0.0 {
            return ProjectionInfo {
                feature: AB,
                params: Vec3::ZERO,
            };
        }

        let vb = -n * perp(ac, _cp);
        if vb < 0.0 && ac_ap >= 0.0 && ac_cp <= 0.0 {
            return ProjectionInfo {
                feature: AC,
                params: Vec3::ZERO,
            };
        }

        let va = n * perp(bc, bp);
        if va < 0.0 && ac_bp - ab_bp >= 0.0 && ab_cp - ac_cp >= 0.0 {
            return ProjectionInfo {
                feature: BC,
                params: Vec3::ZERO,
            };
        }

        ProjectionInfo {
            feature: FACE_CW,
            params: Vec3::new(va, vb, vc),
        }
    }
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
}
