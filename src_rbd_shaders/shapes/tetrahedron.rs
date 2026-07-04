//! Tetrahedron Shape Module (3D only)
//!
//! This module provides the tetrahedron shape definition from its four vertices.

use crate::Vector;
use crate::queries::ProjectionWithLocation;
use glamx::{Vec2, Vec3};

// TODO: group all the epsilon in the same place.
const FLT_EPS: f32 = 1.0e-7;

/// A tetrahedron defined by four vertices.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Tetrahedron {
    /// First vertex of the tetrahedron.
    pub a: Vector,
    /// Second vertex of the tetrahedron.
    pub b: Vector,
    /// Third vertex of the tetrahedron.
    pub c: Vector,
    /// Fourth vertex of the tetrahedron.
    pub d: Vector,
}

impl Tetrahedron {
    /// Creates a new tetrahedron from four vertices.
    #[inline]
    pub fn new(a: Vector, b: Vector, c: Vector, d: Vector) -> Self {
        Self { a, b, c, d }
    }

    /// Projects a point onto a tetrahedron and returns location information.
    pub fn project_local_point_and_get_location(
        &self,
        pt: Vector,
        _solid_flag: bool,
    ) -> ProjectionWithLocation {
        let ab = self.b - self.a;
        let ac = self.c - self.a;
        let ad = self.d - self.a;
        let ap = pt - self.a;

        // Voronoi regions of vertices.
        let ap_ab = ap.dot(ab);
        let ap_ac = ap.dot(ac);
        let ap_ad = ap.dot(ad);

        if ap_ab <= 0.0 && ap_ac <= 0.0 && ap_ad <= 0.0 {
            // Voronoi region of `a`.
            return ProjectionWithLocation::vertex(self.a, 0, false);
        }

        let bc = self.c - self.b;
        let bd = self.d - self.b;
        let bp = pt - self.b;

        let bp_bc = bp.dot(bc);
        let bp_bd = bp.dot(bd);
        let bp_ab = bp.dot(ab);

        if bp_bc <= 0.0 && bp_bd <= 0.0 && bp_ab >= 0.0 {
            // Voronoi region of `b`.
            return ProjectionWithLocation::vertex(self.b, 1, false);
        }

        let cd = self.d - self.c;
        let cp = pt - self.c;

        let cp_ac = cp.dot(ac);
        let cp_bc = cp.dot(bc);
        let cp_cd = cp.dot(cd);

        if cp_cd <= 0.0 && cp_bc >= 0.0 && cp_ac >= 0.0 {
            // Voronoi region of `c`.
            return ProjectionWithLocation::vertex(self.c, 2, false);
        }

        let dp = pt - self.d;

        let dp_cd = dp.dot(cd);
        let dp_bd = dp.dot(bd);
        let dp_ad = dp.dot(ad);

        if dp_ad >= 0.0 && dp_bd >= 0.0 && dp_cd >= 0.0 {
            // Voronoi region of `d`.
            return ProjectionWithLocation::vertex(self.d, 3, false);
        }

        // Voronoi region of ab.
        let nabc = ab.cross(ac);
        let nabd = ab.cross(ad);
        let res_ab = check_edge(0, self.a, self.b, nabc, nabd, ap, ab, ap_ab, bp_ab);
        if res_ab.valid {
            return res_ab.proj;
        }

        let dabc = res_ab.du;
        let dabd = res_ab.dv;

        // Voronoi region of ac.
        let nacd = ac.cross(ad);
        let res_ac = check_edge(1, self.a, self.c, nacd, -nabc, ap, ac, ap_ac, cp_ac);
        if res_ac.valid {
            return res_ac.proj;
        }

        let dacd = res_ac.du;
        let dacb = res_ac.dv;

        // Voronoi region of ad.
        let res_ad = check_edge(2, self.a, self.d, -nabd, -nacd, ap, ad, ap_ad, dp_ad);
        if res_ad.valid {
            return res_ad.proj;
        }

        let dadb = res_ad.du;
        let dadc = res_ad.dv;

        // Voronoi region of bc.
        let nbcd = bc.cross(bd);
        let res_bc = check_edge(3, self.b, self.c, nabc, nbcd, bp, bc, bp_bc, cp_bc);
        if res_bc.valid {
            return res_bc.proj;
        }

        let dbca = res_bc.du;
        let dbcd = res_bc.dv;

        // Voronoi region of bd.
        let res_bd = check_edge(4, self.b, self.d, -nbcd, nabd, bp, bd, bp_bd, dp_bd);
        if res_bd.valid {
            return res_bd.proj;
        }

        let dbdc = res_bd.du;
        let dbda = res_bd.dv;

        // Voronoi region of cd.
        let res_cd = check_edge(5, self.c, self.d, nacd, nbcd, cp, cd, cp_cd, dp_cd);
        if res_cd.valid {
            return res_cd.proj;
        }

        let dcda = res_cd.du;
        let dcdb = res_cd.dv;

        // Face abc.
        let res_abc = check_face(
            0, self.a, self.b, self.c, ap, bp, cp, ab, ac, ad, dabc, dbca, dacb,
        );
        if res_abc.valid {
            return res_abc.proj;
        }

        // Face abd.
        let res_abd = check_face(
            1, self.a, self.b, self.d, ap, bp, dp, ab, ad, ac, dadb, dabd, dbda,
        );
        if res_abd.valid {
            return res_abd.proj;
        }

        // Face acd.
        let res_acd = check_face(
            2, self.a, self.c, self.d, ap, cp, dp, ac, ad, ab, dacd, dcda, dadc,
        );
        if res_acd.valid {
            return res_acd.proj;
        }

        // Face bcd.
        let res_bcd = check_face(
            3, self.b, self.c, self.d, bp, cp, dp, bc, bd, -ab, dbcd, dcdb, dbdc,
        );
        if res_bcd.valid {
            return res_bcd.proj;
        }

        ProjectionWithLocation::solid(pt)
    }
}

struct EdgeCheck {
    proj: ProjectionWithLocation,
    du: f32,
    dv: f32,
    valid: bool,
}

/// Voronoi regions of edges.
fn check_edge(
    i: u32,
    a: Vector,
    _b: Vector,
    nabc: Vector,
    nabd: Vector,
    ap: Vector,
    ab: Vector,
    ap_ab: f32,
    bp_ab: f32,
) -> EdgeCheck {
    let ab_ab = ap_ab - bp_ab;

    let ap_x_ab = ap.cross(ab);
    let dabc = ap_x_ab.dot(nabc);
    let dabd = ap_x_ab.dot(nabd);

    // TODO: the case where ab_ab == 0.0 is not well defined.
    if ab_ab != 0.0 && dabc >= 0.0 && dabd >= 0.0 && ap_ab >= 0.0 && ap_ab <= ab_ab {
        // Voronoi region of `ab`.
        let u = ap_ab / ab_ab;
        let bcoords = Vec2::new(1.0 - u, u);
        let res = a + ab * u;
        let proj = ProjectionWithLocation::edge(res, bcoords, i, false);
        EdgeCheck {
            proj,
            du: dabc,
            dv: dabd,
            valid: true,
        }
    } else {
        EdgeCheck {
            proj: ProjectionWithLocation::default(),
            du: dabc,
            dv: dabd,
            valid: false,
        }
    }
}

/// Voronoi regions of faces.
struct FaceCheck {
    proj: ProjectionWithLocation,
    valid: bool,
}

fn check_face(
    i: u32,
    a: Vector,
    b: Vector,
    c: Vector,
    ap: Vector,
    bp: Vector,
    cp: Vector,
    ab: Vector,
    ac: Vector,
    ad: Vector,
    dabc: f32,
    dbca: f32,
    dacb: f32,
) -> FaceCheck {
    if dabc < 0.0 && dbca < 0.0 && dacb < 0.0 {
        let n = ab.cross(ac); // TODO: is is possible to avoid this cross product?
        if n.dot(ad) * n.dot(ap) < 0.0 {
            // Voronoi region of the face.

            // NOTE: the normalization may fail even if the dot products
            // above were < 0. This happens, e.g., when we use fixed-point
            // numbers and there are not enough decimal bits to perform
            // the normalization.
            let n_length = n.length();

            if n_length < FLT_EPS {
                return FaceCheck {
                    proj: ProjectionWithLocation::default(),
                    valid: false,
                };
            }

            let normal = n / n_length;
            let vc = normal.dot(ap.cross(bp));
            let va = normal.dot(bp.cross(cp));
            let vb = normal.dot(cp.cross(ap));

            let denom = va + vb + vc;
            let inv_denom = 1.0 / denom;

            let bcoords = Vec3::new(va * inv_denom, vb * inv_denom, vc * inv_denom);
            let res = a * bcoords.x + b * bcoords.y + c * bcoords.z;
            return FaceCheck {
                proj: ProjectionWithLocation::face(res, bcoords, i, false),
                valid: true,
            };
        }
    }

    FaceCheck {
        proj: ProjectionWithLocation::default(),
        valid: false,
    }
}
