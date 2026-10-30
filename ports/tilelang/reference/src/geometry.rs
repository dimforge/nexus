//! Live native geometry oracle. Private GJK source is linked into this binary
//! without changing production visibility or duplicating its implementation.
pub use nexus_rbd_shaders3d::{DIM, MAX_FLT, Pad, PaddedVector, Pose, Vector, shapes, sqrt, utils};
#[path = "../../../../src_rbd_shaders/queries/gjk/cso_point.rs"]
pub mod cso_point;
#[path = "../../../../src_rbd_shaders/queries/gjk/epa3.rs"]
pub mod epa3;
#[path = "../../../../src_rbd_shaders/queries/gjk/gjk.rs"]
pub mod gjk;
#[path = "../../../../src_rbd_shaders/queries/gjk/voronoi_simplex3.rs"]
pub mod voronoi_simplex3;
mod queries {
    pub mod projection {
        pub use nexus_rbd_shaders3d::queries::{FEATURE_EDGE, FEATURE_FACE, FEATURE_VERTEX};
    }
    pub mod gjk {
        pub use crate::cso_point;
        pub use crate::epa3;
        pub use crate::gjk;
        pub use crate::voronoi_simplex3::VoronoiSimplex;
    }
}
mod feature_fixture;
mod feature_layout;
mod geometry_layout;
mod polygonal_fixture;
mod projection_fixture;
mod simplex_fixture;
mod support_fixture;
mod support_layout;
use serde_json::json;
use std::{fs, path::Path};
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let root = Path::new(&path);
    fs::create_dir_all(root).unwrap();
    fs::write(
        root.join("layout.json"),
        serde_json::to_vec_pretty(&feature_layout::layout()).unwrap(),
    )
    .unwrap();
    for kind in [
        "segment_projection",
        "triangle_projection",
        "tetrahedron_projection",
        "simplex_init",
        "simplex_add",
        "simplex_project",
        "support_local",
        "support_world",
        "cso_support",
        "gjk_closest",
        "epa_closest",
        "gjk_epa",
        "support_face",
        "polygonal_contacts",
    ] {
        let mut cases = vec![];
        let support = kind.starts_with("support_")
            || kind == "cso_support"
            || kind == "gjk_closest"
            || kind == "epa_closest"
            || kind == "gjk_epa";
        let counts: &[usize] = if kind == "support_face" {
            &[0, 1, 7, 33, 65, 129, 168]
        } else if kind == "polygonal_contacts" {
            &[0, 1, 25, 65, 129, 300]
        } else if support {
            &[0, 1, 49, 65, 129, 343]
        } else {
            &[0, 1, 24, 33, 65, 129]
        };
        for &count in counts {
            let directory = root.join(kind).join(format!("n{count}"));
            cases.push(if kind == "support_face" {
                feature_fixture::fixture(&directory, count)
            } else if kind == "polygonal_contacts" {
                polygonal_fixture::fixture(&directory, count)
            } else if support {
                support_fixture::fixture(&directory, kind, count)
            } else if kind.starts_with("simplex_") {
                simplex_fixture::fixture(&directory, kind, count, root)
            } else {
                projection_fixture::fixture(&directory, kind, count)
            });
        }
        if kind == "epa_closest" || kind == "gjk_epa" {
            cases.push(support_fixture::touching_fixture(
                &root.join(kind).join("point_touch"),
                kind,
            ));
        }
        fs::write(
            root.join(kind).join("suite.json"),
            serde_json::to_vec_pretty(&json!({"cases":cases})).unwrap(),
        )
        .unwrap();
    }
}
