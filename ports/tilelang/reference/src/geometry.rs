//! Live native geometry oracle. Private GJK source is linked into this binary
//! without changing production visibility or duplicating its implementation.
pub use nexus_rbd_shaders3d::{Vector, shapes};
#[path = "../../../../src_rbd_shaders/queries/gjk/cso_point.rs"]
pub mod cso_point;
#[path = "../../../../src_rbd_shaders/queries/gjk/voronoi_simplex3.rs"]
pub mod voronoi_simplex3;
mod queries {
    pub mod projection {
        pub use nexus_rbd_shaders3d::queries::{FEATURE_EDGE, FEATURE_FACE, FEATURE_VERTEX};
    }
    pub mod gjk {
        pub use crate::cso_point;
        pub use crate::voronoi_simplex3::VoronoiSimplex;
    }
}
mod geometry_layout;
mod projection_fixture;
mod simplex_fixture;
use serde_json::json;
use std::{fs, path::Path};
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let root = Path::new(&path);
    fs::create_dir_all(root).unwrap();
    fs::write(
        root.join("layout.json"),
        serde_json::to_vec_pretty(&geometry_layout::layout()).unwrap(),
    )
    .unwrap();
    for kind in [
        "segment_projection",
        "triangle_projection",
        "tetrahedron_projection",
        "simplex_init",
        "simplex_add",
        "simplex_project",
    ] {
        let mut cases = vec![];
        for count in [0, 1, 24, 33, 65, 129] {
            let directory = root.join(kind).join(format!("n{count}"));
            cases.push(if kind.starts_with("simplex_") {
                simplex_fixture::fixture(&directory, kind, count, root)
            } else {
                projection_fixture::fixture(&directory, kind, count)
            });
        }
        fs::write(
            root.join(kind).join("suite.json"),
            serde_json::to_vec_pretty(&json!({"cases":cases})).unwrap(),
        )
        .unwrap();
    }
}
