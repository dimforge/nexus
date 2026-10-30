use serde_json::json;
use std::{fs, path::Path};
mod bucket_program;
mod coloring_program;
mod dispatch;
mod prefix_program;
#[allow(dead_code)]
#[path = "radix_program.rs"]
mod radix;
mod scene_layout;
mod scene_program;
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let root = Path::new(&path);
    fs::create_dir_all(root).unwrap();
    fs::write(
        root.join("layout.json"),
        serde_json::to_vec_pretty(&scene_layout::layout()).unwrap(),
    )
    .unwrap();
    let directory = root.join("general_scene");
    fs::create_dir_all(&directory).unwrap();
    let mut cases = Vec::new();
    for (label, columns, levels, nb, frames) in [
        ("general_capsule", 1, 1, 1, 1),
        ("general_cone", 1, 1, 1, 1),
        ("general_cylinder", 1, 1, 1, 1),
        ("general_convex", 1, 1, 1, 1),
        ("general_triangle", 1, 1, 1, 1),
        ("general_mesh", 1, 1, 1, 1),
        ("general_polyline", 1, 1, 1, 1),
        ("general_mixed", 7, 1, 2, 8),
        ("general_capsule", 1, 1, 1, 120),
        ("general_cone", 1, 1, 1, 120),
        ("general_cylinder", 1, 1, 1, 120),
        ("general_convex", 1, 3, 1, 120),
        ("general_triangle", 1, 1, 1, 120),
        ("general_mesh", 1, 1, 1, 120),
        ("general_polyline", 1, 1, 1, 120),
    ] {
        let name = format!("{label}-c{columns}-l{levels}-b{nb}-f{frames}");
        cases.push(scene_program::fixture(
            &directory.join(name),
            label,
            columns,
            levels,
            nb,
            frames,
        ));
    }
    fs::write(
        directory.join("suite.json"),
        serde_json::to_vec_pretty(&json!({"cases":cases})).unwrap(),
    )
    .unwrap();
}
