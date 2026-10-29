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
    let directory = root.join("analytic_scene");
    fs::create_dir_all(&directory).unwrap();
    let mut cases = Vec::new();
    for (label, columns, levels, nb, frames) in [
        ("empty", 0, 0, 1, 1),
        ("ground", 0, 0, 2, 1),
        ("stack", 1, 3, 1, 1),
        ("free", 1, 1, 1, 1),
        ("friction", 3, 1, 1, 1),
        ("boxes", 2, 3, 1, 1),
        ("impact", 2, 1, 1, 1),
        ("stack", 1, 3, 1, 120),
        ("free", 1, 1, 1, 120),
        ("friction", 3, 1, 1, 120),
        ("boxes", 2, 3, 1, 120),
        ("impact", 2, 1, 1, 120),
        ("stack", 2, 3, 3, 8),
        ("friction", 33, 1, 2, 1),
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
