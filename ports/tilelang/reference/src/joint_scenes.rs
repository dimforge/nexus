use serde_json::json;
use std::{fs, path::Path};
mod bucket_program;
mod coloring_program;
mod dispatch;
mod joint_layout;
mod prefix_program;
#[allow(dead_code)]
#[path = "radix_program.rs"]
mod radix;
mod scene_program;
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let root = Path::new(&path);
    fs::create_dir_all(root).unwrap();
    fs::write(
        root.join("layout.json"),
        serde_json::to_vec_pretty(&joint_layout::layout()).unwrap(),
    )
    .unwrap();
    let directory = root.join("joint_scene");
    fs::create_dir_all(&directory).unwrap();
    let mut cases = Vec::new();
    for (label, columns, levels, nb, frames) in [
        ("joint_fixed", 1, 1, 1, 1),
        ("joint_ball", 1, 1, 1, 1),
        ("joint_revolute", 1, 1, 1, 1),
        ("joint_prismatic", 1, 1, 1, 1),
        ("joint_motor", 1, 1, 1, 1),
        ("joint_impact", 1, 1, 1, 1),
        ("joint_chain", 1, 3, 1, 1),
        ("joint_fixed", 1, 1, 1, 120),
        ("joint_ball", 1, 1, 1, 120),
        ("joint_revolute", 1, 1, 1, 120),
        ("joint_prismatic", 1, 1, 1, 120),
        ("joint_motor", 1, 1, 1, 120),
        ("joint_impact", 1, 1, 1, 120),
        ("joint_chain", 1, 3, 1, 120),
        ("joint_chain", 1, 3, 3, 8),
        ("joint_fixed", 3, 1, 2, 8),
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
