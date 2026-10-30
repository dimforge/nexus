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
#[allow(dead_code)]
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
    let mut cases = [Vec::new(), Vec::new()];
    for (label, levels, snapshots) in [
        (
            "general_triangle",
            1,
            vec![0u32, 16, 18, 19, 21, 26, 28, 33, 35, 39, 119],
        ),
        ("joint_chain", 3, vec![3u32, 80, 90, 98, 99, 100, 109, 119]),
    ] {
        let source = root.join("native_trajectories").join(label);
        let case =
            scene_program::fixture_with_snapshots(&source, label, 1, levels, 1, 120, &snapshots);
        for frame in snapshots {
            let name = format!("{label}-restart-{frame}");
            let index = usize::from(label.starts_with("joint_"));
            let operation = if index == 0 {
                "general_restart"
            } else {
                "joint_restart"
            };
            let dir = root.join(operation).join(&name);
            fs::create_dir_all(&dir).unwrap();
            let snapshot = source.join(format!("restart-{frame}"));
            for key in case["buffer_words"].as_object().unwrap().keys() {
                for suffix in ["", ".expected"] {
                    let filename = format!("{key}{suffix}.bin");
                    let captured = snapshot.join(&filename);
                    let parent = source.join(&filename);
                    let chosen = if captured.exists() {
                        captured
                    } else if parent.exists() {
                        parent
                    } else {
                        source.join(format!("{key}.bin"))
                    };
                    fs::copy(chosen, dir.join(filename)).unwrap();
                }
            }
            let mut restart = case.clone();
            restart["name"] = json!(name);
            restart["start_frame"] = json!(frame);
            restart["iterations"] = json!(1);
            fs::write(
                dir.join("case.json"),
                serde_json::to_vec_pretty(&restart).unwrap(),
            )
            .unwrap();
            cases[index].push(restart);
        }
    }
    for (op, rows) in ["general_restart", "joint_restart"].into_iter().zip(cases) {
        fs::write(
            root.join(op).join("suite.json"),
            serde_json::to_vec_pretty(&json!({"cases":rows})).unwrap(),
        )
        .unwrap();
    }
}
