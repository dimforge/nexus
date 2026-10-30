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
    let selected = std::env::args().nth(2);
    let use_lbvh = matches!(
        selected.as_deref(),
        Some("general_lbvh_restart" | "joint_lbvh_restart")
    );
    let mut layout = joint_layout::layout();
    if use_lbvh {
        use nexus_rbd_shaders3d::broad_phase::LbvhNode;
        macro_rules! fields {($ty:ty,$($f:ident),+)=>{json!({$(stringify!($f):std::mem::offset_of!($ty,$f)/4),+})}}
        layout["max_flt_bits"] = json!(nexus_rbd_shaders3d::MAX_FLT.to_bits());
        layout["lbvh_node_words"] = json!(std::mem::size_of::<LbvhNode>() / 4);
        layout["lbvh_node"] = fields!(
            LbvhNode,
            aabb,
            left,
            right,
            parent,
            refit_count,
            first_leaf,
            last_leaf,
            escape,
            max_extent
        );
    }
    fs::write(
        root.join("layout.json"),
        serde_json::to_vec_pretty(&layout).unwrap(),
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
        if use_lbvh
            && (label.starts_with("joint_") != (selected.as_deref() == Some("joint_lbvh_restart")))
        {
            continue;
        }
        let snapshots = if use_lbvh && label == "general_triangle" {
            vec![0, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 119]
        } else {
            snapshots
        };
        let source = root.join("native_trajectories").join(label);
        let case =
            scene_program::fixture_config(&source, label, 1, levels, 1, 120, &snapshots, use_lbvh);
        for frame in snapshots {
            let name = format!("{label}-restart-{frame}");
            let index = usize::from(label.starts_with("joint_"));
            let operation = if use_lbvh {
                if index == 0 {
                    "general_lbvh_restart"
                } else {
                    "joint_lbvh_restart"
                }
            } else if index == 0 {
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
    for (op, rows) in [
        if use_lbvh {
            "general_lbvh_restart"
        } else {
            "general_restart"
        },
        if use_lbvh {
            "joint_lbvh_restart"
        } else {
            "joint_restart"
        },
    ]
    .into_iter()
    .zip(cases)
    {
        if rows.is_empty() {
            continue;
        }
        fs::write(
            root.join(op).join("suite.json"),
            serde_json::to_vec_pretty(&json!({"cases":rows})).unwrap(),
        )
        .unwrap();
    }
}
