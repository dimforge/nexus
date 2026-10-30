//! Native initial storage ladders and full-capacity reference trajectories.
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
    let mut l = joint_layout::layout();
    macro_rules! fields {($ty:ty,$($f:ident),+)=>{json!({$(stringify!($f):std::mem::offset_of!($ty,$f)/4),+})}}
    use nexus_rbd_shaders3d::broad_phase::LbvhNode;
    l["max_flt_bits"] = json!(nexus_rbd_shaders3d::MAX_FLT.to_bits());
    l["lbvh_node_words"] = json!(std::mem::size_of::<LbvhNode>() / 4);
    l["lbvh_node"] = fields!(
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
    fs::write(
        root.join("layout.json"),
        serde_json::to_vec_pretty(&l).unwrap(),
    )
    .unwrap();
    let selected = std::env::args().nth(2);
    for (op, cases) in [
        (
            "analytic_frame_retry",
            vec![("capacity_growth", 2, 1, 1, 120, 48)],
        ),
        (
            "general_frame_retry",
            vec![
                ("general_mesh", 1, 1, 1, 1, 1),
                ("general_mixed", 7, 1, 2, 8, 1),
            ],
        ),
        ("joint_frame_retry", vec![("joint_impact", 3, 1, 2, 8, 1)]),
    ] {
        if selected.as_deref().is_some_and(|name| name != op) {
            continue;
        }
        let directory = root.join(op);
        fs::create_dir_all(&directory).unwrap();
        let mut rows = vec![];
        for (label, columns, levels, nb, frames, growth_frame) in cases {
            let name = format!("{label}-c{columns}-l{levels}-b{nb}-f{frames}");
            let dir = directory.join(&name);
            let mut case =
                scene_program::fixture_config(&dir, label, columns, levels, nb, frames, &[], true);
            let final_capacity = case["pair_capacity"].as_u64().unwrap() as usize;
            let capacities = [1, final_capacity];
            for (i, capacity) in capacities.iter().enumerate() {
                scene_program::fixture_storage(
                    &dir.join("variants").join(i.to_string()),
                    label,
                    columns,
                    levels,
                    nb,
                    frames,
                    &[],
                    true,
                    Some(*capacity),
                    false,
                );
            }
            case["capacity_ladder"] = json!(capacities);
            case["expected_growth_frames"] = json!([growth_frame]);
            fs::write(
                dir.join("case.json"),
                serde_json::to_vec_pretty(&case).unwrap(),
            )
            .unwrap();
            rows.push(case);
        }
        fs::write(
            directory.join("suite.json"),
            serde_json::to_vec_pretty(&json!({"cases":rows})).unwrap(),
        )
        .unwrap();
    }
}
