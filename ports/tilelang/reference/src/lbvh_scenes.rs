//! Connected native LBVH scenes sharing the ordinary scene and shader implementation.
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
    if selected
        .as_deref()
        .is_none_or(|s| s == "analytic_lbvh_scene")
    {
        let directory = root.join("analytic_lbvh_scene");
        fs::create_dir_all(&directory).unwrap();
        let mut cases = vec![];
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
            cases.push(scene_program::fixture_config(
                &directory.join(name),
                label,
                columns,
                levels,
                nb,
                frames,
                &[],
                true,
            ));
        }
        fs::write(
            directory.join("suite.json"),
            serde_json::to_vec_pretty(&json!({"cases":cases})).unwrap(),
        )
        .unwrap();
    }
    if selected
        .as_deref()
        .is_none_or(|s| s == "general_lbvh_scene")
    {
        let directory = root.join("general_lbvh_scene");
        fs::create_dir_all(&directory).unwrap();
        let mut cases = vec![];
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
            cases.push(scene_program::fixture_config(
                &directory.join(name),
                label,
                columns,
                levels,
                nb,
                frames,
                &[],
                true,
            ));
        }
        fs::write(
            directory.join("suite.json"),
            serde_json::to_vec_pretty(&json!({"cases":cases})).unwrap(),
        )
        .unwrap();
    }
    if selected.as_deref().is_none_or(|s| s == "joint_lbvh_scene") {
        let directory = root.join("joint_lbvh_scene");
        fs::create_dir_all(&directory).unwrap();
        let mut cases = vec![];
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
            cases.push(scene_program::fixture_config(
                &directory.join(name),
                label,
                columns,
                levels,
                nb,
                frames,
                &[],
                true,
            ));
        }
        fs::write(
            directory.join("suite.json"),
            serde_json::to_vec_pretty(&json!({"cases":cases})).unwrap(),
        )
        .unwrap();
    }
}
