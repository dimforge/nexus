//! Native typed specialization and differential oracle for public scene uploads.
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
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        6,
        "runtime-scenes OUTPUT INPUT FRAMES PAIR_CAPACITY execute|upload"
    );
    let root = Path::new(&args[1]);
    let upload = scene_program::scene_upload::SceneUpload::read(Path::new(&args[2])).unwrap();
    let frames: u32 = args[3].parse().unwrap();
    let capacity: usize = args[4].parse().unwrap();
    let execute = match args[5].as_str() {
        "execute" => true,
        "upload" => false,
        _ => panic!("Unknown runtime export mode"),
    };
    assert!(frames > 0 && capacity > 0);
    fs::create_dir_all(root).unwrap();
    let mut layout = joint_layout::layout();
    use nexus_rbd_shaders3d::broad_phase::LbvhNode;
    layout["max_flt_bits"] = json!(nexus_rbd_shaders3d::MAX_FLT.to_bits());
    layout["lbvh_node_words"] = json!(std::mem::size_of::<LbvhNode>() / 4);
    macro_rules! fields { ($($field:ident),+) => { json!({$(stringify!($field): std::mem::offset_of!(LbvhNode, $field) / 4),+}) }; }
    layout["lbvh_node"] = fields!(
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
        serde_json::to_vec_pretty(&layout).unwrap(),
    )
    .unwrap();
    let directory = root.join("runtime_scene");
    let case = scene_program::scene_storage(
        &directory.join("native-upload"),
        "runtime",
        0,
        1,
        upload.batch.num_batches as usize,
        frames,
        &[],
        true,
        Some(capacity),
        execute,
        Some(&upload),
    );
    fs::write(
        directory.join("suite.json"),
        serde_json::to_vec_pretty(&json!({"cases":[case]})).unwrap(),
    )
    .unwrap();
}
