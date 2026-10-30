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
#[path = "twist_layout.rs"]
mod twist_layout;
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        (args.len() == 6 || args.len() == 7)
            || (args.len() == 4 && matches!(args[3].as_str(), "suite" | "suite-twist")),
        "runtime-scenes OUTPUT INPUT FRAMES PAIR_CAPACITY execute|upload [reduced|unreduced] OR OUTPUT INPUT_ROOT suite"
    );
    let root = Path::new(&args[1]);
    fs::create_dir_all(root).unwrap();
    let is_suite = matches!(args[3].as_str(), "suite" | "suite-twist");
    let twist_model = args[3] == "suite-twist"
        || (!is_suite
            && scene_program::scene_upload::SceneUpload::read(Path::new(&args[2]))
                .unwrap()
                .params
                .friction_model
                == nexus_rbd_shaders3d::dynamics::FrictionModel::Simplified);
    let mut layout = joint_layout::layout();
    if twist_model {
        twist_layout::add_layout(&mut layout);
    }
    layout["rbd_params"]["friction_model"] = json!(
        std::mem::offset_of!(nexus_rbd_shaders3d::dynamics::RbdSimParams, friction_model) / 4
    );
    layout["friction_models"] = json!({"coulomb":bytemuck::cast::<_,u32>(nexus_rbd_shaders3d::dynamics::FrictionModel::Coulomb),"twist":bytemuck::cast::<_,u32>(nexus_rbd_shaders3d::dynamics::FrictionModel::Simplified)});
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
    let directory = root.join(if twist_model {
        "runtime_twist_scene"
    } else {
        "runtime_scene"
    });
    let configurations = if matches!(args[3].as_str(), "suite" | "suite-twist") {
        [
            ("multi-collider-joint", 64),
            ("mesh-contact-joint", 64),
            ("wide-contacts", 512),
            ("many-batches", 64),
            ("large-batch-storage", 1024),
            ("split-hub", 128),
            ("split-hubs", 128),
            ("mesh-overlap-reduced", 64),
            ("mesh-overlap-unreduced", 64),
        ]
        .into_iter()
        .flat_map(|(name, capacity)| {
            [1, 8].map(|frames| {
                (
                    format!("{name}-f{frames}"),
                    Path::new(&args[2]).join(name),
                    frames,
                    capacity,
                    true,
                )
            })
        })
        .collect::<Vec<_>>()
    } else {
        let frames: u32 = args[3].parse().unwrap();
        let capacity: usize = args[4].parse().unwrap();
        let execute = match args[5].as_str() {
            "execute" => true,
            "upload" => false,
            _ => panic!("Unknown runtime export mode"),
        };
        assert!(frames > 0 && capacity > 0);
        vec![(
            "native-upload".to_owned(),
            Path::new(&args[2]).to_owned(),
            frames,
            capacity,
            execute,
        )]
    };
    let mut cases = Vec::new();
    for (name, input, frames, capacity, execute) in configurations {
        let mut upload = scene_program::scene_upload::SceneUpload::read(&input).unwrap();
        upload.contact_reduction = if matches!(args[3].as_str(), "suite" | "suite-twist") {
            !name.contains("unreduced")
        } else {
            match args.get(6).map(String::as_str).unwrap_or("unreduced") {
                "reduced" => true,
                "unreduced" => false,
                _ => panic!("Invalid reduction mode"),
            }
        };
        assert_eq!(
            upload.params.friction_model
                == nexus_rbd_shaders3d::dynamics::FrictionModel::Simplified,
            twist_model
        );
        let case = scene_program::scene_storage(
            &directory.join(name),
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
        cases.push(case);
    }
    fs::write(
        directory.join("suite.json"),
        serde_json::to_vec_pretty(&json!({"cases":cases})).unwrap(),
    )
    .unwrap();
}
