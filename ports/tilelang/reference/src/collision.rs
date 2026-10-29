//! Native ordered analytic collision pipeline oracle.
#[allow(dead_code)]
#[path = "collision_layout.rs"]
mod collision_layout;
use collision_layout::contacts::pilot;
use nexus_rbd_shaders3d::{
    PaddedVector, Pose,
    bounding_volumes::Aabb,
    broad_phase::*,
    dynamics::RbdSimParams,
    glamx::{Quat, UVec2, UVec3, Vec3},
    queries::{ColliderMaterial, IndexedManifold},
    rapier::geometry::{Group, InteractionGroups, InteractionTestMode},
    shapes::*,
    utils::BatchIndices,
};
use serde_json::{Value, json};
use std::{fs, path::Path};
fn layout() -> Value {
    collision_layout::layout()
}
fn fixture(root: &Path, n: usize, nb: usize, mode: &str) -> Value {
    let name = format!("n{n}-b{nb}-{mode}");
    let dir = root.join(&name);
    fs::create_dir_all(&dir).unwrap();
    let cap = n + 3;
    let storage = cap * nb + 3;
    let pair_capacity = (n * n * nb).max(1);
    let pair_storage = pair_capacity + 3;
    let contact_storage = pair_capacity * 2 + 3;
    let ids = BatchIndices {
        num_batches: nb as u32,
        colliders_len: n as u32,
        colliders_batch_capacity: cap as u32,
        collision_pairs_capacity: pair_capacity as u32,
        contacts_capacity: (pair_capacity * 2) as u32,
        ..Default::default()
    };
    let mut poses = vec![Pose::IDENTITY; storage];
    let mut shapes = vec![Shape::ball(0.5); storage];
    let mut aabbs = vec![Aabb::new(Vec3::splat(-17.), Vec3::splat(23.)); storage];
    for a in &mut aabbs {
        a.padding0 = 0xdeadbeef;
        a.padding1 = 0xabcdef12;
    }
    let mut materials = vec![ColliderMaterial::default(); storage];
    let mut parents: Vec<u32> = (0..storage as u32).collect();
    let mut filters: Vec<[u32; 2]> = (0..storage as u32).map(|i| [i, 0]).collect();
    let mut groups = vec![InteractionGroups::all(); storage];
    for i in 0..n {
        for batch in 0..nb {
            let k = i * nb + batch;
            let spacing = if mode == "separated" { 3.0 } else { 0.91 };
            poses[k] = Pose::from_parts(
                Vec3::new(
                    (i % 8) as f32 * spacing,
                    ((i / 8) % 8) as f32 * spacing,
                    (i / 64) as f32 * spacing + batch as f32 * 8.0,
                ),
                if mode == "rotated" {
                    Quat::from_scaled_axis(Vec3::new(0.07 * i as f32, 0.13, -0.17))
                } else {
                    Quat::IDENTITY
                },
            );
            poses[k].padding = 0x12345678;
            if mode == "boxes" || mode == "rotated" || (mode != "balls" && i % 2 == 1) {
                shapes[k] = Shape::cuboid(Vec3::new(0.51, 0.43, 0.47));
            }
            if mode == "filters" {
                parents[k] = ((i / 2) * nb + batch) as u32;
                filters[k] = [parents[k], if i % 3 == 0 { 7 } else { 0 }];
            }
            if mode == "groups" {
                groups[k] = InteractionGroups::new(
                    Group::from_bits_retain(1 << (i % 4)),
                    Group::from_bits_retain(if i % 3 == 0 { 15 } else { 1 << ((i + 1) % 4) }),
                    if i % 2 == 0 {
                        InteractionTestMode::And
                    } else {
                        InteractionTestMode::Or
                    },
                );
            }
            materials[k].friction = 0.2 + (i % 4) as f32 * 0.1;
            materials[k].restitution = (i % 3) as f32 * 0.2;
            materials[k].friction_combine_rule = (i % 5) as u32;
            materials[k].restitution_combine_rule = ((i + 1) % 5) as u32;
        }
    }
    let vertices = [PaddedVector::new(Vec3::ZERO)];
    let mut pairs = vec![
        CollisionPair {
            colliders: UVec2::splat(0xdeadbeef)
        };
        pair_storage
    ];
    let mut pair_count = [0, 0xdeadbeef, 0xabcdef12];
    let mut pfm_count = [0xabcdef12, 0xdeadbeef, 0x12345678];
    let mut plan = [ContactPlan {
        bound: 0xdeadbeef,
        pfm_base: 0xabcdef12,
        pfm_len: 0x12345678,
    }; 2];
    let mut sort_len = [0xdeadbeef; 3];
    let mut contacts_grid = [0xdeadbeef; 6];
    let mut pfm_grid = [0xdeadbeef; 6];
    let mut mb_grid = [0xdeadbeef; 6];
    let mut output = vec![IndexedManifold::default(); contact_storage];
    bytemuck::cast_slice_mut::<_, u32>(&mut output).fill(0xdeadbeef);
    let mut p = RbdSimParams::tgs_soft();
    p.normalized_prediction_distance = 0.08;
    macro_rules! buffers {
        ($suffix:expr) => {{
            macro_rules! w {
                ($n:expr,$v:expr) => {
                    pilot::write_buffer(&dir, &format!("{}{}", $n, $suffix), $v)
                };
            }
            w!("Poses", &poses);
            w!("Shapes", &shapes);
            w!("Aabbs", &aabbs);
            w!("BatchIds", &[ids]);
            w!("Vertices", &vertices);
            w!("Pairs", &pairs);
            w!("PairCount", &pair_count);
            w!("Filters", &filters);
            w!("RbdParams", &[p]);
            w!("Parents", &parents);
            w!("Materials", &materials);
            w!("Contacts", &output);
            w!("PfmCount", &pfm_count);
            w!("Plan", &plan);
            w!("PfmSortLen", &sort_len);
            w!("ContactsIndirect", &contacts_grid);
            w!("PfmIndirect", &pfm_grid);
            w!("MbIndirect", &mb_grid);
            fs::write(
                dir.join(format!("Groups{}", $suffix)),
                bytemuck::cast_slice(&groups),
            )
            .unwrap();
        }};
    }
    buffers!(".bin");
    gpu_reset_narrow_phase(UVec3::ZERO, &mut pfm_count[..1]);
    for gid in 0..(n * nb).div_ceil(64).max(1) * 64 {
        gpu_bf_compute_aabbs(
            UVec3::new(gid as u32, 0, 0),
            &poses,
            &shapes,
            &mut aabbs,
            &ids,
            &vertices,
        );
    }
    let pair_groups = (n * n * nb).div_ceil(64).max(1);
    for gid in 0..pair_groups * 64 {
        gpu_bf_find_pairs(
            UVec3::new(gid as u32, 0, 0),
            &aabbs,
            &mut pairs,
            &mut pair_count,
            &groups,
            &ids,
            &filters,
            &p,
        );
    }
    gpu_contact_plan(
        &mut pair_count,
        &mut pfm_count,
        &mut plan[0],
        &mut sort_len,
        (&mut contacts_grid[..3]).try_into().unwrap(),
        (&mut pfm_grid[..3]).try_into().unwrap(),
        (&mut mb_grid[..3]).try_into().unwrap(),
        &ids,
    );
    // A fixed grid consumes the device plan through the native grid-stride loop.
    for gid in 0..64 {
        gpu_narrow_phase_shape_shape(
            UVec3::new(gid, 0, 0),
            UVec3::ONE,
            &pairs,
            &plan[0],
            &poses,
            &shapes,
            &mut output,
            &parents,
            &materials,
            &p,
        );
    }
    buffers!(".expected.bin");
    let sizes: serde_json::Map<String, Value> = [
        "Poses",
        "Shapes",
        "Aabbs",
        "BatchIds",
        "Vertices",
        "Pairs",
        "PairCount",
        "Groups",
        "Filters",
        "RbdParams",
        "Parents",
        "Materials",
        "Contacts",
        "PfmCount",
        "Plan",
        "PfmSortLen",
        "ContactsIndirect",
        "PfmIndirect",
        "MbIndirect",
    ]
    .into_iter()
    .map(|n| {
        (
            n.to_owned(),
            json!(fs::metadata(dir.join(format!("{n}.bin"))).unwrap().len() / 4),
        )
    })
    .collect();
    json!({"name":name,"kind":"analytic_collision","count":n*nb,"colliders":n,"batches":nb,"capacity":cap,"storage":storage,"pair_capacity":pair_capacity,"pair_storage":pair_storage,"vertices":1,"dispatch":n*n*nb,"groups":pair_groups,"contact_groups":1,"contact_storage":contact_storage,"iterations":1,"initial_count":0,"mode":mode,"buffer_words":sizes})
}
fn main() {
    let out = std::env::args().nth(1).unwrap();
    let root = Path::new(&out);
    fs::create_dir_all(root).unwrap();
    pilot::write_json(root.join("layout.json"), &layout());
    let dir = root.join("analytic_collision");
    fs::create_dir_all(&dir).unwrap();
    let mut cases = Vec::new();
    for (n, nb, m) in [
        (0, 1, "mixed"),
        (1, 1, "mixed"),
        (3, 2, "mixed"),
        (7, 3, "mixed"),
        (31, 1, "mixed"),
        (32, 2, "mixed"),
        (33, 1, "mixed"),
        (63, 1, "mixed"),
        (64, 1, "mixed"),
        (65, 2, "mixed"),
        (129, 1, "mixed"),
        (33, 2, "balls"),
        (33, 2, "boxes"),
        (33, 2, "rotated"),
        (33, 2, "filters"),
        (33, 2, "groups"),
        (33, 2, "separated"),
    ] {
        cases.push(fixture(&dir, n, nb, m));
    }
    pilot::write_json(dir.join("suite.json"), &json!({"cases":cases}));
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn repeated_export_keeps_the_input_abi() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "nexus-collision-export-{}-{nonce}",
            std::process::id()
        ));
        let first = fixture(&root, 7, 2, "mixed");
        let dir = root.join(first["name"].as_str().unwrap());
        let output = fs::read(dir.join("Contacts.expected.bin")).unwrap();
        fs::write(dir.join("Contacts.actual.bin"), b"runtime output").unwrap();
        let second = fixture(&root, 7, 2, "mixed");
        assert_eq!(first, second);
        assert_eq!(output, fs::read(dir.join("Contacts.expected.bin")).unwrap());
        assert_eq!(
            fs::read(dir.join("Contacts.actual.bin")).unwrap(),
            b"runtime output"
        );
        fs::remove_dir_all(root).unwrap();
    }
}
