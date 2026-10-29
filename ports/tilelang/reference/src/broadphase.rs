//! Native world bounds and brute-force candidate pair oracles.
#[allow(dead_code)]
#[path = "main.rs"]
mod pilot;
use nexus_rbd_shaders3d::{
    PaddedVector, Pose,
    bounding_volumes::Aabb,
    broad_phase::*,
    dynamics::RbdSimParams,
    glamx::{Quat, UVec2, UVec3, Vec3, Vec4},
    rapier::geometry::{Group, InteractionGroups, InteractionTestMode},
    shapes::*,
    utils::BatchIndices,
};
use serde_json::{Value, json};
use std::{
    fs,
    mem::{offset_of, size_of},
    path::Path,
};
fn layout() -> Value {
    let mut l = pilot::layout();
    macro_rules! f {($ty:ty,$($field:ident),+)=>{json!({$(stringify!($field):offset_of!($ty,$field)/4),+})}}
    l["shape_words"] = json!(size_of::<Shape>() / 4);
    l["shape"] = f!(Shape, a, b, c);
    l["shape_types"] = json!({"ball":SHAPE_TYPE_BALL,"cuboid":SHAPE_TYPE_CUBOID,"capsule":SHAPE_TYPE_CAPSULE,"cone":SHAPE_TYPE_CONE,"cylinder":SHAPE_TYPE_CYLINDER,"polyline":SHAPE_TYPE_POLYLINE,"trimesh":SHAPE_TYPE_TRIMESH,"convex_poly":SHAPE_TYPE_CONVEX_POLY,"triangle":SHAPE_TYPE_TRIANGLE});
    l["polyline_thickness_bits"] = json!(POLYLINE_THICKNESS.to_bits());
    l["aabb_words"] = json!(size_of::<Aabb>() / 4);
    l["aabb"] = f!(Aabb, mins, maxs, padding0, padding1);
    l["vertex_words"] = json!(size_of::<PaddedVector>() / 4);
    l["vertex_vector"] = json!(offset_of!(PaddedVector, 0) / 4);
    l["group_words"] = json!(size_of::<InteractionGroups>() / 4);
    l["groups"] = f!(InteractionGroups, memberships, filter, test_mode);
    l["pair_words"] = json!(size_of::<CollisionPair>() / 4);
    l["pair_colliders"] = json!(offset_of!(CollisionPair, colliders) / 4);
    l["batch_words"] = json!(size_of::<BatchIndices>() / 4);
    l["batch"] = f!(
        BatchIndices,
        num_batches,
        colliders_batch_capacity,
        colliders_len,
        collision_pairs_capacity
    );
    l["rbd_params_words"] = json!(size_of::<RbdSimParams>() / 4);
    l["rbd_params"] = f!(RbdSimParams, normalized_prediction_distance, length_unit);
    l
}
fn shape(ty: u32) -> Shape {
    match ty {
        SHAPE_TYPE_BALL => Shape::ball(0.75),
        SHAPE_TYPE_CUBOID => Shape::cuboid(Vec3::new(0.3, 0.5, 0.9)),
        SHAPE_TYPE_CAPSULE => {
            Shape::capsule(Vec3::new(-0.4, -0.7, 0.2), Vec3::new(0.3, 0.8, -0.1), 0.25)
        }
        SHAPE_TYPE_CONE => Shape::cone(0.8, 0.4),
        SHAPE_TYPE_CYLINDER => Shape::cylinder(0.6, 0.5),
        SHAPE_TYPE_POLYLINE => Shape::polyline(
            17,
            23,
            11,
            Vec3::new(-0.3, -0.7, -0.9),
            Vec3::new(0.5, 0.4, 0.2),
        ),
        SHAPE_TYPE_TRIMESH => Shape::trimesh(
            17,
            23,
            11,
            5,
            7,
            Vec3::new(-0.3, -0.7, -0.9),
            Vec3::new(0.5, 0.4, 0.2),
        ),
        SHAPE_TYPE_CONVEX_POLY => Shape::convex_poly(1, 6, 0, 0),
        SHAPE_TYPE_TRIANGLE => Shape::from_triangle(&Triangle::new(
            Vec3::new(-0.3, 0.7, -0.4),
            Vec3::new(0.2, -0.8, 0.1),
            Vec3::new(0.6, 0.2, 0.4),
        )),
        _ => Shape::new(
            Vec4::new(1., 2., 3., f32::from_bits(ty)),
            Vec4::ZERO,
            Vec4::ZERO,
        ),
    }
}
fn fixture(root: &Path, n: usize, nb: usize, mode: &str, op: &str) -> Value {
    let name = format!("n{n}-b{nb}-{mode}");
    let dir = root.join(&name);
    fs::create_dir_all(&dir).unwrap();
    let capacity = n + 3;
    let storage = (capacity * nb).max(1) + 3;
    let is_bounds = op == "world_aabbs";
    let program = op == "broadphase_pairs";
    let pair_capacity = if mode == "overflow" {
        3
    } else if mode == "zero-capacity" {
        0
    } else {
        n * n * nb + 4
    };
    let initial = if mode == "append" { 2 } else { 0 };
    let ids = BatchIndices {
        num_batches: nb as u32,
        colliders_len: n as u32,
        colliders_batch_capacity: capacity as u32,
        collision_pairs_capacity: pair_capacity as u32,
        ..Default::default()
    };
    let mut p = RbdSimParams::tgs_soft();
    p.normalized_prediction_distance = if mode == "touching" { 0. } else { 0.125 };
    p.length_unit = if mode == "scaled" { 2. } else { 1. };
    let mut poses = vec![Pose::IDENTITY; storage];
    let mut shapes = vec![Shape::ball(0.); storage];
    let mut boxes = vec![Aabb::new(Vec3::splat(-17.), Vec3::splat(23.)); storage];
    for a in &mut boxes {
        a.padding0 = 0xdeadbeef;
        a.padding1 = 0xfedabc12;
    }
    let mut groups = vec![InteractionGroups::all(); storage];
    let mut filters = vec![[0u32; 2]; storage];
    let mut pairs = vec![
        CollisionPair {
            colliders: UVec2::new(0xdeadbeef, 0xabcdef12)
        };
        pair_capacity.max(1) + 3
    ];
    let mut len = [initial, 0xdeadbeef, 0xabcdef12];
    let vertices = vec![
        PaddedVector::new(Vec3::splat(-99.)),
        PaddedVector::new(Vec3::new(-0.7, -0.3, 0.2)),
        PaddedVector::new(Vec3::new(0.6, 0.1, -0.8)),
        PaddedVector::new(Vec3::new(0.2, 0.8, 0.3)),
        PaddedVector::new(Vec3::new(-0.1, 0.2, 0.9)),
        PaddedVector::new(Vec3::new(0.3, -0.6, -0.2)),
        PaddedVector::new(Vec3::splat(99.)),
    ];
    for batch in 0..nb {
        for i in 0..n {
            let index = i * nb + batch;
            poses[index] = Pose::from_parts(
                Vec3::new(i as f32 * 0.125 - 2., batch as f32 * 0.25, -0.3),
                Quat::from_scaled_axis(Vec3::new(0.17 * i as f32, -0.3, 0.2)),
            );
            poses[index].padding = 0x12345678;
            let ty = if let Some(s) = mode.strip_prefix("shape-") {
                s.parse().unwrap()
            } else {
                (i % 10) as u32
            };
            shapes[index] = shape(ty);
            if mode == "identity-box" || mode == "negative-zero" {
                shapes[index] = Shape::cuboid(Vec3::new(
                    if mode == "negative-zero" { -0.0 } else { 0.3 },
                    0.5,
                    0.9,
                ));
                poses[index].rotation = Quat::from_xyzw(-0.0, 0.0, -0.0, 1.0);
            }
            if mode == "degenerate" {
                shapes[index] = Shape::capsule(Vec3::ZERO, Vec3::ZERO, 0.);
            }
            if mode == "empty-convex" {
                shapes[index] = Shape::convex_poly(2, 2, 0, 0);
            }
            let lo = Vec3::new(
                (i % 4) as f32 * 0.25,
                ((i / 4) % 3) as f32 * 0.25,
                batch as f32 * 4.,
            );
            let hi = lo + Vec3::splat(if mode == "touching" { 0.25 } else { 0.2 });
            boxes[batch * capacity + i] = Aabb::new(lo, hi);
            filters[index] = [i as u32, 0];
            if mode == "groups" || mode == "filters" {
                groups[index] = InteractionGroups::new(
                    Group::from_bits_retain(1u32 << (i % 32)),
                    Group::from_bits_retain(if i % 4 == 0 {
                        0xffffffff
                    } else {
                        1u32 << ((i + 1) % 32)
                    }),
                    if (i / 2) % 2 == 0 {
                        InteractionTestMode::Or
                    } else {
                        InteractionTestMode::And
                    },
                );
            }
            if mode == "filters" {
                filters[index] = [(i / 2) as u32, if i % 3 == 0 { 7 } else { 0 }];
            }
        }
    }
    if is_bounds || program {
        for a in &mut boxes {
            a.mins = Vec3::splat(-17.);
            a.maxs = Vec3::splat(23.);
            a.padding0 = 0xdeadbeef;
            a.padding1 = 0xfedabc12;
        }
    }
    macro_rules! w {
        ($name:expr,$data:expr) => {
            pilot::write_buffer(&dir, $name, $data)
        };
    }
    w!("Poses.bin", &poses);
    w!("Shapes.bin", &shapes);
    w!("Aabbs.bin", &boxes);
    w!("Vertices.bin", &vertices);
    w!("BatchIds.bin", &[ids]);
    w!("RbdParams.bin", &[p]);
    w!("Filters.bin", &filters);
    w!("Pairs.bin", &pairs);
    w!("PairCount.bin", &len);
    // InteractionGroups has a valid enum and NoUninit representation, not Pod.
    fs::write(dir.join("Groups.bin"), bytemuck::cast_slice(&groups)).unwrap();
    let dispatch = if is_bounds || op == "brute_pairs_serial" {
        n * nb
    } else {
        n * n * nb
    };
    let workgroups = dispatch.div_ceil(64).max(1);
    if is_bounds || program {
        for i in 0..(n * nb).div_ceil(64).max(1) * 64 {
            gpu_bf_compute_aabbs(
                UVec3::new(i as u32, 0, 0),
                &poses,
                &shapes,
                &mut boxes,
                &ids,
                &vertices,
            );
        }
    }
    if !is_bounds {
        for i in 0..workgroups * 64 {
            if op == "brute_pairs_serial" {
                gpu_bf_find_pairs_serial(
                    UVec3::new(i as u32, 0, 0),
                    &boxes,
                    &mut pairs,
                    &mut len,
                    &groups,
                    &ids,
                    &filters,
                    &p,
                );
            } else {
                gpu_bf_find_pairs(
                    UVec3::new(i as u32, 0, 0),
                    &boxes,
                    &mut pairs,
                    &mut len,
                    &groups,
                    &ids,
                    &filters,
                    &p,
                );
            }
        }
    }
    w!("Aabbs.expected.bin", &boxes);
    w!("Pairs.expected.bin", &pairs);
    w!("PairCount.expected.bin", &len);
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
    ]
    .into_iter()
    .map(|name| {
        (
            name.to_owned(),
            json!(fs::metadata(dir.join(format!("{name}.bin"))).unwrap().len() / 4),
        )
    })
    .collect();
    json!({"name":name,"kind":op,"count":n*nb,"colliders":n,"batches":nb,"capacity":capacity,"storage":storage,"pair_capacity":pair_capacity,"pair_storage":pairs.len(),"vertices":vertices.len(),"dispatch":dispatch,"groups":workgroups,"iterations":1,"initial_count":initial,"mode":mode,"buffer_words":sizes})
}
fn main() {
    let out = std::env::args().nth(1).unwrap();
    let root = Path::new(&out);
    fs::create_dir_all(root).unwrap();
    pilot::write_json(root.join("layout.json"), &layout());
    for op in [
        "world_aabbs",
        "brute_pairs",
        "brute_pairs_serial",
        "broadphase_pairs",
    ] {
        let d = root.join(op);
        fs::create_dir_all(&d).unwrap();
        let mut cases = Vec::new();
        for (n, nb, mode) in [
            (0, 1, "mixed"),
            (1, 1, "mixed"),
            (31, 1, "mixed"),
            (32, 2, "mixed"),
            (33, 3, "mixed"),
            (63, 1, "mixed"),
            (64, 1, "mixed"),
            (65, 2, "mixed"),
            (129, 1, "mixed"),
            (7, 3, "groups"),
            (17, 2, "filters"),
            (17, 2, "overflow"),
            (17, 2, "zero-capacity"),
            (5, 2, "append"),
            (8, 1, "touching"),
            (8, 2, "scaled"),
        ] {
            cases.push(fixture(&d, n, nb, mode, op));
        }
        if op == "world_aabbs" {
            for ty in 0..10 {
                cases.push(fixture(&d, 1, 2, &format!("shape-{ty}"), op));
            }
            for mode in [
                "degenerate",
                "empty-convex",
                "identity-box",
                "negative-zero",
            ] {
                cases.push(fixture(&d, 1, 1, mode, op));
            }
        }
        pilot::write_json(d.join("suite.json"), &json!({"cases":cases}));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn repeated_export_keeps_a_fixed_input_abi() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("nexus-broad-export-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let first = fixture(&root, 5, 2, "append", "broadphase_pairs");
        let d = root.join(first["name"].as_str().unwrap());
        let state = fs::read(d.join("Pairs.expected.bin")).unwrap();
        fs::write(d.join("Pairs.actual.bin"), b"runtime output").unwrap();
        let second = fixture(&root, 5, 2, "append", "broadphase_pairs");
        let second_state = fs::read(d.join("Pairs.expected.bin")).unwrap();
        let artifact = fs::read(d.join("Pairs.actual.bin")).unwrap();
        fs::remove_dir_all(root).unwrap();
        assert_eq!(first, second);
        assert_eq!(state, second_state);
        assert_eq!(artifact, b"runtime output");
    }
}
