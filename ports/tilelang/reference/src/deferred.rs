#[allow(dead_code)]
#[path = "collision_layout.rs"]
mod collision_layout;
use glamx::{Quat, UVec2, UVec3, Vec3};
use nexus_rbd_shaders3d::{
    PaddedVector, Pose,
    broad_phase::*,
    dynamics::RbdSimParams,
    queries::{ColliderMaterial, IndexedManifold},
    shapes::*,
    utils::BatchIndices,
};
use serde_json::{Value, json};
use std::{fs, path::Path};
fn layout() -> Value {
    crate::collision_layout::layout()
}
fn shape(ty: usize) -> Shape {
    match ty {
        0 => Shape::ball(0.5),
        1 => Shape::cuboid(Vec3::new(0.5, 0.7, 0.3)),
        2 => Shape::capsule(Vec3::new(0.1, -0.5, 0.), Vec3::new(-0.1, 0.5, 0.1), 0.2),
        3 => Shape::cone(0.7, 0.5),
        4 => Shape::cylinder(0.7, 0.5),
        5 => Shape::convex_poly(2, 10, 3, 42),
        6 => Shape::from_triangle(&Triangle::new(
            Vec3::new(-0.5, -0.5, 0.),
            Vec3::new(0.5, -0.5, 0.),
            Vec3::new(0., 0.5, 0.1),
        )),
        7 => Shape::trimesh(
            10,
            43,
            3,
            2,
            4,
            Vec3::new(-0.6, -0.5, 0.),
            Vec3::new(0.6, 0.5, 0.),
        ),
        _ => Shape::polyline(
            10,
            58,
            3,
            Vec3::new(-0.6, -0.5, 0.),
            Vec3::new(0.6, 0.5, 0.),
        ),
    }
}
fn fixture(root: &Path, n: usize, capacity: usize, groups: usize, mode: &str, kind: &str) -> Value {
    let name = format!("n{n}-c{capacity}-g{groups}-{mode}");
    let dir = root.join(&name);
    fs::create_dir_all(&dir).unwrap();
    let guard = 0xdeadbeef;
    let storage = n.max(capacity) + 3;
    let colliders = n * 2 + 3;
    let mut pairs = vec![CollisionPair::default(); storage];
    let mut pair_count = [n as u32, guard, guard];
    let mut poses = vec![Pose::IDENTITY; colliders];
    let mut shapes = vec![shape(0); colliders];
    let mut parents: Vec<u32> = (0..colliders as u32).collect();
    let mut materials = vec![ColliderMaterial::default(); colliders];
    let mut pfm = vec![NarrowPhasePfmPair::default(); capacity + 3];
    bytemuck::cast_slice_mut::<_, u32>(&mut pfm).fill(guard);
    let mut pfm_count = [0, guard, guard];
    let mut vertices = vec![PaddedVector::new(Vec3::splat(19.)); 2];
    vertices.extend((0..8).map(|i| {
        PaddedVector::new(Vec3::new(
            if i & 1 == 0 { -0.5 } else { 0.5 },
            if i & 2 == 0 { -0.7 } else { 0.7 },
            if i & 4 == 0 { -0.3 } else { 0.3 },
        ))
    }));
    for _ in 0..3 {
        vertices.push(PaddedVector::new(Vec3::new(-0.6, -0.5, 0.)));
        vertices.push(PaddedVector::new(Vec3::new(0.6, 0.5, 0.)));
    }
    for p in [
        Vec3::new(-0.6, -0.5, 0.),
        Vec3::new(0.6, -0.5, 0.),
        Vec3::new(0.6, 0.5, 0.),
        Vec3::new(-0.6, 0.5, 0.),
    ] {
        vertices.push(PaddedVector::new(p));
    }
    vertices.push(PaddedVector::new(Vec3::splat(23.)));
    let mut indices = vec![
        guard, guard, guard, 0, 0, 0, 0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5, 0, 1, 5, 0, 5, 4, 2, 6,
        7, 2, 7, 3, 0, 2, 3, 0, 3, 1, 4, 5, 7, 4, 7, 6, guard,
    ];
    indices.extend([1, 3, 0, u32::MAX, 2, 0, u32::MAX, 3, 1, 0, 1, 2, 0, 2, 3]);
    indices.extend([1, 3, 0, u32::MAX, 2, 0, u32::MAX, 3, 1, 0, 1, 2, 3, guard]);
    for i in 0..n {
        pairs[i].colliders = UVec2::new((i * 2) as u32, (i * 2 + 1) as u32);
        shapes[i * 2] = shape(i % 9);
        shapes[i * 2 + 1] = shape((i / 9) % 9);
        let profile = if mode == "overflow" { 2 } else { (i / 81) % 7 };
        if mode == "overflow" {
            shapes[i * 2] = shape(7);
            shapes[i * 2 + 1] = shape(0);
        }
        let shift = match profile {
            0 => Vec3::new(3., 0.1, 0.),
            1 => Vec3::new(1.3, 0.2, -0.1),
            2 => Vec3::new(0.2, -0.1, 0.1),
            3 => Vec3::ZERO,
            4 => Vec3::new(0.9, 0.1, 0.),
            5 => Vec3::new(0.1, 2., 0.3),
            _ => Vec3::new(-0.4, 0.2, -1.2),
        };
        poses[i * 2] = Pose::from_parts(
            Vec3::new((i % 3) as f32 * 0.1, -0.2, 0.3),
            Quat::from_scaled_axis(Vec3::new(0.1, -0.2, 0.05) * (i % 3) as f32),
        );
        poses[i * 2 + 1] = poses[i * 2]
            * Pose::from_parts(
                shift,
                if profile % 2 == 0 {
                    Quat::IDENTITY
                } else {
                    Quat::from_scaled_axis(Vec3::new(0.17, -0.11, 0.23) * (i % 5) as f32)
                },
            );
        if mode == "same-body" && i % 3 == 0 {
            parents[i * 2 + 1] = parents[i * 2];
        }
        for j in 0..2 {
            materials[i * 2 + j].friction = 0.2 + ((i + j) % 5) as f32 * 0.1;
            materials[i * 2 + j].friction_combine_rule = ((i + j) % 5) as u32;
            materials[i * 2 + j].restitution = 0.1 * ((i + j) % 3) as f32;
            materials[i * 2 + j].restitution_combine_rule = ((i + j + 2) % 5) as u32;
        }
    }
    let ids = BatchIndices {
        num_batches: 1,
        colliders_len: colliders as u32,
        colliders_batch_capacity: colliders as u32,
        collision_pairs_capacity: capacity as u32,
        contacts_capacity: (capacity * 2) as u32,
        ..Default::default()
    };
    let mut params = RbdSimParams::tgs_soft();
    params.normalized_prediction_distance = 0.02;
    params.length_unit = 1.;
    if mode == "merge-all" {
        params.contact_merge_cos = -1.;
    }
    macro_rules! write {
        ($name:expr,$values:expr) => {
            fs::write(dir.join($name), bytemuck::cast_slice($values)).unwrap()
        };
    }
    write!("Pairs.bin", &pairs);
    write!("PairCount.bin", &pair_count);
    write!("Poses.bin", &poses);
    write!("Shapes.bin", &shapes);
    write!("Parents.bin", &parents);
    write!("Materials.bin", &materials);
    write!("PfmPairs.bin", &pfm);
    write!("PfmCount.bin", &pfm_count);
    write!("Vertices.bin", &vertices);
    write!("Indices.bin", &indices);
    write!("BatchIds.bin", &[ids]);
    write!("RbdParams.bin", &[params]);
    for gid in 0..groups * 64 {
        gpu_narrow_phase_shape_shape_deferred(
            UVec3::new(gid as u32, 0, 0),
            UVec3::new(groups as u32, 1, 1),
            &pairs,
            &mut pair_count,
            &poses,
            &shapes,
            &mut pfm,
            &mut pfm_count,
            &vertices,
            &indices,
            &ids,
            &params,
        );
    }
    write!("PfmPairs.expected.bin", &pfm);
    write!("PfmCount.expected.bin", &pfm_count);
    // Native candidate superset is diagnostic evidence only: no device kernel
    // reads it. Overflow accepts a legal unique subset, independent of atomic order.
    let candidate_capacity = capacity.max(pfm_count[0] as usize);
    let mut candidates = vec![NarrowPhasePfmPair::default(); candidate_capacity + 3];
    bytemuck::cast_slice_mut::<_, u32>(&mut candidates).fill(guard);
    let candidate_ids = BatchIndices {
        collision_pairs_capacity: candidate_capacity as u32,
        ..ids
    };
    let mut candidate_input = [n.min(capacity) as u32, guard, guard];
    let mut candidate_count = [0, guard, guard];
    for gid in 0..groups * 64 {
        gpu_narrow_phase_shape_shape_deferred(
            UVec3::new(gid as u32, 0, 0),
            UVec3::new(groups as u32, 1, 1),
            &pairs,
            &mut candidate_input,
            &poses,
            &shapes,
            &mut candidates,
            &mut candidate_count,
            &vertices,
            &indices,
            &candidate_ids,
            &params,
        );
    }
    assert_eq!(candidate_count[0], pfm_count[0]);
    write!("PfmCandidates.bin", &candidates);

    let len = pfm_count[0].min(capacity as u32);
    let mut order: Vec<u32> = (0..capacity as u32 + 3).collect();
    order[..len as usize].sort_by_key(|i| {
        let pair = pfm[*i as usize];
        (pair.pair_index, pair.subshape)
    });
    let plan = ContactPlan {
        bound: (n.min(capacity)) as u32 + len,
        pfm_base: n.min(capacity) as u32,
        pfm_len: len,
    };
    let mut contacts = vec![IndexedManifold::default(); capacity * 2 + 3];
    bytemuck::cast_slice_mut::<_, u32>(&mut contacts).fill(guard);
    write!("Contacts.bin", &contacts);
    write!("Order.bin", &order);
    write!("Plan.bin", &[plan]);
    for gid in 0..groups * 64 {
        gpu_narrow_phase_pfm_pfm(
            UVec3::new(gid as u32, 0, 0),
            UVec3::new(groups as u32, 1, 1),
            &mut contacts,
            &pfm,
            &order,
            &plan,
            &vertices,
            &indices,
            &parents,
            &materials,
            &params,
        );
    }
    write!("Contacts.expected.bin", &contacts);
    if kind != "deferred_pairs" {
        write!("PfmPairs.bin", &pfm);
    }
    let mut keys = vec![guard; capacity + 3];
    write!("Keys.bin", &keys);
    if kind == "reduce_contacts" {
        for i in 0..len as usize {
            keys[i] = pfm[order[i] as usize].pair_index;
        }
        write!("Keys.bin", &keys);
        write!("Contacts.bin", &contacts);
        for gid in 0..groups * 64 {
            gpu_reduce_contacts(
                UVec3::new(gid as u32, 0, 0),
                UVec3::new(groups as u32, 1, 1),
                &mut contacts,
                &keys,
                &plan,
                &params,
            );
        }
        write!("Contacts.expected.bin", &contacts);
    } else if kind == "pfm_pair_keys" || kind == "pfm_subshape_keys" || kind == "pfm_permuted_keys"
    {
        for gid in 0..groups * 64 {
            let gid = UVec3::new(gid as u32, 0, 0);
            let groups = UVec3::new(groups as u32, 1, 1);
            if kind == "pfm_pair_keys" {
                gpu_pfm_sort_keys(gid, groups, &pfm, &plan, &mut keys);
            } else if kind == "pfm_subshape_keys" {
                gpu_pfm_sub_shape_sort_keys(gid, groups, &pfm, &plan, &mut keys);
            } else {
                gpu_pfm_pair_sort_keys_permuted(gid, groups, &pfm, &plan, &order, &mut keys);
            }
        }
    }
    write!("Keys.expected.bin", &keys);
    let names = [
        "Pairs",
        "PairCount",
        "Poses",
        "Shapes",
        "Parents",
        "Materials",
        "PfmPairs",
        "PfmCount",
        "Vertices",
        "Indices",
        "BatchIds",
        "RbdParams",
        "Contacts",
        "Order",
        "Plan",
        "Keys",
        "PfmCandidates",
    ];
    let sizes = names
        .into_iter()
        .map(|name| {
            (
                name.to_owned(),
                json!(fs::metadata(dir.join(format!("{name}.bin"))).unwrap().len() / 4),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let c = json!({"name":name,"kind":kind,"count":n,"capacity":capacity,"candidate_count":candidate_count[0],"candidate_capacity":candidate_capacity,"groups":groups,"iterations":1,"buffer_words":sizes});
    fs::write(
        dir.join("case.json"),
        serde_json::to_vec_pretty(&c).unwrap(),
    )
    .unwrap();
    c
}
fn main() {
    let arg = std::env::args().nth(1).unwrap();
    let p = Path::new(&arg);
    fs::create_dir_all(p).unwrap();
    fs::write(
        p.join("layout.json"),
        serde_json::to_vec_pretty(&layout()).unwrap(),
    )
    .unwrap();
    for kind in [
        "deferred_pairs",
        "deferred_manifolds",
        "pfm_pair_keys",
        "pfm_subshape_keys",
        "pfm_permuted_keys",
        "reduce_contacts",
    ] {
        let mut cases = vec![];
        for (n, cap, g, mode) in [
            (0, 1, 1, "normal"),
            (1, 4, 1, "normal"),
            (81, 200, 1, "normal"),
            (129, 400, 2, "same-body"),
            (567, 1200, 3, "normal"),
            (567, 17, 2, "overflow"),
            (567, 1200, 2, "merge-all"),
        ] {
            cases.push(fixture(&p.join(kind), n, cap, g, mode, kind));
        }
        fs::write(
            p.join(kind).join("suite.json"),
            serde_json::to_vec_pretty(&json!({"cases":cases})).unwrap(),
        )
        .unwrap();
    }
}
