//! Current Rust link initialization, persistence matching and color-seeding oracle.
#[allow(dead_code)]
#[path = "contacts.rs"]
mod contacts;
use contacts::pilot;
use nexus_rbd_shaders3d::{
    Pose,
    broad_phase::ContactPlan,
    dynamics::{
        ContactLink, ContactRecycleOffsets, ContactRecycleState, RbdSimParams,
        gpu_init_contact_links, gpu_match_contact_links, gpu_seed_colors_from_warmstart,
    },
    glamx::{Quat, UVec2, UVec3, Vec3},
    queries::IndexedManifold,
};
use serde_json::{Value, json};
use std::{
    fs,
    mem::{offset_of, size_of},
    path::Path,
};
fn layout() -> Value {
    let mut l = contacts::layout();
    macro_rules! add {($key:literal,$ty:ty,$($f:ident),+) => {$(l[$key][stringify!($f)]=json!(offset_of!($ty,$f)/4);)+};}
    add!(
        "link",
        ContactLink,
        warmstart_collider_a,
        warmstart_collider_b,
        warmstart_subshape,
        _padding
    );
    add!("contact", IndexedManifold, recycle_extent);
    l["state"] = json!({"pose_a":offset_of!(ContactRecycleState,pose_a)/4,
        "pose_b":offset_of!(ContactRecycleState,pose_b)/4,"colliders":offset_of!(ContactRecycleState,colliders)/4,
        "max_extent":offset_of!(ContactRecycleState,max_extent)/4,"max_drift":offset_of!(ContactRecycleState,max_drift)/4});
    l["params_words"] = json!(size_of::<RbdSimParams>() / 4);
    l["recycle_params"] = json!({"normalized_distance":offset_of!(RbdSimParams,normalized_contact_recycle_distance)/4,
        "length_unit":offset_of!(RbdSimParams,length_unit)/4});
    l
}
fn fixture(root: &Path, n: usize, op: &str) -> Value {
    let name = format!("n{n}");
    let dir = root.join(&name);
    fs::create_dir_all(&dir).unwrap();
    let storage = n.max(1) + 3;
    let old_storage = storage * 4 + 2;
    let bodies = 8;
    let colliders = storage * 2 + 5;
    let flip = n == 65;
    let (old_base, new_base) = if flip {
        (storage + 7, 2)
    } else {
        (3, old_storage + 7)
    };
    let state_storage = (old_base + old_storage).max(new_base + storage) + 3;
    let mut ims = vec![IndexedManifold::default(); storage];
    let mut links = vec![ContactLink::default(); storage];
    bytemuck::cast_slice_mut::<_, u32>(&mut links).fill(0xa5a5a5a5);
    let mut old_links = vec![ContactLink::default(); old_storage];
    let old_indices: Vec<_> = (0..old_storage)
        .map(|i| (old_storage - 1 - i) as u32)
        .collect();
    let old_colors: Vec<u32> = (0..old_storage)
        .map(|i| [0, 1, 63, 64, u32::MAX, 7][(i / 4) % 6])
        .collect();
    let mut colors = vec![0xabcdef01u32; storage];
    let mut colored = vec![0xabcdef02u32; storage];
    let mut is_mb = vec![0u32; bodies];
    is_mb[6] = 1;
    is_mb[7] = 0x80000000;
    let mut poses = vec![Pose::IDENTITY; colliders];
    let mut states = vec![ContactRecycleState::default(); state_storage];
    bytemuck::cast_slice_mut::<_, u32>(&mut states).fill((-19.0f32).to_bits());
    let mut rows = vec![Vec::<u32>::new(); bodies];
    let mut params = RbdSimParams::tgs_soft();
    params.length_unit = if flip { 2.0 } else { 1.0 };
    params.normalized_contact_recycle_distance = 0.05;
    for (i, im_dst) in ims.iter_mut().take(n).enumerate() {
        let mode = i % 18;
        let mut a = if mode == 2 { 0 } else { i % 4 + 1 };
        let mut b = (i * 3 + 1) % 4 + 1;
        if a == b {
            b = b % 4 + 1;
        }
        if mode == 11 {
            a = if (i / 18) % 2 == 0 { 6 } else { 7 };
        }
        if mode == 17 {
            a = 0;
            b = 0;
        }
        let pair = UVec2::new((2 * i + 1) as u32, (2 * i + 2) as u32);
        let mut im = IndexedManifold {
            bodies: UVec2::new(a as u32, b as u32),
            colliders: pair,
            restitution: 0.3,
            friction: 0.4,
            subshape: (i % 5) as u32,
            recycle_extent: if mode == 9 { -1.0 } else { 0.9 },
            ..Default::default()
        };
        im.contact.len = if mode == 10 { 0 } else { (i % 4 + 1) as u32 };
        *im_dst = im;
        let mut old_a = Pose::from_parts(
            Vec3::new(0.17, 0.03 * i as f32, -0.2),
            Quat::from_scaled_axis(Vec3::new(0.12, -0.08, 0.04)),
        );
        let mut old_b = Pose::from_parts(
            Vec3::new(-0.2, 0.03 * i as f32, 0.1),
            Quat::from_scaled_axis(Vec3::new(-0.05, 0.14, 0.01)),
        );
        if (13..=16).contains(&mode) {
            old_a = Pose::IDENTITY;
            old_b = Pose::IDENTITY;
        }
        let mut new_a = old_a;
        let mut new_b = old_b;
        match mode {
            2 => new_b.translation.x += 0.02,
            3 | 9 => new_b.translation.x += 0.12,
            4 => new_b.rotation = Quat::from_rotation_x(0.02) * old_b.rotation,
            5 => new_b.rotation = Quat::from_rotation_x(0.15) * old_b.rotation,
            6 => {
                let global =
                    Pose::from_parts(Vec3::new(0.3, -0.2, 0.1), Quat::from_rotation_z(0.25));
                new_a = global * old_a;
                new_b = global * old_b;
            }
            12 => {
                new_a.rotation = -old_a.rotation;
                new_b.rotation = -old_b.rotation;
            }
            13 => new_b.translation.x = 0.05,
            14 => new_b.translation.x = 0.0501,
            15 => new_b.rotation = Quat::from_rotation_x(0.19),
            16 => new_b.rotation = Quat::from_rotation_x(0.21),
            _ => {}
        }
        old_a.padding = 0x12345678;
        old_b.padding = 0x23456789;
        new_a.padding = 0x34567890;
        new_b.padding = 0x45678901;
        poses[pair.x as usize] = new_a;
        poses[pair.y as usize] = new_b;
        let state = ContactRecycleState {
            pose_a: old_a,
            pose_b: old_b,
            colliders: pair,
            max_extent: if mode == 15 || mode == 16 {
                0.0
            } else if mode == 8 {
                -1.0
            } else {
                0.9
            },
            max_drift: if mode == 7 || mode == 8 { 0.0 } else { 0.05 },
        };
        for j in 0..4 {
            let cid = i * 4 + j;
            old_links[cid] = ContactLink {
                len: 1,
                solver_body_a: a as u32,
                solver_body_b: b as u32,
                warmstart_collider_a: pair.x,
                warmstart_collider_b: pair.y,
                warmstart_subshape: im.subshape + u32::from(j == 0 || mode == 1),
                ..Default::default()
            };
            if j == 2 {
                old_links[cid].warmstart_collider_b += 1;
            }
            states[old_base + cid] = state;
            if a != 0 {
                rows[a].push(cid as u32);
            }
            if b != 0 && b != a {
                rows[b].push(cid as u32);
            }
        }
    }
    let mut old_ids = Vec::new();
    let mut old_counts = Vec::new();
    for (i, row) in rows.iter_mut().enumerate() {
        if i % 2 == 0 {
            row.reverse();
        }
        old_ids.extend(row.iter());
        old_counts.push(old_ids.len() as u32);
    }
    // Unused adjacency storage must not be traversed.
    old_ids.extend([u32::MAX, 0xabcdef01, 0xabcdef02]);
    let plan = ContactPlan {
        bound: n as u32,
        ..Default::default()
    };
    let offsets = ContactRecycleOffsets {
        old_base: old_base as u32,
        new_base: new_base as u32,
        ..Default::default()
    };
    if op == "match_links" || op == "seed_colors" {
        for i in 0..n {
            gpu_init_contact_links(UVec3::new(i as u32, 0, 0), &ims, &mut links, &is_mb, &plan);
        }
        // Gap slots may contain stale IDs. Match and seed must check len first.
        for link in links.iter_mut().take(n) {
            if link.len == 0 {
                link.solver_body_a = u32::MAX;
                link.solver_body_b = u32::MAX;
                link.previous_constraint = u32::MAX;
            }
        }
    }
    if op == "seed_colors" {
        for i in 0..n {
            gpu_match_contact_links(
                UVec3::new(i as u32, 0, 0),
                &ims,
                &mut links,
                &old_counts,
                &old_ids,
                &old_links,
                &old_indices,
                &mut states,
                &poses,
                &plan,
                &params,
                &offsets,
            );
        }
    }
    macro_rules! write {($($name:literal=>$value:expr),+ $(,)?)=>{$(pilot::write_buffer(&dir,$name,$value);)+};}
    write!("Contacts.bin"=>&ims,"Links.bin"=>&links,"BodyIsMultibody.bin"=>&is_mb,
        "OldCounts.bin"=>&old_counts,"OldIds.bin"=>&old_ids,"OldLinks.bin"=>&old_links,
        "OldIndices.bin"=>&old_indices,"States.bin"=>&states,"ColliderPoses.bin"=>&poses,
        "Params.bin"=>&[params],"OldColors.bin"=>&old_colors,"Colors.bin"=>&colors,"Colored.bin"=>&colored,
        "ContactPlan.bin"=>&[plan],"RecycleOffsets.bin"=>&[offsets]);
    for _ in 0..3 {
        for i in 0..n.div_ceil(64).max(1) * 64 {
            let gid = UVec3::new(i as u32, 0, 0);
            if op == "init_links" || op == "contact_persistence" {
                gpu_init_contact_links(gid, &ims, &mut links, &is_mb, &plan);
            }
            if op == "match_links" || op == "contact_persistence" {
                gpu_match_contact_links(
                    gid,
                    &ims,
                    &mut links,
                    &old_counts,
                    &old_ids,
                    &old_links,
                    &old_indices,
                    &mut states,
                    &poses,
                    &plan,
                    &params,
                    &offsets,
                );
            }
            if op == "seed_colors" || op == "contact_persistence" {
                gpu_seed_colors_from_warmstart(
                    gid,
                    &links,
                    &old_colors,
                    &mut colors,
                    &mut colored,
                    &plan,
                );
            }
        }
    }
    write!("Links.expected.bin"=>&links,"States.expected.bin"=>&states,"Colors.expected.bin"=>&colors,"Colored.expected.bin"=>&colored);
    json!({"name":name,"kind":op,"count":n,"storage":storage,"bodies":bodies,"colliders":colliders,
        "old_storage":old_storage,"ids_storage":old_ids.len(),"state_storage":state_storage,
        "old_base":old_base,"new_base":new_base,"iterations":3})
}
fn main() {
    let out = std::env::args().nth(1).expect("Expected output directory");
    let out = Path::new(&out);
    fs::create_dir_all(out).unwrap();
    pilot::write_json(out.join("layout.json"), &layout());
    for op in [
        "init_links",
        "match_links",
        "seed_colors",
        "contact_persistence",
    ] {
        let root = out.join(op);
        fs::create_dir_all(&root).unwrap();
        let cases: Vec<_> = [0, 1, 31, 32, 33, 63, 64, 65, 129]
            .into_iter()
            .map(|n| fixture(&root, n, op))
            .collect();
        pilot::write_json(root.join("suite.json"), &json!({"cases":cases}));
    }
}
