//! Current native twist preparation, nearest-anchor transfer and recycled geometry oracle.
use twist::contacts::pilot;
#[allow(dead_code)]
mod twist;
use nexus_rbd_shaders3d::{
    Pose,
    broad_phase::ContactPlan,
    dynamics::{
        ContactLink, ContactRecycleOffsets, ContactRecycleState,
        TwistConstraint as TwoBodyConstraint, Velocity, WorldMassProperties,
        twist_tiles::{TwistTile as CoulombTile, gpu_prepare_constraints},
    },
    glamx::{Mat4, Quat, UVec2, UVec3, Vec2, Vec3},
    queries::{ContactPoint, IndexedManifold},
    utils::Slice,
};
use serde_json::{Value, json};
use std::{fs, path::Path};

pub(crate) fn layout() -> Value {
    twist::layout()
}

fn fixture(root: &Path, n: usize, mode: &str) -> Value {
    let name = format!("n{n}-{mode}");
    let dir = root.join(&name);
    fs::create_dir_all(&dir).unwrap();
    let storage = n.max(1) + 3;
    let bodies = 7;
    let new_base = storage + 2;
    let tile_count = storage.div_ceil(32);
    let poses: Vec<_> = (0..bodies)
        .map(|i| {
            Pose::from_parts(
                Vec3::new(i as f32 * 0.03, -0.2, 0.1),
                Quat::from_scaled_axis(Vec3::new(0.02 * i as f32, 0.1, -0.13)),
            )
        })
        .collect();
    let vels: Vec<_> = (0..bodies)
        .map(|i| {
            let mut v = Velocity::new(
                Vec3::new(0.3, -0.17 * i as f32, 0.2),
                Vec3::new(-0.2, 0.13, 0.07 * i as f32),
            );
            v.padding0 = 0x12345678;
            v.padding1 = 0x76543210;
            v
        })
        .collect();
    let mprops: Vec<_> = (0..bodies)
        .map(|i| WorldMassProperties {
            inv_mass: if i == 0 {
                Vec3::ZERO
            } else {
                Vec3::new(0.7, 0.0, 0.9)
            },
            inv_inertia: if i == 0 {
                Mat4::ZERO
            } else {
                Mat4::from_cols_array(&[
                    0.8, 0.03, -0.02, 0.0, 0.03, 0.6, 0.04, 0.0, -0.02, 0.04, 0.9, 0.0, 0.0, 0.0,
                    0.0, 0.0,
                ])
            },
            com: poses[i].translation,
            padding0: 0x12345678,
            padding1: 0x12345678,
        })
        .collect();
    let mut contacts = vec![IndexedManifold::default(); storage];
    let mut links = vec![
        ContactLink {
            contact: u32::MAX,
            previous_constraint: u32::MAX,
            previous_constraint_index: u32::MAX,
            ..Default::default()
        };
        storage
    ];
    let mut states = vec![ContactRecycleState::default(); new_base + storage];
    let mut old = vec![CoulombTile::default(); tile_count];
    let mut tiles = vec![CoulombTile::default(); tile_count];
    // Poison every output word, including lanes/points that must remain untouched.
    bytemuck::cast_slice_mut::<_, u32>(&mut tiles).fill((-17.0f32).to_bits());
    for (i, link) in links.iter_mut().take(n).enumerate() {
        let source = n - 1 - i;
        let old_index = (i + 1) % n;
        let a = i % 6 + 1;
        let b = if i % 3 == 0 { 0 } else { (i + 2) % 6 + 1 };
        let len = if n == 1 { 4 } else { i % 4 + 1 };
        let kind = if mode == "mixed" {
            match i % 5 {
                0 => "fresh",
                1 => "nearest",
                2 => "rotated",
                3 => "recycled",
                _ => "no-match",
            }
        } else {
            mode
        };
        let mut cp = Pose::from_parts(
            Vec3::new(0.07, 0.1, 0.03 * i as f32),
            Quat::from_scaled_axis(Vec3::new(0.2, -0.1, 0.05)),
        );
        if kind.starts_with("normal-") {
            cp.rotation = Quat::IDENTITY;
        }
        let mut im = IndexedManifold {
            colliders: UVec2::new(100 + i as u32, 200 + i as u32),
            bodies: UVec2::new(a as u32, b as u32),
            friction: 0.3 + 0.1 * (i % 3) as f32,
            restitution: 0.4,
            subshape: i as u32 + 2,
            ..Default::default()
        };
        im.contact.len = len as u32;
        im.contact.normal_a = Vec3::new(0.2, -1.0, 0.1).normalize();
        im.contact.normal_a = match kind {
            "normal-positive-z" => Vec3::Z,
            "normal-negative-z" => -Vec3::Z,
            "normal-zero-z" => Vec3::Y,
            _ => im.contact.normal_a,
        };
        for k in 0..len {
            im.contact.points_a[k] = ContactPoint::new(
                Vec3::new(k as f32 * 0.02, 0.0, 0.01 * (k % 2) as f32),
                -0.01 * (k + 1) as f32,
            );
        }
        let mut c = TwoBodyConstraint::default();
        c.init_contact_with_poses(
            &im,
            &Slice(&mprops, 0),
            cp,
            poses[a],
            poses[b],
            &Slice(&vels, 0),
        );
        for k in 0..len {
            c.points[k].normal_impulse = 0.15 * (k + 1) as f32;
        }
        c.friction.tangent_impulse = Vec2::new(0.03, -0.02);
        c.friction.twist_impulse = if len > 1 { 0.01 } else { 0. };
        // Reverse old point order: first-candidate matching would be incorrect.
        c.points[..len].reverse();
        if kind == "rotated" {
            c.tangent_a = c.dir_a.cross(c.tangent_a);
        }
        if kind == "no-match" {
            for k in 0..len {
                c.points[k].r_a += c.frame_a * Vec3::splat(0.4);
            }
        }
        if kind == "ties" {
            for k in 1..len {
                c.points[k].r_a = c.points[0].r_a;
            }
        }
        if kind == "threshold" {
            for k in 0..len {
                let delta = if k % 2 == 0 { 0.099 } else { 0.101 };
                c.points[k].r_a += c.frame_a * (Vec3::X * delta);
            }
        }
        if kind == "empty-old" || kind == "recycled-empty" {
            c.len = 0;
        }
        old[old_index / 32].write_constraint(old_index % 32, &c);
        if kind == "recycled" || kind == "recycled-empty" {
            im.contact.len = 1;
        }
        contacts[source] = im;
        states[new_base + source] = ContactRecycleState {
            pose_a: cp,
            pose_b: Pose::IDENTITY,
            colliders: im.colliders,
            max_extent: 1.0,
            max_drift: 0.1,
        };
        *link = ContactLink {
            len: if kind == "inactive" || (mode == "mixed" && i % 11 == 10) {
                0
            } else {
                im.contact.len
            },
            solver_body_a: a as u32,
            solver_body_b: b as u32,
            vel_slot_a: (50 + i * 2) as u32,
            vel_slot_b: (51 + i * 2) as u32,
            warmstart_collider_a: im.colliders.x,
            warmstart_collider_b: im.colliders.y,
            warmstart_subshape: im.subshape,
            previous_constraint: if kind == "fresh" {
                u32::MAX
            } else {
                1000 + i as u32
            },
            previous_constraint_index: old_index as u32,
            recycled: u32::from(kind == "recycled" || kind == "recycled-empty"),
            mass_scale_a: if kind == "zero-mass" {
                0.0
            } else {
                1.0 + (i % 3) as f32
            },
            mass_scale_b: if kind == "zero-mass" {
                0.0
            } else {
                1.0 + (i % 2) as f32
            },
            restitution: 0.7,
            contact: source as u32,
            _padding: 0,
        };
    }
    let plan = ContactPlan {
        bound: n as u32,
        ..Default::default()
    };
    let offsets = ContactRecycleOffsets {
        old_base: 0,
        new_base: new_base as u32,
        ..Default::default()
    };
    pilot::write_buffer(&dir, "Links.bin", &links);
    pilot::write_buffer(&dir, "Contacts.bin", &contacts);
    pilot::write_buffer(&dir, "WorldMass.bin", &mprops);
    pilot::write_buffer(&dir, "SolverPoses.bin", &poses);
    pilot::write_buffer(&dir, "Velocities.bin", &vels);
    pilot::write_buffer(&dir, "States.bin", &states);
    pilot::write_buffer(&dir, "OldTiles.bin", &old);
    pilot::write_buffer(&dir, "Tiles.bin", &tiles);
    pilot::write_buffer(
        &dir,
        "Plan.bin",
        &[
            plan,
            ContactPlan {
                bound: 0xdeadbeef,
                pfm_base: 0xabcdef12,
                pfm_len: 0x12345678,
            },
        ],
    );
    pilot::write_buffer(&dir, "ContactPlan.bin", &[plan]);
    pilot::write_buffer(&dir, "RecycleOffsets.bin", &[offsets]);
    for _ in 0..3 {
        for i in 0..n.div_ceil(64).max(1) * 64 {
            gpu_prepare_constraints(
                UVec3::new(i as u32, 0, 0),
                &links,
                &contacts,
                &mprops,
                &poses,
                &vels,
                &states,
                &old,
                &mut tiles,
                &plan,
                &offsets,
            );
        }
    }
    pilot::write_buffer(&dir, "Tiles.expected.bin", &tiles);
    json!({"name":name,"kind":"twist_prepare","count":n,"launch_capacity":storage,"storage":storage,"bodies":bodies,
        "new_base":new_base,"states":states.len(),"tiles":tile_count,"old_tiles":old.len(),"iterations":3})
}
fn main() {
    let out = std::env::args().nth(1).expect("Expected output directory");
    let out = Path::new(&out);
    fs::create_dir_all(out).unwrap();
    pilot::write_json(out.join("layout.json"), &layout());
    let root = out.join("twist_prepare");
    fs::create_dir_all(&root).unwrap();
    let mut cases = Vec::new();
    for n in [0, 31, 32, 33, 63, 64, 65, 129] {
        cases.push(fixture(&root, n, "mixed"));
    }
    for mode in [
        "fresh",
        "nearest",
        "rotated",
        "recycled",
        "no-match",
        "threshold",
        "ties",
        "inactive",
        "normal-positive-z",
        "normal-negative-z",
        "normal-zero-z",
        "empty-old",
        "recycled-empty",
        "zero-mass",
    ] {
        cases.push(fixture(&root, 1, mode));
    }
    pilot::write_json(root.join("suite.json"), &json!({"cases":cases}));
}
