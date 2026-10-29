//! Ordered TGS frame over caller-supplied, color-ordered contact manifolds.
#[allow(dead_code)]
#[path = "colored.rs"]
mod colored;
use colored::contacts::pilot;
use nexus_rbd_shaders3d::{
    Pose,
    broad_phase::ContactPlan,
    dynamics::{coulomb_tiles::*, *},
    glamx::{UVec2, UVec3, Vec3, Vec4},
    queries::{ContactPoint, IndexedManifold},
    utils::BatchIndices,
};
use serde_json::{Value, json};
use std::{
    fs,
    mem::{offset_of, size_of},
    path::Path,
};
pub(crate) fn layout() -> Value {
    let mut layout = colored::layout();
    for (key, value) in [
        ("pose_padding", offset_of!(Pose, padding) / 4),
        ("local_words", size_of::<LocalMassProperties>() / 4),
        ("local_com", offset_of!(LocalMassProperties, com) / 4),
        (
            "local_inertia_frame",
            offset_of!(LocalMassProperties, inertia_ref_frame) / 4,
        ),
        (
            "local_inv_inertia",
            offset_of!(LocalMassProperties, inv_principal_inertia) / 4,
        ),
        (
            "local_inv_mass",
            offset_of!(LocalMassProperties, inv_mass) / 4,
        ),
        (
            "world_inertia",
            offset_of!(WorldMassProperties, inv_inertia) / 4,
        ),
        ("world_com", offset_of!(WorldMassProperties, com) / 4),
        (
            "world_padding0",
            offset_of!(WorldMassProperties, padding0) / 4,
        ),
        (
            "world_padding1",
            offset_of!(WorldMassProperties, padding1) / 4,
        ),
        ("world_words", size_of::<WorldMassProperties>() / 4),
        (
            "world_inv_mass",
            offset_of!(WorldMassProperties, inv_mass) / 4,
        ),
        ("hub_words", size_of::<HubCounts>() / 4),
        ("hub_hubs", offset_of!(HubCounts, hubs) / 4),
        ("hub_slots", offset_of!(HubCounts, slots) / 4),
    ] {
        layout[key] = json!(value);
    }

    layout["rbd_params"]["normalized_max_linear_velocity"] =
        json!(offset_of!(RbdSimParams, normalized_max_linear_velocity) / 4);
    layout["rbd_params"]["num_solver_iterations"] =
        json!(offset_of!(RbdSimParams, num_solver_iterations) / 4);
    layout["rbd_params"]["warmstart_coefficient"] =
        json!(offset_of!(RbdSimParams, warmstart_coefficient) / 4);
    layout["constants"]["pi_over_four_bits"] = json!(std::f32::consts::FRAC_PI_4.to_bits());
    layout["batch"]["colliders_batch_capacity"] =
        json!(offset_of!(BatchIndices, colliders_batch_capacity) / 4);
    layout["rbd_params"]["num_internal_pgs_iterations"] =
        json!(offset_of!(RbdSimParams, num_internal_pgs_iterations) / 4);
    layout["rbd_params"]["friction_in_bias_pass"] =
        json!(offset_of!(RbdSimParams, friction_in_bias_pass) / 4);
    layout["params_words"] = json!(size_of::<RbdSimParams>() / 4);
    layout["warmstart_coefficient"] = json!(offset_of!(RbdSimParams, warmstart_coefficient) / 4);
    layout
}
#[allow(clippy::too_many_arguments)]
fn fixture(
    root: &Path,
    label: &str,
    columns: usize,
    levels: usize,
    nb: usize,
    substeps: u32,
    pgs: u32,
    mode: u32,
    scale: f32,
    warmstart: f32,
) -> Value {
    let n = if label == "free" {
        0
    } else {
        columns * levels * nb
    };
    let active = columns * (levels + 1) * nb;
    let storage = active.max(1) + 3;
    let capacity = n.max(1) + 3;
    let tiles_len = capacity.div_ceil(32);
    let dir = root.join(label);
    fs::create_dir_all(&dir).unwrap();
    let mut poses = vec![Pose::IDENTITY; storage];
    let mut solver_poses = poses.clone();
    let mut vels = vec![Velocity::default(); storage];
    let mut solver_vels = vels.clone();
    let mut inc = vels.clone();
    let mut local = vec![LocalMassProperties::default(); storage];
    let mut world = vec![WorldMassProperties::default(); storage];
    let mut counts = vec![0xabcdef12u32; storage];
    let mut first = vec![0x12345678u32; storage];
    let mut hubs = HubCounts {
        hubs: 3,
        slots: 7,
        base: 0xdead1234,
        pool: 0xabcde123,
    };
    let mut p = RbdSimParams::tgs_soft();
    p.dt = 1. / 60. / substeps as f32;
    p.num_solver_iterations = substeps;
    p.length_unit = scale;
    p.warmstart_coefficient = warmstart;
    p.friction_in_bias_pass = u32::from(mode == 2);
    p.num_internal_pgs_iterations = pgs;
    let gravity = Vec4::new(0., -9.81 * scale, 0., 87.);
    let gap = if label == "impact" { 0.02 } else { -0.005 };
    let restitution = if label == "impact" { 0.8 } else { 0. };
    for i in 0..storage {
        local[i].inv_mass = Vec3::ZERO;
        local[i].inv_principal_inertia = Vec3::ZERO;
        poses[i].padding = 0x12345678;
        solver_poses[i].padding = 0x23456789;
        vels[i].padding0 = 0x34567890;
        vels[i].padding1 = 0x45678901;
        solver_vels[i].padding0 = 0x56789012;
        solver_vels[i].padding1 = 0x67890123;
        inc[i].padding0 = 0x78901234;
        inc[i].padding1 = 0x89012345;
        if i < active {
            let level = (i / nb) % (levels + 1);
            let column = i / nb / (levels + 1);
            poses[i].translation = Vec3::new(
                column as f32 * 3.,
                if level == 0 {
                    -0.5
                } else {
                    0.5 + (level - 1) as f32 * (1. + gap) + gap
                },
                (i % nb) as f32 * 3.,
            ) * scale;
            if level > 0 {
                local[i].inv_mass = Vec3::ONE;
                local[i].inv_principal_inertia = Vec3::splat(6. / (scale * scale));
                if label == "offset-com" {
                    local[i].com = Vec3::new(0.03, 0.02, -0.01) * scale;
                }
                vels[i].linear = Vec3::new(
                    if label == "friction" { 2. } else { 0. },
                    if label == "impact" { -2. } else { -0.1 },
                    0.,
                ) * scale;
                if label == "free" {
                    vels[i].angular = Vec3::new(0.1, -0.2, 0.3);
                }
            }
        }
        world[i] = local[i].to_world(&poses[i]);
    }
    let mut links = vec![
        ContactLink {
            previous_constraint: u32::MAX,
            previous_constraint_index: u32::MAX,
            ..Default::default()
        };
        capacity
    ];
    let mut contacts = vec![IndexedManifold::default(); capacity];
    let mut states = vec![ContactRecycleState::default(); capacity];
    let mut tiles = vec![CoulombTile::default(); tiles_len];
    bytemuck::cast_slice_mut::<_, u32>(&mut tiles).fill((-17f32).to_bits());
    let old = vec![CoulombTile::default(); tiles_len];
    let mut buckets = vec![0u32; 4 * nb];
    let mut index = 0;
    for color in 1..=2 {
        for batch in 0..nb {
            if n > 0 {
                for column in 0..columns {
                    for level in 1..=levels {
                        if (level - 1) % 2 + 1 != color {
                            continue;
                        }
                        let a = (column * (levels + 1) + level) * nb + batch;
                        let b = a - nb;
                        let len = if label == "impact" { 1 } else { 4 };
                        let im = &mut contacts[index];
                        im.bodies = UVec2::new(a as u32, b as u32);
                        im.colliders = im.bodies;
                        im.friction = 0.5;
                        im.restitution = restitution;
                        im.contact.normal_a = -Vec3::Y;
                        im.contact.len = len;
                        for k in 0..len as usize {
                            im.contact.points_a[k] = ContactPoint::new(
                                Vec3::new(
                                    if len == 1 {
                                        0.
                                    } else if k % 2 == 0 {
                                        -0.25
                                    } else {
                                        0.25
                                    },
                                    -0.5,
                                    if len == 1 {
                                        0.
                                    } else if k / 2 == 0 {
                                        -0.25
                                    } else {
                                        0.25
                                    },
                                ) * scale,
                                gap * scale,
                            );
                        }
                        states[index].pose_a = poses[a];
                        links[index] = ContactLink {
                            contact: index as u32,
                            solver_body_a: a as u32,
                            solver_body_b: b as u32,
                            vel_slot_a: a as u32,
                            vel_slot_b: b as u32,
                            len,
                            mass_scale_a: 1.,
                            mass_scale_b: 1.,
                            restitution,
                            previous_constraint: u32::MAX,
                            previous_constraint_index: u32::MAX,
                            ..Default::default()
                        };
                        index += 1;
                    }
                }
            }
            buckets[color * nb + batch] = index as u32;
        }
    }
    assert_eq!(index, n);
    for b in &mut buckets[3 * nb..] {
        *b = n as u32;
    }
    let ids = BatchIndices {
        bodies_len: (active / nb) as u32,
        colliders_batch_capacity: (active / nb) as u32,
        num_batches: nb as u32,
        solver_color_buckets_stride: 4,
        ..Default::default()
    };
    let plan = ContactPlan {
        bound: n as u32,
        ..Default::default()
    };
    let recycle = ContactRecycleOffsets::default();
    macro_rules! w {
        ($name:expr,$data:expr) => {
            pilot::write_buffer(&dir, $name, $data)
        };
    }
    macro_rules! outputs {
        ($suffix:expr) => {{
            w!(&format!("Poses{}.bin", $suffix), &poses);
            w!(&format!("SolverPoses{}.bin", $suffix), &solver_poses);
            w!(&format!("Velocities{}.bin", $suffix), &vels);
            w!(&format!("SolverVelocities{}.bin", $suffix), &solver_vels);
            w!(&format!("Increments{}.bin", $suffix), &inc);
            w!(&format!("WorldMass{}.bin", $suffix), &world);
            w!(&format!("Counts{}.bin", $suffix), &counts);
            w!(&format!("FirstSlot{}.bin", $suffix), &first);
            w!(&format!("HubCounts{}.bin", $suffix), &[hubs]);
            w!(&format!("Tiles{}.bin", $suffix), &tiles);
        }};
    }
    outputs!("");
    w!("LocalMass.bin", &local);
    w!("RbdParams.bin", &[p]);
    w!("Gravity.bin", &[gravity]);
    w!("BatchIds.bin", &[ids]);
    w!("Buckets.bin", &buckets);
    w!("Links.bin", &links);
    w!("Contacts.bin", &contacts);
    w!("States.bin", &states);
    w!("OldTiles.bin", &old);
    for c in 0..=2u32 {
        w!(&format!("Color{c}.bin"), &[c, 0xdeadbeef]);
    }
    w!("Mode.bin", &[mode, 0xdeadbeef]);
    w!("Params.bin", &[0xabcdef12u32; 6]);
    w!(
        "Params.expected.bin",
        &[
            p.dt,
            p.max_linear_velocity(),
            p.max_angular_velocity(),
            gravity.x,
            gravity.y,
            gravity.z
        ]
    );
    let bg = (active as u32).div_ceil(64).max(1);
    let cg = (n as u32).div_ceil(64).max(1);
    for i in 0..bg * 64 {
        gpu_update_mprops(
            UVec3::new(i, 0, 0),
            UVec3::new(bg, 1, 1),
            &mut world,
            &local,
            &poses,
            &ids,
        );
    }
    for i in 0..bg * 64 {
        gpu_solver_cleanup(
            UVec3::new(i, 0, 0),
            UVec3::new(bg, 1, 1),
            &mut counts,
            &mut solver_vels,
            &vels,
            &world,
            &ids,
            &mut first,
            &mut hubs,
        );
    }
    for i in 0..bg * 64 {
        gpu_init_solver_bodies(UVec3::new(i, 0, 0), &poses, &local, &mut solver_poses, &ids);
    }
    for i in 0..cg * 64 {
        gpu_prepare_constraints(
            UVec3::new(i, 0, 0),
            &links,
            &contacts,
            &world,
            &solver_poses,
            &vels,
            &states,
            &old,
            &mut tiles,
            &plan,
            &recycle,
        );
    }
    for i in 0..bg * 64 {
        gpu_init_solver_vels_inc(UVec3::new(i, 0, 0), &mut inc, &world, &p, &ids, &gravity);
    }
    for _ in 0..substeps {
        for i in 0..bg * 64 {
            gpu_apply_solver_vels_inc(UVec3::new(i, 0, 0), &mut solver_vels, &inc, &ids);
        }
        if warmstart != 1. {
            for i in 0..cg * 64 {
                gpu_scale_constraint_impulses(UVec3::new(i, 0, 0), &buckets, &mut tiles, &ids, &p);
            }
        }
        for color in 1..=2 {
            for i in 0..cg * 64 {
                gpu_warmstart_constraints(
                    UVec3::new(i, 0, 0),
                    UVec3::new(cg, 1, 1),
                    &tiles,
                    &mut solver_vels,
                    &buckets,
                    &color,
                    &ids,
                );
            }
        }
        for _ in 0..pgs.max(1) {
            for color in 1..=2 {
                for i in 0..cg * 64 {
                    gpu_solve_constraints(
                        UVec3::new(i, 0, 0),
                        UVec3::new(cg, 1, 1),
                        &mut tiles,
                        &mut solver_vels,
                        &buckets,
                        &solver_poses,
                        &color,
                        &ids,
                        &mode,
                        &p,
                    );
                }
            }
        }
        for i in 0..bg * 64 {
            gpu_integrate_linearized(
                UVec3::new(i, 0, 0),
                &mut solver_poses,
                &mut solver_vels,
                &p,
                &ids,
            );
        }
        for color in 1..=2 {
            for i in 0..cg * 64 {
                gpu_solve_constraints(
                    UVec3::new(i, 0, 0),
                    UVec3::new(cg, 1, 1),
                    &mut tiles,
                    &mut solver_vels,
                    &buckets,
                    &solver_poses,
                    &color,
                    &ids,
                    &0,
                    &p,
                );
            }
        }
    }
    for i in 0..bg * 64 {
        gpu_solver_finalize(
            UVec3::new(i, 0, 0),
            &mut vels,
            &solver_vels,
            &mut poses,
            &solver_poses,
            &local,
            &ids,
        );
    }
    outputs!(".expected");
    let sizes: serde_json::Map<String, Value> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "bin") && !p.to_string_lossy().contains(".expected.")
        })
        .map(|p| {
            (
                p.file_stem().unwrap().to_str().unwrap().to_owned(),
                json!(fs::metadata(p).unwrap().len() / 4),
            )
        })
        .collect();
    json!({"name":label,"kind":"tgs_frame","count":n,"body_count":active,"body_storage":storage,"contact_storage":capacity,"batches":nb,"columns":columns,"levels":levels,"substeps":substeps,"pgs":pgs,"mode":mode,"scale_warmstart":warmstart!=1.,"storage":capacity,"bodies":storage,"states":capacity,"new_base":0,"tiles":tiles_len,"old_tiles":tiles_len,"groups":cg,"iterations":1,"buffer_words":sizes})
}
fn main() {
    let output = std::env::args().nth(1).unwrap();
    let root = Path::new(&output);
    fs::create_dir_all(root).unwrap();
    pilot::write_json(root.join("layout.json"), &layout());
    coefficient_fixtures(root);
    let root = root.join("tgs_frame");
    fs::create_dir_all(&root).unwrap();
    let mut cases = Vec::new();
    for (label, columns, levels, batches, substeps, pgs, mode, scale, warm) in [
        ("empty", 0, 0, 1, 1, 1, 1, 1., 1.),
        ("free", 1, 1, 1, 4, 1, 1, 1., 1.),
        ("impact", 1, 1, 1, 1, 1, 2, 1., 1.),
        ("rest", 1, 1, 1, 4, 1, 1, 1., 1.),
        ("friction", 1, 1, 1, 4, 3, 2, 1., 0.8),
        ("stack", 1, 4, 1, 4, 2, 1, 1., 1.),
        ("wide31", 31, 2, 1, 4, 2, 2, 1., 1.),
        ("wide33", 33, 3, 2, 3, 1, 1, 0.01, 0.7),
        ("wide65", 65, 2, 1, 2, 2, 2, 2., 1.),
        ("wide129", 129, 1, 1, 4, 0, 1, 1., 1.),
        ("offset-com", 1, 3, 3, 4, 2, 2, 1., 0.9),
    ] {
        cases.push(fixture(
            &root, label, columns, levels, batches, substeps, pgs, mode, scale, warm,
        ));
    }
    pilot::write_json(root.join("suite.json"), &json!({"cases":cases}));
}
fn coefficient_fixtures(root: &Path) {
    let root = root.join("body_coefficients");
    fs::create_dir_all(&root).unwrap();
    let mut cases = Vec::new();
    for kind in 0..10 {
        let mut p = RbdSimParams::tgs_soft();
        let mut gravity = Vec4::new(0.7, -9.81, -0.2, 99.);
        match kind {
            1 => p.dt = 0.,
            2 => p.num_solver_iterations = 0,
            3 => p.num_solver_iterations = 1,
            4 => {
                p.normalized_max_linear_velocity = nexus_rbd_shaders3d::MAX_FLT;
                p.length_unit = 2.;
            }
            5 => {
                p.dt = 1. / 240.;
                p.num_solver_iterations = 16;
                p.length_unit = 0.01;
            }
            6 => gravity = Vec4::ZERO,
            7 => {
                p.normalized_max_linear_velocity = 0.;
                p.num_solver_iterations = 64;
            }
            8 | 9 => {
                let bits = nexus_rbd_shaders3d::MAX_FLT.to_bits();
                p.normalized_max_linear_velocity =
                    f32::from_bits(if kind == 8 { bits - 1 } else { bits + 1 });
                p.length_unit = 2.;
            }
            _ => {}
        }
        let d = root.join(format!("p{kind}"));
        fs::create_dir_all(&d).unwrap();
        pilot::write_buffer(&d, "RbdParams.bin", &[p]);
        pilot::write_buffer(&d, "Gravity.bin", &[gravity]);
        pilot::write_buffer(&d, "Params.bin", &[0xdeadbeefu32; 9]);
        let mut expected = [0xdeadbeefu32; 9];
        for (i, value) in [
            p.dt,
            p.max_linear_velocity(),
            p.max_angular_velocity(),
            gravity.x,
            gravity.y,
            gravity.z,
        ]
        .into_iter()
        .enumerate()
        {
            expected[i] = value.to_bits();
        }
        pilot::write_buffer(&d, "Params.expected.bin", &expected);
        cases.push(json!({"name":format!("p{kind}"),"kind":"body_coefficients","count":6,"iterations":3,"buffer_words":{"RbdParams":size_of::<RbdSimParams>()/4,"Gravity":4,"Params":9}}));
    }
    pilot::write_json(root.join("suite.json"), &json!({"cases":cases}));
}
