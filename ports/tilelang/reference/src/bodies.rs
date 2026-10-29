//! Body-stage fixtures use the production Rust entry points, never copied arithmetic.
#[allow(dead_code)]
#[path = "main.rs"]
mod pilot;
use nexus_rbd_shaders3d::{
    Pose,
    dynamics::{
        HubCounts, LocalMassProperties, RbdSimParams, Velocity, WorldMassProperties,
        gpu_apply_solver_vels_inc, gpu_init_solver_bodies, gpu_init_solver_vels_inc,
        gpu_integrate_linearized, gpu_solver_cleanup, gpu_solver_finalize,
    },
    glamx::{Quat, UVec3, Vec3, Vec4},
    utils::BatchIndices,
};
use serde_json::json;
use std::{fs, mem::offset_of, path::Path};

fn main() {
    let output = std::env::args().nth(1).expect("Expected output directory");
    let output = Path::new(&output);
    fs::create_dir_all(output).unwrap();
    let mut layout = pilot::layout();
    for (key, value) in [
        ("pose_padding", offset_of!(Pose, padding) / 4),
        ("local_words", size_of::<LocalMassProperties>() / 4),
        ("local_com", offset_of!(LocalMassProperties, com) / 4),
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
    pilot::write_json(output.join("layout.json"), &layout);
    for kind in [
        "cleanup",
        "init_bodies",
        "init_increments",
        "apply_increments",
        "finalize",
        "free_step",
    ] {
        let root = output.join(kind);
        fs::create_dir_all(&root).unwrap();
        let mut cases = Vec::new();
        // Count and capacity are deliberately distinct; include zero batches and dispatch tails.
        for (bodies, batches, capacity, substeps, frames, zero_gravity) in [
            (0u32, 1u32, 0u32, 1u32, 3u32, false),
            (1, 1, 3, 1, 3, false),
            (63, 1, 65, 4, 32, false),
            (64, 1, 64, 2, 3, true),
            (65, 1, 70, 4, 32, false),
            (33, 3, 35, 3, 16, false),
            (129, 0, 131, 1, 3, false),
        ] {
            let n = bodies * batches;
            let slots = capacity * batches;
            let storage = slots.max(1) as usize + 3;
            let iterations = if kind == "free_step" { frames } else { 3 };
            let name = format!(
                "n{bodies}-b{batches}-c{capacity}-s{substeps}-i{iterations}-g{}",
                u32::from(!zero_gravity)
            );
            let dir = root.join(&name);
            fs::create_dir_all(&dir).unwrap();
            let mut poses = vec![Pose::IDENTITY; storage];
            let mut solver_poses = vec![Pose::IDENTITY; storage];
            let mut velocities = vec![Velocity::default(); storage];
            let mut solver_vels = vec![Velocity::default(); storage];
            let mut increments = vec![Velocity::default(); storage];
            let mut local = vec![LocalMassProperties::default(); storage];
            let mut world = vec![WorldMassProperties::default(); storage];
            let mut counts = vec![0xf1234567u32; storage];
            let mut first_slot = vec![0xe7654321u32; storage];
            let mut hubs = HubCounts {
                hubs: 17,
                slots: 23,
                base: 0x12345678,
                pool: 0x76543210,
            };
            for i in 0..storage {
                let t = i as f32 * 0.013;
                poses[i] = Pose::from_parts(
                    Vec3::new(t, -t * 0.5, 0.2),
                    Quat::from_scaled_axis(Vec3::new(t, 0.1, -0.2)),
                );
                poses[i].padding = 0x12345678;
                solver_poses[i] = Pose::from_parts(
                    Vec3::new(-t, 0.2, t * 0.3),
                    Quat::from_scaled_axis(Vec3::new(-0.2, t, 0.3)),
                );
                solver_poses[i].padding = 0x23456789;
                velocities[i] = Velocity::new(Vec3::new(0.3, -0.7, t), Vec3::new(t, 0.2, -0.4));
                if i % 6 == 1 {
                    velocities[i].linear *= 100.0;
                    velocities[i].angular *= 1000.0;
                }
                velocities[i].padding0 = 0x34567890;
                velocities[i].padding1 = 0x45678901;
                solver_vels[i] = Velocity::new(Vec3::new(-0.4, t, 0.3), Vec3::new(0.7, -0.2, t));
                solver_vels[i].padding0 = 0x56789012;
                solver_vels[i].padding1 = 0x67890123;
                increments[i] = Velocity::new(Vec3::new(0.2, 0.3, -0.1), Vec3::new(-0.3, 0.4, 0.1));
                increments[i].padding0 = 0x78901234;
                increments[i].padding1 = 0x89012345;
                local[i].com = if i % 7 == 0 {
                    Vec3::ZERO
                } else {
                    Vec3::new(0.17, -0.23, t)
                };
                local[i].inv_mass = match i % 4 {
                    0 => Vec3::ZERO,
                    1 => Vec3::X,
                    2 => Vec3::new(0.0, 2.0, 3.0),
                    _ => Vec3::ONE,
                };
                world[i] = local[i].to_world(&poses[i]);
            }
            let mut params = RbdSimParams::tgs_soft();
            params.dt = 1.0 / (60.0 * substeps as f32);
            params.normalized_max_linear_velocity = 0.5;
            let gravity = if zero_gravity {
                Vec4::ZERO
            } else {
                Vec4::new(0.7, -9.81, -0.2, 99.0)
            };
            let ids = BatchIndices {
                bodies_len: bodies,
                colliders_batch_capacity: capacity,
                num_batches: batches,
                ..Default::default()
            };
            macro_rules! write {
                ($suffix:expr) => {{
                    for (name, bytes) in [
                        ("Velocities", bytemuck::cast_slice(&velocities)),
                        ("SolverVelocities", bytemuck::cast_slice(&solver_vels)),
                        ("Increments", bytemuck::cast_slice(&increments)),
                        ("Poses", bytemuck::cast_slice(&poses)),
                        ("SolverPoses", bytemuck::cast_slice(&solver_poses)),
                        ("Counts", bytemuck::cast_slice(&counts)),
                        ("FirstSlot", bytemuck::cast_slice(&first_slot)),
                        ("HubCounts", bytemuck::bytes_of(&hubs)),
                    ] {
                        fs::write(dir.join(format!("{name}{}.bin", $suffix)), bytes).unwrap();
                    }
                }};
            }
            write!("");
            pilot::write_buffer(&dir, "LocalMass.bin", &local);
            pilot::write_buffer(&dir, "WorldMass.bin", &world);
            pilot::write_buffer(
                &dir,
                "Params.bin",
                &[
                    params.dt,
                    params.max_linear_velocity(),
                    params.max_angular_velocity(),
                    gravity.x,
                    gravity.y,
                    gravity.z,
                ],
            );
            // Keep full typed uniforms for independent reproduction of the Rust calls.
            pilot::write_buffer(&dir, "RbdParams.bin", &[params]);
            pilot::write_buffer(&dir, "BatchIndices.bin", &[ids]);
            pilot::write_buffer(&dir, "Gravity.bin", &[gravity]);
            let groups = slots.div_ceil(64).max(1);
            for _ in 0..iterations {
                if kind == "cleanup" || kind == "free_step" {
                    for i in 0..groups * 64 {
                        gpu_solver_cleanup(
                            UVec3::new(i, 0, 0),
                            UVec3::new(groups, 1, 1),
                            &mut counts,
                            &mut solver_vels,
                            &velocities,
                            &world,
                            &ids,
                            &mut first_slot,
                            &mut hubs,
                        );
                    }
                }
                if kind == "init_bodies" || kind == "free_step" {
                    for i in 0..groups * 64 {
                        gpu_init_solver_bodies(
                            UVec3::new(i, 0, 0),
                            &poses,
                            &local,
                            &mut solver_poses,
                            &ids,
                        );
                    }
                }
                if kind == "init_increments" || kind == "free_step" {
                    for i in 0..groups * 64 {
                        gpu_init_solver_vels_inc(
                            UVec3::new(i, 0, 0),
                            &mut increments,
                            &world,
                            &params,
                            &ids,
                            &gravity,
                        );
                    }
                }
                for _ in 0..if kind == "free_step" { substeps } else { 1 } {
                    if kind == "apply_increments" || kind == "free_step" {
                        for i in 0..groups * 64 {
                            gpu_apply_solver_vels_inc(
                                UVec3::new(i, 0, 0),
                                &mut solver_vels,
                                &increments,
                                &ids,
                            );
                        }
                    }
                    if kind == "free_step" {
                        for i in 0..groups * 64 {
                            gpu_integrate_linearized(
                                UVec3::new(i, 0, 0),
                                &mut solver_poses,
                                &mut solver_vels,
                                &params,
                                &ids,
                            );
                        }
                    }
                }
                if kind == "finalize" || kind == "free_step" {
                    for i in 0..groups * 64 {
                        gpu_solver_finalize(
                            UVec3::new(i, 0, 0),
                            &mut velocities,
                            &solver_vels,
                            &mut poses,
                            &solver_poses,
                            &local,
                            &ids,
                        );
                    }
                }
            }
            write!(".expected");
            cases.push(json!({"name":name, "kind":kind, "count":n, "bodies_per_batch":bodies, "batches":batches, "capacity_per_batch":capacity, "slots":slots, "storage":storage, "iterations":iterations, "substeps":substeps}));
        }
        pilot::write_json(root.join("suite.json"), &json!({"cases":cases}));
    }
}
