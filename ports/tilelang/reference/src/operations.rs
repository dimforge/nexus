//! Live Rust oracles for individually ported solver stages.
#[allow(dead_code)]
#[path = "main.rs"]
mod pilot;
use nexus_rbd_shaders3d::{
    Pose,
    dynamics::{RbdSimParams, Velocity, coulomb_tiles::CoulombTile, gpu_integrate_linearized},
    glamx::{UVec3, Vec3},
    utils::BatchIndices,
};
use serde_json::json;
use std::{fs, path::Path};
fn read<T: bytemuck::Pod>(path: &Path) -> Vec<T> {
    fs::read(path)
        .unwrap()
        .chunks_exact(size_of::<T>())
        .map(bytemuck::pod_read_unaligned)
        .collect()
}
fn main() {
    let output = std::env::args().nth(1).expect("Expected output directory");
    let output = Path::new(&output);
    fs::create_dir_all(output).unwrap();
    pilot::write_json(output.join("layout.json"), &pilot::layout());
    for kind in ["effective_mass", "warmstart", "integrate"] {
        let root = output.join(kind);
        fs::create_dir_all(&root).unwrap();
        let mut cases = Vec::new();
        for n in [0usize, 1, 32, 65, 129] {
            let mut case = pilot::fixture(&root, n, true, true, 3, 20261100 + n as u64, false);
            case["kind"] = json!(kind);
            case["active_bodies"] = json!(n);
            let dir = root.join(case["name"].as_str().unwrap());
            let mut tiles = read::<CoulombTile>(&dir.join("Tiles.bin"));
            let mut velocities = read::<Velocity>(&dir.join("Velocities.bin"));
            let mut poses = read::<Pose>(&dir.join("Poses.bin"));
            let mut params = read::<RbdSimParams>(&dir.join("RbdParams.bin"))[0];
            if kind == "effective_mass" {
                for i in 0..n {
                    let mut c = tiles[i / 32].constraint(i % 32);
                    for p in &mut c.points[..c.len as usize] {
                        p.normal_mass = -17.0;
                        p.tangent_k = [-19.0; 3];
                    }
                    tiles[i / 32].write_constraint(i % 32, &c);
                }
                pilot::write_buffer(&dir, "Tiles.bin", &tiles);
            }
            if kind == "integrate" {
                params.normalized_max_linear_velocity = 0.5;
                for (i, v) in velocities.iter_mut().take(n).enumerate() {
                    if i % 4 == 0 {
                        v.linear = Vec3::ZERO;
                        v.angular = Vec3::ZERO;
                    } else if i % 4 == 1 {
                        v.linear *= 100.0;
                        v.angular *= 1000.0;
                    }
                }
                pilot::write_buffer(&dir, "RbdParams.bin", &[params]);
                pilot::write_buffer(&dir, "Velocities.bin", &velocities);
                pilot::write_buffer(
                    &dir,
                    "Params.bin",
                    &[
                        params.dt,
                        params.max_linear_velocity(),
                        params.max_angular_velocity(),
                        0.0,
                        0.0,
                        0.0,
                    ],
                );
            }
            for _ in 0..3 {
                if kind == "integrate" {
                    let ids = BatchIndices {
                        bodies_len: n as u32,
                        num_batches: 1,
                        ..Default::default()
                    };
                    for i in 0..n {
                        gpu_integrate_linearized(
                            UVec3::new(i as u32, 0, 0),
                            &mut poses,
                            &mut velocities,
                            &params,
                            &ids,
                        );
                    }
                } else {
                    for i in 0..n {
                        let mut c = tiles[i / 32].constraint(i % 32);
                        if kind == "effective_mass" {
                            c.compute_effective_masses();
                            tiles[i / 32].write_constraint(i % 32, &c);
                        } else {
                            let (a, b) = (c.vel_slot_a as usize, c.vel_slot_b as usize);
                            let (mut va, mut vb) = (velocities[a], velocities[b]);
                            c.warmstart_constraint(&mut va, &mut vb);
                            velocities[a] = va;
                            velocities[b] = vb;
                        }
                    }
                }
            }
            pilot::write_buffer(&dir, "Tiles.expected.bin", &tiles);
            pilot::write_buffer(&dir, "Velocities.expected.bin", &velocities);
            pilot::write_buffer(&dir, "Poses.expected.bin", &poses);
            cases.push(case);
        }
        pilot::write_json(root.join("suite.json"), &json!({"cases":cases}));
    }
}
