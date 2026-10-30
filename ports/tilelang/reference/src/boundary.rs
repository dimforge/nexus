//! Fresh native cancellation fixtures and complete CFM branch alternatives.
#[allow(dead_code)]
#[path = "main.rs"]
mod fixture;
#[allow(dead_code, unexpected_cfgs, clippy::too_many_arguments)]
mod solver {
    include!(concat!(env!("OUT_DIR"), "/boundary_solver.rs"));
}
mod boundary_interval;
pub use nexus_rbd_shaders3d::AngVector;
use nexus_rbd_shaders3d::{
    Pose,
    dynamics::{RbdSimParams, Velocity, contact_tiles::TILE_LEN, coulomb_tiles::CoulombTile},
    utils::Slice,
};
use serde_json::{Value, json};
use std::{fs, mem::size_of, path::Path};
// Generic size_of::<T>() cannot be used as a stable array-length const argument.
#[allow(clippy::chunks_exact_to_as_chunks)]
fn read<T: bytemuck::Pod>(dir: &Path, name: &str) -> Vec<T> {
    fs::read(dir.join(name))
        .unwrap()
        .chunks_exact(size_of::<T>())
        .map(bytemuck::pod_read_unaligned)
        .collect()
}
fn alternatives(dir: &Path, c: &mut Value) {
    let n = c["count"].as_u64().unwrap() as usize;
    let poses = read::<Pose>(dir, "Poses.bin");
    let params = read::<RbdSimParams>(dir, "RbdParams.bin")[0];
    let before = read::<CoulombTile>(dir, "Tiles.bin");
    let mut masks = vec![0xdeadbeefu32; n.max(1)];
    for i in 0..n {
        masks[i] = boundary_interval::mask(&before[i / TILE_LEN].constraint(i % TILE_LEN), &poses);
        assert!(
            masks[i].count_ones() <= 1,
            "This policy supports one ambiguous point per manifold"
        );
    }
    fixture::write_buffer(dir, "Uncertain.bin", &masks);
    for mode in 0..3 {
        let mut tiles = before.clone();
        let mut vels = read::<Velocity>(dir, "Velocities.bin");
        for _ in 0..c["iterations"].as_u64().unwrap() {
            for i in 0..n {
                let tile = &mut tiles[i / TILE_LEN];
                let mut constraint = tile.constraint(i % TILE_LEN);
                let header = solver::ContactSolverData::from_constraint(&constraint);
                let (a, b) = (
                    constraint.vel_slot_a as usize,
                    constraint.vel_slot_b as usize,
                );
                let (mut va, mut vb) = (vels[a], vels[b]);
                header.solve_points(
                    &mut constraint.points,
                    &Slice(&poses, 0),
                    &params,
                    &mut va,
                    &mut vb,
                    c["use_bias"].as_bool().unwrap(),
                    c["solve_friction"].as_bool().unwrap(),
                    if mode == 1 { masks[i] } else { 0 },
                    if mode == 2 { masks[i] } else { 0 },
                );
                vels[a] = va;
                vels[b] = vb;
                for k in 0..constraint.len as usize {
                    tile.points[k].arm_a[i % TILE_LEN].scalar = constraint.points[k].normal_impulse;
                    tile.points[k].friction[i % TILE_LEN].impulse =
                        constraint.points[k].tangent_impulse;
                }
            }
        }
        if mode == 0 {
            // Prove that source extraction preserved this fixture's original
            // solver output before either counterfactual may be used as evidence.
            assert_eq!(
                bytemuck::cast_slice::<_, u8>(&tiles),
                fs::read(dir.join("Tiles.expected.bin")).unwrap()
            );
            assert_eq!(
                bytemuck::cast_slice::<_, u8>(&vels),
                fs::read(dir.join("Velocities.expected.bin")).unwrap()
            );
        } else {
            let label = if mode == 1 { "Soft" } else { "Hard" };
            fixture::write_buffer(dir, &format!("{label}Tiles.bin"), &tiles);
            fixture::write_buffer(dir, &format!("{label}Velocities.bin"), &vels);
        }
    }
    let mut sizes = serde_json::Map::new();
    for name in [
        "Tiles",
        "Velocities",
        "Poses",
        "Params",
        "SoftTiles",
        "HardTiles",
        "SoftVelocities",
        "HardVelocities",
        "Uncertain",
    ] {
        sizes.insert(
            name.to_owned(),
            json!(fs::metadata(dir.join(format!("{name}.bin"))).unwrap().len() / 4),
        );
    }
    c["buffer_words"] = json!(sizes);
    c["policy"] = json!("single-ambiguous-cfm-v1");
    fs::write(dir.join("case.json"), serde_json::to_vec_pretty(c).unwrap()).unwrap();
}
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let root = Path::new(&path);
    let directory = root.join("solve_boundary");
    fs::create_dir_all(&directory).unwrap();
    fixture::write_json(root.join("layout.json"), &fixture::layout());
    let mut configurations = Vec::new();
    for n in [0, 1, 31, 32, 33, 63, 64, 65, 127] {
        configurations.push((n, true, true, 1, 20261029 + n as u64));
    }
    for bias in [false, true] {
        for friction in [false, true] {
            for seed in [17, 42] {
                configurations.push((129, bias, friction, 3, seed));
            }
        }
    }
    let mut cases = Vec::new();
    for (n, bias, friction, iterations, seed) in configurations {
        let mut c = fixture::fixture(
            &directory,
            n,
            bias,
            friction,
            iterations,
            seed,
            fixture::FixtureGeometry::BoundaryStress,
        );
        alternatives(&directory.join(c["name"].as_str().unwrap()), &mut c);
        cases.push(c);
    }
    fixture::write_json(directory.join("suite.json"), &json!({"cases":cases}));
}
