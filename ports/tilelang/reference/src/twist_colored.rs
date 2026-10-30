//! Native color-bucket solve and warmstart oracle; bodies are shared across colors.
#[allow(dead_code)]
mod twist;
use nexus_rbd_shaders3d::{
    Pose,
    dynamics::{
        RbdSimParams, SymInertia, TwistConstraint, TwistContactPoint, Velocity,
        twist_tiles::{
            TwistTile as CoulombTile, gpu_solve_constraints, gpu_solve_constraints_unbiased_cached,
            gpu_warmstart_constraints,
        },
    },
    glamx::{Quat, UVec3, Vec2, Vec3},
    utils::BatchIndices,
};
use serde_json::{Value, json};
use std::{
    fs,
    mem::{offset_of, size_of},
    path::Path,
};
use twist::contacts::pilot;
#[allow(clippy::chunks_exact_to_as_chunks)]
fn read<T: bytemuck::Pod>(p: &Path) -> Vec<T> {
    fs::read(p)
        .unwrap()
        .chunks_exact(size_of::<T>())
        .map(bytemuck::pod_read_unaligned)
        .collect()
}
pub(crate) fn layout() -> Value {
    let mut l = twist::layout();
    l["constants"] = json!({"max_flt_bits":nexus_rbd_shaders3d::MAX_FLT.to_bits(), "two_pi_bits":std::f32::consts::TAU.to_bits()});
    macro_rules! fields {($ty:ty,$($f:ident),+)=>{json!({$(stringify!($f):offset_of!($ty,$f)/4),+})}}
    l["rbd_params_words"] = json!(size_of::<RbdSimParams>() / 4);
    l["rbd_params"] = fields!(
        RbdSimParams,
        dt,
        contact_damping_ratio,
        contact_natural_frequency,
        static_contact_damping_ratio,
        static_contact_natural_frequency,
        normalized_max_corrective_velocity,
        length_unit
    );
    l["batch_words"] = json!(size_of::<BatchIndices>() / 4);
    l["batch"] = fields!(
        BatchIndices,
        num_batches,
        bodies_len,
        solver_color_buckets_stride
    );
    l
}
#[allow(clippy::too_many_arguments)]
fn fixture(
    root: &Path,
    width: usize,
    batches: usize,
    mode: u32,
    op: &str,
    seed: u64,
    shared_static: bool,
) -> Value {
    let n = width * batches * 4;
    let old = pilot::fixture(
        root,
        n,
        true,
        true,
        1,
        seed,
        pilot::FixtureGeometry::Portable,
    );
    let name = format!(
        "w{width}-b{batches}-m{mode}-s{seed}{}",
        if shared_static { "-shared-static" } else { "" }
    );
    let directory = root.join(&name);
    // Refresh only the seed files, retaining generated kernels and replay artifacts.
    let seed_directory = root.join(old["name"].as_str().unwrap());
    fs::create_dir_all(&directory).unwrap();
    for entry in fs::read_dir(&seed_directory).unwrap() {
        let entry = entry.unwrap();
        assert!(entry.file_type().unwrap().is_file());
        fs::copy(entry.path(), directory.join(entry.file_name())).unwrap();
        fs::remove_file(entry.path()).unwrap();
    }
    fs::remove_dir(seed_directory).unwrap();
    let seed_tiles = read::<nexus_rbd_shaders3d::dynamics::coulomb_tiles::CoulombTile>(
        &directory.join("Tiles.bin"),
    );
    let mut tiles = vec![CoulombTile::default(); seed_tiles.len()];
    bytemuck::cast_slice_mut::<_, u32>(&mut tiles).fill((-19.0f32).to_bits());
    for i in 0..n {
        let old = seed_tiles[i / 32].constraint(i % 32);
        let mut c = TwistConstraint {
            len: old.len,
            dir_a: old.dir_a,
            tangent_a: old.tangent_a,
            limit: old.limit,
            im_a: old.im_a,
            im_b: old.im_b,
            ii_a: old.ii_a,
            ii_b: old.ii_b,
            solver_body_a: old.solver_body_a,
            solver_body_b: old.solver_body_b,
            vel_slot_a: old.vel_slot_a,
            vel_slot_b: old.vel_slot_b,
            offset_b: old.points[0].r_b - old.points[0].r_a,
            frame_a: Quat::IDENTITY,
            frame_b: Quat::IDENTITY,
            ..Default::default()
        };
        for k in 0..c.len as usize {
            let p = old.points[k];
            c.points[k] = TwistContactPoint {
                r_a: p.r_a,
                normal_impulse: p.normal_impulse,
                normal_vel: p.normal_vel,
                dist: p.dist,
                ..Default::default()
            };
        }
        c.compute_effective_masses();
        c.friction.tangent_impulse = Vec2::new(0.01, -0.02);
        c.friction.twist_impulse = if c.len > 1 { 0.02 } else { 0. };
        tiles[i / 32].write_constraint(i % 32, &c);
    }

    let mut velocities = read::<Velocity>(&directory.join("Velocities.bin"));
    let poses = read::<Pose>(&directory.join("Poses.bin"));
    let mut params = read::<RbdSimParams>(&directory.join("RbdParams.bin"))[0];
    // Exercise runtime parameter loads, including zero-frequency and uncapped limits.
    if seed.is_multiple_of(3) {
        params.contact_natural_frequency = 0.;
    }
    if seed.is_multiple_of(5) {
        params.normalized_max_corrective_velocity = nexus_rbd_shaders3d::MAX_FLT;
        params.length_unit = 2.;
    }
    if shared_static {
        for velocity in velocities.iter_mut().take(batches) {
            velocity.linear = Vec3::splat(-0.0);
            velocity.angular = Vec3::splat(-0.0);
        }
    }
    params.friction_model = nexus_rbd_shaders3d::dynamics::FrictionModel::Simplified;
    let mut buckets = vec![0u32; 5 * batches];
    for color in 1..5 {
        for batch in 0..batches {
            let bucket = color * batches + batch;
            let start = ((color - 1) * batches + batch) * width;
            buckets[bucket] = (start + width) as u32;
            for lane in 0..width {
                let index = start + lane;
                let mut c = tiles[index / 32].constraint(index % 32);
                let a = if shared_static {
                    batch
                } else {
                    2 * lane * batches + batch
                };
                let b = if shared_static {
                    (lane + 1) * batches + batch
                } else {
                    (2 * lane + 1) * batches + batch
                };
                c.solver_body_a = a as u32;
                c.solver_body_b = b as u32;
                c.vel_slot_a = a as u32;
                c.vel_slot_b = b as u32;
                let inertia = SymInertia {
                    diag: Vec3::new(0.7, 0.9, 0.6),
                    off: Vec3::new(0.04, -0.03, 0.02),
                    ..Default::default()
                };
                c.im_a = Vec3::new(0.7, 0.6, 0.8);
                c.im_b = Vec3::splat(0.4);
                c.ii_a = inertia;
                c.ii_b = inertia;
                if shared_static || lane % 5 == 1 {
                    c.im_a = Vec3::ZERO;
                    c.ii_a = SymInertia::default();
                }
                for k in 0..c.len as usize {
                    let (pa, pb) = c.local_anchors(k);
                    let separation = (poses[a] * pa - poses[b] * pb).dot(c.dir_a);
                    let p = &mut c.points[k];
                    p.dist = [-0.03, 0.025, -0.5][(index + k) % 3] - separation;
                }
                c.compute_effective_masses();
                tiles[index / 32].write_constraint(index % 32, &c);
            }
        }
    }
    let ids = BatchIndices {
        num_batches: batches as u32,
        bodies_len: (2 * width) as u32,
        solver_color_buckets_stride: 5,
        ..Default::default()
    };
    let color = [2u32, 0xdeadbeef];
    let modes = [mode, 0xdeadbeef];
    let groups = if width == 129 { 1 } else { 3 };
    let repeats = if op == "colored_program" { 1 } else { 3 };
    macro_rules! w {
        ($name:expr,$data:expr) => {
            pilot::write_buffer(&directory, $name, $data)
        };
    }
    w!("Tiles.bin", &tiles);
    w!("Velocities.bin", &velocities);
    w!("RbdParams.bin", &[params]);
    w!("Buckets.bin", &buckets);
    w!("BatchIds.bin", &[ids]);
    w!("Color.bin", &color);
    w!("Mode.bin", &modes);
    for c in 1..=3u32 {
        w!(&format!("Color{c}.bin"), &[c, 0xdeadbeef]);
    }
    let mut pass = |kind: &str, c: u32, m: u32| {
        for i in 0..groups * 64 {
            let gid = UVec3::new(i, 0, 0);
            let grid = UVec3::new(groups, 1, 1);
            match kind {
                "colored_warmstart" => gpu_warmstart_constraints(
                    gid,
                    grid,
                    &tiles,
                    &mut velocities,
                    &buckets,
                    &c,
                    &ids,
                ),
                "colored_cached" => gpu_solve_constraints_unbiased_cached(
                    gid,
                    grid,
                    &mut tiles,
                    &mut velocities,
                    &buckets,
                    &poses,
                    &c,
                    &ids,
                    &m,
                    &params,
                ),
                _ => gpu_solve_constraints(
                    gid,
                    grid,
                    &mut tiles,
                    &mut velocities,
                    &buckets,
                    &poses,
                    &c,
                    &ids,
                    &m,
                    &params,
                ),
            }
        }
    };
    for _ in 0..repeats {
        if op == "colored_program" {
            for c in 1..=3 {
                pass("colored_warmstart", c, mode);
            }
            for _ in 0..3 {
                for c in 1..=3 {
                    pass("colored_solve", c, mode);
                }
            }
            for c in 1..=3 {
                pass("colored_cached", c, 0);
            }
        } else {
            pass(op, 2, mode);
        }
    }
    w!("Tiles.expected.bin", &tiles);
    w!("Velocities.expected.bin", &velocities);
    let l = layout();
    let sizes = json!({"Tiles":tiles.len()*l["tile_words"].as_u64().unwrap() as usize,"Velocities":velocities.len()*l["velocity_words"].as_u64().unwrap() as usize,"Poses":poses.len()*l["pose_words"].as_u64().unwrap() as usize,"RbdParams":l["rbd_params_words"],"Buckets":buckets.len(),"BatchIds":l["batch_words"],"Color":2,"Color1":2,"Color2":2,"Color3":2,"Mode":2});
    json!({"name":name,"kind":op,"count":n,"width":width,"batches":batches,"mode":mode,"groups":groups,"iterations":repeats,"bodies":velocities.len(),"tiles":tiles.len(),"seed":seed,"shared_static":shared_static,"buffer_words":sizes})
}
fn main() {
    let output = std::env::args().nth(1).expect("output directory");
    let root = Path::new(&output);
    fs::create_dir_all(root).unwrap();
    pilot::write_json(root.join("layout.json"), &layout());
    for op in [
        "colored_solve",
        "colored_warmstart",
        "colored_cached",
        "colored_program",
    ] {
        let dir = root.join(format!("twist_{op}"));
        fs::create_dir_all(&dir).unwrap();
        let mut cases = Vec::new();
        for (width, batches, mode, seed) in [
            (0, 1, 0, 17),
            (1, 1, 1, 18),
            (31, 1, 2, 19),
            (32, 2, 0, 20),
            (33, 3, 1, 21),
            (65, 2, 2, 22),
            (129, 1, 99, 23),
        ] {
            cases.push(fixture(&dir, width, batches, mode, op, seed, false));
        }
        for width in [1, 33, 65, 129] {
            cases.push(fixture(&dir, width, 1, 2, op, 42, true));
        }
        pilot::write_json(dir.join("suite.json"), &json!({"cases":cases}));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_export_preserves_replay_artifacts() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "nexus-colored-export-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let first = fixture(&root, 1, 1, 1, "colored_program", 18, false);
        let directory = root.join(first["name"].as_str().unwrap());
        let expected: Vec<_> = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| {
                let path = entry.unwrap().path();
                let bytes = fs::read(&path).unwrap();
                (path, bytes)
            })
            .collect();
        let artifact = directory.join("keep.note");
        fs::write(&artifact, b"replay artifact").unwrap();
        assert_eq!(fixture(&root, 1, 1, 1, "colored_program", 18, false), first);
        for (path, bytes) in expected {
            assert_eq!(fs::read(path).unwrap(), bytes);
        }
        assert_eq!(fs::read(artifact).unwrap(), b"replay artifact");
        fs::remove_dir_all(root).unwrap();
    }
}
