//! Collider propagation fixtures execute the current Rust shader entry function.
#[allow(dead_code)]
#[path = "main.rs"]
mod pilot;
use nexus_rbd_shaders3d::{
    Pose,
    dynamics::gpu_sync_collider_poses,
    glamx::{Quat, UVec3, Vec3},
    utils::BatchIndices,
};
use serde_json::json;
use std::{fs, mem::offset_of, path::Path};

fn main() {
    let out = std::env::args().nth(1).expect("Expected output directory");
    let out = Path::new(&out);
    fs::create_dir_all(out).unwrap();
    let mut layout = pilot::layout();
    layout["pose_padding"] = json!(offset_of!(Pose, padding) / 4);
    pilot::write_json(out.join("layout.json"), &layout);
    let root = out.join("sync_colliders");
    fs::create_dir_all(&root).unwrap();
    let mut cases = Vec::new();
    for (count, batches, bodies) in [
        (0u32, 1u32, 1u32),
        (1, 1, 1),
        (63, 1, 7),
        (64, 1, 5),
        (65, 1, 3),
        (33, 3, 5),
        (129, 0, 7),
    ] {
        let n = count * batches;
        let body_count = bodies * batches;
        let storage = n.max(1) as usize + 3;
        let body_storage = body_count.max(1) as usize + 3;
        let name = format!("n{count}-b{batches}-p{bodies}");
        let dir = root.join(&name);
        fs::create_dir_all(&dir).unwrap();
        let body_poses: Vec<_> = (0..body_storage)
            .map(|i| {
                let t = i as f32 * 0.13;
                let mut p = Pose::from_parts(
                    Vec3::new(t, -t * 0.7, 0.3),
                    Quat::from_scaled_axis(Vec3::new(0.2, t, -0.3)),
                );
                p.padding = 0x12345678;
                p
            })
            .collect();
        let local: Vec<_> = (0..storage)
            .map(|i| {
                let t = i as f32 * 0.07;
                let mut p = if i % 4 == 0 {
                    Pose::IDENTITY
                } else {
                    Pose::from_parts(
                        Vec3::new(-0.7, t, 0.2),
                        Quat::from_scaled_axis(Vec3::new(t, 0.4, -0.1)),
                    )
                };
                p.padding = 0x23456789;
                p
            })
            .collect();
        let mut world =
            vec![
                Pose::from_parts(Vec3::splat(-17.0), Quat::from_xyzw(-1.0, -2.0, -3.0, -4.0));
                storage
            ];
        for p in &mut world {
            p.padding = 0x34567890;
        }
        let parents: Vec<u32> = (0..storage)
            .map(|i| {
                if i < n as usize {
                    let i = i as u32;
                    (i / batches * 7 % bodies) * batches + i % batches
                } else {
                    u32::MAX
                }
            })
            .collect();
        let ids = BatchIndices {
            bodies_len: bodies,
            colliders_len: count,
            colliders_batch_capacity: count,
            num_batches: batches,
            ..Default::default()
        };
        pilot::write_buffer(&dir, "BodyPoses.bin", &body_poses);
        pilot::write_buffer(&dir, "LocalPoses.bin", &local);
        pilot::write_buffer(&dir, "WorldPoses.bin", &world);
        pilot::write_buffer(&dir, "Parents.bin", &parents);
        pilot::write_buffer(&dir, "BatchIndices.bin", &[ids]);
        let groups = n.div_ceil(64).max(1);
        for _ in 0..3 {
            for i in 0..groups * 64 {
                gpu_sync_collider_poses(
                    UVec3::new(i, 0, 0),
                    UVec3::new(groups, 1, 1),
                    &body_poses,
                    &local,
                    &mut world,
                    &parents,
                    &ids,
                );
            }
        }
        pilot::write_buffer(&dir, "WorldPoses.expected.bin", &world);
        cases.push(json!({"name":name,"kind":"sync_colliders","count":n,"batches":batches,"colliders_per_batch":count,"body_count":body_count,"storage":storage,"body_storage":body_storage,"iterations":3}));
    }
    pilot::write_json(root.join("suite.json"), &json!({"cases":cases}));
}
