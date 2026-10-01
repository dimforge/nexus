//! 2D probe for contact warmstarting on polylines and multi-collider bodies.
//!
//! A polyline ground emits one manifold per segment, and a body with two
//! colliders one per collider, so a body pair alone does not identify a
//! manifold across frames. Bodies resting on a zigzag polyline must stay at
//! rest: no ejection, no fall-through, no resting jitter. Two failures are
//! guarded: a broad phase that ignored the segment capsules found pairs only
//! once bodies were deep inside them and launched every body, and matching
//! manifolds by body pair alone left a resting jitter of a few tenths of a
//! millimeter. Run with
//! `cargo test -p nexus_rbd2d --features metal test_polyline_warmstart -- --nocapture --ignored`.

use crate::math::Pose;
use crate::pipeline::{RbdCapacities, RbdPipeline, RbdState};
use crate::rapier::prelude::*;
use crate::shaders::dynamics::RbdSimParams;
use khal::backend::{Backend, GpuBackend};

async fn test_backend() -> GpuBackend {
    #[cfg(feature = "metal")]
    {
        GpuBackend::Metal(khal::backend::metal::Metal::new().unwrap())
    }
    #[cfg(not(feature = "metal"))]
    {
        GpuBackend::WebGpu(khal::backend::WebGpu::default().await.unwrap())
    }
}

/// Number of dynamic bodies per environment.
const NUM_BODIES: usize = 9;

/// The polyline narrow phase sees each segment as a capsule of this radius, so
/// the ground surface sits this far above the polyline's vertices.
const POLYLINE_THICKNESS: f32 = 0.4;

/// A zigzag polyline ground (2 cm teeth every 10 cm; with the segment capsules
/// every resting body touches about ten segments, one manifold each) with
/// boxes, balls and two-collider bodies on it.
fn build_env() -> (RigidBodySet, ColliderSet) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let ground = bodies.insert(RigidBodyBuilder::fixed());
    let vertices: Vec<Vector> = (0..=60)
        .map(|i| Vector::new(i as f32 * 0.1 - 3.0, 0.02 * (i % 2) as f32))
        .collect();
    let indices: Vec<[u32; 2]> = (0..60).map(|i| [i, i + 1]).collect();
    colliders.insert_with_parent(
        ColliderBuilder::polyline(vertices, Some(indices)),
        ground,
        &mut bodies,
    );
    for k in 0..NUM_BODIES {
        let x = k as f32 * 0.6 - 2.4;
        let y = POLYLINE_THICKNESS + 0.2;
        let body = bodies.insert(RigidBodyBuilder::dynamic().translation(Vector::new(x, y)));
        match k % 3 {
            0 => {
                colliders.insert_with_parent(
                    ColliderBuilder::cuboid(0.15, 0.08),
                    body,
                    &mut bodies,
                );
            }
            1 => {
                colliders.insert_with_parent(ColliderBuilder::ball(0.1), body, &mut bodies);
            }
            _ => {
                for dx in [-0.09, 0.09] {
                    colliders.insert_with_parent(
                        ColliderBuilder::cuboid(0.08, 0.06).translation(Vector::new(dx, 0.0)),
                        body,
                        &mut bodies,
                    );
                }
            }
        }
    }
    (bodies, colliders)
}

/// The worst resting motion over `steps` steps after settling, across every
/// dynamic body of `num_envs` environments.
struct RestReport {
    /// Largest height excursion from the settled pose (m).
    max_excursion: f32,
    /// Largest upward displacement in a single step (m).
    max_step_rise: f32,
    /// Bodies that left the scene (ejected or fell through).
    lost: usize,
}

async fn rest_report(num_envs: usize, settle: u32, steps: u32) -> RestReport {
    let backend = test_backend().await;
    let pipeline = RbdPipeline::new(&backend).unwrap();
    let envs: Vec<_> = (0..num_envs).map(|_| build_env()).collect();
    let impulse_joints = ImpulseJointSet::new();
    let multibody_joints = MultibodyJointSet::new();
    let params = RbdSimParams::tgs_soft();
    let refs: Vec<_> = envs
        .iter()
        .map(|(b, c)| (b, c, &impulse_joints, &multibody_joints, &params))
        .collect();
    let capacities = RbdCapacities {
        batches: num_envs as u32,
        ..Default::default()
    };
    let mut state = RbdState::from_rapier(&backend, &refs, capacities);

    for _ in 0..settle {
        pipeline.step(&backend, &mut state, None).unwrap();
    }
    let rest: Vec<Pose> = backend
        .slow_read_vec(state.body_poses().buffer())
        .await
        .unwrap();
    let mut prev = rest.clone();
    let mut report = RestReport {
        max_excursion: 0.0,
        max_step_rise: 0.0,
        lost: 0,
    };
    // GPU pose slot of body `b` in environment `e` is `b * num_envs + e`; body 0
    // is the fixed ground and slots past the bodies are spare capacity
    let dynamic = num_envs..(NUM_BODIES + 1) * num_envs;
    let mut lost = vec![false; rest.len()];
    for _ in 0..steps {
        pipeline.step(&backend, &mut state, None).unwrap();
        let poses: Vec<Pose> = backend
            .slow_read_vec(state.body_poses().buffer())
            .await
            .unwrap();
        for k in dynamic.clone() {
            let y = poses[k].translation.y;
            if !(POLYLINE_THICKNESS - 0.1..=POLYLINE_THICKNESS + 1.0).contains(&y) {
                lost[k] = true;
                continue;
            }
            report.max_excursion = report.max_excursion.max((y - rest[k].translation.y).abs());
            report.max_step_rise = report.max_step_rise.max(y - prev[k].translation.y);
        }
        prev = poses;
    }
    report.lost = lost.iter().filter(|l| **l).count();
    println!(
        "polyline rest, {num_envs} env(s): excursion {:.4} m, step rise {:.4} m, lost {}/{}",
        report.max_excursion,
        report.max_step_rise,
        report.lost,
        num_envs * NUM_BODIES
    );
    report
}

#[futures_test::test]
#[serial_test::serial]
#[ignore]
async fn test_polyline_warmstart() {
    for num_envs in [1, 256] {
        let report = rest_report(num_envs, 240, 240).await;
        assert_eq!(report.lost, 0, "bodies left a resting polyline scene");
        // body-pair matching measured 0.2 to 0.3 mm here; per-manifold matching 0
        assert!(
            report.max_excursion < 1.0e-4,
            "resting bodies moved {} m",
            report.max_excursion
        );
        assert!(
            report.max_step_rise < 1.0e-4,
            "a resting body popped {} m",
            report.max_step_rise
        );
    }
}
