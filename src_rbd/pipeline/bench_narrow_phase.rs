//! Headless timing harness for the batched narrow phase.
//!
//! Builds `num_envs` copies of a small scene (the many-small-environments shape
//! the flat dispatch targets) and reports wall-clock ms per step. Run with
//! `cargo test -p nexus_rbd3d --features metal bench_narrow_phase -- --nocapture --ignored`.

use crate::pipeline::{RbdCapacities, RbdPipeline, RbdState};
use crate::rapier::prelude::*;
use crate::shaders::dynamics::RbdSimParams;
use khal::backend::{Backend, GpuBackend};

/// One environment: a ground cuboid plus `num_boxes` stacked dynamic cuboids.
fn build_env(num_boxes: usize) -> (RigidBodySet, ColliderSet) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();

    let ground = bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0.0, -0.5, 0.0)));
    colliders.insert_with_parent(ColliderBuilder::cuboid(5.0, 0.5, 5.0), ground, &mut bodies);

    for i in 0..num_boxes {
        let y = 0.6 + i as f32 * 0.45;
        let handle = bodies.insert(RigidBodyBuilder::dynamic().translation(Vec3::new(0.0, y, 0.0)));
        colliders.insert_with_parent(ColliderBuilder::cuboid(0.2, 0.2, 0.2), handle, &mut bodies);
    }

    (bodies, colliders)
}

async fn run_bench(num_envs: u32, num_boxes: usize, num_steps: u32) {
    // Metal, not WebGPU: `gpu_mb_init_joint_constraints` currently binds 10
    // storage buffers, over WebGPU's per-stage limit of 8.
    let backend = GpuBackend::Metal(khal::backend::metal::Metal::new().unwrap());

    let envs: Vec<_> = (0..num_envs).map(|_| build_env(num_boxes)).collect();
    let joints = ImpulseJointSet::new();
    let mb_joints = MultibodyJointSet::new();
    let params = RbdSimParams::tgs_soft();
    let refs: Vec<_> = envs
        .iter()
        .map(|(b, c)| (b, c, &joints, &mb_joints, &params))
        .collect();

    let capacities = RbdCapacities {
        batches: num_envs,
        collisions_capacity: 256,
        ..Default::default()
    };
    let mut state = RbdState::from_rapier(&backend, &refs, capacities);
    let pipeline = RbdPipeline::new(&backend).unwrap();

    // Warm up: buffer growth and coloring settle over the first few steps.
    for _ in 0..20 {
        pipeline.step(&backend, &mut state, None).unwrap();
    }
    backend.synchronize().unwrap();

    let start = web_time::Instant::now();
    for _ in 0..num_steps {
        pipeline.step(&backend, &mut state, None).unwrap();
    }
    backend.synchronize().unwrap();
    let elapsed = start.elapsed();

    let per_step = elapsed.as_secs_f64() * 1000.0 / num_steps as f64;
    println!(
        "envs={num_envs:5} boxes/env={num_boxes} -> {per_step:8.3} ms/step  \
         ({:.0} env-steps/s)",
        num_envs as f64 / (per_step / 1000.0)
    );
}

#[futures_test::test]
#[serial_test::serial]
#[ignore]
async fn bench_narrow_phase_sweep() {
    for envs in [1u32, 64, 256, 1024, 4096] {
        run_bench(envs, 4, 200).await;
    }
}
