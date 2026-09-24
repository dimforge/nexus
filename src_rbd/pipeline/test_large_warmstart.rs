//! Exercises the large-scene warmstart scratch on WebGPU, in both dimensions.
use super::{RbdCapacities, RbdPipeline, RbdState};
use crate::math::Pose;
use crate::rapier::prelude::*;
use crate::shaders::dynamics::RbdSimParams;
use khal::backend::{Backend, GpuBackend, WebGpu};

#[futures_test::test]
#[serial_test::serial]
#[ignore = "requires a WebGPU adapter"]
async fn large_warmstart_keeps_all_boxes_on_ground() {
    check_large_warmstart(16, RbdSimParams::tgs_soft(), false).await;
}

#[futures_test::test]
#[serial_test::serial]
#[ignore = "requires a WebGPU adapter"]
async fn large_warmstart_supports_default_binding_limit() {
    check_large_warmstart(8, RbdSimParams::tgs_soft(), false).await;
    #[cfg(feature = "dim3")]
    {
        let mut params = RbdSimParams::tgs_soft();
        params.friction_model = crate::shaders::dynamics::FrictionModel::Simplified;
        check_large_warmstart(8, params, false).await;
    }
}

#[futures_test::test]
#[serial_test::serial]
#[ignore = "requires a WebGPU adapter"]
async fn large_warmstart_resizes_within_default_binding_limit() {
    check_large_warmstart(8, RbdSimParams::tgs_soft(), true).await;
    #[cfg(feature = "dim3")]
    {
        let mut params = RbdSimParams::tgs_soft();
        params.friction_model = crate::shaders::dynamics::FrictionModel::Simplified;
        check_large_warmstart(8, params, true).await;
    }
}

async fn check_large_warmstart(storage_buffers: u32, params: RbdSimParams, resize: bool) {
    let backend = GpuBackend::WebGpu(
        WebGpu::new(
            Default::default(),
            khal::re_exports::wgpu::Limits {
                max_storage_buffers_per_shader_stage: storage_buffers,
                ..Default::default()
            },
        )
        .await
        .unwrap(),
    );
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    #[cfg(feature = "dim3")]
    let (ground_pos, ground_shape) = (
        Vector::new(0.0, -0.5, 0.0),
        ColliderBuilder::cuboid(50.0, 0.5, 50.0),
    );
    #[cfg(feature = "dim2")]
    let (ground_pos, ground_shape) = (Vector::new(0.0, -0.5), ColliderBuilder::cuboid(2000.0, 0.5));
    let ground = bodies.insert(RigidBodyBuilder::fixed().translation(ground_pos));
    colliders.insert_with_parent(ground_shape, ground, &mut bodies);
    for i in 0..4096 {
        #[cfg(feature = "dim3")]
        let (pos, shape) = (
            Vector::new((i % 64) as f32 * 0.7, 0.25, (i / 64) as f32 * 0.7),
            ColliderBuilder::cuboid(0.2, 0.2, 0.2),
        );
        #[cfg(feature = "dim2")]
        let (pos, shape) = (
            Vector::new((i as f32 - 2048.0) * 0.7, 0.25),
            ColliderBuilder::cuboid(0.2, 0.2),
        );
        let handle = bodies.insert(
            RigidBodyBuilder::dynamic()
                .can_sleep(false)
                .translation(pos),
        );
        colliders.insert_with_parent(shape, handle, &mut bodies);
    }
    let mut state = RbdState::from_rapier(
        &backend,
        &[(
            &bodies,
            &colliders,
            &ImpulseJointSet::new(),
            &MultibodyJointSet::new(),
            &params,
        )],
        RbdCapacities {
            collisions_capacity: if resize { 64 } else { 8192 },
            ..Default::default()
        },
    );
    let pipeline = RbdPipeline::new(&backend).unwrap();
    let initial_contact_capacity = state.contacts_capacity_cpu;
    for step in 0..120 {
        if resize {
            // Readbacks must finish so the deliberately undersized buffers grow
            // before falling boxes cross the ground. Toggle the ordinary
            // deterministic path after both arena phases have been exercised.
            backend.synchronize().unwrap();
            if step == 61 || step == 80 {
                state.set_deterministic(&backend, step == 61);
            }
        }
        pipeline.auto_resize_buffers(&backend, &mut state).unwrap();
        pipeline.step(&backend, &mut state, None).unwrap();
    }
    backend.synchronize().unwrap();
    if resize {
        assert!(state.contacts_capacity_cpu > initial_contact_capacity);
    }
    let poses: Vec<Pose> = backend
        .slow_read_vec(state.body_poses().buffer())
        .await
        .unwrap();
    for (i, pose) in poses.iter().enumerate().skip(1).take(4096) {
        assert!(pose.translation.is_finite(), "box {i}: non-finite pose");
        assert!(
            (pose.translation.y - 0.2).abs() < 0.01,
            "box {i}: height {}",
            pose.translation.y
        );
    }
}
