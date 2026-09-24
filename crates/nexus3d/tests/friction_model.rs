//! Live testbed friction changes must not rebuild bodies from their initial poses.
#![cfg(feature = "rbd")]

use khal::backend::{Backend, GpuBackend, WebGpu};
use nexus3d::prelude::{NexusCapacities, NexusPipeline, NexusState, RbdCoupling};
use nexus3d::rbd::dynamics::FrictionModel;
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder, Vec3};

#[test]
#[ignore = "requires a WebGPU adapter"]
fn switching_friction_preserves_running_bodies() {
    pollster::block_on(async {
        let backend = GpuBackend::WebGpu(
            WebGpu::new(Default::default(), Default::default())
                .await
                .unwrap(),
        );
        let mut pipeline = NexusPipeline::default();
        for (batches, reserve) in [(2, 0), (1, 8)] {
            let mut state = NexusState::new(
                NexusCapacities::default()
                    .rbd_batches(batches)
                    .rbd_collisions(64),
            );
            state.reserve_rigid_bodies(reserve);
            state.set_rbd_friction_model(FrictionModel::Simplified);
            for env in 0..batches as usize {
                if env > 0 {
                    state.add_environment();
                }
                state.insert_rigid_body_in(
                    env,
                    RigidBodyBuilder::dynamic()
                        .translation(Vec3::new(env as f32, 3.0, 0.0))
                        .linvel(Vec3::X)
                        .can_sleep(false)
                        .build(),
                    ColliderBuilder::cuboid(0.2, 0.2, 0.2).build(),
                    RbdCoupling::None,
                );
            }
            state.finalize(&backend).await.unwrap();
            assert_eq!(
                state.rbd.as_ref().unwrap().friction_model(),
                FrictionModel::Simplified
            );
            for _ in 0..6 {
                pipeline.simulate(&backend, &mut state, None).await.unwrap();
            }
            for model in [FrictionModel::Coulomb, FrictionModel::Simplified] {
                let rbd = state.rbd.as_ref().unwrap();
                let poses = backend
                    .slow_read_vec(rbd.body_poses().buffer())
                    .await
                    .unwrap();
                let velocities = backend.slow_read_vec(rbd.vels().buffer()).await.unwrap();
                assert!(poses.iter().any(|pose| pose.translation.y < 2.99));
                let key = rbd.graph_key(1);
                state.set_rbd_friction_model(model);
                state.finalize(&backend).await.unwrap();
                let rbd = state.rbd.as_ref().unwrap();
                assert_eq!(rbd.friction_model(), model);
                assert_ne!(
                    rbd.graph_key(1),
                    key,
                    "model switch must invalidate captured work"
                );
                for env in 0..batches as usize {
                    assert_eq!(state.rbd_sim_params(env).unwrap().friction_model, model);
                }
                let after_poses = backend
                    .slow_read_vec(rbd.body_poses().buffer())
                    .await
                    .unwrap();
                let after_velocities = backend.slow_read_vec(rbd.vels().buffer()).await.unwrap();
                assert_eq!(poses.len(), after_poses.len());
                assert_eq!(velocities.len(), after_velocities.len());
                for (before, after) in poses.iter().zip(after_poses) {
                    assert_eq!(before.translation, after.translation);
                    assert_eq!(before.rotation, after.rotation);
                }
                for (before, after) in velocities.iter().zip(after_velocities) {
                    assert_eq!(before.linear, after.linear);
                    assert_eq!(before.angular, after.angular);
                }
                let key = state.rbd.as_ref().unwrap().graph_key(1);
                // The UI pushes its selection each frame; unchanged selections are cheap.
                state.set_rbd_friction_model(model);
                state.finalize(&backend).await.unwrap();
                assert_eq!(state.rbd.as_ref().unwrap().graph_key(1), key);
                pipeline.simulate(&backend, &mut state, None).await.unwrap();
            }
        }
    });
}
