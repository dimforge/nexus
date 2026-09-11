//! Live TGS substep changes preserve body state and the full physics-step duration.
#![cfg(feature = "rbd")]

use khal::backend::{Backend, GpuBackend, WebGpu};
use nexus3d::prelude::{NexusCapacities, NexusPipeline, NexusState, RbdCoupling};
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder, Vec3};

#[test]
#[ignore = "requires a WebGPU adapter"]
fn changing_substeps_preserves_state_and_elapsed_time() {
    pollster::block_on(async {
        let backend = GpuBackend::WebGpu(
            WebGpu::new(Default::default(), Default::default())
                .await
                .unwrap(),
        );
        let mut pipeline = NexusPipeline::default();
        const DT: f32 = 1.0 / 50.0;
        for (batches, reserve) in [(2, 0), (1, 8)] {
            let mut state = NexusState::new(
                NexusCapacities::default()
                    .rbd_batches(batches)
                    .rbd_collisions(64),
            );
            state.reserve_rigid_bodies(reserve);
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
            state.set_rbd_timestep(DT, 4);
            state.set_rbd_substeps(&backend, 3);
            state.finalize(&backend).await.unwrap();
            state.set_rbd_gravity(&backend, [0.0; 3]);
            assert_eq!(state.rbd.as_ref().unwrap().num_solver_iterations(), 3);
            for _ in 0..3 {
                pipeline.simulate(&backend, &mut state, None).await.unwrap();
            }

            for substeps in [1, 8, 2, 20, 0] {
                let rbd = state.rbd.as_ref().unwrap();
                let poses = backend
                    .slow_read_vec(rbd.body_poses().buffer())
                    .await
                    .unwrap();
                let velocities = backend.slow_read_vec(rbd.vels().buffer()).await.unwrap();
                let key = rbd.graph_key(1);
                state.set_rbd_substeps(&backend, substeps);
                state.finalize(&backend).await.unwrap();
                let rbd = state.rbd.as_ref().unwrap();
                assert_eq!(rbd.num_solver_iterations(), substeps.max(1));
                assert_eq!(rbd.multibodies().num_solver_iterations(), substeps.max(1));
                assert_eq!(state.rbd_substeps(), substeps.max(1));
                assert_ne!(
                    rbd.graph_key(1),
                    key,
                    "invalidate captured substep dispatches"
                );
                for env in 0..batches as usize {
                    let params = state.rbd_sim_params(env).unwrap();
                    assert_eq!(params.num_solver_iterations, substeps.max(1));
                    assert_eq!(params.dt, DT);
                }
                let after_poses = backend
                    .slow_read_vec(rbd.body_poses().buffer())
                    .await
                    .unwrap();
                let after_velocities = backend.slow_read_vec(rbd.vels().buffer()).await.unwrap();
                for (before, after) in poses.iter().zip(after_poses) {
                    assert_eq!(before.translation, after.translation);
                    assert_eq!(before.rotation, after.rotation);
                }
                for (before, after) in velocities.iter().zip(after_velocities) {
                    assert_eq!(before.linear, after.linear);
                    assert_eq!(before.angular, after.angular);
                }
                let key = rbd.graph_key(1);
                // The viewer pushes the selection every frame; unchanged is a no-op.
                state.set_rbd_substeps(&backend, substeps);
                assert_eq!(state.rbd.as_ref().unwrap().graph_key(1), key);
                pipeline.simulate(&backend, &mut state, None).await.unwrap();
                let rbd = state.rbd.as_ref().unwrap();
                let after = backend
                    .slow_read_vec(rbd.body_poses().buffer())
                    .await
                    .unwrap();
                for env in 0..batches as usize {
                    let dx = after[env].translation.x - poses[env].translation.x;
                    assert!((dx - DT).abs() < 1.0e-5, "{substeps} substeps: dx={dx}");
                }
            }
        }
    });
}
