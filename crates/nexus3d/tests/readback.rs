//! Reads multibody state back on the WebGPU backend, which validates buffer usages (native
//! Metal does not). A readback from a buffer without `COPY_SRC` panics with a wgpu error.
//! Needs a GPU: `cargo test -p nexus3d --release --features rbd,mpm --test readback -- --ignored`.

use khal::backend::{GpuBackend, WebGpu};
use nexus3d::prelude::{NexusCapacities, NexusPipeline, NexusState, RbdCoupling};
use rapier3d::prelude::{ColliderBuilder, RevoluteJointBuilder, RigidBodyBuilder, Vec3};

const NUM_LINKS: usize = 4;

#[test]
#[ignore = "needs a GPU"]
fn multibody_state_reads_back_on_webgpu() {
    pollster::block_on(async {
        let backend = GpuBackend::WebGpu(WebGpu::default().await.unwrap());
        let mut state = NexusState::new(NexusCapacities::default());
        state.insert_rigid_body(
            RigidBodyBuilder::fixed().build(),
            ColliderBuilder::cuboid(40.0, 1.0, 40.0).build(),
            RbdCoupling::None,
        );
        let half = 0.4f32;
        let mut links = Vec::new();
        for l in 0..NUM_LINKS {
            let handle = state.insert_rigid_body(
                RigidBodyBuilder::dynamic()
                    .translation(Vec3::new(0.0, 8.0 - l as f32 * 2.0 * half, 0.0))
                    .build(),
                ColliderBuilder::cuboid(half, half, half).build(),
                RbdCoupling::None,
            );
            if let Some(&parent) = links.last() {
                let joint = RevoluteJointBuilder::new(Vec3::Z)
                    .local_anchor1(Vec3::new(0.0, -half, 0.0))
                    .local_anchor2(Vec3::new(0.0, half, 0.0));
                state.insert_multibody_joint(parent, handle, joint);
            }
            links.push(handle);
        }

        let mut pipeline = NexusPipeline::default();
        state.finalize(&backend).await.unwrap();
        for _ in 0..3 {
            pipeline.simulate(&backend, &mut state, None).await.unwrap();
        }

        // `dof_state`: the generalized velocities.
        let vels = state
            .multibody_joint_velocities(&backend, 0, links[0])
            .await
            .unwrap();
        // A free root (6 DoFs) and one revolute DoF per joint.
        assert_eq!(vels.len(), 6 + NUM_LINKS - 1);

        // `links_static` and `dof_state`, read by the env-reset snapshot.
        let rbd = state.rbd.as_mut().unwrap();
        let _ = rbd.snapshot(&backend).await;

        // `multibody_info`, reallocated when joint friction reserves constraint slots.
        let ndofs = (rbd.multibodies().dofs_per_batch() * rbd.multibodies().num_batches()) as usize;
        rbd.set_dof_frictionloss(&backend, &vec![0.1; ndofs]);
        let (layout, _) = rbd.multibodies().debug_cons_layout(&backend).await;
        assert!(!layout.is_empty());
    });
}
