#![cfg(feature = "cpu")]
use khal::backend::GpuBackend;
use nexus_guide_3d::*;
use nexus3d::prelude::*;

#[test]
fn documented_scenes_run() {
    pollster::block_on(async {
        let backend = GpuBackend::Cpu;
        let mut state = NexusState::default();
        let ball = bodies(&mut state);
        let initial_y = state.rbd_world(0).bodies[ball].translation().y;
        lifecycle(&backend, &mut state).await.unwrap();
        let pose = readback(&backend, &state, ball).await.unwrap();
        assert!(pose.translation.y < initial_y);

        let mut state = batches();
        determinism(&backend, &mut state);
        friction_model(&mut state);
        lifecycle(&backend, &mut state).await.unwrap();
        assert!(state.deterministic());
        assert_eq!(state.rbd_num_batches(), 4);

        let mut state = insertion(&backend).await.unwrap();
        lifecycle(&backend, &mut state).await.unwrap();
        assert_eq!(state.counts().rigid_bodies, 2);

        let mut state = NexusState::default();
        colliders(&mut state);
        state.finalize(&backend).await.unwrap();
        assert_eq!(state.counts().colliders, 2);
        assert_eq!(state.counts().rigid_bodies, 1);

        let mut state = NexusState::default();
        let chunk = particles(&backend, &mut state).await.unwrap();
        coupling(&mut state);
        lifecycle(&backend, &mut state).await.unwrap();
        remove_particles(&backend, &mut state, chunk).unwrap();
        assert_eq!(state.counts().particles, 0);

        let mut state = NexusState::default();
        joints(&mut state);
        lifecycle(&backend, &mut state).await.unwrap();
    });
}
