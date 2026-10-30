#![cfg(feature = "runtime")]
//! Public runtime growth must retain native hub behavior, with no host physics.
use khal::backend::{Backend, GpuBackend};
use nexus_rbd3d::{pipeline::RbdState, rapier::prelude::*, shaders::dynamics::RbdSimParams};
use nexus_tilelang_native::{
    runtime::{RuntimeCheckpoint, RuntimeOptions, TileLangBackend, TileLangSimulation},
    upload_cases::{capacities, hub_scene},
};
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};
fn exercise(target: TileLangBackend) {
    futures::executor::block_on(async {
        let backend = GpuBackend::Cpu;
        let scenes = [hub_scene(0.), hub_scene(10.)];
        let mb = MultibodyJointSet::new();
        let mut params = RbdSimParams::tgs_soft();
        params.dt = 0.03;
        params.num_solver_iterations = 3;
        params.num_internal_pgs_iterations = 5;
        let refs: Vec<_> = scenes
            .iter()
            .map(|(b, c, j)| (b, c, j, &mb, &params))
            .collect();
        let mut caps = capacities(2);
        caps.body_capacity = 34;
        caps.collisions_capacity = 128;
        let mut state = RbdState::from_rapier(&backend, &refs, caps);
        state.set_gravity(&backend, [0., -9.81, 0.]);
        state.set_deterministic(&backend, true);
        backend.synchronize().unwrap();
        let upload = state.read_scene_upload(&backend).await.unwrap();
        let root =
            PathBuf::from(std::env::var_os("NEXUS_TILELANG_RUNTIME_WORK").expect("Set workspace"));
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let workspace = root.join(format!("hubs-{target:?}-{}-{now}", std::process::id()));
        println!("Hub runtime artifacts: {}", workspace.display());
        let options = RuntimeOptions {
            backend: target,
            python: PathBuf::from(
                std::env::var_os("NEXUS_TILELANG_RUNTIME_PYTHON").expect("Set compiler"),
            ),
            port_directory: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .to_owned(),
            workspace,
            initial_pair_capacity: 1,
            max_pair_capacity: 256,
            trace_steps: None,
            contact_reduction: true,
        };
        let mut simulation = unsafe { TileLangSimulation::from_upload(&upload, options) }.unwrap();
        let expected = fs::read(
            PathBuf::from(
                std::env::var_os("NEXUS_TILELANG_HUB_EXPECTED")
                    .expect("Set fresh split-hubs-f8 fixture"),
            )
            .join("Trajectory.expected.bin"),
        )
        .unwrap();
        let trace: Vec<f32> = expected
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect();
        let bodies = (upload.batch_indices.bodies_len * upload.batch_indices.num_batches) as usize;
        let stride = bodies
            * (size_of::<nexus_rbd3d::math::Pose>() + size_of::<nexus_rbd3d::Velocity>())
            / 4;
        let mut saved = None;
        let mut observed = Vec::new();
        for frame in 0..8 {
            let progress = simulation.step().unwrap();
            assert_eq!(progress.step, frame as u64 + 1);
            if frame == 0 {
                assert!(progress.retries > 0);
            }
            let poses = simulation.body_poses().unwrap();
            let velocities = simulation.velocities().unwrap();
            let actual: Vec<f32> = bytemuck::cast_slice::<_, f32>(&poses)
                .iter()
                .chain(bytemuck::cast_slice::<_, f32>(&velocities))
                .copied()
                .collect();
            assert_eq!(actual.len(), stride);
            observed.push(actual.clone());
            if frame == 3 {
                saved = Some(simulation.checkpoint().unwrap());
            }
            for (i, (&a, &e)) in actual
                .iter()
                .zip(&trace[frame * stride..(frame + 1) * stride])
                .enumerate()
            {
                assert!(
                    a.is_finite() && (a - e).abs() <= 2e-6 + 2e-5 * e.abs(),
                    "frame {frame} word {i}: {a} != {e}"
                );
            }
            // Both dynamic hubs are first in the interleaved native body layout.
            for (a, e) in poses[2..].iter().zip(&upload.body_poses[2..bodies]) {
                assert_eq!(bytemuck::bytes_of(a), bytemuck::bytes_of(e));
            }
        }
        assert_eq!(simulation.step().unwrap().step, 9);
        let saved = saved.unwrap();
        let encoded = serde_json::to_vec(&saved).unwrap();
        let restored: RuntimeCheckpoint = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(restored.step(), 4);
        let before = serde_json::to_value(simulation.checkpoint().unwrap()).unwrap();
        let mut wrong = serde_json::to_value(&restored).unwrap();
        wrong["signature"] = serde_json::json!({"invalid":true});
        let wrong: RuntimeCheckpoint = serde_json::from_value(wrong).unwrap();
        assert!(unsafe { simulation.restore_checkpoint(&wrong) }.is_err());
        assert_eq!(
            serde_json::to_value(simulation.checkpoint().unwrap()).unwrap(),
            before
        );
        simulation.set_gravity([0., 0., 0.]).unwrap();
        unsafe { simulation.restore_checkpoint(&restored) }.unwrap();
        assert_eq!(
            serde_json::to_value(simulation.checkpoint().unwrap()).unwrap(),
            serde_json::to_value(&restored).unwrap()
        );
        for (frame, expected) in observed.iter().enumerate().skip(4) {
            assert_eq!(simulation.step().unwrap().step, frame as u64 + 1);
            let poses = simulation.body_poses().unwrap();
            let velocities = simulation.velocities().unwrap();
            for (&a, &e) in bytemuck::cast_slice::<_, f32>(&poses)
                .iter()
                .chain(bytemuck::cast_slice::<_, f32>(&velocities))
                .zip(expected)
            {
                assert!(
                    a.is_finite() && (a - e).abs() <= 2e-6 + 2e-5 * e.abs(),
                    "restored frame {frame}: {a} != {e}"
                );
            }
        }
        simulation.reset().unwrap();
        assert_eq!(simulation.checkpoint().unwrap().step(), 0);
        assert_eq!(
            bytemuck::cast_slice::<_, u32>(&simulation.body_poses().unwrap()),
            bytemuck::cast_slice::<_, u32>(&upload.body_poses[..bodies])
        );
        assert_eq!(simulation.step().unwrap().step, 1);
        let poses = simulation.body_poses().unwrap();
        let velocities = simulation.velocities().unwrap();
        for (&a, &e) in bytemuck::cast_slice::<_, f32>(&poses)
            .iter()
            .chain(bytemuck::cast_slice::<_, f32>(&velocities))
            .zip(&observed[0])
        {
            assert!(
                a.is_finite() && (a - e).abs() <= 2e-6 + 2e-5 * e.abs(),
                "reset: {a} != {e}"
            );
        }
    });
}
#[test]
#[ignore = "requires compiler and fresh split-hubs-f8 fixture"]
fn c_hubs_survive_runtime_growth() {
    exercise(TileLangBackend::C);
}
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires Metal, compiler and fresh split-hubs-f8 fixture"]
fn metal_hubs_survive_runtime_growth() {
    exercise(TileLangBackend::Metal);
}
