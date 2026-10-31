#![cfg(feature = "runtime")]
//! A dense non-hub graph must grow before solving, including native tail colors.
use khal::backend::{Backend, GpuBackend};
use nexus_rbd3d::{
    pipeline::{RbdPipeline, RbdState},
    rapier::prelude::*,
    shaders::dynamics::{FrictionModel, RbdSimParams},
};
use nexus_tilelang_native::{
    runtime::{RuntimeOptions, TileLangBackend, TileLangSimulation},
    upload_cases::{capacities, dense_color_scene},
};
use std::{
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

fn compare(actual: &[f32], expected: &[f32], frame: usize) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && e.is_finite() && (a - e).abs() <= 2e-6 + 2e-5 * e.abs(),
            "dense frame {frame} word {i}: {a} != {e}"
        );
    }
}
fn exercise(target: TileLangBackend) {
    futures::executor::block_on(async {
        assert_eq!(std::env::var("NEXUS_DISABLE_BF").as_deref(), Ok("1"));
        let backend = GpuBackend::Cpu;
        let (bodies, colliders, joints) = dense_color_scene(0.);
        let mb = MultibodyJointSet::new();
        let mut params = RbdSimParams::tgs_soft();
        params.dt = 0.03;
        params.num_solver_iterations = 3;
        params.num_internal_pgs_iterations = 5;
        params.friction_model = FrictionModel::Coulomb;
        let mut caps = capacities(1);
        caps.body_capacity = 34;
        caps.collisions_capacity = 64;
        // A full native palette makes the production reference valid on its first
        // frame. The port starts at the usual eight-color native budget below.
        caps.solver_colors = 31;
        let mut state = RbdState::from_rapier(
            &backend,
            &[(&bodies, &colliders, &joints, &mb, &params)],
            caps,
        );
        state.set_gravity(&backend, [0., -9.81, 0.]);
        state.set_deterministic(&backend, true);
        backend.synchronize().unwrap();
        let mut upload = state.read_scene_upload(&backend).await.unwrap();
        upload.batch_indices.solver_color_buckets_stride = 11;
        let count = (upload.batch_indices.bodies_len * upload.batch_indices.num_batches) as usize;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let workspace =
            PathBuf::from(std::env::var_os("NEXUS_TILELANG_RUNTIME_WORK").expect("Set workspace"))
                .join(format!("coloring-{target:?}-{}-{now}", std::process::id()));
        println!("Coloring runtime artifacts: {}", workspace.display());
        let options = RuntimeOptions {
            kernel_cache: None,
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
            max_pair_capacity: 128,
            trace_steps: None,
            contact_reduction: true,
        };
        let mut runtime =
            unsafe { TileLangSimulation::from_upload(&upload, options.clone()) }.unwrap();
        let mut pipeline = RbdPipeline::new(&backend).unwrap();
        pipeline.contact_reduction = true;
        let mut saved = None;
        let mut continuation = Vec::new();
        for frame in 0..120 {
            if frame == 60 {
                // Live control must survive the expanded coloring schedule.
                runtime.set_gravity([0., -5., 0.]).unwrap();
                state.set_gravity(&backend, [0., -5., 0.]);
            }
            let progress = runtime.step().unwrap();
            assert_eq!(progress.step, frame as u64 + 1);
            assert_eq!(progress.counts["coloring_rounds"], 0);
            if frame == 0 {
                assert!(progress.retries >= 2);
                assert!(progress.counts["colors"] >= 24);
            }
            pipeline.step(&backend, &mut state, None).unwrap();
            backend.synchronize().unwrap();
            let poses = runtime.body_poses().unwrap();
            let velocities = runtime.velocities().unwrap();
            let native_poses = backend
                .slow_read_vec(state.body_poses().buffer())
                .await
                .unwrap();
            let native_velocities = backend.slow_read_vec(state.vels().buffer()).await.unwrap();
            compare(
                bytemuck::cast_slice(&poses),
                bytemuck::cast_slice(&native_poses[..count]),
                frame,
            );
            compare(
                bytemuck::cast_slice(&velocities),
                bytemuck::cast_slice(&native_velocities[..count]),
                frame,
            );
            for (actual, initial) in poses[1..].iter().zip(&upload.body_poses[1..count]) {
                assert_eq!(bytemuck::bytes_of(actual), bytemuck::bytes_of(initial));
            }
            // Independent stability checks: a supported plate cannot fall through
            // its supports or gain unbounded motion while all contacts are solved.
            assert!(poses[0].translation.y > -0.04 && poses[0].translation.y < 0.2);
            assert!(poses[0].rotation.is_finite());
            for &value in bytemuck::cast_slice::<_, f32>(&velocities[..1]) {
                assert!(value.is_finite() && value.abs() < 3.);
            }
            let feedback = runtime.read_words("ColorRequirements").unwrap();
            assert_eq!(&feedback[2..], &[0xDEADBEEF; 3]);
            if frame == 116 {
                saved = Some(runtime.checkpoint().unwrap());
            }
            if frame > 116 {
                continuation.push((poses, velocities));
            }
            if frame % 10 == 9 {
                println!("{target:?}: dense native comparison frame {}", frame + 1);
            }
        }
        let saved = saved.unwrap();
        let encoded = serde_json::to_value(&saved).unwrap();
        assert!(encoded["frame"]["capacities"]["colors"].as_u64().unwrap() >= 32);
        // A fresh runtime starts with the small palette. Restore must use named
        // limits, never mistake a color/round limit for a pair allocation count.
        let mut restart_options = options;
        restart_options.workspace = restart_options.workspace.with_extension("restart");
        let mut restart =
            unsafe { TileLangSimulation::from_upload(&upload, restart_options) }.unwrap();
        unsafe { restart.restore_checkpoint(&saved) }.unwrap();
        for (i, (poses, velocities)) in continuation.iter().enumerate() {
            assert_eq!(restart.step().unwrap().step, 118 + i as u64);
            compare(
                bytemuck::cast_slice(&restart.body_poses().unwrap()),
                bytemuck::cast_slice(poses),
                117 + i,
            );
            compare(
                bytemuck::cast_slice(&restart.velocities().unwrap()),
                bytemuck::cast_slice(velocities),
                117 + i,
            );
        }
    });
}
#[test]
#[ignore = "requires the local compiler and independent native CPU pipeline"]
fn c_dense_coloring_growth_and_native_tail_parity() {
    exercise(TileLangBackend::C);
}
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires Metal, the local compiler and independent native CPU pipeline"]
fn metal_dense_coloring_growth_and_native_tail_parity() {
    exercise(TileLangBackend::Metal);
}
