#![cfg(feature = "runtime")]
use khal::backend::{Backend, GpuBackend};
use nexus_rbd3d::{
    pipeline::{RbdPipeline, RbdState},
    rapier::prelude::*,
    shaders::dynamics::RbdSimParams,
};
use nexus_tilelang_native::{
    runtime::{RuntimeOptions, TileLangBackend, TileLangSimulation},
    upload_cases::{capacities, small_contact_scene},
};
use std::{
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

fn compare<T: bytemuck::Pod>(a: &[T], b: &[T]) {
    let a: &[f32] = bytemuck::cast_slice(a);
    let b: &[f32] = bytemuck::cast_slice(b);
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-6 + 2e-5 * b.abs(),
            "{a} != {b}"
        );
    }
}
fn exercise(target: TileLangBackend) {
    assert!(
        std::env::var("NEXUS_DISABLE_BF").is_ok(),
        "Set NEXUS_DISABLE_BF=1"
    );
    futures::executor::block_on(async {
        let backend = GpuBackend::Cpu;
        let scenes = [small_contact_scene(0.), small_contact_scene(10.)];
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
        caps.collisions_capacity = 64;
        let mut native = RbdState::from_rapier(&backend, &refs, caps);
        native.set_deterministic(&backend, true);
        native.set_gravity(&backend, [0., -9.81, 0.]);
        let upload = native.read_scene_upload(&backend).await.unwrap();
        let root = PathBuf::from(
            std::env::var_os("NEXUS_TILELANG_RUNTIME_WORK").expect("Set runtime output directory"),
        );
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let workspace = root.join(format!(
            "configuration-{target:?}-{}-{now}",
            std::process::id()
        ));
        println!("Runtime configuration artifacts: {}", workspace.display());
        let options = RuntimeOptions {
            kernel_cache: None,
            backend: target,
            python: PathBuf::from(
                std::env::var_os("NEXUS_TILELANG_RUNTIME_PYTHON").expect("Set compiler Python"),
            ),
            port_directory: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .to_owned(),
            workspace,
            initial_pair_capacity: 1,
            max_pair_capacity: 64,
            trace_steps: None,
            contact_reduction: false,
        };
        let mut tilelang = unsafe { TileLangSimulation::from_upload(&upload, options) }.unwrap();
        let pipeline = RbdPipeline::new(&backend).unwrap();
        for frame in 0..8 {
            if frame == 2 {
                let before = serde_json::to_value(tilelang.checkpoint().unwrap()).unwrap();
                let mut updated = tilelang.solver_parameters();
                let mut invalid = updated;
                invalid.dt = f32::NAN;
                assert!(tilelang.set_solver_parameters(invalid).is_err());
                invalid = updated;
                invalid.num_solver_iterations = 0;
                assert!(tilelang.set_solver_parameters(invalid).is_err());
                assert_eq!(
                    before,
                    serde_json::to_value(tilelang.checkpoint().unwrap()).unwrap()
                );
                updated.dt = 0.02 / 2.0;
                updated.num_solver_iterations = 2;
                updated.num_internal_pgs_iterations = 3;
                updated.contact_merge_cos = 0.95;
                updated.normalized_contact_recycle_distance = 0.001;
                tilelang.set_solver_parameters(updated).unwrap();
                let after = serde_json::to_value(tilelang.checkpoint().unwrap()).unwrap();
                assert_eq!(
                    before["frame"], after["frame"],
                    "Reconfiguration must preserve every cached word"
                );
                assert_ne!(before["signature"], after["signature"]);
                assert!(tilelang.solver_parameters() == updated);
                // Identical setters do not compile or reset the live simulation.
                tilelang.set_timestep(0.02, 2).unwrap();
                tilelang.set_num_internal_pgs_iterations(3).unwrap();
                tilelang.set_contact_merge_cos(0.95).unwrap();
                tilelang.set_contact_recycle_distance(0.001).unwrap();
                assert_eq!(
                    after,
                    serde_json::to_value(tilelang.checkpoint().unwrap()).unwrap()
                );
                native.set_timestep(&backend, 0.02, 2);
                native.set_num_internal_pgs_iterations(&backend, 3);
                native.set_contact_merge_cos(&backend, 0.95);
                native.set_contact_recycle_distance(&backend, 0.001);
            }
            pipeline.step(&backend, &mut native, None).unwrap();
            backend.synchronize().unwrap();
            let progress = tilelang.step().unwrap();
            assert_eq!(progress.step, frame + 1);
            if frame == 0 {
                assert!(progress.retries >= 1);
            }
            let poses = backend
                .slow_read_vec(native.body_poses().buffer())
                .await
                .unwrap();
            let velocities = backend.slow_read_vec(native.vels().buffer()).await.unwrap();
            compare(&tilelang.body_poses().unwrap(), &poses[..4]);
            compare(&tilelang.velocities().unwrap(), &velocities[..4]);
        }
    });
}
#[test]
#[ignore = "requires the compiler, runtime workspace, and NEXUS_DISABLE_BF=1"]
fn c_configuration_updates_match_production_pipeline() {
    exercise(TileLangBackend::C);
}
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires Metal, the compiler, runtime workspace, and NEXUS_DISABLE_BF=1"]
fn metal_configuration_updates_match_production_pipeline() {
    exercise(TileLangBackend::Metal);
}
