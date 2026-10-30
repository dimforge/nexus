#![cfg(feature = "runtime")]
use khal::backend::{Backend, GpuBackend};
use nexus_rbd3d::{
    Velocity,
    math::Quat,
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
        let mut poses = backend
            .slow_read_vec(native.body_poses().buffer())
            .await
            .unwrap();
        for pose in &mut poses[2..4] {
            pose.translation.y = 3.;
        }
        backend
            .write_buffer(native.body_poses_mut().buffer_mut(), 0, &poses)
            .unwrap();
        let upload = native.read_scene_upload(&backend).await.unwrap();
        let root = PathBuf::from(
            std::env::var_os("NEXUS_TILELANG_RUNTIME_WORK").expect("Set runtime output directory"),
        );
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let workspace = root.join(format!("control-{target:?}-{}-{now}", std::process::id()));
        println!("Runtime control artifacts: {}", workspace.display());
        let options = RuntimeOptions {
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
            if frame == 1 {
                let before = tilelang.body_poses().unwrap();
                let mut invalid = before[2];
                invalid.rotation = Quat::from_xyzw(0., 0., 0., 2.);
                assert!(tilelang.set_body_poses(2, &[invalid]).is_err());
                assert!(tilelang.set_body_poses(usize::MAX, &[]).is_err());
                assert!(tilelang.set_gravity([f32::NAN, 0., 0.]).is_err());
                assert!(
                    tilelang
                        .set_velocities(2, &[Velocity::new(Vec3::splat(f32::NAN), Vec3::ZERO)])
                        .is_err()
                );
                assert_eq!(
                    bytemuck::cast_slice::<_, u8>(&before),
                    bytemuck::cast_slice::<_, u8>(&tilelang.body_poses().unwrap())
                );
                let mut poses = backend
                    .slow_read_vec(native.body_poses().buffer())
                    .await
                    .unwrap();
                for pose in &mut poses[2..4] {
                    pose.translation.y = 0.49;
                }
                tilelang.set_body_poses(2, &poses[2..4]).unwrap();
                backend
                    .write_buffer(native.body_poses_mut().buffer_mut(), 0, &poses)
                    .unwrap();
                let mut velocities = backend.slow_read_vec(native.vels().buffer()).await.unwrap();
                for v in &mut velocities[2..4] {
                    *v = Velocity::new(Vec3::new(0.2, -0.5, 0.1), Vec3::ZERO);
                }
                tilelang.set_velocities(2, &velocities[2..4]).unwrap();
                backend
                    .write_buffer(native.vels_mut().buffer_mut(), 0, &velocities)
                    .unwrap();
                tilelang.set_gravity([0., -3., 0.5]).unwrap();
                native.set_gravity(&backend, [0., -3., 0.5]);
            }
            pipeline.step(&backend, &mut native, None).unwrap();
            backend.synchronize().unwrap();
            let progress = tilelang.step().unwrap();
            assert_eq!(progress.step, frame + 1);
            if frame == 0 {
                assert_eq!(progress.retries, 0);
            }
            if frame == 1 {
                assert!(progress.retries >= 1);
            }
            let poses = backend
                .slow_read_vec(native.body_poses().buffer())
                .await
                .unwrap();
            let velocities = backend.slow_read_vec(native.vels().buffer()).await.unwrap();
            compare(&tilelang.body_poses().unwrap(), &poses[..4]);
            compare(&tilelang.velocities().unwrap(), &velocities[..4]);
            if frame >= 1 {
                assert_eq!(
                    tilelang.read_words("Gravity").unwrap(),
                    [0., -3.0f32, 0.5, 0.].map(f32::to_bits)
                );
            }
        }
    });
}
#[test]
#[ignore = "requires the compiler, runtime workspace, and NEXUS_DISABLE_BF=1"]
fn c_live_updates_match_production_pipeline() {
    exercise(TileLangBackend::C);
}
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires Metal, the compiler, runtime workspace, and NEXUS_DISABLE_BF=1"]
fn metal_live_updates_match_production_pipeline() {
    exercise(TileLangBackend::Metal);
}
