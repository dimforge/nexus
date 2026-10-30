#![cfg(feature = "runtime")]
use khal::backend::{Backend, GpuBackend};
use nexus_rbd3d::{
    pipeline::{RbdPipeline, RbdState},
    rapier::prelude::*,
    shaders::dynamics::{FrictionModel, RbdSimParams},
};
use nexus_tilelang_native::{
    runtime::{RuntimeOptions, TileLangBackend, TileLangSimulation},
    upload_cases::capacities,
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
fn spinning_box(shift: f32) -> (RigidBodySet, ColliderSet, ImpulseJointSet) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let ground = bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0., -0.5, shift)));
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(3., 0.5, 3.).friction(0.7),
        ground,
        &mut bodies,
    );
    let moving = bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vec3::new(0., 0.49, shift))
            .linvel(Vec3::new(0.2, -0.1, 0.03))
            .angvel(Vec3::new(0., 0.8, 0.))
            .can_sleep(false),
    );
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(0.5, 0.5, 0.5).friction(0.7),
        moving,
        &mut bodies,
    );
    for (_, body) in bodies.iter_mut() {
        body.recompute_mass_properties_from_colliders(&colliders);
    }
    (bodies, colliders, ImpulseJointSet::new())
}
fn exercise(target: TileLangBackend) {
    assert!(
        std::env::var("NEXUS_DISABLE_BF").is_ok(),
        "Set NEXUS_DISABLE_BF=1"
    );
    futures::executor::block_on(async {
        let backend = GpuBackend::Cpu;
        let scenes = [spinning_box(0.), spinning_box(10.)];
        let mb = MultibodyJointSet::new();
        let mut params = RbdSimParams::tgs_soft();
        params.dt = 0.01;
        params.friction_model = FrictionModel::Simplified;
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
        let workspace = root.join(format!("friction-{target:?}-{}-{now}", std::process::id()));
        println!("Runtime friction artifacts: {}", workspace.display());
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
        let mut snapshot = None;
        for frame in 0..12 {
            if frame == 3 || frame == 6 {
                let before = serde_json::to_value(tilelang.checkpoint().unwrap()).unwrap();
                let model = if frame == 3 {
                    FrictionModel::Coulomb
                } else {
                    FrictionModel::Simplified
                };
                tilelang.set_friction_model(model).unwrap();
                native.set_friction_model(&backend, model);
                assert_eq!(tilelang.friction_model(), model);
                let after = serde_json::to_value(tilelang.checkpoint().unwrap()).unwrap();
                assert_eq!(before["frame"]["steps"], after["frame"]["steps"]);
                assert_ne!(before["signature"], after["signature"]);
                let old = before["frame"]["regions"].as_object().unwrap();
                let new = after["frame"]["regions"].as_object().unwrap();
                for (name, words) in old {
                    if name != "OldCounts" && name != "OldTiles" {
                        assert_eq!(words, &new[name], "Preserved region {name}");
                    }
                }
                assert!(
                    new["OldCounts"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .all(|v| v.as_u64() == Some(0))
                );
                assert_ne!(
                    old["OldTiles"].as_array().unwrap().len(),
                    new["OldTiles"].as_array().unwrap().len()
                );
                tilelang.set_friction_model(model).unwrap();
                assert_eq!(
                    after,
                    serde_json::to_value(tilelang.checkpoint().unwrap()).unwrap()
                );
                if frame == 6 {
                    assert!(
                        unsafe { tilelang.restore_checkpoint(snapshot.as_ref().unwrap()) }.is_err()
                    );
                    assert_eq!(
                        after,
                        serde_json::to_value(tilelang.checkpoint().unwrap()).unwrap()
                    );
                }
            }
            if frame == 5 {
                snapshot = Some(tilelang.checkpoint().unwrap());
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
fn c_twist_and_model_switches_match_production_pipeline() {
    exercise(TileLangBackend::C);
}
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires Metal, the compiler, runtime workspace, and NEXUS_DISABLE_BF=1"]
fn metal_twist_and_model_switches_match_production_pipeline() {
    exercise(TileLangBackend::Metal);
}
