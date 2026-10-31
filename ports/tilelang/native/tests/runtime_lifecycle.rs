#![cfg(feature = "runtime")]
use khal::backend::{Backend, GpuBackend};
use nexus_rbd3d::{
    pipeline::{RbdPipeline, RbdState},
    rapier::prelude::*,
    shaders::dynamics::RbdSimParams,
};
use nexus_tilelang_native::{
    runtime::{RuntimeOptions, TileLangBackend, TileLangSimulation},
    upload_cases::capacities,
};
use std::{
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

fn scene(phase: usize, z: f32) -> (RigidBodySet, ColliderSet, ImpulseJointSet) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut joints = ImpulseJointSet::new();
    let ground = bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0., -0.5, z)));
    colliders.insert_with_parent(ColliderBuilder::cuboid(4., 0.5, 2.), ground, &mut bodies);
    if phase > 0 {
        let b = bodies.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vec3::new(1., 3., z))
                .can_sleep(false),
        );
        colliders.insert_with_parent(ColliderBuilder::cuboid(0.3, 0.3, 0.3), b, &mut bodies);
        colliders.insert_with_parent(
            ColliderBuilder::ball(0.2).translation(Vec3::new(0.4, 0., 0.)),
            b,
            &mut bodies,
        );
        joints.insert(
            ground,
            b,
            PrismaticJointBuilder::new(Vec3::X)
                .local_anchor1(Vec3::new(1., 3.5, 0.))
                .limits([-1., 1.])
                .motor_velocity(0.2, 1.),
            true,
        );
    }
    if phase < 2 {
        let b = bodies.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vec3::new(0., 0.49, z))
                .linvel(Vec3::new(0.2, -0.1, 0.1))
                .can_sleep(false),
        );
        colliders.insert_with_parent(ColliderBuilder::ball(0.5), b, &mut bodies);
    }
    for (_, b) in bodies.iter_mut() {
        b.recompute_mass_properties_from_colliders(&colliders);
    }
    (bodies, colliders, joints)
}
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
        let mb = MultibodyJointSet::new();
        let mut params = RbdSimParams::tgs_soft();
        params.dt = 0.03;
        params.num_solver_iterations = 3;
        params.num_internal_pgs_iterations = 5;
        let mut native: Option<RbdState> = None;
        let mut tilelang: Option<TileLangSimulation> = None;
        let pipeline = RbdPipeline::new(&backend).unwrap();
        for phase in 0..3 {
            let scenes = [scene(phase, 0.), scene(phase, 10.)];
            let refs: Vec<_> = scenes
                .iter()
                .map(|(b, c, j)| (b, c, j, &mb, &params))
                .collect();
            let mut caps = capacities(2);
            caps.collisions_capacity = 64;
            let mut replacement = RbdState::from_rapier(&backend, &refs, caps);
            replacement.set_deterministic(&backend, true);
            replacement.set_gravity(&backend, [0., -3., 0.5]);
            let upload = replacement.read_scene_upload(&backend).await.unwrap();
            let count = upload.batch_indices.bodies_len as usize * 2;
            if let (Some(runtime), Some(old)) = (&mut tilelang, &mut native) {
                let old_poses = backend
                    .slow_read_vec(old.body_poses().buffer())
                    .await
                    .unwrap();
                let old_velocities = backend.slow_read_vec(old.vels().buffer()).await.unwrap();
                let exported = runtime.read_scene_upload().unwrap();
                let active = exported.batch_indices.bodies_len as usize * 2;
                compare(&exported.body_poses[..active], &old_poses[..active]);
                compare(&exported.velocities[..active], &old_velocities[..active]);
                assert_eq!(exported.gravity.to_array(), [0., -3., 0.5, 0.]);
                let before = serde_json::to_value(runtime.checkpoint().unwrap()).unwrap();
                assert!(runtime.replace_scene(&upload, &[(0, 0), (1, 0)]).is_err());
                assert!(runtime.replace_scene(&upload, &[(0, usize::MAX)]).is_err());
                assert_eq!(
                    before,
                    serde_json::to_value(runtime.checkpoint().unwrap()).unwrap()
                );
                let remap = if phase == 1 {
                    vec![(0, 0), (1, 1), (2, 4), (3, 5)]
                } else {
                    vec![(0, 0), (1, 1), (2, 2), (3, 3)]
                };
                runtime.replace_scene(&upload, &remap).unwrap();
                assert_eq!(runtime.checkpoint().unwrap().step(), 0);
                let mut poses = upload.body_poses.clone();
                let mut velocities = upload.velocities.clone();
                for &(from, to) in &remap {
                    poses[to] = old_poses[from];
                    velocities[to] = old_velocities[from];
                }
                backend
                    .write_buffer(replacement.body_poses_mut().buffer_mut(), 0, &poses)
                    .unwrap();
                backend
                    .write_buffer(replacement.vels_mut().buffer_mut(), 0, &velocities)
                    .unwrap();
                let actual = runtime.body_poses().unwrap();
                for &(from, to) in &remap {
                    assert_eq!(
                        bytemuck::bytes_of(&actual[to]),
                        bytemuck::bytes_of(&exported.body_poses[from])
                    );
                }
                compare(&actual, &poses[..count]);
            } else {
                let root = PathBuf::from(
                    std::env::var_os("NEXUS_TILELANG_RUNTIME_WORK")
                        .expect("Set runtime output directory"),
                );
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                let workspace =
                    root.join(format!("lifecycle-{target:?}-{}-{now}", std::process::id()));
                println!("Runtime lifecycle artifacts: {}", workspace.display());
                let options = RuntimeOptions {
                    kernel_cache: None,
                    backend: target,
                    python: PathBuf::from(
                        std::env::var_os("NEXUS_TILELANG_RUNTIME_PYTHON")
                            .expect("Set compiler Python"),
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
                tilelang =
                    Some(unsafe { TileLangSimulation::from_upload(&upload, options) }.unwrap());
            }
            native = Some(replacement);
            let runtime = tilelang.as_mut().unwrap();
            let state = native.as_mut().unwrap();
            for frame in 0..4 {
                pipeline.step(&backend, state, None).unwrap();
                backend.synchronize().unwrap();
                assert_eq!(runtime.step().unwrap().step, frame + 1);
                let poses = backend
                    .slow_read_vec(state.body_poses().buffer())
                    .await
                    .unwrap();
                let velocities = backend.slow_read_vec(state.vels().buffer()).await.unwrap();
                compare(&runtime.body_poses().unwrap(), &poses[..count]);
                compare(&runtime.velocities().unwrap(), &velocities[..count]);
            }
        }
    });
}
#[test]
#[ignore = "requires compiler, runtime workspace and NEXUS_DISABLE_BF=1"]
fn c_reconstructs_inserted_and_removed_scenes() {
    exercise(TileLangBackend::C);
}
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires Metal, compiler, runtime workspace and NEXUS_DISABLE_BF=1"]
fn metal_reconstructs_inserted_and_removed_scenes() {
    exercise(TileLangBackend::Metal);
}
