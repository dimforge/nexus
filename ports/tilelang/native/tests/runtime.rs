#![cfg(feature = "runtime")]
use khal::backend::{Backend, GpuBackend};
use nexus_rbd3d::{
    pipeline::{RbdSceneUpload, RbdState},
    rapier::prelude::*,
    shaders::dynamics::RbdSimParams,
};
use nexus_tilelang_native::{
    runtime::{RuntimeOptions, TileLangBackend, TileLangSimulation},
    upload_cases::{capacities, scene},
};
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

fn upload(overlapping_mesh: bool) -> RbdSceneUpload {
    futures::executor::block_on(async {
        let backend = GpuBackend::Cpu;
        let mesh = SharedShape::trimesh_with_flags(
            vec![
                Vec3::new(-2., 0., -2.),
                Vec3::new(2., 0., -2.),
                Vec3::new(0., 0., 2.),
            ],
            vec![[0, 1, 2]; if overlapping_mesh { 3 } else { 1 }],
            TriMeshFlags::MERGE_DUPLICATE_VERTICES,
        )
        .unwrap();
        let scenes = [scene(0., mesh.clone()), scene(10., mesh)];
        let mb = MultibodyJointSet::new();
        let mut params = RbdSimParams::tgs_soft();
        params.num_solver_iterations = 3;
        params.num_internal_pgs_iterations = 5;
        params.dt = 0.03;
        let refs: Vec<_> = scenes
            .iter()
            .map(|(b, c, j)| (b, c, j, &mb, &params))
            .collect();
        let mut state = RbdState::from_rapier(&backend, &refs, capacities(2));
        state.set_gravity(&backend, [0., -3., 0.5]);
        backend.synchronize().unwrap();
        let mut upload = state.read_scene_upload(&backend).await.unwrap();
        for i in 0..4 {
            upload.body_poses[i].translation.y -= 2.45;
        }
        upload
    })
}
fn floats(path: PathBuf) -> Vec<f32> {
    fs::read(path)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect()
}
fn compare(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && e.is_finite() && (a - e).abs() <= 2e-6 + 2e-5 * e.abs(),
            "{label}[{i}]: {a} != {e}"
        );
    }
}
fn exercise(backend: TileLangBackend, contact_reduction: bool) {
    let python = PathBuf::from(
        std::env::var_os("NEXUS_TILELANG_RUNTIME_PYTHON").expect("Set compiler Python"),
    );
    let expected = PathBuf::from(
        std::env::var_os(if contact_reduction {
            "NEXUS_TILELANG_RUNTIME_EXPECTED"
        } else {
            "NEXUS_TILELANG_RUNTIME_EXPECTED_UNREDUCED"
        })
        .expect("Set the fresh native fixture for the selected contact-reduction mode"),
    );
    let root = PathBuf::from(
        std::env::var_os("NEXUS_TILELANG_RUNTIME_WORK").expect("Set runtime test output directory"),
    );
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let workspace = root.join(format!(
        "{backend:?}-reduced-{contact_reduction}-{}-{now}",
        std::process::id()
    ));
    println!("Native runtime artifacts: {}", workspace.display());
    let upload = upload(!contact_reduction);
    let count =
        upload.batch_indices.bodies_len as usize * upload.batch_indices.num_batches as usize;
    let poses = count * size_of::<nexus_rbd3d::math::Pose>() / 4;
    let velocities = count * size_of::<nexus_rbd3d::Velocity>() / 4;
    let trace = floats(expected.join("Trajectory.expected.bin"));
    let options = RuntimeOptions {
        backend,
        python,
        port_directory: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_owned(),
        workspace,
        initial_pair_capacity: 1,
        max_pair_capacity: 64,
        trace_steps: None,
        contact_reduction,
    };
    let mut runtime = unsafe { TileLangSimulation::from_upload(&upload, options) }.unwrap();
    for frame in 0..8 {
        let result = runtime.step().unwrap();
        assert_eq!(result.step, frame as u64 + 1);
        if frame == 0 {
            assert!(result.retries >= 1);
        }
        let p = runtime.body_poses().unwrap();
        let v = runtime.velocities().unwrap();
        let start = frame * (poses + velocities);
        compare(
            bytemuck::cast_slice(&p),
            &trace[start..start + poses],
            "poses",
        );
        compare(
            bytemuck::cast_slice(&v),
            &trace[start + poses..start + poses + velocities],
            "velocities",
        );
        for (actual, initial) in p[2..6].iter().zip(&upload.body_poses[2..6]) {
            assert_eq!(
                bytemuck::bytes_of(actual),
                bytemuck::bytes_of(initial),
                "Fixed native body changed"
            );
        }
    }
    assert!(runtime.read_words("Trajectory").is_err());
    assert_eq!(runtime.step().unwrap().step, 9);
}
#[test]
#[ignore = "requires the local TileLang compiler and fresh native scene fixtures"]
fn c_runtime_steps_real_native_scene_with_lazy_growth() {
    exercise(TileLangBackend::C, true);
}
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires Metal, the local TileLang compiler and fresh native scene fixtures"]
fn metal_runtime_steps_real_native_scene_with_lazy_growth() {
    exercise(TileLangBackend::Metal, true);
}

#[test]
#[ignore = "requires the local TileLang compiler and fresh unreduced native scene fixtures"]
fn c_runtime_preserves_unreduced_contacts_through_lazy_growth() {
    exercise(TileLangBackend::C, false);
}
#[cfg(feature = "metal")]
#[test]
#[ignore = "requires Metal, the local TileLang compiler and fresh unreduced scene fixtures"]
fn metal_runtime_preserves_unreduced_contacts_through_lazy_growth() {
    exercise(TileLangBackend::Metal, false);
}
