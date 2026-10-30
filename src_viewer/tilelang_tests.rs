use super::*;
use glamx::{Quat, Vec3};
use nexus::prelude::{NexusCapacities, RbdCoupling};
use rapier::prelude::{ColliderBuilder, RigidBodyBuilder, RigidBodyHandle};

fn scene() -> (NexusState, RigidBodyHandle) {
    let mut state = NexusState::new(NexusCapacities::default().rbd_collisions(32));
    state.insert_rigid_body(
        RigidBodyBuilder::fixed()
            .translation(Vec3::new(0.0, -0.5, 0.0))
            .build(),
        ColliderBuilder::cuboid(4.0, 0.5, 4.0).build(),
        RbdCoupling::None,
    );
    let body = state.insert_rigid_body(
        RigidBodyBuilder::dynamic()
            .can_sleep(false)
            .translation(Vec3::new(0.0, 2.0, 0.0))
            .build(),
        ColliderBuilder::ball(0.5).build(),
        RbdCoupling::None,
    );
    (state, body)
}

#[test]
fn comparison_checks_rotation_sign_and_nonfinite_state() {
    let slots = BodySlots::from([((0, 0, 0), 0)]);
    let a = Pose::IDENTITY;
    let mut b = a;
    b.rotation = -a.rotation;
    let velocity = Velocity::default();
    let compare = |b: Pose| Comparison::measure(&slots, &[a], &[b], &[velocity], &[velocity]);
    assert_eq!(compare(b).unwrap().max_angle, 0.0);
    b.rotation = Quat::from_rotation_y(0.03);
    assert!((compare(b).unwrap().max_angle - 0.03).abs() < 1.0e-6);
    b.translation.x = f32::NAN;
    assert!(compare(b).is_err());
}

#[test]
fn unrelated_backend_is_rejected() {
    assert!(options(BackendType::Cpu, 32).is_err());
}

async fn exercise(backend_type: BackendType) {
    let cpu = GpuBackend::Cpu;
    let (mut state, body) = scene();
    let mut ui = TileLangUi {
        compare: true,
        ..Default::default()
    };
    let mut pipeline = NexusPipeline::default();
    let options =
        super::options(backend_type, 32).expect("Set NEXUS_TILELANG_PYTHON to the pinned compiler");
    // SAFETY: these opt-in tests run against the explicitly selected trusted compiler.
    let mut driver = unsafe {
        TileLangDriver::new(&mut state, options, true)
            .await
            .unwrap()
    };
    for _ in 0..5 {
        driver
            .step(&mut state, &mut pipeline, &mut ui)
            .await
            .unwrap();
    }
    let slot = state.rigid_body_gpu_index(0, body).unwrap() as usize;
    assert!(driver.poses[slot].translation.y < 2.0);
    assert!(
        ui.comparison.as_ref().unwrap().max_position < 1.0e-4,
        "{:?}",
        ui.comparison
    );
    assert_eq!(state.steps(), 5);
    // Deliberately perturb only TileLang. Continued divergence proves that later
    // comparison frames do not silently feed Rust results back to TileLang.
    let mut displaced = driver.poses[slot];
    displaced.translation.x += 0.125;
    driver
        .simulation
        .set_body_poses(slot, &[displaced])
        .unwrap();
    for _ in 0..2 {
        driver
            .step(&mut state, &mut pipeline, &mut ui)
            .await
            .unwrap();
    }
    assert!((ui.comparison.as_ref().unwrap().max_position - 0.125).abs() < 1.0e-4);
    // A live caller edit reaches both solvers and does not rebuild the scene.
    state
        .set_rigid_body_pose(
            &cpu,
            0,
            body,
            Pose::from_translation(Vec3::new(0.0, 3.0, 0.0)),
        )
        .unwrap();
    driver
        .step(&mut state, &mut pipeline, &mut ui)
        .await
        .unwrap();
    assert!(ui.comparison.as_ref().unwrap().max_position < 1.0e-4);
    assert_eq!(ui.reconstructions, 0);
    state.set_rbd_gravity(&cpu, [0.0, 0.0, 0.0]);
    state.set_rbd_substeps(&cpu, 2);
    driver
        .step(&mut state, &mut pipeline, &mut ui)
        .await
        .unwrap();
    assert!(ui.comparison.as_ref().unwrap().max_linear_velocity < 1.0e-4);
    // Native full rebuilds can reset bodies from their host world. The viewer
    // marks this as a fresh comparison segment, never a continuous trajectory.
    state.insert_rigid_body(
        RigidBodyBuilder::dynamic()
            .translation(Vec3::new(2.0, 2.0, 0.0))
            .can_sleep(false)
            .build(),
        ColliderBuilder::ball(0.25).build(),
        RbdCoupling::None,
    );
    driver
        .step(&mut state, &mut pipeline, &mut ui)
        .await
        .unwrap();
    assert_eq!(ui.reconstructions, 1);
    assert_eq!(ui.comparison.as_ref().unwrap().bodies, 3);
    assert!(ui.comparison.as_ref().unwrap().max_position < 1.0e-4);

    let (mut state, body) = scene();
    let options = super::options(backend_type, 32).unwrap();
    let mut driver = unsafe {
        TileLangDriver::new(&mut state, options, false)
            .await
            .unwrap()
    };
    let mut ui = TileLangUi::default();
    driver
        .step(&mut state, &mut pipeline, &mut ui)
        .await
        .unwrap();
    assert_eq!(
        state.steps(),
        0,
        "TileLang-only mode must not execute Rust physics"
    );
    let slot = state.rigid_body_gpu_index(0, body).unwrap() as usize;
    assert!(driver.poses[slot].translation.y < 2.0);
    let published = state.read_rigid_body_poses(&cpu).await;
    assert_eq!(
        bytemuck::bytes_of(&published[slot]),
        bytemuck::bytes_of(&driver.poses[slot])
    );
}

#[test]
#[ignore = "requires the pinned TileLang compiler and C toolchain"]
fn c_live_adapter() {
    futures::executor::block_on(exercise(BackendType::TileLangC));
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires the pinned TileLang compiler and an Apple Metal device"]
fn metal_live_adapter() {
    futures::executor::block_on(exercise(BackendType::TileLangMetal));
}
