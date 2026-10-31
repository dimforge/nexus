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

async fn exercise_cache(backend_type: BackendType) {
    let (mut state, body) = scene();
    let mut opts = super::options(backend_type, 32).unwrap();
    opts.kernel_cache = Some(opts.workspace.with_extension("kernel-cache"));
    let cache = opts.kernel_cache.clone();
    let workspace = opts.workspace.clone();
    let mut ui = TileLangUi {
        compare: true,
        ..Default::default()
    };
    let mut pipeline = NexusPipeline::default();
    // SAFETY: explicitly selected trusted local compiler, as in exercise().
    let mut driver = unsafe {
        TileLangDriver::start(&mut state, opts, None, &mut ui)
            .await
            .unwrap()
    };
    eprintln!(
        "{backend_type:?} cold: {:.3}s {:?}",
        ui.preparation_ms / 1000.0,
        ui.cache
    );
    let cold = ui.cache.as_ref().unwrap();
    assert!(
        cold.lowering_misses > 0 && cold.binary_misses > 0,
        "{cold:?}"
    );
    let initial_poses = bytemuck::cast_slice::<Pose, u8>(&driver.poses).to_vec();
    let initial = driver.simulation.checkpoint().unwrap();
    for _ in 0..3 {
        driver
            .step(&mut state, &mut pipeline, &mut ui)
            .await
            .unwrap();
    }
    let expected = bytemuck::cast_slice::<Pose, u8>(&driver.poses).to_vec();
    assert_ne!(initial_poses, expected);

    let (mut restarted, _) = scene();
    let mut opts = super::options(backend_type, 32).unwrap();
    opts.kernel_cache = cache.clone();
    let unused_workspace = opts.workspace.clone();
    ui.clear_run();
    driver = unsafe {
        TileLangDriver::start(&mut restarted, opts, Some(driver), &mut ui)
            .await
            .unwrap()
    };
    eprintln!(
        "{backend_type:?} restart: {:.3}s",
        ui.preparation_ms / 1000.0
    );
    assert!(ui.reused_runtime);
    assert_eq!(ui.workspace.as_ref(), Some(&workspace));
    assert!(
        !unused_workspace.exists(),
        "Restart must not start the compiler/exporter"
    );
    assert_eq!(driver.simulation.checkpoint().unwrap().step(), 0);
    assert_eq!(
        bytemuck::cast_slice::<Pose, u8>(&driver.poses),
        initial_poses
    );
    // Equality includes all semantic state, contact caches and frame count.
    assert_eq!(driver.simulation.checkpoint().unwrap(), initial);
    for _ in 0..3 {
        driver
            .step(&mut restarted, &mut pipeline, &mut ui)
            .await
            .unwrap();
    }
    assert_eq!(bytemuck::cast_slice::<Pose, u8>(&driver.poses), expected);
    assert!(ui.comparison.as_ref().unwrap().max_position < 1.0e-4);

    // A -> B -> A: shape-compatible kernels hit, but initial state is never reused.
    for height in [3.0, 2.0] {
        let (mut fresh, body) = scene();
        fresh.finalize(&GpuBackend::Cpu).await.unwrap();
        fresh
            .set_rigid_body_pose(
                &GpuBackend::Cpu,
                0,
                body,
                Pose::from_translation(Vec3::new(0.0, height, 0.0)),
            )
            .unwrap();
        let mut opts = super::options(backend_type, 32).unwrap();
        opts.kernel_cache = cache.clone();
        ui.clear_run();
        driver = unsafe {
            TileLangDriver::start(&mut fresh, opts, Some(driver), &mut ui)
                .await
                .unwrap()
        };
        eprintln!(
            "{backend_type:?} switched height {height}: {:.3}s {:?}",
            ui.preparation_ms / 1000.0,
            ui.cache
        );
        assert!(!ui.reused_runtime);
        let stats = ui.cache.as_ref().unwrap();
        assert_eq!(
            (stats.lowering_misses, stats.binary_misses),
            (0, 0),
            "{stats:?}"
        );
        assert!(
            stats.lowering_hits > 0 && stats.binary_hits > 0,
            "{stats:?}"
        );
        let slot = fresh.rigid_body_gpu_index(0, body).unwrap() as usize;
        assert_eq!(driver.poses[slot].translation.y, height);
        for _ in 0..3 {
            driver
                .step(&mut fresh, &mut pipeline, &mut ui)
                .await
                .unwrap();
        }
        assert!(ui.comparison.as_ref().unwrap().max_position < 1.0e-4);
        if height == 2.0 {
            assert_eq!(bytemuck::cast_slice::<Pose, u8>(&driver.poses), expected);
        }
    }
    // A new runtime with the cache disabled is the numerical comparison baseline.
    let (mut plain, _) = scene();
    let mut opts = super::options(backend_type, 32).unwrap();
    opts.kernel_cache = None;
    ui.clear_run();
    let mut plain_driver = unsafe {
        TileLangDriver::start(&mut plain, opts, None, &mut ui)
            .await
            .unwrap()
    };
    assert!(ui.cache.is_none());
    for _ in 0..3 {
        plain_driver
            .step(&mut plain, &mut pipeline, &mut ui)
            .await
            .unwrap();
    }
    assert_eq!(
        bytemuck::cast_slice::<Pose, u8>(&plain_driver.poses),
        expected
    );
    // Exercise collision preparation, contact matching, warmstarts and solving
    // through both native-library paths, including reset after nonempty contacts.
    let slot = state.rigid_body_gpu_index(0, body).unwrap() as usize;
    for simulation in [&mut driver.simulation, &mut plain_driver.simulation] {
        simulation.reset().unwrap();
        simulation
            .set_body_poses(slot, &[Pose::from_translation(Vec3::new(0.0, 0.49, 0.0))])
            .unwrap();
    }
    for _ in 0..8 {
        driver.simulation.step().unwrap();
        plain_driver.simulation.step().unwrap();
        assert_eq!(
            bytemuck::cast_slice::<Pose, u8>(&driver.simulation.body_poses().unwrap()),
            bytemuck::cast_slice::<Pose, u8>(&plain_driver.simulation.body_poses().unwrap())
        );
        assert_eq!(
            bytemuck::cast_slice::<Velocity, u8>(&driver.simulation.velocities().unwrap()),
            bytemuck::cast_slice::<Velocity, u8>(&plain_driver.simulation.velocities().unwrap())
        );
    }
    assert!(driver.simulation.read_words("OldCounts").unwrap()[0] > 0);
    driver.simulation.reset().unwrap();
    assert_eq!(driver.simulation.read_words("OldCounts").unwrap()[0], 0);
}

#[test]
#[ignore = "requires pinned TileLang and the C compiler"]
fn c_kernel_cache_and_restart() {
    futures::executor::block_on(exercise_cache(BackendType::TileLangC));
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires pinned TileLang, Metal device and offline Metal compiler"]
fn metal_kernel_cache_and_restart() {
    futures::executor::block_on(exercise_cache(BackendType::TileLangMetal));
}
