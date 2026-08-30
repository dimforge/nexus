use crate::math::Pose;
use crate::pipeline::{RbdCapacities, RbdPipeline, RbdResizePolicy, RbdState};
use crate::rapier::prelude::*;
use crate::shaders::dynamics::RbdSimParams;
use khal::backend::{Backend, GpuBackend};

async fn test_backend() -> GpuBackend {
    #[cfg(feature = "metal")]
    {
        GpuBackend::Metal(khal::backend::metal::Metal::new().unwrap())
    }
    #[cfg(not(feature = "metal"))]
    {
        GpuBackend::WebGpu(khal::backend::WebGpu::default().await.unwrap())
    }
}

fn build_env(num_stacks: usize, stack_height: usize) -> (RigidBodySet, ColliderSet) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();

    let ground = bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0.0, -0.5, 0.0)));
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(50.0, 0.5, 50.0),
        ground,
        &mut bodies,
    );

    for s in 0..num_stacks {
        let x = (s % 8) as f32 * 2.0;
        let z = (s / 8) as f32 * 2.0;
        for i in 0..stack_height {
            let y = 0.25 + i as f32 * 0.45;
            let handle = bodies.insert(RigidBodyBuilder::dynamic().translation(Vec3::new(x, y, z)));
            colliders.insert_with_parent(
                ColliderBuilder::cuboid(0.2, 0.2, 0.2),
                handle,
                &mut bodies,
            );
        }
    }

    (bodies, colliders)
}

async fn run_case(num_envs: u32, num_stacks: usize, stack_height: usize, collisions_capacity: u32) {
    let backend = test_backend().await;

    let envs: Vec<_> = (0..num_envs)
        .map(|_| build_env(num_stacks, stack_height))
        .collect();
    let joints = ImpulseJointSet::new();
    let mb_joints = MultibodyJointSet::new();
    let params = RbdSimParams::tgs_soft();
    let refs: Vec<_> = envs
        .iter()
        .map(|(b, c)| (b, c, &joints, &mb_joints, &params))
        .collect();

    let capacities = RbdCapacities {
        batches: num_envs,
        collisions_capacity,
        ..Default::default()
    };
    let mut state = RbdState::from_rapier(&backend, &refs, capacities);
    let pipeline = RbdPipeline::new(&backend).unwrap();

    for _ in 0..250 {
        pipeline.step(&backend, &mut state, None).unwrap();
        pipeline.auto_resize_buffers(&backend, &mut state).unwrap();
    }
    backend.synchronize().unwrap();

    let poses: Vec<Pose> = backend
        .slow_read_vec(state.body_poses().buffer())
        .await
        .unwrap();
    let nb = num_envs as usize;
    let boxes_per_env = num_stacks * stack_height;
    for env in 0..num_envs as usize {
        for b in 0..boxes_per_env {
            let pose = poses[(1 + b) * nb + env];
            assert!(
                pose.translation.is_finite(),
                "env {env} box {b}: non-finite pose {:?}",
                pose.translation
            );
            let level = b % stack_height;
            let expected_y = 0.2 + level as f32 * 0.4;
            let y = pose.translation.y;
            assert!(
                (y - expected_y).abs() < 0.1,
                "env {env} box {b}: y = {y}, expected ~{expected_y}"
            );
            if env > 0 {
                let ref_pose = poses[(1 + b) * nb];
                let d = (pose.translation - ref_pose.translation).length();
                assert!(d < 5.0e-2, "env {env} box {b}: diverged from env 0 by {d}");
            }
        }
    }
    println!(
        "OK: envs={num_envs} stacks={num_stacks} height={stack_height} cap={collisions_capacity}"
    );
}
#[futures_test::test]
#[serial_test::serial]
#[ignore]
async fn test_stacks_1_tiny() {
    run_case(1, 1, 4, 64).await;
}
#[futures_test::test]
#[serial_test::serial]
#[ignore]
async fn test_stacks_2_batched() {
    run_case(64, 1, 4, 64).await;
}
#[futures_test::test]
#[serial_test::serial]
#[ignore]
async fn test_stacks_3_lbvh() {
    run_case(1, 32, 4, 1024).await;
}
#[futures_test::test]
#[serial_test::serial]
#[ignore]
async fn test_stacks_4_lbvh_batched() {
    run_case(4, 32, 4, 1024).await;
}
#[futures_test::test]
#[serial_test::serial]
#[ignore]
async fn test_stacks_5_overflow_resize() {
    run_case(64, 1, 4, 8).await;
}
#[futures_test::test]
#[serial_test::serial]
#[ignore]
async fn test_stacks_8_impulse_joint() {
    let backend = test_backend().await;

    let num_envs = 4u32;
    let build = || {
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let mut joints = ImpulseJointSet::new();
        let anchor = bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0.0, 2.0, 0.0)));
        colliders.insert_with_parent(ColliderBuilder::ball(0.1), anchor, &mut bodies);
        let bob = bodies.insert(RigidBodyBuilder::dynamic().translation(Vec3::new(1.0, 2.0, 0.0)));
        colliders.insert_with_parent(ColliderBuilder::ball(0.1), bob, &mut bodies);
        let joint = RevoluteJointBuilder::new(Vec3::Z)
            .local_anchor1(Vec3::ZERO)
            .local_anchor2(Vec3::new(-1.0, 0.0, 0.0));
        joints.insert(anchor, bob, joint, true);
        (bodies, colliders, joints)
    };
    let envs: Vec<_> = (0..num_envs).map(|_| build()).collect();
    let mb_joints = MultibodyJointSet::new();
    let params = RbdSimParams::tgs_soft();
    let refs: Vec<_> = envs
        .iter()
        .map(|(b, c, j)| (b, c, j, &mb_joints, &params))
        .collect();

    let capacities = RbdCapacities {
        batches: num_envs,
        collisions_capacity: 16,
        ..Default::default()
    };
    let mut state = RbdState::from_rapier(&backend, &refs, capacities);
    let pipeline = RbdPipeline::new(&backend).unwrap();
    for _ in 0..400 {
        pipeline.step(&backend, &mut state, None).unwrap();
    }
    backend.synchronize().unwrap();

    let poses: Vec<Pose> = backend
        .slow_read_vec(state.body_poses().buffer())
        .await
        .unwrap();
    let nb = num_envs as usize;
    for env in 0..nb {
        let p = poses[nb + env].translation;
        assert!(p.is_finite(), "env {env}: non-finite bob pose {p:?}");
        let d = (p - Vec3::new(0.0, 2.0, 0.0)).length();
        assert!(
            (0.9..1.1).contains(&d),
            "env {env}: bob at distance {d} from anchor, expected ~1"
        );
    }
    println!("OK: impulse-joint pendulum holds, envs={num_envs}");
}
#[futures_test::test]
#[serial_test::serial]
#[ignore]
async fn test_stacks_9_env_reset() {
    let backend = test_backend().await;

    let num_envs = 4u32;
    let envs: Vec<_> = (0..num_envs).map(|_| build_env(1, 4)).collect();
    let joints = ImpulseJointSet::new();
    let mb_joints = MultibodyJointSet::new();
    let params = RbdSimParams::tgs_soft();
    let refs: Vec<_> = envs
        .iter()
        .map(|(b, c)| (b, c, &joints, &mb_joints, &params))
        .collect();

    let capacities = RbdCapacities {
        batches: num_envs,
        collisions_capacity: 32,
        ..Default::default()
    };
    let mut state = RbdState::from_rapier(&backend, &refs, capacities);
    let pipeline = RbdPipeline::new(&backend).unwrap();

    for _ in 0..50 {
        pipeline.step(&backend, &mut state, None).unwrap();
    }
    backend.synchronize().unwrap();
    let snap = state.snapshot(&backend).await;
    state.publish_reset_templates(&backend, &[&snap]);

    for _ in 0..50 {
        pipeline.step(&backend, &mut state, None).unwrap();
    }
    state.reset_envs_from_templates(&backend, &[(1, 0)], &[crate::math::Vector::ZERO], &[]);
    backend.synchronize().unwrap();

    let poses: Vec<Pose> = backend
        .slow_read_vec(state.body_poses().buffer())
        .await
        .unwrap();
    let nb = num_envs as usize;
    for b in 0..5 {
        let restored = poses[b * nb + 1].translation;
        let template = snap_pose(&snap, b);
        let d = (restored - template).length();
        assert!(
            d < 1.0e-6,
            "body {b}: env 1 not restored to the template (off by {d})"
        );
    }
    println!("OK: env reset restores the template, envs={num_envs}");
}
#[cfg(feature = "dim3")]
fn snap_pose(snap: &crate::pipeline::RbdSnapshot, b: usize) -> Vec3 {
    snap.debug_body_pose(b).translation
}
#[futures_test::test]
#[serial_test::serial]
#[ignore]
async fn test_stacks_7_pfm_shapes() {
    let backend = test_backend().await;

    let num_envs = 4u32;
    let build = || {
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let ground =
            bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0.0, -0.5, 0.0)));
        colliders.insert_with_parent(
            ColliderBuilder::cuboid(20.0, 0.5, 20.0),
            ground,
            &mut bodies,
        );
        for i in 0..3 {
            let b = bodies.insert(RigidBodyBuilder::dynamic().translation(Vec3::new(
                i as f32 * 1.5,
                0.6,
                0.0,
            )));
            colliders.insert_with_parent(ColliderBuilder::capsule_y(0.15, 0.1), b, &mut bodies);
        }
        for i in 0..2 {
            let b = bodies.insert(RigidBodyBuilder::dynamic().translation(Vec3::new(
                i as f32 * 1.5 + 0.5,
                0.6,
                1.5,
            )));
            colliders.insert_with_parent(ColliderBuilder::ball(0.2), b, &mut bodies);
        }
        (bodies, colliders)
    };
    let envs: Vec<_> = (0..num_envs).map(|_| build()).collect();
    let joints = ImpulseJointSet::new();
    let mb_joints = MultibodyJointSet::new();
    let params = RbdSimParams::tgs_soft();
    let refs: Vec<_> = envs
        .iter()
        .map(|(b, c)| (b, c, &joints, &mb_joints, &params))
        .collect();

    let capacities = RbdCapacities {
        batches: num_envs,
        collisions_capacity: 32,
        ..Default::default()
    };
    let mut state = RbdState::from_rapier(&backend, &refs, capacities);
    let pipeline = RbdPipeline::new(&backend).unwrap();
    for _ in 0..250 {
        pipeline.step(&backend, &mut state, None).unwrap();
        pipeline.auto_resize_buffers(&backend, &mut state).unwrap();
    }
    backend.synchronize().unwrap();

    let poses: Vec<Pose> = backend
        .slow_read_vec(state.body_poses().buffer())
        .await
        .unwrap();
    let nb = num_envs as usize;
    for env in 0..num_envs as usize {
        for b in 0..5 {
            let pose = poses[(1 + b) * nb + env];
            assert!(pose.translation.is_finite());
            let y = pose.translation.y;
            assert!(
                (0.05..0.4).contains(&y),
                "env {env} body {b}: y = {y}, expected resting on the ground"
            );
        }
    }
    println!("OK: pfm shapes rest, envs={num_envs}");
}

fn build_mb_env(ball_colliders: bool) -> (RigidBodySet, ColliderSet, MultibodyJointSet) {
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    let mut mb_joints = MultibodyJointSet::new();

    let ground = bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0.0, -0.5, 0.0)));
    colliders.insert_with_parent(
        ColliderBuilder::cuboid(20.0, 0.5, 20.0),
        ground,
        &mut bodies,
    );

    let mut prev = None;
    for i in 0..3 {
        let x = i as f32 * 0.5;
        let link = bodies.insert(RigidBodyBuilder::dynamic().translation(Vec3::new(x, 1.0, 0.0)));
        let shape = if ball_colliders {
            ColliderBuilder::ball(0.1)
        } else {
            ColliderBuilder::cuboid(0.2, 0.1, 0.1)
        };
        colliders.insert_with_parent(shape, link, &mut bodies);
        if let Some(prev) = prev {
            let joint = RevoluteJointBuilder::new(Vec3::Z)
                .local_anchor1(Vec3::new(0.25, 0.0, 0.0))
                .local_anchor2(Vec3::new(-0.25, 0.0, 0.0));
            mb_joints.insert(prev, link, joint, true);
        }
        prev = Some(link);
    }

    (bodies, colliders, mb_joints)
}
#[futures_test::test]
#[serial_test::serial]
#[ignore]
async fn test_stacks_6_multibody() {
    let backend = test_backend().await;

    let num_envs = 4u32;
    let envs: Vec<_> = (0..num_envs).map(|_| build_mb_env(false)).collect();
    let joints = ImpulseJointSet::new();
    let params = RbdSimParams::tgs_soft();
    let refs: Vec<_> = envs
        .iter()
        .map(|(b, c, mb)| (b, c, &joints, mb, &params))
        .collect();

    let mut capacities = RbdCapacities {
        batches: num_envs,
        collisions_capacity: 32,
        ..Default::default()
    };
    if std::env::var("NEXUS_TEST_FIXED").is_ok() {
        capacities.collisions_resize_policy = RbdResizePolicy::Fixed;
        capacities.solver_colors_resize_policy = RbdResizePolicy::Fixed;
    }
    let mut state = RbdState::from_rapier(&backend, &refs, capacities);
    let pipeline = RbdPipeline::new(&backend).unwrap();

    let debug = std::env::var("NEXUS_TEST_DEBUG").is_ok();
    for step in 0..300 {
        pipeline.step(&backend, &mut state, None).unwrap();
        pipeline.auto_resize_buffers(&backend, &mut state).unwrap();
        if debug && (20..40).contains(&step) {
            backend.synchronize().unwrap();
            let (layout, demand) = state.multibodies().debug_cons_layout(&backend).await;
            let poses: Vec<Pose> = backend
                .slow_read_vec(state.body_poses().buffer())
                .await
                .unwrap();
            let finite = poses.iter().all(|p| p.translation.is_finite());
            println!(
                "step {step}: cap={} demand={demand} layout={layout:?} finite={finite}",
                state.multibodies().contact_constraints_capacity()
            );
            if demand > 0 {
                let cons: Vec<crate::shaders::dynamics::MultibodyContactConstraint> = backend
                    .slow_read_vec(state.multibodies().contact_constraints().buffer())
                    .await
                    .unwrap_or_default();
                for env in [0usize, 2] {
                    let (st, ct, _, _) = layout[env];
                    for k in 0..ct.min(2) {
                        let c = &cons[(st + k) as usize];
                        println!(
                            "  env{env} slot{k}: kind={} imp={} inv_lhs={} rhs={}",
                            c.kind, c.impulse, c.inv_lhs, c.rhs
                        );
                    }
                }
            }
            if !finite {
                for (i, p) in poses.iter().enumerate() {
                    if !p.translation.is_finite() {
                        println!("  non-finite body slot {i}");
                    }
                }
                break;
            }
        }
    }
    backend.synchronize().unwrap();

    let poses: Vec<Pose> = backend
        .slow_read_vec(state.body_poses().buffer())
        .await
        .unwrap();
    let nb = num_envs as usize;
    for env in 0..num_envs as usize {
        for l in 0..3 {
            let pose = poses[(1 + l) * nb + env];
            assert!(
                pose.translation.is_finite(),
                "env {env} link {l}: non-finite pose {:?}",
                pose.translation
            );
            let y = pose.translation.y;
            assert!(
                (0.0..0.5).contains(&y),
                "env {env} link {l}: y = {y}, expected resting near 0.1"
            );
        }
    }
    println!("OK: multibody chain rests, envs={num_envs}");
}
#[futures_test::test]
#[serial_test::serial]
#[ignore]
async fn test_stacks_11_mb_point_contacts() {
    let backend = test_backend().await;

    let num_envs = 4u32;
    let envs: Vec<_> = (0..num_envs).map(|_| build_mb_env(true)).collect();
    let joints = ImpulseJointSet::new();
    let params = RbdSimParams::tgs_soft();
    let refs: Vec<_> = envs
        .iter()
        .map(|(b, c, mb)| (b, c, &joints, mb, &params))
        .collect();

    let capacities = RbdCapacities {
        batches: num_envs,
        collisions_capacity: 32,
        ..Default::default()
    };
    let mut state = RbdState::from_rapier(&backend, &refs, capacities);
    let pipeline = RbdPipeline::new(&backend).unwrap();
    for _ in 0..300 {
        pipeline.step(&backend, &mut state, None).unwrap();
        pipeline.auto_resize_buffers(&backend, &mut state).unwrap();
    }
    backend.synchronize().unwrap();

    let poses: Vec<Pose> = backend
        .slow_read_vec(state.body_poses().buffer())
        .await
        .unwrap();
    let nb = num_envs as usize;
    for env in 0..nb {
        for l in 0..3 {
            let pose = poses[(1 + l) * nb + env];
            assert!(
                pose.translation.is_finite(),
                "env {env} link {l}: non-finite pose {:?}",
                pose.translation
            );
            let y = pose.translation.y;
            assert!(
                (0.05..0.3).contains(&y),
                "env {env} link {l}: y = {y}, expected resting near 0.1"
            );
        }
    }
    println!("OK: 1-point-manifold multibody contacts rest, envs={num_envs}");
}
#[futures_test::test]
#[serial_test::serial]
#[ignore]
async fn test_stacks_10_mb_impulse_joint() {
    let backend = test_backend().await;

    let num_envs = 4u32;
    let anchor_y = |e: u32| 2.0 + 0.25 * e as f32;
    let build = |e: u32| {
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let mut joints = ImpulseJointSet::new();
        let mut mb_joints = MultibodyJointSet::new();
        let y = anchor_y(e);

        let anchor = bodies.insert(RigidBodyBuilder::fixed().translation(Vec3::new(0.0, y, 0.0)));
        colliders.insert_with_parent(ColliderBuilder::ball(0.05), anchor, &mut bodies);
        let mut links = Vec::new();
        let mut prev = anchor;
        for i in 0..2 {
            let x = 0.5 + i as f32 * 0.5;
            let link = bodies.insert(RigidBodyBuilder::dynamic().translation(Vec3::new(x, y, 0.0)));
            colliders.insert_with_parent(ColliderBuilder::ball(0.05), link, &mut bodies);
            let joint = RevoluteJointBuilder::new(Vec3::Z)
                .local_anchor1(Vec3::new(if i == 0 { 0.0 } else { 0.25 }, 0.0, 0.0))
                .local_anchor2(Vec3::new(if i == 0 { -0.5 } else { -0.25 }, 0.0, 0.0));
            mb_joints.insert(prev, link, joint, true);
            links.push(link);
            prev = link;
        }
        for (i, &link) in links.iter().enumerate() {
            let x = 0.5 + i as f32 * 0.5;
            let bob =
                bodies.insert(RigidBodyBuilder::dynamic().translation(Vec3::new(x, y - 0.4, 0.0)));
            colliders.insert_with_parent(ColliderBuilder::ball(0.05), bob, &mut bodies);
            let joint = RevoluteJointBuilder::new(Vec3::Z)
                .local_anchor1(Vec3::ZERO)
                .local_anchor2(Vec3::new(0.0, 0.4, 0.0));
            joints.insert(link, bob, joint, true);
        }

        (bodies, colliders, joints, mb_joints)
    };
    let envs: Vec<_> = (0..num_envs).map(build).collect();
    let params = RbdSimParams::tgs_soft();
    let refs: Vec<_> = envs
        .iter()
        .map(|(b, c, j, mb)| (b, c, j, mb, &params))
        .collect();

    let capacities = RbdCapacities {
        batches: num_envs,
        collisions_capacity: 16,
        ..Default::default()
    };
    let mut state = RbdState::from_rapier(&backend, &refs, capacities);
    assert!(
        state.multibodies().mb_imp_joint_num_colors() >= 2,
        "expected >= 2 impulse-joint colors (both joints touch the same multibody)"
    );
    let pipeline = RbdPipeline::new(&backend).unwrap();
    for _ in 0..400 {
        pipeline.step(&backend, &mut state, None).unwrap();
    }
    backend.synchronize().unwrap();

    let poses: Vec<Pose> = backend
        .slow_read_vec(state.body_poses().buffer())
        .await
        .unwrap();
    let nb = num_envs as usize;
    for env in 0..nb {
        let at = |slot: usize| poses[slot * nb + env];
        let ay = at(0).translation.y;
        assert!(
            (ay - anchor_y(env as u32)).abs() < 1.0e-4,
            "env {env}: anchor y = {ay}, expected {}",
            anchor_y(env as u32)
        );
        for slot in 0..5 {
            let p = at(slot).translation;
            assert!(
                p.is_finite(),
                "env {env} slot {slot}: non-finite pose {p:?}"
            );
        }
        for (link_slot, bob_slot) in [(1usize, 3usize), (2, 4)] {
            let link = at(link_slot);
            let bob = at(bob_slot);
            let bob_anchor = bob.translation + bob.rotation * Vec3::new(0.0, 0.4, 0.0);
            let err = (bob_anchor - link.translation).length();
            assert!(
                err < 0.05,
                "env {env}: bob {bob_slot} anchor drifts {err} from link {link_slot}"
            );
        }
    }
    println!("OK: multibody impulse joints hold, envs={num_envs}");
}

#[futures_test::test]
#[serial_test::serial]
#[ignore]
async fn test_stacks_12_trimesh_reduction() {
    let backend = test_backend().await;

    let num_envs = 4u32;
    let build = || {
        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();

        let ground = bodies.insert(RigidBodyBuilder::fixed());
        let nsubdivs = 8;
        let heights = Array2::from_fn(nsubdivs + 1, nsubdivs + 1, |_, _| 0.0f32);
        let (vertices, indices) = HeightField::new(heights, Vec3::new(8.0, 1.0, 8.0)).to_trimesh();
        colliders.insert_with_parent(
            ColliderBuilder::trimesh_with_flags(
                vertices,
                indices,
                TriMeshFlags::MERGE_DUPLICATE_VERTICES,
            )
            .unwrap(),
            ground,
            &mut bodies,
        );

        let wide = bodies.insert(RigidBodyBuilder::dynamic().translation(Vec3::new(0.0, 0.3, 0.0)));
        colliders.insert_with_parent(ColliderBuilder::cuboid(1.5, 0.2, 1.5), wide, &mut bodies);

        for i in 0..3 {
            let b = bodies.insert(RigidBodyBuilder::dynamic().translation(Vec3::new(
                0.0,
                0.7 + i as f32 * 0.45,
                0.0,
            )));
            colliders.insert_with_parent(ColliderBuilder::cuboid(0.2, 0.2, 0.2), b, &mut bodies);
        }
        (bodies, colliders)
    };
    let envs: Vec<_> = (0..num_envs).map(|_| build()).collect();
    let joints = ImpulseJointSet::new();
    let mb_joints = MultibodyJointSet::new();
    let params = RbdSimParams::tgs_soft();
    let refs: Vec<_> = envs
        .iter()
        .map(|(b, c)| (b, c, &joints, &mb_joints, &params))
        .collect();

    let capacities = RbdCapacities {
        batches: num_envs,
        collisions_capacity: 64,
        ..Default::default()
    };
    let mut state = RbdState::from_rapier(&backend, &refs, capacities);
    let mut pipeline = RbdPipeline::new(&backend).unwrap();
    pipeline.contact_reduction = true;
    for _ in 0..250 {
        pipeline.step(&backend, &mut state, None).unwrap();
        pipeline.auto_resize_buffers(&backend, &mut state).unwrap();
    }
    backend.synchronize().unwrap();

    let poses: Vec<Pose> = backend
        .slow_read_vec(state.body_poses().buffer())
        .await
        .unwrap();
    let nb = num_envs as usize;
    for env in 0..num_envs as usize {
        let expected = [0.2f32, 0.6, 1.0, 1.4];
        for (b, expected_y) in expected.iter().enumerate() {
            let pose = poses[(1 + b) * nb + env];
            assert!(
                pose.translation.is_finite(),
                "env {env} box {b}: non-finite pose {:?}",
                pose.translation
            );
            let y = pose.translation.y;
            assert!(
                (y - expected_y).abs() < 0.1,
                "env {env} box {b}: y = {y}, expected ~{expected_y}"
            );
            if env > 0 {
                let ref_pose = poses[(1 + b) * nb];
                let d = (pose.translation - ref_pose.translation).length();
                assert!(d < 5.0e-2, "env {env} box {b}: diverged from env 0 by {d}");
            }
        }
    }
    println!("OK: trimesh floor + contact reduction rests, envs={num_envs}");
}
