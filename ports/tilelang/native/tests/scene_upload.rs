#![cfg(feature = "rust-reference")]
use khal::backend::{Backend, GpuBackend, Metal};
use nexus_rbd3d::{
    math::Pose, pipeline::RbdState, rapier::prelude::*, shaders::dynamics::RbdSimParams,
};

use nexus_tilelang_native::upload_cases::{capacities, scene};

#[test]
fn upload_preserves_native_multi_collider_and_batch_layout() {
    futures::executor::block_on(async {
        let backend = GpuBackend::Metal(Metal::new().unwrap());
        let mesh = SharedShape::trimesh_with_flags(
            vec![
                Vec3::new(-2.0, 0.0, -2.0),
                Vec3::new(2.0, 0.0, -2.0),
                Vec3::new(0.0, 0.0, 2.0),
            ],
            vec![[0, 1, 2]],
            TriMeshFlags::MERGE_DUPLICATE_VERTICES,
        )
        .unwrap();
        let scenes = [scene(0.0, mesh.clone()), scene(10.0, mesh)];
        let multibodies = MultibodyJointSet::new();
        let mut params = RbdSimParams::tgs_soft();
        params.num_solver_iterations = 3;
        params.num_internal_pgs_iterations = 5;
        params.dt = 0.03;
        let refs: Vec<_> = scenes
            .iter()
            .map(|(b, c, j)| (b, c, j, &multibodies, &params))
            .collect();
        let mut state = RbdState::from_rapier(&backend, &refs, capacities(2));
        state.set_gravity(&backend, [0.0, -3.0, 0.5]);
        backend.synchronize().unwrap();
        let before = backend
            .slow_read_vec(state.body_poses().buffer())
            .await
            .unwrap();
        let upload = state.read_scene_upload(&backend).await.unwrap();
        let after = backend
            .slow_read_vec(state.body_poses().buffer())
            .await
            .unwrap();
        assert_eq!(
            bytemuck::cast_slice::<Pose, u8>(&before),
            bytemuck::cast_slice::<Pose, u8>(&after)
        );
        assert_eq!(upload.batch_indices.num_batches, 2);
        assert_eq!(upload.batch_indices.bodies_len, 3);
        assert_eq!(upload.batch_indices.colliders_len, 4);
        assert_eq!(upload.batch_indices.colliders_batch_capacity, 4);
        assert_eq!(upload.collider_parents, [0, 1, 0, 1, 2, 3, 4, 5]);
        assert_eq!(upload.body_poses.len(), 8);
        assert_eq!(upload.params.dt, params.dt / 3.0);
        assert_eq!(upload.params.num_internal_pgs_iterations, 5);
        assert_eq!(upload.gravity.truncate(), Vec3::new(0.0, -3.0, 0.5));
        assert_eq!(upload.joints_per_batch, 1);
        assert_eq!(upload.batch_indices.impulse_joints_len, 1);
        assert_eq!(upload.joint_color_ends, [1]);
        assert_eq!(upload.joints.len(), 2);
        for (batch, scene) in scenes.iter().enumerate() {
            let shift = batch as f32 * 10.0;
            assert_eq!(
                upload.body_poses[batch].translation,
                Vec3::new(shift, 2.0, 0.0)
            );
            assert_eq!(
                upload.velocities[batch].linear,
                Vec3::new(1.0 + shift, 0.0, 0.0)
            );
            assert_eq!(upload.velocities[batch].angular, Vec3::new(0.0, 0.3, 0.0));
            assert_eq!(upload.body_poses[4 + batch], Pose::IDENTITY);
            assert_eq!(
                upload.collider_local_poses[batch].translation,
                Vec3::new(0.4, 0.0, 0.0)
            );
            assert_eq!(
                upload.collider_local_poses[6 + batch].translation,
                Vec3::new(shift, -1.0, 0.0)
            );
            let body = scene.0.iter().next().unwrap().1;
            assert_eq!(
                upload.local_mass[batch].com,
                body.mass_properties().local_mprops.local_com
            );
            assert!(upload.local_mass[batch].com.x.abs() > 0.01);
            assert_eq!(upload.local_mass[batch + 2].inv_mass, Vec3::ZERO);
            assert_eq!(upload.local_mass[batch + 4].inv_mass, Vec3::ZERO);
            assert_eq!(upload.materials[batch].friction, 0.7);
            assert_eq!(upload.materials[batch].restitution, 0.3);
            assert_eq!(
                upload.materials[batch].friction_combine_rule,
                CoefficientCombineRule::Max as u32
            );
            assert_eq!(
                upload.groups[batch],
                InteractionGroups::new(Group::GROUP_2, Group::GROUP_3, InteractionTestMode::Or)
            );
            assert_eq!(upload.pair_filters[batch], upload.pair_filters[batch + 2]);
            let joint = &upload.joints[batch];
            assert_eq!((joint.body_a, joint.body_b), (0, 1));
            assert_eq!(
                joint.data.local_frame_a.translation,
                Vec3::new(0.2, 0.3, 0.4)
            );
            assert_eq!(joint.data.limits[0].min, -0.25);
            assert_eq!(joint.data.limits[0].max, 0.75);
            assert_eq!(joint.data.motors[0].target_vel, 0.15);
        }
        // Cloned mesh geometry is shared across batches by the native converter.
        assert_eq!(
            bytemuck::bytes_of(&upload.shapes[6]),
            bytemuck::bytes_of(&upload.shapes[7])
        );
        assert!(!upload.vertices.is_empty());
        assert!(!upload.indices.is_empty());
        if let Some(directory) = std::env::var_os("NEXUS_TILELANG_UPLOAD_DIR") {
            nexus_tilelang_native::upload::write_inputs(&upload, std::path::Path::new(&directory))
                .unwrap();
        }
        let mut invalid = upload.clone();
        invalid.collider_parents[0] = 1;
        let error = nexus_tilelang_native::upload::write_inputs(
            &invalid,
            std::path::Path::new("must-not-exist"),
        )
        .unwrap_err();
        assert!(error.contains("active batch"));
        invalid = upload.clone();
        invalid.joint_color_ends[0] = 2;
        assert!(
            nexus_tilelang_native::upload::write_inputs(
                &invalid,
                std::path::Path::new("must-not-exist")
            )
            .unwrap_err()
            .contains("joint coloring")
        );
        invalid = upload.clone();
        invalid.joints.extend_from_slice(&upload.joints);
        invalid.joints_per_batch = 2;
        invalid.batch_indices.impulse_joints_len = 2;
        invalid.joint_color_ends[0] = 2;
        assert!(
            nexus_tilelang_native::upload::write_inputs(
                &invalid,
                std::path::Path::new("must-not-exist")
            )
            .unwrap_err()
            .contains("aliases a velocity writer")
        );
        invalid = upload.clone();
        invalid.params.dt = f32::NAN;
        assert!(
            nexus_tilelang_native::upload::write_inputs(
                &invalid,
                std::path::Path::new("must-not-exist")
            )
            .unwrap_err()
            .contains("timestep")
        );
    });
}

#[test]
fn empty_upload_retains_capacity_without_inventing_active_objects() {
    futures::executor::block_on(async {
        let backend = GpuBackend::Metal(Metal::new().unwrap());
        let state = RbdState::empty(&backend, capacities(2));
        let upload = state.read_scene_upload(&backend).await.unwrap();
        assert_eq!(upload.batch_indices.num_batches, 2);
        assert_eq!(upload.batch_indices.bodies_len, 0);
        assert_eq!(upload.batch_indices.colliders_len, 0);
        assert_eq!(upload.body_poses.len(), 16);
        assert!(upload.joints.is_empty());
        assert!(upload.joint_color_ends.is_empty());
        assert!(
            upload
                .groups
                .iter()
                .all(|g| g.memberships == Group::NONE && g.filter == Group::NONE)
        );
    });
}

#[test]
fn articulated_upload_is_rejected_explicitly() {
    futures::executor::block_on(async {
        let backend = GpuBackend::Metal(Metal::new().unwrap());
        let (bodies, colliders, _) = scene(0.0, SharedShape::ball(0.4));
        let handles: Vec<_> = bodies.iter().map(|(h, _)| h).collect();
        let joints = ImpulseJointSet::new();
        let mut multibodies = MultibodyJointSet::new();
        multibodies
            .insert(handles[0], handles[1], FixedJointBuilder::new(), true)
            .unwrap();
        let params = RbdSimParams::tgs_soft();
        let state = RbdState::from_rapier(
            &backend,
            &[(&bodies, &colliders, &joints, &multibodies, &params)],
            capacities(1),
        );
        match state.read_scene_upload(&backend).await {
            Err(khal::backend::GpuBackendError::Unsupported(message)) => {
                assert!(message.contains("articulated"))
            }
            _ => panic!("articulated upload must fail before discarding its dynamics"),
        }
    });
}
