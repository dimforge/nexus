use khal::backend::GpuTimestamps;
use nexus_viewer3d::NexusViewer;
use nexus3d::prelude::{NexusPipeline, NexusState};
use rapier3d::prelude::*;
use rapier3d_urdf::{UrdfLoaderOptions, UrdfMultibodyOptions, UrdfRobot};
use std::path::PathBuf;

fn urdf_path() -> PathBuf {
    // let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    // PathBuf::from(home).join("Downloads/robot.urdf")
    // PathBuf::from("/Users/sebcrozet/work/rapier/assets/3d/T12/urdf/T12.URDF")
    PathBuf::from("/Users/sebcrozet/work/nexus-demos/XoQ/js/examples/assets/openarm_v10.urdf")
}

/// The example owns its loop. Every 5 simulated seconds it re-randomizes each
/// multibody joint's angular-X motor target velocity within `[-0.6, 0.6]` rad/s
/// so the robot stays in slow continuous motion with periodically changing
/// direction.
pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    use rand::RngExt;

    let mut state = NexusState::default();

    /*
     * Robot loaded from URDF.
     */
    let path = urdf_path();
    let scale = 40.0;
    let options = UrdfLoaderOptions {
        create_colliders_from_collision_shapes: true,
        // Many URDFs (including openarm_v10) only ship visual meshes and leave the
        // collision tags empty, so fall back to visuals when collisions are missing.
        create_colliders_from_visual_shapes: true,
        apply_imported_mass_props: true,
        make_roots_fixed: true,
        scale,
        // Use cheap oriented bounding boxes for physics. The loader keeps the
        // original triangle meshes attached to each collider as a `UrdfVisual`,
        // which we forward to the viewer as a per-body visual override.
        mesh_converter: None, // Some(MeshConverter::Obb),
        // Lift the robot above the ground. URDF is Z-up but the viewer is Y-up,
        // so rotate -90° around X so the robot stands upright.
        shift: Pose::from_parts(
            Vec3::new(0.0, scale, 0.0),
            Rotation::from_rotation_x(-std::f32::consts::FRAC_PI_2),
        ),
        collider_blueprint: ColliderBuilder::ball(0.5).collision_groups(InteractionGroups::none()),
        ..UrdfLoaderOptions::default()
    };

    // Per-body render shapes collected during loading and registered with the
    // viewer once the rapier-world borrow has ended.
    let mut render_shapes: Vec<(RigidBodyHandle, SharedShape, Pose)> = Vec::new();
    let mut num_links = 0u32;

    match UrdfRobot::from_file(&path, options, None) {
        Ok((mut robot, _)) => {
            // Switch every joint's `AngX` motor to acceleration-based mode so the
            // per-frame motor target velocity feels right regardless of link mass.
            // Initial target velocity is 0 — the loop below re-randomizes it every
            // 5 simulated seconds.
            for urdf_joint in &mut robot.joints {
                urdf_joint
                    .joint
                    .set_motor_model(JointAxis::AngX, MotorModel::AccelerationBased);
                urdf_joint
                    .joint
                    .set_motor_velocity(JointAxis::AngX, 0.0, 1.0);
            }

            let world = state.rbd_world_mut(0);
            let handles = robot.insert_using_multibody_joints(
                &mut world.bodies,
                &mut world.colliders,
                &mut world.multibody_joints,
                UrdfMultibodyOptions::DISABLE_SELF_CONTACTS,
            );

            num_links = handles.links.len() as u32;
            for link in &handles.links {
                for collider in &link.colliders {
                    // Prefer the attached visual mesh; otherwise render the
                    // collider's own shape. Render against the link's body, since
                    // poses are body-keyed.
                    let (shape, local_pose) = match &collider.visual {
                        Some(visual) => (visual.shape.clone(), visual.local_pose),
                        None => (
                            world.colliders[collider.handle].shared_shape().clone(),
                            Pose::IDENTITY,
                        ),
                    };
                    render_shapes.push((link.body, shape, local_pose));
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to load URDF file at {}: {e}.", path.display());
        }
    }

    for (body, shape, local_pose) in &render_shapes {
        viewer.insert_visual_shape(0, *body, shape, *local_pose);
    }

    let mut timestamps = GpuTimestamps::new(viewer.backend(), 2048);
    state.finalize(viewer.backend()).await?;

    let mut rng = rand::rng();
    let dt = 1.0 / 60.0_f64;
    let mut sim_time = 0.0_f64;
    let mut next_change_at = 0.0_f64;
    let interval = 5.0;

    while viewer.render_frame().await {
        if sim_time >= next_change_at {
            next_change_at = sim_time + interval;
            let num_batches = state.rbd_num_batches();
            for batch in 0..num_batches {
                for link_id in 0..num_links {
                    let target_vel: f32 = rng.random_range(-0.6f32..=0.6);
                    let _ = state.set_multibody_motor_velocity(
                        viewer.backend(),
                        batch,
                        link_id,
                        JointAxis::AngX,
                        target_vel,
                    );
                }
            }
        }

        if viewer.simulating() {
            pipeline
                .simulate(viewer.backend(), &mut state, Some(&mut timestamps))
                .await?;
            sim_time += dt;
        }
        viewer.sync(&mut state, Some(&mut timestamps)).await?;
    }

    Ok(state)
}
