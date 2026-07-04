use khal::backend::GpuTimestamps;
use nexus_viewer3d::NexusViewer;
use nexus3d::prelude::{NexusCapacities, NexusPipeline, NexusState};
use rapier3d::prelude::*;

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let capacities = NexusCapacities::default().rbd_collisions(32);
    let mut state = NexusState::new(capacities);

    let rad = 0.4;
    let num = 10;
    let shift = 2.0;
    let nk = 10;
    let nj = 50;

    // Environment 0 already exists; the first chain reuses it, the rest get a
    // fresh environment each.
    let mut first = true;

    for k in 0..nk {
        for l in 0..4 {
            let y = l as f32 * shift * (num as f32) * 3.0;
            for j in 0..nj {
                let env = if first {
                    first = false;
                    0
                } else {
                    state.add_environment()
                };

                let x = (j as f32 - nj as f32 / 2.0) * shift * 4.0;
                let z = (k as f32 - nk as f32 / 2.0) * num as f32 * shift * 2.1;

                let ground = RigidBodyBuilder::fixed()
                    .translation(Vec3::new(x, y, z))
                    .build();
                let ground_collider = ColliderBuilder::cuboid(rad, rad, rad).build();
                let ground_shape = ground_collider.shared_shape().clone();
                let mut curr_parent = state.insert_rigid_body_in(env, ground, ground_collider);
                viewer.insert_shape_in(
                    env as u32,
                    curr_parent,
                    &ground_shape,
                    Pose::IDENTITY,
                    None,
                );

                for i in 0..num {
                    // Create four bodies.
                    let z = z + i as f32 * shift * 2.0 + shift;
                    let positions = [
                        Pose3::translation(x, y, z),
                        Pose3::translation(x + shift, y, z),
                        Pose3::translation(x + shift, y, z + shift),
                        Pose3::translation(x, y, z + shift),
                    ];

                    let mut handles = [curr_parent; 4];
                    for k in 0..4 {
                        let density = 1.0;
                        let body = RigidBodyBuilder::dynamic().pose(positions[k]).build();
                        let collider = ColliderBuilder::cuboid(rad, rad, rad)
                            .density(density)
                            .build();
                        let shape = collider.shared_shape().clone();
                        handles[k] = state.insert_rigid_body_in(env, body, collider);
                        viewer.insert_shape_in(
                            env as u32,
                            handles[k],
                            &shape,
                            Pose::IDENTITY,
                            None,
                        );
                    }

                    // Setup four impulse_joints.
                    let x = Vec3::X;
                    let z = Vec3::Z;

                    let revs = [
                        RevoluteJointBuilder::new(z).local_anchor2(Vec3::new(0.0, 0.0, -shift)),
                        RevoluteJointBuilder::new(x).local_anchor2(Vec3::new(-shift, 0.0, 0.0)),
                        RevoluteJointBuilder::new(z).local_anchor2(Vec3::new(0.0, 0.0, -shift)),
                        RevoluteJointBuilder::new(x).local_anchor2(Vec3::new(shift, 0.0, 0.0)),
                    ];

                    state.insert_impulse_joint_in(env, curr_parent, handles[0], revs[0]);
                    state.insert_impulse_joint_in(env, handles[0], handles[1], revs[1]);
                    state.insert_impulse_joint_in(env, handles[1], handles[2], revs[2]);
                    state.insert_impulse_joint_in(env, handles[2], handles[3], revs[3]);

                    curr_parent = handles[3];
                }
            }
        }
    }

    let mut timestamps = GpuTimestamps::new(viewer.backend(), 2048);
    viewer
        .scene3d_mut()
        .add_directional_light(glamx::Vec3::new(1.0, -2.0, 3.0));
    state.finalize(viewer.backend()).await?;

    while viewer.render_frame().await {
        if viewer.simulating() {
            pipeline
                .simulate(viewer.backend(), &mut state, Some(&mut timestamps))
                .await?;
        }
        viewer.sync(&mut state, Some(&mut timestamps)).await?;
    }

    Ok(state)
}
