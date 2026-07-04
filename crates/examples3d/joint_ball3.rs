use khal::backend::GpuTimestamps;
use nexus_viewer3d::NexusViewer;
use nexus3d::prelude::{NexusCapacities, NexusPipeline, NexusState};
use rapier3d::prelude::*;

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    /*
     * World
     */
    let capacities = NexusCapacities::default().rbd_collisions(350_000);
    let mut state = NexusState::new(capacities);

    let rad = 0.4;
    let ni = 200;
    let nk = 301;
    let shift = 1.0;
    let center = Vec3::new(nk as f32 * shift / 2.0, 0.0, ni as f32 * shift / 2.0);

    let mut body_handles = Vec::new();

    // A lot of joints. Kind of like a piece of cloth.
    for k in 0..nk {
        for i in 0..ni {
            let fk = k as f32;
            let fi = i as f32;

            let status = if ((i == 0 || i == ni - 1) && (k % 4 == 0 || k == ni - 1))
                || ((k == 0 || k == nk - 1) && (i % 4 == 0 || i == nk - 1))
            {
                RigidBodyType::Fixed
            } else {
                RigidBodyType::Dynamic
            };

            let rigid_body = RigidBodyBuilder::new(status)
                .translation(Vec3::new(fk * shift, 0.0, fi * shift) - center)
                .build();
            let collider = if status == RigidBodyType::Fixed {
                ColliderBuilder::cuboid(rad, rad, rad).build()
            } else {
                ColliderBuilder::ball(rad).density(10.0).build()
            };
            let shape = collider.shared_shape().clone();
            let child_handle = state.insert_rigid_body(rigid_body, collider);
            viewer.insert_shape(child_handle, &shape, Pose::IDENTITY);

            // Vertical joint.
            if i > 0 {
                let parent_handle = *body_handles.last().unwrap();
                let joint = SphericalJointBuilder::new().local_anchor2(Vec3::new(0.0, 0.0, -shift));
                state.insert_impulse_joint(parent_handle, child_handle, joint);
            }

            // Horizontal joint.
            if k > 0 {
                let parent_index = body_handles.len() - ni;
                let parent_handle = body_handles[parent_index];
                let joint = SphericalJointBuilder::new().local_anchor2(Vec3::new(-shift, 0.0, 0.0));
                state.insert_impulse_joint(parent_handle, child_handle, joint);
            }

            body_handles.push(child_handle);
        }
    }

    // Some rigid-bodies to fall on top.
    let nj = 10;
    let nk = nk / 3;
    let ni = ni / 6;
    let rad = rad * 2.5;

    for k in 0..nk {
        for i in 0..ni {
            for j in 0..nj {
                let body = RigidBodyBuilder::dynamic()
                    .translation(Vec3::new(
                        (k as f32 - nk as f32 / 2.0) * rad * 2.1,
                        j as f32 * rad * 2.1 + 2.0,
                        (i as f32 - ni as f32 / 2.0) * rad * 2.1,
                    ))
                    .build();
                let collider = ColliderBuilder::cuboid(rad, rad, rad).build();
                let shape = collider.shared_shape().clone();
                let handle = state.insert_rigid_body(body, collider);
                viewer.insert_shape(handle, &shape, Pose::IDENTITY);
            }
        }
    }

    // Optional, useful so we can render even before starting the simulation.
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
