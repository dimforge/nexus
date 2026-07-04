use khal::backend::GpuTimestamps;
use nexus_viewer3d::NexusViewer;
use nexus3d::prelude::{NexusCapacities, NexusPipeline, NexusState};
use rapier3d::prelude::*;

/// Inserts a body + collider into the state and registers its render shape.
fn add_body(
    state: &mut NexusState,
    viewer: &mut NexusViewer,
    body: RigidBody,
    collider: Collider,
) -> RigidBodyHandle {
    let shape = collider.shared_shape().clone();
    let handle = state.insert_rigid_body(body, collider);
    viewer.insert_shape(handle, &shape, Pose::IDENTITY);
    handle
}

fn create_pyramid(
    state: &mut NexusState,
    viewer: &mut NexusViewer,
    offset: Vector,
    stack_height: usize,
    rad: f32,
) {
    let shift = rad * 2.0;

    for i in 0usize..stack_height {
        for j in i..stack_height {
            let fj = j as f32;
            let fi = i as f32;
            let x = (fi * shift / 2.0) + (fj - fi) * shift;
            let y = fi * shift;

            add_body(
                state,
                viewer,
                RigidBodyBuilder::dynamic()
                    .translation(Vec3::new(x, y, 0.0) + offset)
                    .build(),
                ColliderBuilder::cuboid(rad, rad, rad).build(),
            );
        }
    }
}

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let capacities = NexusCapacities::default().rbd_collisions(420_000);
    let mut state = NexusState::new(capacities);

    let rad = 0.5;
    let pyramid_count = 40;
    let spacing = 4.0;

    /*
     * Ground
     */
    let ground_size = 100.0;
    let ground_height = 0.1;

    add_body(
        &mut state,
        viewer,
        RigidBodyBuilder::fixed()
            .translation(Vec3::new(0.0, -ground_height, 0.0))
            .build(),
        ColliderBuilder::cuboid(
            ground_size,
            ground_height,
            pyramid_count as f32 * spacing / 2.0 + ground_size,
        )
        .build(),
    );

    /*
     * Create the cubes
     */
    for pyramid_index in 0..pyramid_count {
        let bottomy = rad;
        create_pyramid(
            &mut state,
            viewer,
            Vec3::new(
                0.0,
                bottomy,
                (pyramid_index as f32 - pyramid_count as f32 / 2.0) * spacing,
            ),
            60,
            rad,
        );

        create_pyramid(
            &mut state,
            viewer,
            Vec3::new(
                -75.0,
                bottomy,
                (pyramid_index as f32 - pyramid_count as f32 / 2.0) * spacing,
            ),
            60,
            rad,
        );
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
    // viewer.look_at(point![100.0, 100.0, 100.0], Point::origin());
}
