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
    half_extents: Vector,
) {
    let shift = half_extents * 2.5;
    for i in 0usize..stack_height {
        for j in i..stack_height {
            for k in i..stack_height {
                let fi = i as f32;
                let fj = j as f32;
                let fk = k as f32;
                let x = (fi * shift.x / 2.0) + (fk - fi) * shift.x + offset.x
                    - stack_height as f32 * half_extents.x;
                let y = fi * shift.y + offset.y;
                let z = (fi * shift.z / 2.0) + (fj - fi) * shift.z + offset.z
                    - stack_height as f32 * half_extents.z;

                add_body(
                    state,
                    viewer,
                    RigidBodyBuilder::dynamic()
                        .translation(Vec3::new(x, y, z))
                        .build(),
                    ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z).build(),
                );
            }
        }
    }
}

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let capacities = NexusCapacities::default().rbd_collisions(170_000);
    let mut state = NexusState::new(capacities);

    /*
     * Ground
     */
    let ground_size = 200.0;
    let ground_height = 0.1;

    add_body(
        &mut state,
        viewer,
        RigidBodyBuilder::fixed()
            .translation(Vec3::new(0.0, -ground_height, 0.0))
            .build(),
        ColliderBuilder::cuboid(ground_size, ground_height, ground_size).build(),
    );

    /*
     * Create the cubes
     */
    let cube_size = 1.0;
    let hext = Vec3::splat(cube_size);
    let bottomy = cube_size;
    create_pyramid(&mut state, viewer, Vec3::new(0.0, bottomy, 0.0), 50, hext);

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
