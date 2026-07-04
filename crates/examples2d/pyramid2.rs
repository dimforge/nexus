use khal::backend::GpuTimestamps;
use nexus_viewer2d::NexusViewer;
use nexus2d::prelude::{NexusPipeline, NexusState};
use rapier2d::prelude::*;

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let mut state = NexusState::default();

    /*
     * Ground
     */
    let ground_size = 500.0;
    let ground_thickness = 1.0;

    let body = RigidBodyBuilder::fixed().build();
    let collider = ColliderBuilder::cuboid(ground_size, ground_thickness).build();
    let shape = collider.shared_shape().clone();
    let handle = state.insert_rigid_body(body, collider);
    viewer.insert_shape(handle, &shape, Pose::IDENTITY);

    /*
     * Create the cubes
     */
    let num = 200;
    let rad = 0.5;

    let shiftx = rad * 2.0 + 0.1;
    let shifty = rad * 2.0;
    let centerx = shiftx * (num as f32) / 2.0;
    let centery = shifty / 2.0 + ground_thickness;

    for k in 0..4 {
        for i in 0usize..num {
            for j in i..num {
                let fj = j as f32;
                let fi = i as f32;
                let x = (fi * shiftx / 2.0) + (fj - fi) * shiftx - centerx
                    + (k as f32 - 1.5) * rad * 2.5 * num as f32;
                let y = fi * shifty + centery;

                // Build the rigid body.
                let body = RigidBodyBuilder::dynamic()
                    .translation(Vec2::new(x, y))
                    .build();
                let collider = ColliderBuilder::cuboid(rad, rad).build();
                let shape = collider.shared_shape().clone();
                let handle = state.insert_rigid_body(body, collider);
                viewer.insert_shape(handle, &shape, Pose::IDENTITY);
            }
        }
    }

    // Optional, useful so we can render even before starting the simulation.
    let mut timestamps = GpuTimestamps::new(viewer.backend(), 2048);
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
