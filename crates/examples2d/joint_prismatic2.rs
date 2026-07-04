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
     * Create the balls
     */
    let rad = 0.4;
    let num = 10;
    let shift = 1.0;

    for l in 0..38 {
        let y = l as f32 * shift * (num as f32 + 2.0) * 2.0;

        for j in 0..300 {
            let x = j as f32 * shift * 4.0;

            let body = RigidBodyBuilder::fixed()
                .translation(Vec2::new(x, y))
                .build();
            let collider = ColliderBuilder::cuboid(rad, rad).build();
            let shape = collider.shared_shape().clone();
            let mut curr_parent = state.insert_rigid_body(body, collider);
            viewer.insert_shape(curr_parent, &shape, Pose::IDENTITY);

            for i in 0..num {
                let y = y - (i + 1) as f32 * shift;
                let density = 1.0;
                let body = RigidBodyBuilder::dynamic()
                    .translation(Vec2::new(x, y))
                    .build();
                let collider = ColliderBuilder::cuboid(rad, rad).density(density).build();
                let shape = collider.shared_shape().clone();
                let curr_child = state.insert_rigid_body(body, collider);
                viewer.insert_shape(curr_child, &shape, Pose::IDENTITY);

                let axis = if i % 2 == 0 {
                    Vec2::new(1.0, 1.0).normalize()
                } else {
                    Vec2::new(-1.0, 1.0).normalize()
                };

                let prism = PrismaticJointBuilder::new(axis)
                    .local_anchor2(Vec2::new(0.0, shift))
                    .limits([-1.5, 1.5]);
                state.insert_impulse_joint(curr_parent, curr_child, prism);

                curr_parent = curr_child;
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
