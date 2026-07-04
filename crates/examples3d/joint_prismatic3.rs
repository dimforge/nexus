use khal::backend::GpuTimestamps;
use nexus_viewer3d::NexusViewer;
use nexus3d::prelude::{NexusPipeline, NexusState};
use rapier3d::prelude::*;

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    /*
     * World
     */
    let mut state = NexusState::default();

    let rad = 0.4;
    let num = 10;
    let shift = 1.0;

    for m in 0..20 {
        let z = m as f32 * shift * (num as f32 + 2.0);

        for l in 0..20 {
            let y = l as f32 * shift * (num as f32) * 2.0;

            for j in 0..30 {
                let x = j as f32 * shift * 4.0;

                let ground = RigidBodyBuilder::fixed()
                    .translation(Vec3::new(x, y, z))
                    .build();
                let collider = ColliderBuilder::cuboid(rad, rad, rad).build();
                let shape = collider.shared_shape().clone();
                let mut curr_parent = state.insert_rigid_body(ground, collider);
                viewer.insert_shape(curr_parent, &shape, Pose::IDENTITY);

                for i in 0..num {
                    let z = z + (i + 1) as f32 * shift;
                    let density = 1.0;
                    let rigid_body = RigidBodyBuilder::dynamic()
                        .translation(Vec3::new(x, y, z))
                        .build();
                    let collider = ColliderBuilder::cuboid(rad, rad, rad)
                        .density(density)
                        .build();
                    let shape = collider.shared_shape().clone();
                    let curr_child = state.insert_rigid_body(rigid_body, collider);
                    viewer.insert_shape(curr_child, &shape, Pose::IDENTITY);

                    let axis = if i % 2 == 0 {
                        Vec3::new(1.0, 1.0, 0.0).normalize()
                    } else {
                        Vec3::new(-1.0, 1.0, 0.0).normalize()
                    };

                    let prism = PrismaticJointBuilder::new(axis)
                        .local_anchor2(Vec3::new(0.0, 0.0, -shift))
                        .limits([-2.0, 0.0]);
                    state.insert_impulse_joint(curr_parent, curr_child, prism);

                    curr_parent = curr_child;
                }
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
