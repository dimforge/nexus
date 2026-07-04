use glamx::Pose2;
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
    let num = 50; // Num vertical nodes.
    let shift = 1.0;

    let mut body_handles = Vec::new();

    for xx in 0..8 {
        let x = xx as f32 * shift * (num as f32 + 2.0);

        for yy in 0..8 {
            let y = yy as f32 * shift * (num as f32 + 4.0);

            for k in 0..num {
                for i in 0..num {
                    let fk = k as f32;
                    let fi = i as f32;

                    let status = if k == 0 {
                        RigidBodyType::Fixed
                    } else {
                        RigidBodyType::Dynamic
                    };

                    let body = RigidBodyBuilder::new(status)
                        .translation(Vec2::new(x + fk * shift, y - fi * shift))
                        .build();
                    let collider = ColliderBuilder::ball(rad).build();
                    let shape = collider.shared_shape().clone();
                    let child_handle = state.insert_rigid_body(body, collider);
                    viewer.insert_shape(child_handle, &shape, Pose::IDENTITY);

                    // Vertical joint.
                    if i > 0 {
                        let parent_handle = *body_handles.last().unwrap();
                        let joint =
                            FixedJointBuilder::new().local_frame2(Pose2::translation(0.0, shift));
                        state.insert_impulse_joint(parent_handle, child_handle, joint);
                    }

                    // Horizontal joint.
                    if k > 0 {
                        let parent_index = body_handles.len() - num;
                        let parent_handle = body_handles[parent_index];
                        let joint =
                            FixedJointBuilder::new().local_frame2(Pose2::translation(-shift, 0.0));
                        state.insert_impulse_joint(parent_handle, child_handle, joint);
                    }

                    body_handles.push(child_handle);
                }
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
