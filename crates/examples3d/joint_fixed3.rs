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

    let mut body_handles = Vec::new();

    for m in 0..10 {
        let z = m as f32 * shift * (num as f32 + 2.0);

        for l in 0..10 {
            let y = l as f32 * shift * 3.0;

            for j in 0..10 {
                let x = j as f32 * shift * (num as f32) * 2.0;

                for k in 0..num {
                    for i in 0..num {
                        let fk = k as f32;
                        let fi = i as f32;

                        // NOTE: the num - 2 test is to avoid two consecutive
                        // fixed bodies. Because physx will crash if we add
                        // a joint between these.

                        let status = if i == 0 && (k % 4 == 0 && k != num - 2 || k == num - 1) {
                            RigidBodyType::Fixed
                        } else {
                            RigidBodyType::Dynamic
                        };

                        let rigid_body = RigidBodyBuilder::new(status)
                            .translation(Vec3::new(x + fk * shift, y, z + fi * shift))
                            .build();
                        let collider = ColliderBuilder::ball(rad).build();
                        let shape = collider.shared_shape().clone();
                        let child_handle = state.insert_rigid_body(rigid_body, collider);
                        viewer.insert_shape(child_handle, &shape, Pose::IDENTITY);

                        // Vertical joint.
                        if i > 0 {
                            let parent_handle = *body_handles.last().unwrap();
                            let joint =
                                FixedJointBuilder::new().local_anchor2(Vec3::new(0.0, 0.0, -shift));
                            state.insert_impulse_joint(parent_handle, child_handle, joint);
                        }

                        // Horizontal joint.
                        if k > 0 {
                            let parent_index = body_handles.len() - num;
                            let parent_handle = body_handles[parent_index];
                            let joint =
                                FixedJointBuilder::new().local_anchor2(Vec3::new(-shift, 0.0, 0.0));
                            state.insert_impulse_joint(parent_handle, child_handle, joint);
                        }

                        body_handles.push(child_handle);
                    }
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
