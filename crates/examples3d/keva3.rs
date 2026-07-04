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

pub fn build_block(
    state: &mut NexusState,
    viewer: &mut NexusViewer,
    half_extents: Vector,
    shift: Vector,
    (mut numx, numy, mut numz): (usize, usize, usize),
) {
    let dimensions = [
        half_extents,
        Vec3::new(half_extents.z, half_extents.y, half_extents.x),
    ];
    let block_width = 2.0 * half_extents.z * numx as f32;
    let block_height = 2.0 * half_extents.y * numy as f32;
    let spacing = (half_extents.z * numx as f32 - half_extents.x) / (numz as f32 - 1.0);
    let mut color0 = [0.7, 0.5, 0.9];
    let mut color1 = [0.6, 1.0, 0.6];

    for i in 0..numy {
        std::mem::swap(&mut numx, &mut numz);
        let dim = dimensions[i % 2];
        let y = dim.y * i as f32 * 2.0;

        for j in 0..numx {
            let x = if i % 2 == 0 {
                spacing * j as f32 * 2.0
            } else {
                dim.x * j as f32 * 2.0
            };

            for k in 0..numz {
                let z = if i % 2 == 0 {
                    dim.z * k as f32 * 2.0
                } else {
                    spacing * k as f32 * 2.0
                };

                add_body(
                    state,
                    viewer,
                    RigidBodyBuilder::dynamic()
                        .translation(Vec3::new(
                            x + dim.x + shift.x,
                            y + dim.y + shift.y,
                            z + dim.z + shift.z,
                        ))
                        .build(),
                    ColliderBuilder::cuboid(dim.x, dim.y, dim.z).build(),
                );

                // viewer.set_initial_body_color(handle, color0);
                std::mem::swap(&mut color0, &mut color1);
            }
        }
    }

    // Close the top.
    let dim = Vec3::new(half_extents.z, half_extents.x, half_extents.y);

    for i in 0..(block_width / (dim.x * 2.0)) as usize {
        for j in 0..(block_width / (dim.z * 2.0)) as usize {
            add_body(
                state,
                viewer,
                RigidBodyBuilder::dynamic()
                    .translation(Vec3::new(
                        i as f32 * dim.x * 2.0 + dim.x + shift.x,
                        dim.y + shift.y + block_height,
                        j as f32 * dim.z * 2.0 + dim.z + shift.z,
                    ))
                    .build(),
                ColliderBuilder::cuboid(dim.x, dim.y, dim.z).build(),
            );
            // viewer.set_initial_body_color(handle, color0);
            std::mem::swap(&mut color0, &mut color1);
        }
    }
}

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let capacities = NexusCapacities::default().rbd_collisions(200_000);
    let mut state = NexusState::new(capacities);

    /*
     * Ground
     */
    let ground_size = 70.0;
    let ground_height = 2.0;

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
    let half_extents = Vec3::new(0.02, 0.1, 0.4) / 2.0 * 10.0;
    let mut block_height = 0.0;
    // These should only be set to odd values otherwise
    // the blocks won't align in the nicest way.
    let numy = [0, 9, 13, 17, 21, 41];

    for i in (1..=5).rev() {
        let numx = (i as f32 * 2.5).ceil() as usize;
        let numy = numy[i];
        let numz = numx * 3 + 1;
        let block_width = numx as f32 * half_extents.z * 2.0;
        build_block(
            &mut state,
            viewer,
            half_extents,
            Vec3::new(-block_width / 2.0, block_height, -block_width / 2.0),
            (numx, numy, numz),
        );
        block_height += numy as f32 * half_extents.y * 2.0 + half_extents.x * 2.0;
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
