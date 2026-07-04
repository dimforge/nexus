use khal::backend::GpuTimestamps;
use nexus_viewer2d::NexusViewer;
use nexus2d::prelude::{NexusCapacities, NexusPipeline, NexusState};
use rapier2d::prelude::*;

/// 2D analogue of rapier's `compound3` demo.
pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let capacities = NexusCapacities::default().rbd_collisions(40_000);
    let mut state = NexusState::new(capacities);

    /*
     * Ground
     */
    let ground_size = 200.0;

    let body = RigidBodyBuilder::fixed().build();
    let collider = ColliderBuilder::cuboid(ground_size, 1.5).build();
    let shape = collider.shared_shape().clone();
    let handle = state.insert_rigid_body(body, collider);
    viewer.insert_shape(handle, &shape, Pose::IDENTITY);

    let body = RigidBodyBuilder::fixed()
        .rotation(std::f32::consts::FRAC_PI_2)
        .translation(Vec2::new(ground_size, ground_size * 2.1))
        .build();
    let collider = ColliderBuilder::cuboid(ground_size * 2.1, 1.5).build();
    let shape = collider.shared_shape().clone();
    let handle = state.insert_rigid_body(body, collider);
    viewer.insert_shape(handle, &shape, Pose::IDENTITY);

    let body = RigidBodyBuilder::fixed()
        .rotation(std::f32::consts::FRAC_PI_2)
        .translation(Vec2::new(-ground_size, ground_size * 2.1))
        .build();
    let collider = ColliderBuilder::cuboid(ground_size * 2.1, 1.5).build();
    let shape = collider.shared_shape().clone();
    let handle = state.insert_rigid_body(body, collider);
    viewer.insert_shape(handle, &shape, Pose::IDENTITY);

    /*
     * "U"-shaped bodies, each made of three cuboid colliders.
     */
    let rad = 0.4f32;
    let parts: [(Vec2, Vec2); 3] = [
        (Vec2::ZERO, Vec2::new(rad * 10.0, rad)),
        (
            Vec2::new(rad * 10.0, rad * 10.0),
            Vec2::new(rad, rad * 10.0),
        ),
        (
            Vec2::new(-rad * 10.0, rad * 10.0),
            Vec2::new(rad, rad * 10.0),
        ),
    ];

    let num = 20;
    let numy = 200;
    // Each U spans ~4 units in x; space the grid out so they don't start
    // interpenetrating.
    let shift = rad * 10.0 * 2.0 + 1.0;
    let centerx = shift * (num as f32) / 2.0;

    for j in 0..numy {
        for i in 0..num {
            let x = i as f32 * shift - centerx + (j % 2) as f32 * 0.3;
            let y = j as f32 * shift + 6.0;

            let body = RigidBodyBuilder::dynamic()
                .translation(Vec2::new(x, y))
                .build();

            // Attach the three colliders to a single rigid body: the base
            // collider comes with the body, the two walls are added with
            // `insert_collider_in`.
            let (base_offset, base_he) = parts[0];
            let base = ColliderBuilder::cuboid(base_he.x, base_he.y)
                .translation(base_offset)
                .build();
            let handle = state.insert_rigid_body(body, base);
            for (offset, he) in &parts[1..] {
                let collider = ColliderBuilder::cuboid(he.x, he.y)
                    .translation(*offset)
                    .build();
                state.insert_collider_in(0, collider, Some(handle));
            }

            // One render shape per collider, at its body-local pose.
            for (offset, he) in parts {
                let shape = SharedShape::cuboid(he.x, he.y);
                viewer.insert_visual_shape(0, handle, &shape, Pose::from_translation(offset));
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
