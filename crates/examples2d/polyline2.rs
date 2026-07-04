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
    let ground_size = 200.0;
    let nsubdivs = 40;
    let step_size = ground_size / (nsubdivs as f32);
    let mut points = Vec::new();

    points.push(Vec2::new(-ground_size / 2.0, 240.0));
    for i in 1..nsubdivs - 1 {
        let x = -ground_size / 2.0 + i as f32 * step_size;
        let y = (i as f32 / nsubdivs as f32 * 10.0).cos() * 20.0;
        points.push(Vec2::new(x, y));
    }
    points.push(Vec2::new(ground_size / 2.0, 240.0));

    let body = RigidBodyBuilder::fixed().build();
    let collider = ColliderBuilder::polyline(points, None).build();
    let shape = collider.shared_shape().clone();
    let handle = state.insert_rigid_body(body, collider);
    viewer.insert_shape(handle, &shape, Pose::IDENTITY);

    // Create 5 predefined convex polygon shapes (so we can render
    // them efficiently with instancing).
    let polygon_shapes: Vec<SharedShape> = {
        let mut shapes = Vec::new();
        let mut rng = oorandom::Rand32::new(42);

        for _ in 0..5 {
            let mut points = Vec::new();
            let scale = 2.0;
            for _ in 0..10 {
                let pt = Vec2::new(rng.rand_float() - 0.5, rng.rand_float() - 0.5);
                points.push(pt * scale);
            }

            // Center the shape at its center-of-mass
            let shape = SharedShape::convex_hull(&points).unwrap();
            let mprops = shape.mass_properties(1.0);
            points.iter_mut().for_each(|pt| *pt -= mprops.local_com);
            shapes.push(SharedShape::convex_hull(&points).unwrap());
        }
        shapes
    };

    /*
     * Create the falling primitives
     */
    let num = 100;
    let rad = 0.5;

    let shift = rad * 2.0 + 0.4;
    let centerx = shift * (num as f32) / 2.0;
    let centery = shift / 2.0;

    for i in 0..num {
        for j in 0usize..num * 3 {
            let x = i as f32 * shift - centerx + (j % 2) as f32 * 0.2;
            let y = j as f32 * shift + centery + 20.0;

            // Build the rigid body.
            let body = RigidBodyBuilder::dynamic()
                .translation(Vec2::new(x, y))
                .build();
            let collider = match j % 4 {
                0 => ColliderBuilder::cuboid(rad, rad),
                1 => ColliderBuilder::capsule_y(rad, rad),
                2 => {
                    // Reuse one of the 5 predefined polygon shapes
                    let shape_idx = (i + j) % polygon_shapes.len();
                    ColliderBuilder::new(polygon_shapes[shape_idx].clone())
                }
                _ => ColliderBuilder::ball(rad),
            }
            .build();
            let shape = collider.shared_shape().clone();
            let handle = state.insert_rigid_body(body, collider);
            viewer.insert_shape(handle, &shape, Pose::IDENTITY);
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
