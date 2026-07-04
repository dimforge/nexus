use khal::backend::GpuTimestamps;
use nexus_viewer3d::NexusViewer;
use nexus3d::prelude::{NexusCapacities, NexusPipeline, NexusState};
use rapier3d::prelude::*;

/// Demonstrates adding rigid-bodies to a live scene WITHOUT rebuilding the whole
/// GPU state each time: the scene reserves spare collider slots up-front
/// (`reserve_rigid_bodies`), then drops a new body every few frames via
/// `add_rigid_body`, which appends it directly to the existing GPU buffers.
pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    // Every `SPAWN_PERIOD` simulated frames, drop in a *batch* of
    // `BODIES_PER_SPAWN` bodies at once (a single batched append), up to
    // `MAX_BODIES`.
    const SPAWN_PERIOD: u32 = 2;
    const BODIES_PER_SPAWN: usize = 100;
    const MAX_BODIES: usize = 50000;
    // Spare GPU slots reserved for the ground + all dynamically-added bodies, so
    // none of the inserts trigger a full scene rebuild.
    const RESERVE: usize = MAX_BODIES + 16;
    // Horizontal grid the falling bodies are laid out on, so a whole batch is
    // inserted without self-overlap; successive layers stack upward.
    const GRID: usize = 20;

    let capacities = NexusCapacities::default().rbd_collisions(310_000);
    let mut state = NexusState::new(capacities);

    /*
     * A boxed ground: a floor plus four low walls to keep the pile contained.
     */
    let floor_half = 50.0;
    let wall_h = 4.0;
    let walls_color = Vec4::new(0.6, 0.8, 1.0, 0.3);
    let ground_parts = [
        (
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(floor_half, 0.5, floor_half),
        ),
        (
            Vec3::new(floor_half, wall_h, 0.0),
            Vec3::new(0.5, wall_h, floor_half),
        ),
        (
            Vec3::new(-floor_half, wall_h, 0.0),
            Vec3::new(0.5, wall_h, floor_half),
        ),
        (
            Vec3::new(0.0, wall_h, floor_half),
            Vec3::new(floor_half, wall_h, 0.5),
        ),
        (
            Vec3::new(0.0, wall_h, -floor_half),
            Vec3::new(floor_half, wall_h, 0.5),
        ),
    ];
    for (pos, half) in ground_parts {
        let body = RigidBodyBuilder::fixed().build();
        let collider = ColliderBuilder::cuboid(half.x, half.y, half.z)
            .translation(pos)
            .build();
        let shape = collider.shared_shape().clone();
        let handle = state.insert_rigid_body(body, collider);
        viewer.insert_shape_with_color(handle, &shape, Pose::from_translation(pos), walls_color);
    }

    // Reserve the spare slots BEFORE the first `finalize`, so the GPU buffers are
    // built large enough to append into later.
    state.reserve_rigid_bodies(RESERVE);

    let mut timestamps = GpuTimestamps::new(viewer.backend(), 2048);
    viewer
        .scene3d_mut()
        .add_directional_light(glamx::Vec3::new(1.0, -2.0, 3.0));
    state.finalize(viewer.backend()).await?;

    let mut frame: u32 = 0;
    let mut added = 0usize;

    while viewer.render_frame().await {
        if viewer.simulating() {
            // Drop in a whole batch of bodies periodically — all inserted with a
            // single batched append, without rebuilding the scene.
            if frame.is_multiple_of(SPAWN_PERIOD) && added < MAX_BODIES {
                let n = BODIES_PER_SPAWN.min(MAX_BODIES - added);
                let mut batch = Vec::with_capacity(n);
                let mut shapes = Vec::with_capacity(n);
                for k in 0..n {
                    let idx = added + k;
                    // Unique grid slot per body (a rising column high above the
                    // floor), so a simultaneously-inserted batch never overlaps.
                    let cell = idx % (GRID * GRID);
                    let gx = (cell % GRID) as f32 - (GRID as f32 - 1.0) * 0.5;
                    let gz = (cell / GRID) as f32 - (GRID as f32 - 1.0) * 0.5;
                    let layer = (idx / (GRID * GRID)) as f32;
                    let pos = Vec3::new(gx * 1.4, 18.0 + layer * 1.3, gz * 1.4);

                    let body = RigidBodyBuilder::dynamic().translation(pos).build();
                    // Alternate between balls and cubes (both primitive shapes,
                    // which is what the in-place append path supports).
                    let collider = if idx.is_multiple_of(3) {
                        ColliderBuilder::ball(0.5).build()
                    } else {
                        ColliderBuilder::cuboid(0.4, 0.4, 0.4).build()
                    };
                    shapes.push(collider.shared_shape().clone());
                    batch.push((body, collider));
                }

                let handles = state.add_rigid_bodies(viewer.backend(), batch)?;
                for (handle, shape) in handles.iter().zip(shapes.iter()) {
                    viewer.insert_shape(*handle, shape, Pose::IDENTITY);
                }
                added += n;
            }

            pipeline
                .simulate(viewer.backend(), &mut state, Some(&mut timestamps))
                .await?;
            frame += 1;
        }
        viewer.sync(&mut state, Some(&mut timestamps)).await?;
    }

    Ok(state)
}
