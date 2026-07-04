use khal::backend::GpuTimestamps;
use nexus_viewer2d::NexusViewer;
use nexus2d::prelude::{NexusPipeline, NexusState};
use rapier2d::prelude::*;

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
    const BODIES_PER_SPAWN: usize = 50;
    const MAX_BODIES: usize = 20000;
    // Spare GPU slots reserved for the ground + all dynamically-added bodies, so
    // none of the inserts trigger a full scene rebuild.
    const RESERVE: usize = MAX_BODIES + 16;
    // Horizontal row the falling bodies are laid out on, so a whole batch is
    // inserted without self-overlap; successive batches stack upward.
    const ROW: usize = 50;

    let mut state = NexusState::default();

    /*
     * A boxed ground: a floor plus two side walls to keep the pile contained.
     */
    let floor_half = 60.0;
    let wall_h = 60.0;
    let ground_parts = [
        (Vec2::new(0.0, -1.5), Vec2::new(floor_half, 1.5)),
        (Vec2::new(floor_half, wall_h), Vec2::new(1.5, wall_h)),
        (Vec2::new(-floor_half, wall_h), Vec2::new(1.5, wall_h)),
    ];
    for (pos, half) in ground_parts {
        let body = RigidBodyBuilder::fixed().build();
        let collider = ColliderBuilder::cuboid(half.x, half.y)
            .translation(pos)
            .build();
        let shape = collider.shared_shape().clone();
        let handle = state.insert_rigid_body(body, collider);
        viewer.insert_shape(handle, &shape, Pose::from_translation(pos));
    }

    // Reserve the spare slots BEFORE the first `finalize`, so the GPU buffers are
    // built large enough to append into later.
    state.reserve_rigid_bodies(RESERVE);

    let mut timestamps = GpuTimestamps::new(viewer.backend(), 2048);
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
                    // Unique row slot per body (a rising line high above the
                    // floor), so a simultaneously-inserted batch never overlaps.
                    let cell = idx % ROW;
                    let gx = cell as f32 - (ROW as f32 - 1.0) * 0.5;
                    let layer = (idx / ROW) as f32;
                    let pos = Vec2::new(gx * 1.4, 40.0 + layer * 1.3);

                    let body = RigidBodyBuilder::dynamic().translation(pos).build();
                    // Alternate between balls and cubes (both primitive shapes,
                    // which is what the in-place append path supports).
                    let collider = if idx.is_multiple_of(3) {
                        ColliderBuilder::ball(0.5).build()
                    } else {
                        ColliderBuilder::cuboid(0.4, 0.4).build()
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
