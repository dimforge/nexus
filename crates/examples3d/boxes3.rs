use khal::backend::GpuTimestamps;
use nexus_viewer3d::NexusViewer;
use nexus3d::prelude::{NexusCapacities, NexusPipeline, NexusState};
use rapier3d::prelude::*;

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    const NXZ: isize = 30;
    const NY: isize = 70;

    let capacities = NexusCapacities::default().rbd_collisions(500_000);
    let mut state = NexusState::new(capacities);

    /*
     * Falling dynamic objects.
     */
    for j in 0..NY {
        let max_ik = NXZ / 2;
        for i in -max_ik..max_ik {
            for k in -max_ik..max_ik {
                let x = i as f32 * 1.1 + j as f32 * 0.01;
                let y = j as f32 * 1.1 + 0.6;
                let z = k as f32 * 1.1 + j as f32 * 0.01;
                let pos = Vec3::new(x, y, z);

                let body = RigidBodyBuilder::dynamic().translation(pos).build();
                let collider = ColliderBuilder::cuboid(0.5, 0.5, 0.5).build();
                let shape = collider.shared_shape().clone();
                let handle = state.insert_rigid_body(body, collider);
                viewer.insert_shape(handle, &shape, Pose::IDENTITY);
            }
        }
    }

    /*
     * Floor made of large cuboids.
     */
    {
        let thick = NXZ as f32 * 1.5;
        let height = 12.0;
        let walls_color = Vec4::new(0.6, 0.8, 1.0, 0.3);
        let walls = [
            (Vec3::new(0.0, -0.5, 0.0), Vec3::new(thick, 0.5, thick)),
            (Vec3::new(thick, height, 0.0), Vec3::new(0.5, height, thick)),
            (
                Vec3::new(-thick, height, 0.0),
                Vec3::new(0.5, height, thick),
            ),
            (Vec3::new(0.0, height, thick), Vec3::new(thick, height, 0.5)),
            (
                Vec3::new(0.0, height, -thick),
                Vec3::new(thick, height, 0.5),
            ),
        ];

        for (wall_pos, wall_sz) in walls {
            let body = RigidBodyBuilder::fixed().build();
            let collider = ColliderBuilder::cuboid(wall_sz.x, wall_sz.y, wall_sz.z)
                .translation(wall_pos)
                .build();
            let shape = collider.shared_shape().clone();
            let handle = state.insert_rigid_body(body, collider);
            viewer.insert_shape_with_color(
                handle,
                &shape,
                Pose::from_translation(wall_pos),
                walls_color,
            );
        }
    }

    let mut timestamps = GpuTimestamps::new(viewer.backend(), 2048);
    viewer
        .scene3d_mut()
        .add_directional_light(glamx::Vec3::new(1.0, -2.0, 3.0));
    // Optional finalize, useful so we can render even before starting the simulation.
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
