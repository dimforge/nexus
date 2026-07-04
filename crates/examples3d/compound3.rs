use khal::backend::GpuTimestamps;
use nexus_viewer3d::NexusViewer;
use nexus3d::prelude::{NexusCapacities, NexusPipeline, NexusState};
use rapier3d::prelude::*;

/// Port of rapier's `compound3` demo, but every "U"-shaped body is assembled
/// from THREE separate colliders attached to one rigid body — exercising
/// multiple-colliders-per-body support — instead of a single compound collider.
pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let capacities = NexusCapacities::default().rbd_collisions(280_000);
    let mut state = NexusState::new(capacities);

    /*
     * Floor made of large cuboids.
     */
    {
        let thick = 50.0;
        let height = 7.0;
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

    /*
     * "U"-shaped bodies, each made of three cuboid colliders: a horizontal base
     * and two vertical walls, forming an upward-opening cup. `(local_pose,
     * half_extents)` for the three parts.
     */
    let rad = 0.2f32;
    let parts: [(Vec3, Vec3); 3] = [
        (Vec3::ZERO, Vec3::new(rad * 10.0, rad, rad)),
        (
            Vec3::new(rad * 10.0, rad * 10.0, 0.0),
            Vec3::new(rad, rad * 10.0, rad),
        ),
        (
            Vec3::new(-rad * 10.0, rad * 10.0, 0.0),
            Vec3::new(rad, rad * 10.0, rad),
        ),
    ];

    let num = 10;
    let numy = 100;
    // Each U spans ~4 units in x; space the grid out so they don't start
    // interpenetrating.
    let shift = rad * 10.0 * 2.0 + 1.0;
    let center = shift * (num as f32) / 2.0;

    for j in 0..numy {
        for i in 0..num {
            for k in 0..num {
                let x = i as f32 * shift - center;
                let y = j as f32 * shift + 5.0;
                let z = k as f32 * shift - center;

                let body = RigidBodyBuilder::dynamic()
                    .translation(Vec3::new(x, y, z))
                    .build();

                let handle = state.insert_body_in(0, body);
                for (offset, he) in parts {
                    let collider = ColliderBuilder::cuboid(he.x, he.y, he.z)
                        .translation(offset)
                        .build();
                    viewer.insert_shape(
                        handle,
                        collider.shared_shape(),
                        Pose::from_translation(offset),
                    );
                    state.insert_collider_in(0, collider, Some(handle));
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
