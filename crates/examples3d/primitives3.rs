use khal::backend::GpuTimestamps;
use nexus_viewer3d::NexusViewer;
use nexus3d::prelude::{NexusCapacities, NexusPipeline, NexusState};
use rapier3d::prelude::*;

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    const NXZ: isize = 20;
    const NY: isize = 40;

    let capacities = NexusCapacities::default().rbd_collisions(150_000);
    let mut state = NexusState::new(capacities);

    /*
     * Falling dynamic objects.
     */
    // Create 5 predefined convex polyhedron shapes (so we can render
    // them efficiently with instancing).
    let polyhedron_shapes: Vec<SharedShape> = {
        let mut shapes = Vec::new();
        let mut rng = oorandom::Rand32::new(42);

        for _ in 0..5 {
            let mut points = Vec::new();
            let scale = 2.0;
            for _ in 0..10 {
                let pt = Vec3::new(
                    rng.rand_float() - 0.5,
                    rng.rand_float() - 0.5,
                    rng.rand_float() - 0.5,
                );
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

    for j in 0..NY {
        let max_ik = NXZ / 2;
        for i in -max_ik..max_ik {
            for k in -max_ik..max_ik {
                let x = i as f32 * 1.1 + j as f32 * 0.01;
                let y = j as f32 * 1.6 + 1.0;
                let z = k as f32 * 1.1 + j as f32 * 0.01;
                let pos = Vec3::new(x, y, z);

                let collider = match j % 6 {
                    0 => ColliderBuilder::cylinder(0.5, 0.5),
                    1 => ColliderBuilder::cuboid(0.5, 0.5, 0.5),
                    2 => ColliderBuilder::cone(0.5, 0.5),
                    3 => ColliderBuilder::capsule_y(0.4, 0.4),
                    4 => ColliderBuilder::ball(0.5),
                    _ => {
                        if i % 2 == 0 || k % 2 == 0 {
                            continue;
                        }
                        // Reuse one of the 5 predefined polyhedron shapes
                        let shape_idx = ((i + max_ik) as usize + (k + max_ik) as usize)
                            % polyhedron_shapes.len();
                        ColliderBuilder::new(polyhedron_shapes[shape_idx].clone())
                    }
                }
                .build();

                let body = RigidBodyBuilder::dynamic().translation(pos).build();
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
        let thick = NXZ as f32 * 1.3;
        let height = 5.0;
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
