use khal::backend::GpuTimestamps;
use nexus_viewer3d::NexusViewer;
use nexus3d::prelude::{NexusCapacities, NexusPipeline, NexusState};
use rapier3d::parry::utils::Array2;
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
                let y = j as f32 * 1.6 + 2.0;
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
     * A trimesh floor.
     */
    let ground_size = Vec3::new(100.0, 1.0, 100.0);
    let nsubdivs = 20;

    let heights = Array2::from_fn(nsubdivs + 1, nsubdivs + 1, |i, j| {
        if i == 0 || i == nsubdivs || j == 0 || j == nsubdivs {
            10.0
        } else {
            let x = i as f32 * ground_size.x / (nsubdivs as f32);
            let z = j as f32 * ground_size.z / (nsubdivs as f32);

            // NOTE: make sure we use the sin/cos from simba to ensure
            // cross-platform determinism of the example when the
            // enhanced_determinism feature is enabled.
            x.sin() + z.cos()
        }
    });

    // Here we will build our trimesh from the mesh representation of an
    // heightfield.
    let heightfield = HeightField::new(heights, ground_size);
    let (vertices, indices) = heightfield.to_trimesh();

    let body = RigidBodyBuilder::fixed().build();
    let collider = ColliderBuilder::trimesh_with_flags(
        vertices,
        indices,
        TriMeshFlags::MERGE_DUPLICATE_VERTICES,
    )
    .unwrap()
    .build();
    let shape = collider.shared_shape().clone();
    let handle = state.insert_rigid_body(body, collider);
    viewer.insert_shape(handle, &shape, Pose::IDENTITY);

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
