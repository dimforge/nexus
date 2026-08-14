//! A three-dimensional dam break running past a square column.
//!
//! The obstacle is offset from the tank centerline, so the surge wraps around it
//! asymmetrically and the two arms collide again downstream.

use khal::backend::GpuTimestamps;
use nexus_viewer3d::NexusViewer;
use nexus3d::mpm::solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams};
use nexus3d::prelude::{NexusPipeline, NexusState, RbdCoupling};

use glamx::{Vec4, vec3};
use rapier3d::prelude::{Collider, ColliderBuilder, Pose, RigidBody, RigidBodyBuilder};

const DENSITY: f32 = 1000.0;
/// Deepest the water gets. The bulk modulus follows from it.
const MAX_DEPTH: f32 = 16.0;
/// See the note in the 2D dam break: the softest fluid whose compression at that
/// depth is still invisible.
const MAX_COMPRESSION: f32 = 0.01;

const TANK_HALF_X: f32 = 30.0;
const TANK_HALF_Z: f32 = 14.0;
const TANK_HEIGHT: f32 = 28.0;

fn insert_boundary(
    state: &mut NexusState,
    viewer: &mut NexusViewer,
    body: RigidBody,
    collider: Collider,
) {
    let shape = collider.shared_shape().clone();
    let coupling = RbdCoupling::MpmOneWay(BoundaryCondition::separate(0.0));
    let handle = state.insert_rigid_body(body, collider, coupling);
    viewer.insert_shape(handle, &shape, Pose::IDENTITY);
}

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let mut state = NexusState::default();

    let cell_width = 0.6;
    let radius = cell_width / 4.0;
    let spacing = radius * 2.0;
    let model = ParticleModel::water_for_depth(DENSITY, 9.81, MAX_DEPTH, MAX_COMPRESSION);

    /*
     * The water column, held against the -X wall.
     */
    let mut particles = vec![];
    let nx = (14.0 / spacing) as i32;
    let ny = (16.0 / spacing) as i32;
    let nz = (TANK_HALF_Z * 2.0 / spacing) as i32;
    // One group per row, shaded by initial height, so the overturning of the
    // surge front stays visible.
    let shades: Vec<_> = (0..ny)
        .map(|j| {
            let t = j as f32 / ny as f32;
            Vec4::new(0.10 + 0.35 * t, 0.45 + 0.35 * t, 0.85 + 0.15 * t, 1.0)
        })
        .collect();
    viewer.set_particle_group_colors(&shades);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let position = vec3(
                    -TANK_HALF_X + 0.5 + (i as f32 + 0.5) * spacing,
                    0.5 + (j as f32 + 0.5) * spacing,
                    -TANK_HALF_Z + 0.5 + (k as f32 + 0.5) * spacing,
                );
                particles.push(Particle::with_group(
                    position, radius, DENSITY, model, j as u32,
                ));
            }
        }
    }

    let params = SimulationParams {
        gravity: vec3(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
    };
    state.set_mpm_params(viewer.backend(), params, cell_width)?;
    state.set_mpm_substeps(25);
    state.add_particles(viewer.backend(), particles)?;

    /*
     * Tank.
     */
    let walls_color = Vec4::new(0.6, 0.8, 1.0, 0.04);
    let walls = [
        (
            vec3(0.0, -1.0, 0.0),
            vec3(TANK_HALF_X + 3.0, 1.0, TANK_HALF_Z + 3.0),
        ),
        (
            vec3(0.0, TANK_HEIGHT + 1.0, 0.0),
            vec3(TANK_HALF_X + 3.0, 1.0, TANK_HALF_Z + 3.0),
        ),
        (
            vec3(-TANK_HALF_X - 1.0, TANK_HEIGHT / 2.0, 0.0),
            vec3(1.0, TANK_HEIGHT / 2.0 + 2.0, TANK_HALF_Z + 3.0),
        ),
        (
            vec3(TANK_HALF_X + 1.0, TANK_HEIGHT / 2.0, 0.0),
            vec3(1.0, TANK_HEIGHT / 2.0 + 2.0, TANK_HALF_Z + 3.0),
        ),
        (
            vec3(0.0, TANK_HEIGHT / 2.0, -TANK_HALF_Z - 1.0),
            vec3(TANK_HALF_X + 3.0, TANK_HEIGHT / 2.0 + 2.0, 1.0),
        ),
        (
            vec3(0.0, TANK_HEIGHT / 2.0, TANK_HALF_Z + 1.0),
            vec3(TANK_HALF_X + 3.0, TANK_HEIGHT / 2.0 + 3.0, 1.0),
        ),
    ];
    for (pos, half_extents) in walls {
        let collider =
            ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z).build();
        let shape = collider.shared_shape().clone();
        let body = RigidBodyBuilder::fixed().translation(pos).build();
        let coupling = RbdCoupling::MpmOneWay(BoundaryCondition::separate(0.0));
        let handle = state.insert_rigid_body(body, collider, coupling);
        viewer.insert_shape_with_color(handle, &shape, Pose::IDENTITY, walls_color);
    }

    /*
     * Column for the surge to wrap around.
     */
    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::fixed()
            .translation(vec3(2.0, 5.0, 0.0))
            .build(),
        ColliderBuilder::cuboid(2.0, 5.0, 6.0).build(),
    );

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
