//! The classic dam break: a column of water collapses and runs along a tank.

use khal::backend::GpuTimestamps;
use nexus_viewer2d::NexusViewer;
use nexus2d::mpm::solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams};
use nexus2d::prelude::{NexusPipeline, NexusState, RbdCoupling};

use glamx::{Vec4, vec2};
use rapier2d::prelude::{Collider, ColliderBuilder, Pose, RigidBody, RigidBodyBuilder};

const DENSITY: f32 = 1000.0;
/// Deepest the water gets. The bulk modulus follows from it.
const MAX_DEPTH: f32 = 22.0;
/// Volume loss tolerated at that depth. Water's real bulk modulus (~2.2 GPa)
/// would force an impractically small timestep; a weakly compressible scheme
/// instead picks the softest fluid whose compression is still invisible.
const MAX_COMPRESSION: f32 = 0.01;

const TANK_HALF_WIDTH: f32 = 30.0;
const TANK_HEIGHT: f32 = 26.0;

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

    let cell_width = 0.2;
    let radius = cell_width / 4.0;
    let spacing = radius * 2.0;
    let model = ParticleModel::water_for_depth(DENSITY, 9.81, MAX_DEPTH, MAX_COMPRESSION);

    /*
     * The water column, held against the left wall.
     */
    let mut particles = vec![];
    let column_width = 32.0;
    let column_height = 25.0;
    let nx = (column_width / spacing) as i32;
    let ny = (column_height / spacing) as i32;
    let shades: Vec<_> = (0..ny)
        .map(|j| {
            let t = j as f32 / ny as f32;
            Vec4::new(0.10 + 0.35 * t, 0.45 + 0.35 * t, 0.85 + 0.15 * t, 1.0)
        })
        .collect();
    viewer.set_particle_group_colors(&shades);
    for i in 0..nx {
        for j in 0..ny {
            let position = vec2(
                -TANK_HALF_WIDTH + 0.5 + (i as f32 + 0.5) * spacing,
                0.5 + (j as f32 + 0.5) * spacing,
            );
            // One group per row: shading by initial height makes the
            // overturning of the surge front visible once the column collapses.
            particles.push(Particle::with_group(
                position, radius, DENSITY, model, j as u32,
            ));
        }
    }

    let params = SimulationParams {
        gravity: vec2(0.0, -9.81),
        padding: 0.0,
        dt: 1.0 / 60.0,
    };
    state.set_mpm_params(viewer.backend(), params, cell_width)?;
    state.set_mpm_substeps(25);
    state.add_particles(viewer.backend(), particles)?;

    /*
     * Tank.
     */
    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::fixed()
            .translation(vec2(0.0, -1.0))
            .build(),
        ColliderBuilder::cuboid(TANK_HALF_WIDTH + 1.0, 1.0).build(),
    );
    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::fixed()
            .translation(vec2(0.0, TANK_HEIGHT + 1.0))
            .build(),
        ColliderBuilder::cuboid(TANK_HALF_WIDTH + 1.0, 1.0).build(),
    );
    for side in [-1.0f32, 1.0] {
        insert_boundary(
            &mut state,
            viewer,
            RigidBodyBuilder::fixed()
                .translation(vec2(side * (TANK_HALF_WIDTH + 1.0), TANK_HEIGHT / 2.0))
                .build(),
            ColliderBuilder::cuboid(1.0, TANK_HEIGHT / 2.0 + 1.0).build(),
        );
    }

    /*
     * An obstacle in the path of the surge, which the front breaks over.
     */
    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::fixed()
            .translation(vec2(8.0, 2.0))
            .build(),
        ColliderBuilder::cuboid(1.5, 2.0).build(),
    );

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
