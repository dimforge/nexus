//! Five identical granular columns released at once, differing only in cohesion.

use khal::backend::GpuTimestamps;
use nexus_viewer2d::NexusViewer;
use nexus2d::mpm::solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams};
use nexus2d::prelude::{NexusPipeline, NexusState, RbdCoupling};

use glamx::{Vec4, vec2};
use rapier2d::prelude::{ColliderBuilder, Pose, RigidBodyBuilder};

const DENSITY: f32 = 1600.0;
const YOUNG_MODULUS: f32 = 1.0e7;
const POISSON_RATIO: f32 = 0.2;

/// Cohesion of each column, left to right.
const COHESIONS: [f32; 5] = [0.0, 0.001, 0.003, 0.01, 0.03];

/// Half-width of each column.
const COLUMN_HALF_WIDTH: f32 = 3.0;
/// Height of each column.
const COLUMN_HEIGHT: f32 = 20.0;
/// Distance between column centers.
const COLUMN_PITCH: f32 = 22.0;

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let mut state = NexusState::default();

    let cell_width = 0.2;
    let radius = cell_width / 4.0;
    let spacing = radius * 2.0;

    let mut particles = vec![];
    let nx = (COLUMN_HALF_WIDTH * 2.0 / spacing) as i32;
    let ny = (COLUMN_HEIGHT / spacing) as i32;
    let count = COHESIONS.len();
    // Dry sand is pale, the most cohesive column is dark.
    let shades: Vec<_> = (0..count)
        .map(|c| {
            let t = c as f32 / (count - 1) as f32;
            Vec4::new(0.92 - 0.5 * t, 0.80 - 0.48 * t, 0.62 - 0.42 * t, 1.0)
        })
        .collect();
    viewer.set_particle_group_colors(&shades);
    for (c, cohesion) in COHESIONS.iter().enumerate() {
        let center_x = (c as f32 - (count - 1) as f32 / 2.0) * COLUMN_PITCH;
        let model = ParticleModel::cohesive_sand(YOUNG_MODULUS, POISSON_RATIO, *cohesion);

        for i in 0..nx {
            for j in 0..ny {
                let position = vec2(
                    center_x - COLUMN_HALF_WIDTH + (i as f32 + 0.5) * spacing,
                    0.2 + (j as f32 + 0.5) * spacing,
                );
                particles.push(Particle::with_group(
                    position, radius, DENSITY, model, c as u32,
                ));
            }
        }
    }

    let params = SimulationParams {
        gravity: vec2(0.0, -9.81),
        padding: 0.0,
        dt: 1.0 / 60.0,
    };
    state.set_mpm_params(viewer.backend(), params, cell_width)?;
    state.set_mpm_substeps(15);
    state.add_particles(viewer.backend(), particles)?;

    /*
     * Setup the floor.
     */
    let half_span = COLUMN_PITCH * count as f32 / 2.0 + 10.0;
    let collider = ColliderBuilder::cuboid(half_span, 1.0).build();
    let shape = collider.shared_shape().clone();
    let body = RigidBodyBuilder::fixed()
        .translation(vec2(0.0, -1.0))
        .build();
    let coupling = RbdCoupling::MpmOneWay(BoundaryCondition::separate(1.0));
    let handle = state.insert_rigid_body(body, collider, coupling);
    viewer.insert_shape(handle, &shape, Pose::IDENTITY);

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
