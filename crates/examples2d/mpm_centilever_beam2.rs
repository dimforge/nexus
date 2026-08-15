use khal::backend::GpuTimestamps;
use nexus_viewer2d::NexusViewer;
use nexus2d::mpm::solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams};
use nexus2d::prelude::{NexusPipeline, NexusState, RbdCoupling};

use rapier2d::prelude::{ColliderBuilder, Pose, RigidBodyBuilder};

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let mut state = NexusState::default();

    let width = 10.0;
    let height = 2.0;
    let fixed_part = 1.0;
    let cell_width = 0.2;
    let particle_per_cell_dim = 2;
    let young_modulus = 1.0e8;
    let poisson_ratio = 0.3;

    let diameter = cell_width / particle_per_cell_dim as f32;
    let ni = ((width + fixed_part) / diameter).ceil() as usize;
    let nj = (height / diameter).ceil() as usize;

    let mut particles = vec![];
    for i in 0..ni {
        for j in 0..nj {
            let position = glamx::vec2(i as f32, j as f32) * diameter;
            let density = 1000.0;
            let radius = diameter / 2.0;
            let model = ParticleModel::elastic_neo_hookean(young_modulus, poisson_ratio);
            particles.push(Particle::new(position, radius, density, model));
        }
    }

    let params = SimulationParams {
        gravity: glamx::vec2(0.0, -9.81),
        padding: 0.0,
        dt: 1.0 / 60.0,
    };
    state.set_mpm_params(viewer.backend(), params, cell_width)?;
    state.set_mpm_substeps(150);
    state.add_particles(viewer.backend(), particles)?;

    // Fixed anchor the beam is cantilevered from (boundary coupled to the MPM
    // continuum).
    let body = RigidBodyBuilder::fixed()
        .translation(glamx::vec2(0.0, height / 2.0))
        .build();
    let collider = ColliderBuilder::cuboid(fixed_part, height).build();
    let shape = collider.shared_shape().clone();
    let stick = BoundaryCondition::stick();
    let handle = state.insert_rigid_body(body, collider, RbdCoupling::MpmOneWay(stick));
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
