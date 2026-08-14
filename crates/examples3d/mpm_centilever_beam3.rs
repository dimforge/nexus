use khal::backend::GpuTimestamps;
use nexus_viewer3d::NexusViewer;
use nexus3d::mpm::solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams};
use nexus3d::prelude::{NexusPipeline, NexusState, RbdCoupling};

use glamx::vec3;
use rapier3d::prelude::{ColliderBuilder, Pose, RigidBodyBuilder};

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let mut state = NexusState::default();
    let coupling = RbdCoupling::MpmOneWay(BoundaryCondition::stick());

    let width = 10.0;
    let height = 2.0;
    let fixed_part = 1.0;
    let cell_width = 0.2;
    let particle_per_cell_dim = 2;
    let young_modulus = 1.0e7;
    let poisson_ratio = 0.3;

    let diameter = cell_width / particle_per_cell_dim as f32;
    let ni = ((width + fixed_part) / diameter).ceil() as usize;
    let njk = (height / diameter).ceil() as usize;

    let mut particles = vec![];
    for i in 0..ni {
        for j in 0..njk {
            for k in 0..njk {
                let position = vec3(i as f32, j as f32, k as f32) * diameter;
                let density = 1000.0;
                let radius = diameter / 2.0;
                let model = ParticleModel::elastic_neo_hookean(young_modulus, poisson_ratio);
                let mut particle = Particle::new(position, radius, density, model);
                particle.dynamics.set_damping(2.0);
                particles.push(particle);
            }
        }
    }

    let params = SimulationParams {
        gravity: vec3(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
    };
    state.set_mpm_params(viewer.backend(), params, cell_width)?;
    state.set_mpm_substeps(20);
    state.add_particles(viewer.backend(), particles)?;

    // Fixed block that clamps one end of the beam.
    let body = RigidBodyBuilder::fixed()
        .translation(vec3(0.0, height / 2.0, height / 2.0))
        .build();
    let collider = ColliderBuilder::cuboid(fixed_part, height, height).build();
    let shape = collider.shared_shape().clone();
    let handle = state.insert_rigid_body(body, collider, coupling);
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
}
