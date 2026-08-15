//! Snowballs thrown into each other and into a snow bank.

use khal::backend::GpuTimestamps;
use nexus_viewer2d::NexusViewer;
use nexus2d::mpm::solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams};
use nexus2d::prelude::{NexusPipeline, NexusState, RbdCoupling};

use glamx::{Vec2, Vec4, vec2};
use rapier2d::prelude::{ColliderBuilder, Pose, RigidBodyBuilder};

const DENSITY: f32 = 400.0;
const YOUNG_MODULUS: f32 = 1.4e6;
const POISSON_RATIO: f32 = 0.2;

const BANK: u32 = 0;
const BLUE_BALL: u32 = 1;
const RED_BALL: u32 = 2;
const GREEN_BALL: u32 = 3;

/// Adds a disc of snow particles centered on `center`, moving at `velocity`.
#[allow(clippy::too_many_arguments)]
fn add_snowball(
    particles: &mut Vec<Particle>,
    center: Vec2,
    velocity: Vec2,
    group_id: u32,
    particle_radius: f32,
    snowball_radius: f32,
    spacing: f32,
    model: ParticleModel,
) {
    let n = (snowball_radius / spacing).ceil() as i32;
    for i in -n..=n {
        for j in -n..=n {
            let offset = vec2(i as f32, j as f32) * spacing;
            if offset.length() > snowball_radius {
                continue;
            }
            let mut particle =
                Particle::with_group(center + offset, particle_radius, DENSITY, model, group_id);
            particle.dynamics.velocity = velocity;
            particles.push(particle);
        }
    }
}

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let mut state = NexusState::default();

    let cell_width = 0.2;
    let radius = cell_width / 4.0;
    let spacing = radius * 2.0;
    let model = ParticleModel::snow(YOUNG_MODULUS, POISSON_RATIO);

    // Indexed by BANK / BLUE_BALL / RED_BALL / GREEN_BALL.
    viewer.set_particle_group_colors(&[
        Vec4::new(0.70, 0.73, 0.78, 1.0),
        Vec4::new(0.35, 0.55, 0.85, 1.0),
        Vec4::new(0.85, 0.45, 0.35, 1.0),
        Vec4::new(0.45, 0.75, 0.45, 1.0),
    ]);

    let mut particles = vec![];

    /*
     * A loose snow bank for the balls to land in.
     */
    let bank_half_width = 24.0;
    let bank_height = 20.0;
    let nx = (bank_half_width * 2.0 / spacing) as i32;
    let ny = (bank_height / spacing) as i32;
    for i in 0..nx {
        for j in 0..ny {
            let position = vec2(
                -bank_half_width + (i as f32 + 0.5) * spacing,
                0.2 + (j as f32 + 0.5) * spacing,
            );
            particles.push(Particle::with_group(position, radius, DENSITY, model, BANK));
        }
    }

    /*
     * Two balls thrown at each other above the bank, and one dropped straight
     * down onto the collision point.
     */
    add_snowball(
        &mut particles,
        vec2(-16.0, 30.0),
        vec2(18.0, 0.0),
        BLUE_BALL,
        radius,
        3.0,
        spacing,
        model,
    );
    add_snowball(
        &mut particles,
        vec2(16.0, 30.0),
        vec2(-18.0, 0.0),
        RED_BALL,
        radius,
        6.0,
        spacing,
        model,
    );
    add_snowball(
        &mut particles,
        vec2(0.0, 45.0),
        vec2(0.0, -12.0),
        GREEN_BALL,
        radius,
        9.0,
        spacing,
        model,
    );

    let params = SimulationParams {
        gravity: vec2(0.0, -9.81),
        padding: 0.0,
        dt: 1.0 / 60.0,
    };
    state.set_mpm_params(viewer.backend(), params, cell_width)?;
    state.set_mpm_substeps(25);
    state.add_particles(viewer.backend(), particles)?;

    /*
     * Ground and side walls.
     */
    let coupling = RbdCoupling::MpmOneWay(BoundaryCondition::separate(1.0));
    for (pos, half_extents) in [
        (vec2(0.0, -1.0), vec2(bank_half_width + 2.0, 1.0)),
        (vec2(-bank_half_width - 1.0, 11.0), vec2(1.0, 13.0)),
        (vec2(bank_half_width + 1.0, 11.0), vec2(1.0, 13.0)),
    ] {
        let collider = ColliderBuilder::cuboid(half_extents.x, half_extents.y).build();
        let shape = collider.shared_shape().clone();
        let body = RigidBodyBuilder::fixed().translation(pos).build();
        let handle = state.insert_rigid_body(body, collider, coupling);
        viewer.insert_shape(handle, &shape, Pose::IDENTITY);
    }

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
