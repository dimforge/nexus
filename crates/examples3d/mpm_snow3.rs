//! Snowballs fired into a snow wall, and a plough sweeping the debris.

use khal::backend::GpuTimestamps;
use nexus_viewer3d::NexusViewer;
use nexus3d::mpm::solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams};
use nexus3d::prelude::{NexusPipeline, NexusState, RbdCoupling};

use glamx::{Vec3, Vec4, vec3};
use rapier3d::prelude::{Collider, ColliderBuilder, Pose, RigidBody, RigidBodyBuilder};

const DENSITY: f32 = 400.0;
const YOUNG_MODULUS: f32 = 1.4e5;
const POISSON_RATIO: f32 = 0.2;

const BALL_RADIUS: f32 = 3.0;

const WALL: u32 = 0;
/// The three projectiles take `FIRST_BALL`, `FIRST_BALL + 1` and `FIRST_BALL + 2`.
const FIRST_BALL: u32 = 1;

fn insert_boundary(
    state: &mut NexusState,
    viewer: &mut NexusViewer,
    body: RigidBody,
    collider: Collider,
) {
    let shape = collider.shared_shape().clone();
    let coupling = RbdCoupling::MpmOneWay(BoundaryCondition::separate(0.7));
    let handle = state.insert_rigid_body(body, collider, coupling);
    viewer.insert_shape(handle, &shape, Pose::IDENTITY);
}

/// Adds a ball of snow particles centered on `center`, moving at `velocity`.
fn add_snowball(
    particles: &mut Vec<Particle>,
    center: Vec3,
    velocity: Vec3,
    group_id: u32,
    radius: f32,
    spacing: f32,
    model: ParticleModel,
) {
    let n = (BALL_RADIUS / spacing).ceil() as i32;
    for i in -n..=n {
        for j in -n..=n {
            for k in -n..=n {
                let offset = vec3(i as f32, j as f32, k as f32) * spacing;
                if offset.length() > BALL_RADIUS {
                    continue;
                }
                let mut particle =
                    Particle::with_group(center + offset, radius, DENSITY, model, group_id);
                particle.dynamics.velocity = velocity;
                particles.push(particle);
            }
        }
    }
}

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let mut state = NexusState::default();

    let cell_width = 0.5;
    let radius = cell_width / 4.0;
    let spacing = radius * 2.0;
    let model = ParticleModel::snow(YOUNG_MODULUS, POISSON_RATIO);

    let mut group_colors = vec![Vec4::new(0.90, 0.93, 0.98, 1.0)];
    group_colors.extend((0..3).map(|n| {
        let t = n as f32 / 2.0;
        Vec4::new(0.30 + 0.55 * t, 0.55, 0.85 - 0.45 * t, 1.0)
    }));
    viewer.set_particle_group_colors(&group_colors);

    let mut particles = vec![];

    /*
     * A standing snow wall.
     */
    let wall_half_width = 14.0;
    let wall_height = 16.0;
    let wall_half_depth = 3.0;
    let nx = (wall_half_width * 2.0 / spacing) as i32;
    let ny = (wall_height / spacing) as i32;
    let nz = (wall_half_depth * 2.0 / spacing) as i32;
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let position = vec3(
                    -wall_half_width + (i as f32 + 0.5) * spacing,
                    0.2 + (j as f32 + 0.5) * spacing,
                    -wall_half_depth + (k as f32 + 0.5) * spacing,
                );
                particles.push(Particle::with_group(position, radius, DENSITY, model, WALL));
            }
        }
    }

    /*
     * Three projectiles aimed at different heights of the wall.
     */
    for (n, (height, speed)) in [(4.0f32, 26.0f32), (10.0, 30.0), (14.0, 22.0)]
        .into_iter()
        .enumerate()
    {
        add_snowball(
            &mut particles,
            vec3((n as f32 - 1.0) * 7.0, height, 26.0),
            vec3(0.0, 0.0, -speed),
            FIRST_BALL + n as u32,
            radius,
            spacing,
            model,
        );
    }

    let params = SimulationParams {
        gravity: vec3(0.0, -9.81, 0.0),
        dt: 1.0 / 60.0,
    };
    state.set_mpm_params(viewer.backend(), params, cell_width)?;
    state.set_mpm_substeps(20);
    state.add_particles(viewer.backend(), particles)?;

    /*
     * Ground.
     */
    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::fixed()
            .translation(vec3(0.0, -1.0, 0.0))
            .build(),
        ColliderBuilder::cuboid(40.0, 1.0, 40.0).build(),
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
