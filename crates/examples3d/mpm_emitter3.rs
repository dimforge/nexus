use khal::backend::GpuTimestamps;
use nexus_viewer3d::NexusViewer;
use nexus3d::mpm::solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams};
use nexus3d::prelude::{NexusParticleChunk, NexusPipeline, NexusState, RbdCoupling};

use glamx::{Vec4, vec3};
use rapier3d::prelude::{ColliderBuilder, Pose, RigidBodyBuilder};

use std::collections::VecDeque;

const DENSITY: f32 = 2700.0;
const YOUNG_MODULUS: f32 = 2.0e8;
const POISSON_RATIO: f32 = 0.2;

/// Emit a fresh patch of particles every `EMIT_EVERY` substeps.
const EMIT_EVERY: u64 = 10;
/// Edge length (in particles) of the emitted cube.
const EMIT_BLOCK: i32 = 30;
const EMIT_BLOCK_Y: i32 = 2;
/// Cap on the live particle count. Once reached, every emit removes as many of
/// the oldest particles as it adds, so the total stays at this budget (the
/// settled sand erodes behind the orbiting emitter like a comet tail).
const MAX_PARTICLES: usize = 250_000;

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let mut state = NexusState::default();
    // MPM boundary colliders are inserted as rigid bodies coupled (one-way) to
    // the continuum: they push the particles but aren't pushed back.
    let coupling = RbdCoupling::MpmOneWay(BoundaryCondition::separate(1.0));

    let cell_width = 1.0;
    let dt = 1.0 / 60.0;

    let params = SimulationParams {
        gravity: vec3(0.0, -9.81, 0.0),
        dt,
    };
    state.set_mpm_params(viewer.backend(), params, cell_width)?;
    state.set_mpm_substeps(10);
    // No particles up-front: they are emitted dynamically in the loop below.

    /*
     * Boundary colliders: a floor and four walls forming an open box that
     * catches the falling sand.
     */
    let thickness = 0.5;
    let walls_color = Vec4::new(0.6, 0.8, 1.0, 0.3);
    let walls = [
        (vec3(0.0, -thickness, 0.0), vec3(30.0, thickness, 30.0)),
        (vec3(0.0, 10.0, -30.0), vec3(30.0, 10.0, thickness)),
        (vec3(0.0, 10.0, 30.0), vec3(30.0, 10.0, thickness)),
        (vec3(-30.0, 10.0, 0.0), vec3(thickness, 10.0, 30.0)),
        (vec3(30.0, 10.0, 0.0), vec3(thickness, 10.0, 30.0)),
    ];
    for (pos, half_extents) in walls {
        let body = RigidBodyBuilder::fixed().translation(pos).build();
        let collider =
            ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z).build();
        let shape = collider.shared_shape().clone();
        let handle = state.insert_rigid_body(body, collider, coupling);
        viewer.insert_shape_with_color(handle, &shape, Pose::IDENTITY, walls_color);
    }

    let mut timestamps = GpuTimestamps::new(viewer.backend(), 2048);
    viewer
        .scene3d_mut()
        .add_directional_light(glamx::Vec3::new(1.0, -2.0, 3.0));
    state.finalize(viewer.backend()).await?;

    /*
     * Dynamic emitter: a small cube of sand spawned at a point that orbits the
     * center, so the stream paints a moving ring of sand into the box.
     */
    let radius = cell_width / 4.0;
    let spacing = radius * 2.0;
    let model = ParticleModel::sand(YOUNG_MODULUS, POISSON_RATIO);
    let emit_height = 40.0;
    let orbit_radius = 10.0;
    let angular_speed = 1.5; // rad/s

    // Oldest-first queue of live chunks paired with their particle counts, plus
    // the running total `MAX_PARTICLES` is enforced against.
    let mut chunks: VecDeque<(NexusParticleChunk, usize)> = VecDeque::new();
    let mut total_particles: usize = 0;
    let mut t: f32 = 0.0;
    let mut step: u64 = 0;

    while viewer.render_frame().await {
        if viewer.simulating() {
            if step.is_multiple_of(EMIT_EVERY) && total_particles < MAX_PARTICLES {
                let angle = t * angular_speed;
                let center = vec3(
                    orbit_radius * angle.cos(),
                    emit_height,
                    orbit_radius * angle.sin(),
                );

                let mut particles =
                    Vec::with_capacity((EMIT_BLOCK as usize).pow(2) * EMIT_BLOCK_Y as usize);
                for i in 0..EMIT_BLOCK {
                    for j in 0..EMIT_BLOCK_Y {
                        for k in 0..EMIT_BLOCK {
                            let offset = vec3(
                                (i - EMIT_BLOCK / 2) as f32,
                                (j - EMIT_BLOCK_Y / 2) as f32,
                                (k - EMIT_BLOCK / 2) as f32,
                            ) * spacing;
                            let mut particle =
                                Particle::new(center + offset, radius, DENSITY, model);
                            particle.dynamics.velocity = vec3(0.0, -8.0, 0.0);
                            particles.push(particle);
                        }
                    }
                }
                let n = particles.len();
                let chunk = state.add_particles(viewer.backend(), particles)?;
                chunks.push_back((chunk, n));
                total_particles += n;
            }

            pipeline
                .simulate(viewer.backend(), &mut state, Some(&mut timestamps))
                .await?;
            t += dt;
            step += 1;
        }
        viewer.sync(&mut state, Some(&mut timestamps)).await?;
    }

    Ok(state)
}
