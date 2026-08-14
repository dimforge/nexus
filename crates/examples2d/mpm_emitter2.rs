use khal::backend::GpuTimestamps;
use nexus_viewer2d::NexusViewer;
use nexus2d::mpm::solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams};
use nexus2d::prelude::{NexusParticleChunk, NexusPipeline, NexusState, RbdCoupling};

use glamx::{Vec4, vec2};
use rapier2d::prelude::{Collider, ColliderBuilder, Pose, RigidBody, RigidBodyBuilder};

use std::collections::VecDeque;

const DENSITY: f32 = 1000.0;
const YOUNG_MODULUS: f32 = 1.0e7;
const POISSON_RATIO: f32 = 0.2;

/// Emit a fresh patch of particles every `EMIT_EVERY` substeps.
const EMIT_EVERY: u64 = 10;
/// Edge length (in particles) of the emitted block.
const EMIT_BLOCK: i32 = 120;
const EMIT_BLOCK_Y: i32 = 20;
/// Cap on the live particle count. Once reached, every emit removes as many of
/// the oldest particles as it adds, so the total stays at this budget.
const MAX_PARTICLES: usize = 250_000;

/// Inserts a boundary collider coupled (one-way) to the MPM particles and
/// registers it for rendering.
fn insert_boundary(
    state: &mut NexusState,
    viewer: &mut NexusViewer,
    body: RigidBody,
    collider: Collider,
) {
    let shape = collider.shared_shape().clone();
    let friction = BoundaryCondition::separate(1.0);
    let handle = state.insert_rigid_body(body, collider, RbdCoupling::MpmOneWay(friction));
    viewer.insert_shape(handle, &shape, Pose::IDENTITY);
}

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let mut state = NexusState::default();

    let cell_width = 0.2;
    let dt = 1.0 / 60.0;

    let params = SimulationParams {
        gravity: vec2(0.0, -9.81),
        padding: 0.0,
        dt,
    };
    state.set_mpm_params(viewer.backend(), params, cell_width)?;
    state.set_mpm_substeps(10);
    // No particles up-front: they are emitted dynamically in the loop below.

    /*
     * Boundary colliders: a floor, two side walls, and a pair of angled ramps
     * in the middle for the sand stream to cascade over.
     */
    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::fixed()
            .translation(vec2(40.0, -1.0))
            .build(),
        ColliderBuilder::cuboid(45.0, 1.0).build(),
    );
    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::fixed()
            .translation(vec2(-4.0, 30.0))
            .build(),
        ColliderBuilder::cuboid(1.0, 32.0).build(),
    );
    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::fixed()
            .translation(vec2(84.0, 30.0))
            .build(),
        ColliderBuilder::cuboid(1.0, 32.0).build(),
    );
    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::fixed()
            .translation(vec2(25.0, 20.0))
            .rotation(-0.5)
            .build(),
        ColliderBuilder::cuboid(12.0, 0.8).build(),
    );
    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::fixed()
            .translation(vec2(58.0, 12.0))
            .rotation(0.5)
            .build(),
        ColliderBuilder::cuboid(12.0, 0.8).build(),
    );
    viewer.set_particle_group_colors(&[Vec4::new(0.95, 0.75, 0.35, 1.0)]);

    let mut timestamps = GpuTimestamps::new(viewer.backend(), 2048);
    state.finalize(viewer.backend()).await?;

    /*
     * Dynamic emitter: a small block of sand spawned at a point that sweeps
     * left and right across the top, pouring a moving curtain of sand.
     */
    let radius = cell_width / 4.0;
    let spacing = radius * 2.0;
    let model = ParticleModel::sand(YOUNG_MODULUS, POISSON_RATIO);
    let emit_height = 55.0;
    let sweep_center = 40.0;
    let sweep_amplitude = 30.0;
    let sweep_speed = 0.9; // rad/s

    // Oldest-first queue of live chunks paired with their particle counts, plus
    // the running total `MAX_PARTICLES` is enforced against.
    let mut chunks: VecDeque<(NexusParticleChunk, usize)> = VecDeque::new();
    let mut total_particles: usize = 0;
    let mut t: f32 = 0.0;
    let mut step: u64 = 0;

    while viewer.render_frame().await {
        if viewer.simulating() {
            if step.is_multiple_of(EMIT_EVERY) && total_particles < MAX_PARTICLES {
                let phase = t * sweep_speed;
                let center = vec2(sweep_center + sweep_amplitude * phase.sin(), emit_height);
                let velocity = vec2(0.0, -12.0);

                let mut particles = Vec::with_capacity((EMIT_BLOCK * EMIT_BLOCK_Y) as usize);
                for i in 0..EMIT_BLOCK {
                    for j in 0..EMIT_BLOCK_Y {
                        let offset =
                            vec2((i - EMIT_BLOCK / 2) as f32, (j - EMIT_BLOCK_Y / 2) as f32)
                                * spacing;
                        let mut particle = Particle::new(center + offset, radius, DENSITY, model);
                        particle.dynamics.velocity = velocity;
                        particles.push(particle);
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
