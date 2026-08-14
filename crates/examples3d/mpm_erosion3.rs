//! Water poured onto a cohesive sand mound, washing it away.

use khal::backend::GpuTimestamps;
use nexus_viewer3d::NexusViewer;
use nexus3d::mpm::solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams};
use nexus3d::prelude::{NexusParticleChunk, NexusPipeline, NexusState, RbdCoupling};

use glamx::{Vec4, vec3};
use rapier3d::prelude::{ColliderBuilder, Pose, RigidBodyBuilder};

use std::collections::VecDeque;

const SAND_DENSITY: f32 = 1800.0;
const SAND_YOUNG_MODULUS: f32 = 1.0e7;
const SAND_POISSON_RATIO: f32 = 0.2;
/// Cohesive shear strength of the sand (Pa).
const SAND_SHEAR_STRENGTH: f32 = 3.0e3;

/// How much of the basin's friction the water feels.
const WATER_BOUNDARY_FRICTION: f32 = 0.05;

const WATER_DENSITY: f32 = 1000.0;
/// Depth the pooled water reaches in the basin. The bulk modulus follows from it.
const WATER_DEPTH: f32 = 8.0;

/// Emit a fresh slug of water every `EMIT_EVERY` steps.
const EMIT_EVERY: u64 = 3;
/// Cap on the live water particle count.
const MAX_WATER_PARTICLES: usize = 200_000;

/// Radius of the sand mound at its base.
const MOUND_RADIUS: f32 = 14.0;
/// Height of the sand mound.
const MOUND_HEIGHT: f32 = 12.0;

/// The sand bands take `SAND_LIGHT` and `SAND_LIGHT + 1`.
const SAND_LIGHT: u32 = 0;
const WATER: u32 = 2;

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let mut state = NexusState::default();
    let coupling = RbdCoupling::MpmOneWay(BoundaryCondition::separate(0.8));

    let cell_width = 0.6;
    let radius = cell_width / 4.0;
    let spacing = radius * 2.0;
    let dt = 1.0 / 60.0;

    /*
     * The sand mound: a cone, sampled by rejection.
     */
    let sand_model = ParticleModel::cohesive_sand_with_strength(
        SAND_YOUNG_MODULUS,
        SAND_POISSON_RATIO,
        SAND_SHEAR_STRENGTH,
    );
    viewer.set_particle_group_colors(&[
        Vec4::new(0.85, 0.70, 0.45, 1.0),
        Vec4::new(0.72, 0.56, 0.34, 1.0),
        Vec4::new(0.25, 0.55, 0.95, 1.0),
    ]);

    let mut particles = vec![];
    let n = (MOUND_RADIUS / spacing).ceil() as i32;
    let ny = (MOUND_HEIGHT / spacing).ceil() as i32;
    for i in -n..=n {
        for j in 0..ny {
            for k in -n..=n {
                let y = (j as f32 + 0.5) * spacing;
                // Cone: the allowed radius shrinks linearly with height.
                let max_radius = MOUND_RADIUS * (1.0 - y / MOUND_HEIGHT);
                let x = i as f32 * spacing;
                let z = k as f32 * spacing;
                if x * x + z * z > max_radius * max_radius {
                    continue;
                }
                // Horizontal bands, so the erosion depth stays readable.
                let band = ((j / 6) % 2) as u32;
                particles.push(Particle::with_group(
                    vec3(x, y, z),
                    radius,
                    SAND_DENSITY,
                    sand_model,
                    SAND_LIGHT + band,
                ));
            }
        }
    }

    let params = SimulationParams {
        gravity: vec3(0.0, -9.81, 0.0),
        dt,
    };
    state.set_mpm_params(viewer.backend(), params, cell_width)?;
    state.set_mpm_substeps(20);
    state.add_particles(viewer.backend(), particles)?;

    /*
     * Basin.
     */
    let walls_color = Vec4::new(0.6, 0.8, 1.0, 0.3);
    let basin = 20.0;
    let walls = [
        (vec3(0.0, -1.0, 0.0), vec3(basin, 1.0, basin)),
        (vec3(0.0, 6.0, -basin), vec3(basin, 7.0, 1.0)),
        (vec3(0.0, 6.0, basin), vec3(basin, 7.0, 1.0)),
        (vec3(-basin, 6.0, 0.0), vec3(1.0, 7.0, basin)),
        (vec3(basin, 6.0, 0.0), vec3(1.0, 7.0, basin)),
    ];
    for (pos, half_extents) in walls {
        let collider =
            ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z).build();
        let shape = collider.shared_shape().clone();
        let body = RigidBodyBuilder::fixed().translation(pos).build();
        let handle = state.insert_rigid_body(body, collider, coupling);
        viewer.insert_shape_with_color(handle, &shape, Pose::IDENTITY, walls_color);
    }

    let mut timestamps = GpuTimestamps::new(viewer.backend(), 2048);
    viewer
        .scene3d_mut()
        .add_directional_light(glamx::Vec3::new(1.0, -2.0, 3.0));
    state.finalize(viewer.backend()).await?;

    /*
     * The jet: a small slab of water spawned above the mound, aimed at the flank
     * so the crater migrates instead of drilling straight down.
     */
    let water_model = ParticleModel::water_for_depth(WATER_DENSITY, 9.81, WATER_DEPTH, 0.01);
    let jet_extent = 5;
    let mut chunks: VecDeque<(NexusParticleChunk, usize)> = VecDeque::new();
    let mut live_water = 0usize;
    let mut t = 0.0f32;
    let mut step = 0u64;

    while viewer.render_frame().await {
        if viewer.simulating() {
            if step.is_multiple_of(EMIT_EVERY) && live_water < MAX_WATER_PARTICLES {
                // Sweep the jet slowly back and forth across the mound.
                let center = vec3((t * 1.35).sin() * 8.0, 26.0, 4.0);
                let mut water = Vec::new();
                for i in -jet_extent..=jet_extent {
                    for j in 0..2 {
                        for k in -jet_extent..=jet_extent {
                            let offset = vec3(i as f32, j as f32, k as f32) * spacing;
                            let mut particle = Particle::with_group(
                                center + offset,
                                radius,
                                WATER_DENSITY,
                                water_model,
                                WATER,
                            );
                            particle.dynamics.velocity = vec3(0.0, -14.0, 0.0);
                            particle
                                .dynamics
                                .set_boundary_friction(WATER_BOUNDARY_FRICTION);
                            water.push(particle);
                        }
                    }
                }

                let n = water.len();
                let chunk = state.add_particles(viewer.backend(), water)?;
                chunks.push_back((chunk, n));
                live_water += n;
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
