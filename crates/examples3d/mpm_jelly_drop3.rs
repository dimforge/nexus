//! A row of neo-Hookean blobs of increasing stiffness dropped onto a floor.

use khal::backend::GpuTimestamps;
use nexus_viewer3d::NexusViewer;
use nexus3d::mpm::solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams};
use nexus3d::prelude::{NexusPipeline, NexusState, RbdCoupling};

use glamx::{Vec4, vec3};
use rapier3d::prelude::{Collider, ColliderBuilder, Pose, RigidBody, RigidBodyBuilder};

const DENSITY: f32 = 1000.0;
const POISSON_RATIO: f32 = 0.3;

/// Young modulus of each blob, left to right.
const YOUNG_MODULI: [f32; 5] = [2.0e5, 5.0e5, 1.5e6, 5.0e6, 2.0e7];
/// Radius of each blob.
const BLOB_RADIUS: f32 = 3.0;
/// Horizontal spacing between blob centers.
const BLOB_SPACING: f32 = 8.0;

fn insert_boundary(
    state: &mut NexusState,
    viewer: &mut NexusViewer,
    body: RigidBody,
    collider: Collider,
) {
    let shape = collider.shared_shape().clone();
    let coupling = RbdCoupling::MpmOneWay(BoundaryCondition::separate(0.5));
    let handle = state.insert_rigid_body(body, collider, coupling);
    viewer.insert_shape(handle, &shape, Pose::IDENTITY);
}

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let mut state = NexusState::default();

    let cell_width = 0.5;
    let radius = cell_width / 4.0;
    let spacing = radius * 2.0;

    /*
     * One blob per stiffness, sampled as a ball of particles.
     */
    let mut particles = vec![];
    let n = (BLOB_RADIUS / spacing).ceil() as i32;
    let count = YOUNG_MODULI.len();
    // One group per blob: the softest is warm, the stiffest is cold.
    let shades: Vec<_> = (0..count)
        .map(|b| {
            let t = b as f32 / (count - 1) as f32;
            Vec4::new(0.95 - 0.7 * t, 0.45, 0.25 + 0.65 * t, 1.0)
        })
        .collect();
    viewer.set_particle_group_colors(&shades);
    for (b, young_modulus) in YOUNG_MODULI.iter().enumerate() {
        let center = vec3(
            (b as f32 - (count - 1) as f32 / 2.0) * BLOB_SPACING,
            10.0,
            0.0,
        );
        let model = ParticleModel::elastic_neo_hookean(*young_modulus, POISSON_RATIO);

        for i in -n..=n {
            for j in -n..=n {
                for k in -n..=n {
                    let offset = vec3(i as f32, j as f32, k as f32) * spacing;
                    if offset.length() > BLOB_RADIUS {
                        continue;
                    }
                    particles.push(Particle::with_group(
                        center + offset,
                        radius,
                        DENSITY,
                        model,
                        b as u32,
                    ));
                }
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

    /*
     * Floor.
     */
    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::fixed()
            .translation(vec3(0.0, -1.0, 0.0))
            .build(),
        ColliderBuilder::cuboid(40.0, 1.0, 20.0).build(),
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
