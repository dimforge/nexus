use khal::backend::GpuTimestamps;
use nexus_viewer2d::NexusViewer;
use nexus2d::mpm::solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams};
use nexus2d::prelude::{NexusPipeline, NexusState, RbdCoupling};

use glamx::{Vec2, Vec4};
use rapier2d::prelude::{Collider, ColliderBuilder, Pose, RigidBody, RigidBodyBuilder};

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

    let offset_y = 46.0;
    // let cell_width = 0.1;
    let cell_width = 0.2;
    let ny = 700;

    let shades: Vec<_> = (0..ny)
        .map(|j| {
            let t = j as f32 / ny as f32;
            Vec4::new(0.95, 0.75 - 0.35 * t, 0.35 - 0.25 * t, 1.0)
        })
        .rev()
        .collect();
    viewer.set_particle_group_colors(&shades);

    let mut particles = vec![];
    for i in 0..700 {
        for j in 0..ny {
            let position =
                glamx::vec2(i as f32 + 0.5, j as f32 + 0.5) * cell_width / 2.0 + Vec2::Y * offset_y;
            let density = 1000.0;
            let radius = cell_width / 4.0;
            let young_modulus = 1.0e7;
            let poisson_ratio = 0.2;
            let model = ParticleModel::sand(young_modulus, poisson_ratio);

            particles.push(Particle::with_group(position, radius, density, model, j));
        }
    }

    let params = SimulationParams {
        gravity: glamx::vec2(0.0, -9.81),
        padding: 0.0,
        dt: 1.0 / 60.0,
    };
    state.set_mpm_params(viewer.backend(), params, cell_width)?;
    state.set_mpm_substeps(10);
    state.add_particles(viewer.backend(), particles)?;

    const ANGVEL: f32 = 2.0;

    /*
     * Static platforms.
     */
    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::fixed()
            .translation(glamx::vec2(35.0, -1.0))
            .build(),
        ColliderBuilder::cuboid(42.0, 1.0).build(),
    );
    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::fixed()
            .translation(glamx::vec2(-25.0, 45.0))
            .rotation(0.5)
            .build(),
        ColliderBuilder::cuboid(1.0, 52.0).build(),
    );
    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::fixed()
            .translation(glamx::vec2(95.0, 45.0))
            .rotation(-0.5)
            .build(),
        ColliderBuilder::cuboid(1.0, 52.0).build(),
    );

    /*
     * Rotating platforms.
     */
    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::kinematic_velocity_based()
            .translation(glamx::vec2(5.0, 35.0))
            .angvel(ANGVEL)
            .build(),
        ColliderBuilder::cuboid(1.0, 10.0).build(),
    );
    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::kinematic_velocity_based()
            .translation(glamx::vec2(35.0, 35.0))
            .angvel(-ANGVEL)
            .build(),
        ColliderBuilder::cuboid(10.0, 1.0).build(),
    );
    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::kinematic_velocity_based()
            .translation(glamx::vec2(65.0, 35.0))
            .angvel(ANGVEL)
            .build(),
        ColliderBuilder::cuboid(1.0, 10.0).build(),
    );
    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::kinematic_velocity_based()
            .translation(glamx::vec2(20.0, 20.0))
            .angvel(-ANGVEL)
            .build(),
        ColliderBuilder::ball(5.0).build(),
    );
    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::kinematic_velocity_based()
            .translation(glamx::vec2(50.0, 20.0))
            .angvel(-ANGVEL)
            .build(),
        ColliderBuilder::capsule_y(5.0, 3.0).build(),
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
