use khal::backend::GpuTimestamps;
use nexus_viewer2d::NexusViewer;
use nexus2d::mpm::solver::{BoundaryCondition, Particle, ParticleModel, SimulationParams};
use nexus2d::prelude::{NexusPipeline, NexusState, RbdCoupling};

use glamx::Vec2;
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
    let separate = BoundaryCondition::separate(1.0);
    let handle = state.insert_rigid_body(body, collider, RbdCoupling::MpmOneWay(separate));
    viewer.insert_shape(handle, &shape, Pose::IDENTITY);
}

pub async fn run(
    viewer: &mut NexusViewer,
    pipeline: &mut NexusPipeline,
) -> anyhow::Result<NexusState> {
    let mut state = NexusState::default();

    let offset_y = 10.0;
    // let cell_width = 0.1;
    let cell_width = 0.2;

    let mut particles = vec![];
    for i in 0..700 {
        for j in 0..700 {
            let position = glamx::vec2(i as f32 + 0.5 + (i / 50) as f32 * 2.0, j as f32 + 0.5)
                * cell_width
                / 2.0
                + Vec2::Y * offset_y;
            let density = 1000.0;
            let radius = cell_width / 4.0;
            let model = ParticleModel::elastic(5.0e6, 0.2);
            particles.push(Particle::new(position, radius, density, model));
        }
    }

    let params = SimulationParams {
        gravity: glamx::vec2(0.0, -9.81) * 2.0,
        padding: 0.0,
        dt: 1.0 / 60.0,
    };
    state.set_mpm_params(viewer.backend(), params, cell_width)?;
    state.set_mpm_substeps(15);
    state.add_particles(viewer.backend(), particles)?;

    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::fixed()
            .translation(glamx::vec2(0.0, -1.0))
            .build(),
        ColliderBuilder::cuboid(1000.0, 1.0).build(),
    );
    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::fixed()
            .translation(glamx::vec2(-20.0, 0.0))
            .rotation(0.5)
            .build(),
        ColliderBuilder::cuboid(1.0, 60.0).build(),
    );
    insert_boundary(
        &mut state,
        viewer,
        RigidBodyBuilder::fixed()
            .translation(glamx::vec2(90.0, 0.0))
            .rotation(-0.5)
            .build(),
        ColliderBuilder::cuboid(1.0, 60.0).build(),
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
